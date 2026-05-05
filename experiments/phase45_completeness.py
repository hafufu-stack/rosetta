"""
Phase 45: Latent Arithmetic Completeness
==========================================
Can we derive complex operations from simple ones?
Test: can we get `power` from `multiply` via a
"self-application" operator in vector space?
Is the Rosetta algebra Turing-complete?
"""
import os, json, time, sys
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    print("=" * 60)
    print("Phase 45: Latent Arithmetic Completeness")
    print("Can vector algebra derive all operations?")
    print("=" * 60)
    t0 = time.time()

    lat_file = os.path.join(DATA_DIR, 'rosetta_latents_v2.npz')
    if not os.path.exists(lat_file):
        lat_file = os.path.join(DATA_DIR, 'rosetta_latents.npz')
    latents = np.load(lat_file)
    z_ast = latents['ast']

    ds_file = os.path.join(DATA_DIR, 'rosetta_dataset_v2.json')
    if not os.path.exists(ds_file):
        ds_file = os.path.join(DATA_DIR, 'rosetta_dataset.json')
    with open(ds_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)['dataset']

    sys.path.insert(0, BASE_DIR)
    with open(os.path.join(DATA_DIR, 'char_vocab.json'), 'r') as f:
        vd = json.load(f)
    idx2char = {int(i): c for c, i in vd['char2idx'].items()}
    V = len(vd['char2idx'])
    from experiments.phase9_generative_decompiler import CodeDecoder, decode_tokens
    decoder = CodeDecoder(64, V, hidden=128, max_len=80).to(DEVICE)
    dec_path = os.path.join(DATA_DIR, 'decoder_robust.pt')
    if not os.path.exists(dec_path):
        dec_path = os.path.join(DATA_DIR, 'decoder.pt')
    decoder.load_state_dict(torch.load(dec_path, map_location=DEVICE, weights_only=True))
    decoder.eval()

    def gen(z):
        with torch.no_grad():
            z_t = torch.tensor(z.astype(np.float32)).unsqueeze(0).to(DEVICE)
            tokens = decoder(z_t)
            return decode_tokens(tokens[0].cpu().numpy(), idx2char)

    src_to_idx = {}
    for i, d in enumerate(dataset):
        if d['source'] not in src_to_idx:
            src_to_idx[d['source']] = i

    def get_vec(src):
        if src in src_to_idx:
            return z_ast[src_to_idx[src]]
        return None

    # === Test 1: Derivation via analogy ===
    print("\n--- Analogy-Based Derivation ---")
    # Can we derive new operations from known relationships?
    # add:sub :: mul:? -> should give div
    # add:mul :: sub:? -> should give div

    analogies = [
        ("def f(x, y): return x + y", "def f(x, y): return x - y",
         "def f(x, y): return x * y", "def f(x, y): return x / y",
         "add:sub :: mul:?=div"),

        ("def f(x, y): return x + y", "def f(x, y): return x * y",
         "def f(x, y): return x - y", "def f(x, y): return x / y",
         "add:mul :: sub:?=div"),

        ("def f(x, y): return x + y", "def f(x): return abs(x)",
         "def f(x, y): return x - y", "def f(x, y): return abs(x - y)",
         "add:abs :: sub:?=abs(sub)"),

        ("def f(x): return -x", "def f(x): return abs(x)",
         "def f(x, y): return x - y", "def f(x, y): return abs(x - y)",
         "neg:abs :: sub:?=abs(sub)"),
    ]

    analogy_results = []
    for src_a, src_b, src_c, expected, desc in analogies:
        va, vb, vc = get_vec(src_a), get_vec(src_b), get_vec(src_c)
        ve = get_vec(expected)
        if va is None or vb is None or vc is None:
            continue

        # Analogy: a:b :: c:? => ? = c + (b - a)
        v_predicted = vc + (vb - va)
        code = gen(v_predicted)

        if ve is not None:
            cos = float(np.dot(v_predicted, ve) /
                       (np.linalg.norm(v_predicted) * np.linalg.norm(ve) + 1e-8))
        else:
            cos = 0.0

        print(f"  {desc}")
        print(f"    Predicted: {code[:45]} (cos to target: {cos:.3f})")

        analogy_results.append({
            'desc': desc, 'predicted': code, 'expected': expected,
            'cos': cos,
        })

    # === Test 2: Composition via addition ===
    print("\n--- Composition via Vector Addition ---")
    compositions = [
        # f + g -> should give f(g(x))
        ("def f(x): return abs(x)", "def f(x, y): return x + y",
         "abs + add = abs(add)?"),
        ("def f(x): return -x", "def f(x): return abs(x)",
         "neg + abs = ?"),
        ("def f(x, y): return x + y", "def f(x, y): return x + y",
         "add + add = double?"),
        ("def f(x, y): return x * y", "def f(x, y): return x * y",
         "mul + mul = square?"),
    ]

    comp_results = []
    for src_a, src_b, desc in compositions:
        va, vb = get_vec(src_a), get_vec(src_b)
        if va is None or vb is None:
            continue

        v_sum = va + vb
        v_sum_norm = v_sum / (np.linalg.norm(v_sum) + 1e-8) * np.linalg.norm(va)
        code = gen(v_sum_norm)
        print(f"  {desc}: {code[:45]}")
        comp_results.append({'desc': desc, 'result': code})

    # === Test 3: Negation as subtraction from zero ===
    print("\n--- Algebraic Identities ---")
    identities = []

    # Identity: f - f = zero function?
    add_vec = get_vec("def f(x, y): return x + y")
    if add_vec is not None:
        zero_vec = add_vec - add_vec  # Should be zero
        code_zero = gen(zero_vec)
        print(f"  add - add = {code_zero[:40]}")
        identities.append({'desc': 'add - add', 'result': code_zero})

    # f + (-f): should also give zero
    neg_vec = get_vec("def f(x): return -x")
    abs_vec = get_vec("def f(x): return abs(x)")
    if neg_vec is not None and abs_vec is not None:
        v = neg_vec + abs_vec
        code = gen(v)
        print(f"  neg + abs = {code[:40]}")
        identities.append({'desc': 'neg + abs', 'result': code})

    # 2*add - sub = ?
    sub_vec = get_vec("def f(x, y): return x - y")
    if add_vec is not None and sub_vec is not None:
        v = 2 * add_vec - sub_vec
        code = gen(v)
        print(f"  2*add - sub = {code[:40]}")
        identities.append({'desc': '2*add - sub', 'result': code})

    # === Test 4: Can we reach ALL functions from a basis? ===
    print("\n--- Spanning Test ---")
    # Use {add, mul, gt, abs, upper} as basis. Can we reconstruct all others?
    basis_srcs = [
        "def f(x, y): return x + y", "def f(x, y): return x * y",
        "def f(x, y): return x > y", "def f(x): return abs(x)",
        "def f(s): return s.upper()",
    ]
    basis_vecs = [z_ast[src_to_idx[s]] for s in basis_srcs if s in src_to_idx]
    B = np.array(basis_vecs)  # (5, 64)

    # For each function, find the best linear combination of basis
    from sklearn.linear_model import LinearRegression
    unique_vecs = np.array([z_ast[src_to_idx[s]] for s in list(src_to_idx.keys())])
    reg = LinearRegression().fit(B.T, unique_vecs.T)  # This is wrong shape

    # Actually: for each target, solve min ||B^T w - target||
    n_well_approximated = 0
    total = min(50, len(src_to_idx))
    for si, (src, idx) in enumerate(list(src_to_idx.items())[:total]):
        target = z_ast[idx]
        # Least squares: B^T @ w = target
        w, res, _, _ = np.linalg.lstsq(B.T, target, rcond=None)
        recon = B.T @ w
        cos = float(np.dot(target, recon) /
                    (np.linalg.norm(target) * np.linalg.norm(recon) + 1e-8))
        if cos > 0.9:
            n_well_approximated += 1

    span_ratio = n_well_approximated / total
    print(f"  5-basis spanning: {n_well_approximated}/{total} ({span_ratio:.0%}) within cos>0.9")

    elapsed = time.time() - t0
    results = {
        'phase': 45, 'name': 'Latent Arithmetic Completeness',
        'analogies': analogy_results,
        'compositions': comp_results,
        'identities': identities,
        'span_ratio': float(span_ratio),
        'elapsed': elapsed, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase45_completeness.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Analogy accuracy
    if analogy_results:
        descs = [a['desc'][:20] for a in analogy_results]
        cos_vals = [a['cos'] for a in analogy_results]
        colors = ['#4CAF50' if c > 0.5 else '#F44336' for c in cos_vals]
        axes[0].barh(descs, cos_vals, color=colors, edgecolor='black')
        axes[0].set_xlabel('Cosine to Target')
        axes[0].set_title('Analogy Derivation\n(a:b :: c:? = target?)', fontweight='bold')
        axes[0].set_xlim(-1, 1)

    # 2. Composition results
    if comp_results:
        axes[1].text(0.1, 0.9, 'Composition Results:', fontsize=12, fontweight='bold',
                    transform=axes[1].transAxes, va='top')
        for ci, cr in enumerate(comp_results):
            axes[1].text(0.1, 0.75 - ci*0.15, f"{cr['desc']}\n  = {cr['result'][:30]}",
                        fontsize=9, transform=axes[1].transAxes, va='top')
        axes[1].set_title('Vector Composition', fontweight='bold')
        axes[1].axis('off')

    # 3. Spanning test
    axes[2].bar(['5-basis\nSpanning'], [span_ratio], color='#9C27B0', edgecolor='black')
    axes[2].text(0, span_ratio+0.05, f'{span_ratio:.0%}', ha='center',
                fontweight='bold', fontsize=16)
    axes[2].set_ylabel('Fraction of functions')
    axes[2].set_title('Can 5 functions span all software?', fontweight='bold')
    axes[2].set_ylim(0, 1.2)

    plt.suptitle('Phase 45: Latent Arithmetic Completeness\n'
                 'Is the Rosetta algebra complete?',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase45_completeness.png'), dpi=150)
    plt.close()
    print(f"\nPhase 45 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
