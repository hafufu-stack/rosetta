"""
Phase 19: Latent Function Composition
=======================================
Add function vectors to create composite functions.
add_vec + abs_vec -> abs(x + y)?
"""
import os, json, time
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    print("=" * 60)
    print("Phase 19: Latent Function Composition")
    print("=" * 60)
    t0 = time.time()

    ds_file = os.path.join(DATA_DIR, 'rosetta_dataset_v2.json')
    if not os.path.exists(ds_file):
        ds_file = os.path.join(DATA_DIR, 'rosetta_dataset.json')
    with open(ds_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)['dataset']

    lat_file = os.path.join(DATA_DIR, 'rosetta_latents_v2.npz')
    if not os.path.exists(lat_file):
        lat_file = os.path.join(DATA_DIR, 'rosetta_latents.npz')
    latents = np.load(lat_file)
    z_ast = latents['ast']

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

    src_to_idx = {}
    for i, d in enumerate(dataset):
        if d['source'] not in src_to_idx:
            src_to_idx[d['source']] = i

    def gen(z_vec):
        with torch.no_grad():
            z_t = torch.tensor(z_vec, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            tokens = decoder(z_t)
            return decode_tokens(tokens[0].cpu().numpy(), idx2char)

    # Composition experiments
    compositions = [
        ("def f(x, y): return x + y", "def f(x): return abs(x)",
         "add + abs -> abs(x+y)?", ["abs"]),
        ("def f(x, y): return x - y", "def f(x): return abs(x)",
         "sub + abs -> abs(x-y)?", ["abs"]),
        ("def f(x, y): return x + y", "def f(x): return x * x",
         "add + square -> (x+y)^2?", ["*"]),
        ("def f(x, y): return x * y", "def f(x): return -x",
         "mul + negate -> -(x*y)?", ["-"]),
        ("def f(s): return s.upper()", "def f(s): return len(s)",
         "upper + len -> len(upper(s))?", ["len"]),
        ("def f(x, y): return x + y", "def f(x, y): return x - y",
         "add + sub -> identity/zero?", []),
        ("def f(x, y): return x > y", "def f(x): return not x",
         "gt + not -> x <= y?", ["<="]),
    ]

    comp_results = []
    print("\n--- Function Composition via Vector Addition ---")

    for src_a, src_b, desc, expected_tokens in compositions:
        if src_a not in src_to_idx or src_b not in src_to_idx:
            print(f"  SKIP: {desc}")
            continue

        va = z_ast[src_to_idx[src_a]]
        vb = z_ast[src_to_idx[src_b]]

        # Method 1: Vector addition
        v_sum = va + vb
        code_sum = gen(v_sum)

        # Method 2: Vector average
        v_avg = (va + vb) / 2
        code_avg = gen(v_avg)

        # Method 3: Weighted (0.7 * a + 0.3 * b)
        v_weighted = 0.7 * va + 0.3 * vb
        code_weighted = gen(v_weighted)

        # Check if any expected tokens appear
        has_expected = any(t in code_sum for t in expected_tokens) if expected_tokens else True

        print(f"\n  {desc}")
        print(f"    A: {src_a}")
        print(f"    B: {src_b}")
        print(f"    Sum:      {code_sum[:60]}")
        print(f"    Average:  {code_avg[:60]}")
        print(f"    Weighted: {code_weighted[:60]}")

        comp_results.append({
            'desc': desc, 'src_a': src_a, 'src_b': src_b,
            'sum': code_sum, 'avg': code_avg, 'weighted': code_weighted,
            'has_expected': has_expected,
        })

    # Bonus: Subtraction as anti-composition
    print("\n--- Anti-Composition (Vector Subtraction) ---")
    anti_pairs = [
        ("def f(x, y): return x + y", "def f(x, y): return x - y",
         "add - sub -> ?"),
        ("def f(x, y): return x * y", "def f(x, y): return x + y",
         "mul - add -> ?"),
    ]
    for src_a, src_b, desc in anti_pairs:
        if src_a not in src_to_idx or src_b not in src_to_idx:
            continue
        v_diff = z_ast[src_to_idx[src_a]] - z_ast[src_to_idx[src_b]]
        code = gen(v_diff)
        print(f"  {desc}: {code[:60]}")

    elapsed = time.time() - t0
    n_expected = sum(1 for r in comp_results if r['has_expected'])
    results = {
        'phase': 19, 'name': 'Latent Function Composition',
        'total': len(comp_results), 'has_expected': n_expected,
        'details': comp_results,
        'elapsed': elapsed, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase19_composition.json'), 'w') as f:
        json.dump(results, f, indent=2)

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    descs = [r['desc'][:30] for r in comp_results]
    colors = ['#4CAF50' if r['has_expected'] else '#FF9800' for r in comp_results]
    ax.barh(descs, [1]*len(comp_results), color=colors, edgecolor='black')
    for i, r in enumerate(comp_results):
        ax.text(0.02, i, f"= {r['sum'][:40]}", va='center', fontsize=9,
               fontfamily='monospace')
    ax.set_title('Phase 19: Latent Function Composition\n'
                 'Vector(A) + Vector(B) = Composite Code',
                 fontsize=14, fontweight='bold')
    ax.set_xticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase19_composition.png'), dpi=150)
    plt.close()
    print(f"\nPhase 19 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
