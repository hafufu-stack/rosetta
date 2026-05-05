"""
Phase 17: Infinite Code Morphing
==================================
Interpolate between two programs in latent space.
Watch code smoothly transform from add to max.
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
    print("Phase 17: Infinite Code Morphing")
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

    # Try robust decoder first, fall back to original
    decoder = CodeDecoder(64, V, hidden=128, max_len=80).to(DEVICE)
    robust_path = os.path.join(DATA_DIR, 'decoder_robust.pt')
    orig_path = os.path.join(DATA_DIR, 'decoder.pt')
    dec_path = robust_path if os.path.exists(robust_path) else orig_path
    decoder.load_state_dict(torch.load(dec_path, map_location=DEVICE, weights_only=True))
    decoder.eval()
    print(f"Using decoder: {os.path.basename(dec_path)}")

    src_to_idx = {}
    for i, d in enumerate(dataset):
        if d['source'] not in src_to_idx:
            src_to_idx[d['source']] = i

    def gen(z_vec):
        with torch.no_grad():
            z_t = torch.tensor(z_vec, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            tokens = decoder(z_t)
            return decode_tokens(tokens[0].cpu().numpy(), idx2char)

    # Define morphing pairs
    morph_pairs = [
        ("def f(x, y): return x + y", "def f(x, y): return x * y",
         "Addition -> Multiplication"),
        ("def f(x, y): return x + y", "def f(x, y): return x > y",
         "Addition -> Comparison"),
        ("def f(x, y): return x + y", "def f(x, y): return max(x, y)",
         "Addition -> Maximum"),
        ("def f(x): return abs(x)", "def f(x): return x * x",
         "Absolute -> Square"),
        ("def f(s): return s.upper()", "def f(s): return len(s)",
         "String upper -> Length"),
        ("def f(x, y): return x - y", "def f(x, y): return x / y",
         "Subtraction -> Division"),
    ]

    N_STEPS = 11  # 0.0, 0.1, ..., 1.0
    all_morphs = []

    for src_a, src_b, desc in morph_pairs:
        if src_a not in src_to_idx or src_b not in src_to_idx:
            print(f"  SKIP: {desc}")
            continue

        ia, ib = src_to_idx[src_a], src_to_idx[src_b]
        va, vb = z_ast[ia], z_ast[ib]

        print(f"\n  === {desc} ===")
        print(f"  t=0.0: {src_a}")

        morph_seq = []
        for step in range(N_STEPS):
            t = step / (N_STEPS - 1)
            v_interp = (1 - t) * va + t * vb
            code = gen(v_interp)
            morph_seq.append({'t': t, 'code': code})
            print(f"  t={t:.1f}: {code[:60]}")

        print(f"  t=1.0: {src_b} (target)")
        all_morphs.append({
            'desc': desc, 'src_a': src_a, 'src_b': src_b,
            'sequence': morph_seq,
        })

    # Analyze: how many unique codes appear in each morph?
    print("\n--- Morphing Diversity ---")
    for m in all_morphs:
        unique = len(set(s['code'].strip() for s in m['sequence']))
        print(f"  {m['desc']}: {unique} unique codes in {N_STEPS} steps")

    elapsed = time.time() - t0
    results = {
        'phase': 17, 'name': 'Infinite Code Morphing',
        'n_pairs': len(all_morphs), 'n_steps': N_STEPS,
        'morphs': all_morphs,
        'elapsed': elapsed, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase17_code_morphing.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Figure: show morph trajectories
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_morphs = len(all_morphs)
    fig, axes = plt.subplots(n_morphs, 1, figsize=(16, 3 * n_morphs))
    if n_morphs == 1: axes = [axes]

    for ax, m in zip(axes, all_morphs):
        ts = [s['t'] for s in m['sequence']]
        codes = [s['code'][:35] for s in m['sequence']]
        # Color by uniqueness
        unique_codes = list(dict.fromkeys(c.strip() for c in codes))
        color_map = {}
        cmap = plt.cm.tab10
        for ci, uc in enumerate(unique_codes):
            color_map[uc] = cmap(ci % 10)
        colors = [color_map[c.strip()] for c in codes]

        for i, (t, code) in enumerate(zip(ts, codes)):
            ax.barh(t, 1, height=0.08, color=colors[i], edgecolor='black', lw=0.5)
            ax.text(0.02, t, f't={t:.1f}: {code}', va='center', fontsize=8,
                   fontfamily='monospace')
        ax.set_xlim(0, 1); ax.set_ylim(-0.1, 1.1)
        ax.set_title(m['desc'], fontweight='bold', fontsize=11)
        ax.set_yticks([]); ax.set_xticks([])

    plt.suptitle('Phase 17: Infinite Code Morphing\n'
                 'Programs smoothly transform in latent space',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase17_code_morphing.png'), dpi=150)
    plt.close()
    print(f"\nPhase 17 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
