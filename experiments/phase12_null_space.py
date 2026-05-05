"""
Phase 12: Null-Space Obfuscation
==================================
Prove that program meaning lives in the top SVD axes.
Inject massive noise into null-space dimensions and decode.
"""
import os, json, time
import numpy as np
import torch
from sklearn.linear_model import Ridge

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    print("=" * 60)
    print("Phase 12: Null-Space Obfuscation")
    print("=" * 60)
    t0 = time.time()

    # Load
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
    N, D = z_ast.shape

    with open(os.path.join(DATA_DIR, 'char_vocab.json'), 'r') as f:
        vd = json.load(f)
    char2idx, idx2char = vd['char2idx'], {int(i): c for c, i in vd['char2idx'].items()}
    V = len(char2idx)

    from experiments.phase9_generative_decompiler import CodeDecoder, decode_tokens
    decoder = CodeDecoder(64, V, hidden=128, max_len=80).to(DEVICE)
    decoder.load_state_dict(torch.load(os.path.join(DATA_DIR, 'decoder.pt'),
                                        map_location=DEVICE, weights_only=True))
    decoder.eval()

    # Get W_compile SVD
    z_bc = latents['bc']
    n_train = int(N * 0.8)
    perm = np.random.RandomState(42).permutation(N)
    reg = Ridge(alpha=1.0).fit(z_ast[perm[:n_train]], z_bc[perm[:n_train]])
    W = reg.coef_
    U, S, Vt = np.linalg.svd(W)

    # Select test samples
    src_to_idx = {}
    for i, d in enumerate(dataset):
        if d['source'] not in src_to_idx:
            src_to_idx[d['source']] = i
    test_srcs = list(src_to_idx.keys())[:20]
    test_idx = [src_to_idx[s] for s in test_srcs]

    def gen(z_vec):
        with torch.no_grad():
            z_t = torch.tensor(z_vec, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            tokens = decoder(z_t)
            return decode_tokens(tokens[0].cpu().numpy(), idx2char)

    # Experiment: inject noise into null-space (dimensions 5+)
    noise_levels = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
    k_signal = 4  # Top 4 axes = signal space

    results_grid = []
    print(f"\nTesting {len(test_srcs)} functions x {len(noise_levels)} noise levels")
    print(f"Signal space: top {k_signal} SVD axes, Null space: remaining {D-k_signal}")

    for ni, noise_std in enumerate(noise_levels):
        exact, semantic, total = 0, 0, 0
        for ti in test_idx:
            z_orig = z_ast[ti].copy()
            true_src = dataset[ti]['source']

            if noise_std > 0:
                # Project to SVD space
                proj = z_orig @ Vt.T  # (D,)
                # Add noise ONLY to null-space dimensions (k_signal onwards)
                noise = np.zeros(D, dtype=np.float32)
                noise[k_signal:] = np.random.randn(D - k_signal).astype(np.float32) * noise_std
                proj_noisy = proj + noise
                # Project back
                z_noisy = proj_noisy @ Vt
            else:
                z_noisy = z_orig

            gen_code = gen(z_noisy)
            is_exact = gen_code.strip() == true_src.strip()
            # Semantic: same operation even if different var names
            key_part = true_src.split('return ')[-1].strip() if 'return ' in true_src else ''
            key_op = key_part.replace('x','').replace('y','').replace('a','').replace('b','').replace(' ','')
            gen_part = gen_code.split('return ')[-1].strip() if 'return ' in gen_code else ''
            gen_op = gen_part.replace('x','').replace('y','').replace('a','').replace('b','').replace('p','').replace('q','').replace('m','').replace('n','').replace(' ','')
            is_semantic = key_op == gen_op if key_op else False
            if is_exact: exact += 1
            if is_semantic: semantic += 1
            total += 1

        results_grid.append({
            'noise_std': noise_std, 'exact': exact/total,
            'semantic': semantic/total, 'total': total
        })
        print(f"  Noise={noise_std:5.1f}: exact={exact}/{total} ({exact/total:.0%}), "
              f"semantic={semantic}/{total} ({semantic/total:.0%})")

    # Also test: noise in SIGNAL space (should destroy meaning)
    print("\n--- Control: noise in SIGNAL space (should destroy) ---")
    for noise_std in [0.5, 2.0, 10.0]:
        exact, total = 0, 0
        for ti in test_idx:
            z_orig = z_ast[ti].copy()
            proj = z_orig @ Vt.T
            noise = np.zeros(D, dtype=np.float32)
            noise[:k_signal] = np.random.randn(k_signal).astype(np.float32) * noise_std
            z_noisy = (proj + noise) @ Vt
            gen_code = gen(z_noisy)
            if gen_code.strip() == dataset[ti]['source'].strip(): exact += 1
            total += 1
        print(f"  Signal noise={noise_std}: exact={exact}/{total} ({exact/total:.0%})")

    elapsed = time.time() - t0
    results = {
        'phase': 12, 'name': 'Null-Space Obfuscation',
        'signal_dims': k_signal, 'null_dims': D - k_signal,
        'noise_results': results_grid,
        'elapsed': elapsed, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase12_null_space.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Figure
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ns = [r['noise_std'] for r in results_grid]
    ax.plot(ns, [r['exact'] for r in results_grid], 'o-', color='#E91E63',
            lw=2, ms=8, label='Exact Match')
    ax.plot(ns, [r['semantic'] for r in results_grid], 's-', color='#4CAF50',
            lw=2, ms=8, label='Semantic Match')
    ax.set_xlabel('Null-Space Noise Magnitude', fontsize=13)
    ax.set_ylabel('Accuracy', fontsize=13)
    ax.set_title('Phase 12: Null-Space Obfuscation\n'
                 f'Meaning lives in top {k_signal} SVD axes',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=12); ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase12_null_space.png'), dpi=150)
    plt.close()
    print(f"\nPhase 12 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
