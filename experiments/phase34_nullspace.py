"""
Phase 34: The Compiler's Null Space
======================================
What does the compiler THROW AWAY?
The null space of W_compile = information invisible to binary.
Decode null-space vectors to reveal what the compiler ignores.
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
    print("Phase 34: The Compiler's Null Space")
    print("What does the compiler throw away?")
    print("=" * 60)
    t0 = time.time()

    lat_file = os.path.join(DATA_DIR, 'rosetta_latents_v2.npz')
    if not os.path.exists(lat_file):
        lat_file = os.path.join(DATA_DIR, 'rosetta_latents.npz')
    latents = np.load(lat_file)
    z_ast, z_bc = latents['ast'], latents['bc']

    ds_file = os.path.join(DATA_DIR, 'rosetta_dataset_v2.json')
    if not os.path.exists(ds_file):
        ds_file = os.path.join(DATA_DIR, 'rosetta_dataset.json')
    with open(ds_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)['dataset']

    # Fit W_compile
    from sklearn.linear_model import Ridge
    reg = Ridge(alpha=1.0).fit(z_ast, z_bc)
    W = reg.coef_  # (64, 64): maps AST -> Binary

    # SVD decomposition
    U, S, Vt = np.linalg.svd(W, full_matrices=True)
    print(f"  Singular values: {[f'{s:.3f}' for s in S[:10]]}")
    print(f"  Min singular value: {S[-1]:.6f}")

    # Null space = right singular vectors with near-zero singular values
    threshold = S[0] * 0.01  # 1% of max
    null_dims = np.where(S < threshold)[0]
    signal_dims = np.where(S >= threshold)[0]
    print(f"\n  Signal dimensions (S >= {threshold:.4f}): {len(signal_dims)}")
    print(f"  Near-null dimensions (S < {threshold:.4f}): {len(null_dims)}")

    # The null-space basis vectors (in AST space)
    null_basis = Vt[null_dims]  # (n_null, 64)
    signal_basis = Vt[signal_dims]

    # Load decoder
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

    # === Experiment 1: Decode null-space directions ===
    print("\n--- Decoded Null-Space Directions ---")
    print("  (What the compiler considers irrelevant)")
    null_decoded = []
    for i, nv in enumerate(null_basis[:8]):
        code = gen(nv)
        print(f"  Null dim {null_dims[i]}: {code[:50]}")
        null_decoded.append({'dim': int(null_dims[i]), 'sv': float(S[null_dims[i]]),
                           'code': code})

    # === Experiment 2: Add null-space noise to real functions ===
    print("\n--- Null-Space Injection Test ---")
    print("  Adding null-space vectors to functions:")
    print("  (should change source but NOT binary)")

    src_to_idx = {}
    for i, d in enumerate(dataset):
        if d['source'] not in src_to_idx:
            src_to_idx[d['source']] = i

    test_srcs = list(src_to_idx.keys())[:5]
    injection_results = []

    for src in test_srcs:
        idx = src_to_idx[src]
        v_orig = z_ast[idx]

        # Binary from original
        bc_orig = W @ v_orig

        for scale in [0.5, 1.0, 2.0]:
            if len(null_basis) == 0:
                break
            # Add null-space perturbation
            null_dir = null_basis[0]
            v_perturbed = v_orig + scale * null_dir

            # Binary from perturbed (should be same!)
            bc_perturbed = W @ v_perturbed
            bc_diff = float(np.linalg.norm(bc_orig - bc_perturbed))

            # But decode should differ
            code_orig = gen(v_orig)
            code_pert = gen(v_perturbed)
            same_code = code_orig.strip() == code_pert.strip()

            print(f"  {src[:35]} + {scale:.1f}*null:")
            print(f"    Binary change: {bc_diff:.6f}")
            print(f"    Code: {code_orig[:25]} -> {code_pert[:25]} "
                  f"({'SAME' if same_code else 'DIFFERENT'})")
            injection_results.append({
                'src': src, 'scale': scale,
                'binary_change': bc_diff, 'same_code': same_code,
                'original': code_orig, 'perturbed': code_pert,
            })

    # === Experiment 3: Project functions onto signal vs null ===
    print("\n--- Signal vs Null Energy Distribution ---")
    signal_energies = []
    null_energies = []
    for src, idx in list(src_to_idx.items())[:50]:
        v = z_ast[idx]
        # Project onto signal space
        sig_proj = sum((np.dot(v, Vt[d])**2) for d in signal_dims)
        null_proj = sum((np.dot(v, Vt[d])**2) for d in null_dims)
        total = sig_proj + null_proj
        signal_energies.append(sig_proj / total if total > 0 else 0)
        null_energies.append(null_proj / total if total > 0 else 0)

    avg_signal = float(np.mean(signal_energies))
    avg_null = float(np.mean(null_energies))
    print(f"  Avg energy in signal space: {avg_signal:.3f}")
    print(f"  Avg energy in null space:   {avg_null:.3f}")

    elapsed = time.time() - t0
    results = {
        'phase': 34, 'name': "Compiler's Null Space",
        'n_signal': len(signal_dims), 'n_null': len(null_dims),
        'threshold': float(threshold),
        'null_decoded': null_decoded,
        'avg_signal_energy': avg_signal, 'avg_null_energy': avg_null,
        'injection_results': injection_results[:6],
        'elapsed': elapsed, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase34_nullspace.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Singular value spectrum with threshold
    axes[0].bar(range(len(S)), S, color='#2196F3', alpha=0.8, edgecolor='black')
    axes[0].axhline(threshold, color='red', ls='--', lw=2,
                   label=f'Null threshold ({threshold:.4f})')
    axes[0].set_xlabel('Dimension')
    axes[0].set_ylabel('Singular Value')
    axes[0].set_title(f'W_compile Spectrum\n{len(signal_dims)} signal, '
                     f'{len(null_dims)} null', fontweight='bold')
    axes[0].legend()

    # 2. Signal vs Null energy
    axes[1].bar(['Signal Space', 'Null Space'], [avg_signal, avg_null],
               color=['#4CAF50', '#F44336'], edgecolor='black')
    for i, v in enumerate([avg_signal, avg_null]):
        axes[1].text(i, v+0.01, f'{v:.3f}', ha='center', fontweight='bold', fontsize=14)
    axes[1].set_ylabel('Fraction of Energy')
    axes[1].set_title('Where do real programs live?\n(signal vs null space)',
                     fontweight='bold')
    axes[1].set_ylim(0, 1.1)

    # 3. Binary change under null injection
    if injection_results:
        scales = [r['scale'] for r in injection_results[:3]]
        changes = [r['binary_change'] for r in injection_results[:3]]
        axes[2].bar([f's={s}' for s in scales], changes,
                   color='#FF9800', edgecolor='black')
        axes[2].set_ylabel('Binary Vector Change (L2)')
        axes[2].set_title('Null-Space Injection\n(should be ~0 if true null)',
                         fontweight='bold')

    plt.suptitle("Phase 34: The Compiler's Null Space\n"
                 "What information does compilation destroy?",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase34_nullspace.png'), dpi=150)
    plt.close()
    print(f"\nPhase 34 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
