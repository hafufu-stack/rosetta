"""
Phase 33: Compilation Eigenfunctions (Opus Bonus)
====================================================
What are the FIXED POINTS of the compiler?
Find eigenvectors of W_compile: functions that survive compilation unchanged.
These are the "platonic ideals" of software.
"""
import os, json, time
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')


def main():
    print("=" * 60)
    print("Phase 33: Compilation Eigenfunctions (Opus Bonus)")
    print("The platonic ideals of software")
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
    W = Ridge(alpha=1.0).fit(z_ast, z_bc).coef_.T  # (64, 64)

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eig(W)
    # Sort by magnitude
    order = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    print(f"  Top 10 eigenvalues (magnitude):")
    for i in range(10):
        ev = eigenvalues[i]
        print(f"    lambda_{i} = {ev.real:.4f} + {ev.imag:.4f}i "
              f"(|lambda| = {abs(ev):.4f})")

    # Find eigenvalues closest to 1 (fixed points of compilation)
    dist_to_1 = np.abs(eigenvalues - 1.0)
    fixed_point_idx = np.argsort(dist_to_1)[:5]
    print(f"\n  Eigenvalues closest to 1 (fixed points):")
    for idx in fixed_point_idx:
        print(f"    lambda_{idx} = {eigenvalues[idx].real:.4f} "
              f"(dist to 1: {dist_to_1[idx]:.4f})")

    # Decode eigenvectors to see what they look like as code
    import torch, sys
    sys.path.insert(0, BASE_DIR)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
            z_t = torch.tensor(z.real.astype(np.float32)).unsqueeze(0).to(DEVICE)
            tokens = decoder(z_t)
            return decode_tokens(tokens[0].cpu().numpy(), idx2char)

    print(f"\n--- Decoded Eigenvectors (Compilation Eigenfunctions) ---")
    eigen_codes = []
    for i in range(min(10, len(eigenvalues))):
        ev = eigenvectors[:, i]
        code = gen(ev)
        # Also find nearest real function
        cos_sims = np.array([
            np.dot(ev.real, z_ast[j]) /
            (np.linalg.norm(ev.real) * np.linalg.norm(z_ast[j]) + 1e-8)
            for j in range(len(z_ast))
        ])
        nearest_idx = np.argmax(cos_sims)
        nearest_src = dataset[nearest_idx]['source']

        print(f"\n  Eigenfunction {i} (|lambda|={abs(eigenvalues[i]):.3f}):")
        print(f"    Decoded: {code[:50]}")
        print(f"    Nearest: {nearest_src[:50]} (cos={cos_sims[nearest_idx]:.3f})")

        eigen_codes.append({
            'index': i, 'eigenvalue_real': float(eigenvalues[i].real),
            'eigenvalue_imag': float(eigenvalues[i].imag),
            'eigenvalue_mag': float(abs(eigenvalues[i])),
            'decoded': code, 'nearest': nearest_src,
            'nearest_cos': float(cos_sims[nearest_idx]),
        })

    # Which real functions are most "eigen-like"?
    print(f"\n--- Most Eigen-like Real Functions ---")
    # For each function, measure how much W*v is parallel to v
    src_to_idx = {}
    for i, d in enumerate(dataset):
        if d['source'] not in src_to_idx:
            src_to_idx[d['source']] = i

    eigen_scores = {}
    for src, idx in src_to_idx.items():
        v = z_ast[idx]
        Wv = W @ v
        cos = float(np.dot(v, Wv) / (np.linalg.norm(v) * np.linalg.norm(Wv) + 1e-8))
        eigen_scores[src] = cos

    sorted_funcs = sorted(eigen_scores.items(), key=lambda x: -x[1])
    print("  Most compilation-invariant functions:")
    for src, score in sorted_funcs[:5]:
        print(f"    {src[:40]} (eigen-score={score:.4f})")
    print("  Least compilation-invariant:")
    for src, score in sorted_funcs[-3:]:
        print(f"    {src[:40]} (eigen-score={score:.4f})")

    elapsed = time.time() - t0
    results = {
        'phase': 33, 'name': 'Compilation Eigenfunctions',
        'eigenvalues': [{'real': float(e.real), 'imag': float(e.imag),
                        'mag': float(abs(e))} for e in eigenvalues[:10]],
        'eigen_codes': eigen_codes,
        'most_invariant': [{'src': s, 'score': sc} for s, sc in sorted_funcs[:5]],
        'least_invariant': [{'src': s, 'score': sc} for s, sc in sorted_funcs[-3:]],
        'elapsed': elapsed, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase33_eigenfunctions.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Eigenvalue spectrum (complex plane)
    axes[0].scatter([e.real for e in eigenvalues], [e.imag for e in eigenvalues],
                   c=np.abs(eigenvalues), cmap='hot', s=60, edgecolor='black')
    axes[0].axhline(0, color='gray', lw=0.5); axes[0].axvline(0, color='gray', lw=0.5)
    circle = plt.Circle((0,0), 1, fill=False, color='blue', ls='--', label='Unit circle')
    axes[0].add_patch(circle)
    axes[0].set_xlabel('Real'); axes[0].set_ylabel('Imaginary')
    axes[0].set_title('Eigenvalues of W_compile\n(complex plane)', fontweight='bold')
    axes[0].set_aspect('equal')
    axes[0].legend()

    # 2. Eigenvalue magnitudes
    axes[1].bar(range(len(eigenvalues)), np.abs(eigenvalues),
               color='#E91E63', alpha=0.8, edgecolor='black')
    axes[1].axhline(1.0, color='blue', ls='--', label='|lambda|=1 (fixed point)')
    axes[1].set_xlabel('Index'); axes[1].set_ylabel('|eigenvalue|')
    axes[1].set_title('Eigenvalue Magnitudes', fontweight='bold')
    axes[1].legend()

    # 3. Eigen-scores of real functions
    all_scores = list(eigen_scores.values())
    axes[2].hist(all_scores, bins=30, color='#4CAF50', edgecolor='black', alpha=0.8)
    axes[2].set_xlabel('Eigen-score (cos between v and Wv)')
    axes[2].set_ylabel('Count')
    axes[2].set_title('How "eigen-like" are real functions?\n(compilation invariance)',
                     fontweight='bold')

    plt.suptitle('Phase 33: Compilation Eigenfunctions\n'
                 'The platonic ideals of software',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase33_eigenfunctions.png'), dpi=150)
    plt.close()
    print(f"\nPhase 33 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
