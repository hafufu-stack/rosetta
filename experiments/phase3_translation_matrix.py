"""
Phase 3: The Translation Matrix
================================
Discover the linear transform W such that V_PL x W ~ V_Bin.

If compiltion can be approximated by a single matrix multiplication
in latent space, this proves that the "compiler" is a simple linear
map in the AI's high-dimensional representation.
"""
import os, json, time
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')


def main():
    print("=" * 60)
    print("Phase 3: The Translation Matrix")
    print("=" * 60)
    t0 = time.time()

    # Load latent vectors from Phase 2
    latents = np.load(os.path.join(DATA_DIR, 'rosetta_latents.npz'))
    z_nl = latents['nl']
    z_ast = latents['ast']
    z_bc = latents['bc']
    labels = latents['labels']

    N, D = z_nl.shape
    print(f"Loaded {N} vectors of dim {D}")

    # Split train/test (80/20)
    n_train = int(N * 0.8)
    perm = np.random.RandomState(42).permutation(N)
    train_idx, test_idx = perm[:n_train], perm[n_train:]

    results_matrices = {}

    # === 1. PL -> Bin (The Compiler Matrix) ===
    print("\n--- PL -> Bin (The Compiler Matrix) ---")
    reg_pl_bin = Ridge(alpha=1.0)
    reg_pl_bin.fit(z_ast[train_idx], z_bc[train_idx])
    z_bc_pred = reg_pl_bin.predict(z_ast[test_idx])

    r2_pl_bin = r2_score(z_bc[test_idx], z_bc_pred)
    cos_sims_pl_bin = [float(np.dot(z_bc_pred[i], z_bc[test_idx][i]) /
                       (np.linalg.norm(z_bc_pred[i]) * np.linalg.norm(z_bc[test_idx][i]) + 1e-8))
                       for i in range(len(test_idx))]
    mean_cos_pl_bin = float(np.mean(cos_sims_pl_bin))
    W_compile = reg_pl_bin.coef_  # D x D matrix!

    print(f"  R2 score: {r2_pl_bin:.4f}")
    print(f"  Mean cosine similarity: {mean_cos_pl_bin:.4f}")
    print(f"  W_compile shape: {W_compile.shape}")
    print(f"  W_compile rank: {np.linalg.matrix_rank(W_compile)}")
    results_matrices['pl_to_bin'] = {
        'r2': float(r2_pl_bin), 'cos_sim': mean_cos_pl_bin,
        'rank': int(np.linalg.matrix_rank(W_compile)),
    }

    # === 2. NL -> Bin (The "Intent to Execution" Matrix) ===
    print("\n--- NL -> Bin (Intent to Execution) ---")
    reg_nl_bin = Ridge(alpha=1.0)
    reg_nl_bin.fit(z_nl[train_idx], z_bc[train_idx])
    z_bc_pred2 = reg_nl_bin.predict(z_nl[test_idx])
    r2_nl_bin = r2_score(z_bc[test_idx], z_bc_pred2)
    cos_sims_nl_bin = [float(np.dot(z_bc_pred2[i], z_bc[test_idx][i]) /
                       (np.linalg.norm(z_bc_pred2[i]) * np.linalg.norm(z_bc[test_idx][i]) + 1e-8))
                       for i in range(len(test_idx))]
    mean_cos_nl_bin = float(np.mean(cos_sims_nl_bin))
    W_intent = reg_nl_bin.coef_

    print(f"  R2 score: {r2_nl_bin:.4f}")
    print(f"  Mean cosine similarity: {mean_cos_nl_bin:.4f}")
    results_matrices['nl_to_bin'] = {
        'r2': float(r2_nl_bin), 'cos_sim': mean_cos_nl_bin,
        'rank': int(np.linalg.matrix_rank(W_intent)),
    }

    # === 3. NL -> PL (The "Understanding" Matrix) ===
    print("\n--- NL -> PL (Understanding) ---")
    reg_nl_pl = Ridge(alpha=1.0)
    reg_nl_pl.fit(z_nl[train_idx], z_ast[train_idx])
    z_ast_pred = reg_nl_pl.predict(z_nl[test_idx])
    r2_nl_pl = r2_score(z_ast[test_idx], z_ast_pred)
    cos_sims_nl_pl = [float(np.dot(z_ast_pred[i], z_ast[test_idx][i]) /
                      (np.linalg.norm(z_ast_pred[i]) * np.linalg.norm(z_ast[test_idx][i]) + 1e-8))
                      for i in range(len(test_idx))]
    mean_cos_nl_pl = float(np.mean(cos_sims_nl_pl))

    print(f"  R2 score: {r2_nl_pl:.4f}")
    print(f"  Mean cosine similarity: {mean_cos_nl_pl:.4f}")
    results_matrices['nl_to_pl'] = {
        'r2': float(r2_nl_pl), 'cos_sim': mean_cos_nl_pl,
    }

    # === 4. Transitivity test: NL->PL->Bin vs NL->Bin directly ===
    print("\n--- Transitivity: NL -> PL -> Bin vs NL -> Bin ---")
    z_ast_from_nl = reg_nl_pl.predict(z_nl[test_idx])
    z_bc_transit = reg_pl_bin.predict(z_ast_from_nl)
    cos_sims_transit = [float(np.dot(z_bc_transit[i], z_bc[test_idx][i]) /
                        (np.linalg.norm(z_bc_transit[i]) * np.linalg.norm(z_bc[test_idx][i]) + 1e-8))
                        for i in range(len(test_idx))]
    mean_cos_transit = float(np.mean(cos_sims_transit))
    print(f"  Transitive (NL->PL->Bin) cosine: {mean_cos_transit:.4f}")
    print(f"  Direct (NL->Bin) cosine:          {mean_cos_nl_bin:.4f}")

    # === 5. Singular value decomposition of W_compile ===
    print("\n--- SVD of W_compile ---")
    U, S, Vt = np.linalg.svd(W_compile)
    energy_90 = np.searchsorted(np.cumsum(S**2) / np.sum(S**2), 0.9) + 1
    energy_95 = np.searchsorted(np.cumsum(S**2) / np.sum(S**2), 0.95) + 1
    print(f"  Singular values: top 5 = {S[:5].round(3)}")
    print(f"  90% energy in {energy_90} components (of {D})")
    print(f"  95% energy in {energy_95} components (of {D})")

    elapsed = time.time() - t0
    results = {
        'phase': 3,
        'name': 'The Translation Matrix',
        'matrices': results_matrices,
        'transitivity': {
            'transitive_cos': mean_cos_transit,
            'direct_cos': mean_cos_nl_bin,
        },
        'svd': {
            'top5_singular_values': S[:5].tolist(),
            'energy_90pct_components': int(energy_90),
            'energy_95pct_components': int(energy_95),
        },
        'elapsed': elapsed,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase3_translation_matrix.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Save W_compile for Phase 4
    np.save(os.path.join(DATA_DIR, 'W_compile.npy'), W_compile)

    # Visualization
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # R2 scores
    pairs = ['PL->Bin\n(Compiler)', 'NL->Bin\n(Intent)', 'NL->PL\n(Understanding)']
    r2s = [r2_pl_bin, r2_nl_bin, r2_nl_pl]
    bars = axes[0].bar(pairs, r2s, color=['#E91E63','#2196F3','#4CAF50'],
                       edgecolor='black')
    for b, v in zip(bars, r2s):
        y = max(0, v) + 0.02
        axes[0].text(b.get_x()+b.get_width()/2, y, f'{v:.3f}',
                     ha='center', fontweight='bold', fontsize=13)
    axes[0].set_ylabel('R2 Score', fontsize=12)
    axes[0].set_title('Translation Matrix Quality', fontsize=13, fontweight='bold')
    axes[0].axhline(y=0, color='gray', ls='--', alpha=0.5)

    # Cosine similarities
    cos_vals = [mean_cos_pl_bin, mean_cos_nl_bin, mean_cos_nl_pl, mean_cos_transit]
    cos_labels = ['PL->Bin', 'NL->Bin', 'NL->PL', 'NL->PL->Bin\n(Transit)']
    bars2 = axes[1].bar(cos_labels, cos_vals,
                        color=['#E91E63','#2196F3','#4CAF50','#FF9800'],
                        edgecolor='black')
    for b, v in zip(bars2, cos_vals):
        axes[1].text(b.get_x()+b.get_width()/2, v+0.02, f'{v:.3f}',
                     ha='center', fontweight='bold', fontsize=11)
    axes[1].set_ylabel('Cosine Similarity', fontsize=12)
    axes[1].set_title('Cross-Modal Alignment', fontsize=13, fontweight='bold')
    axes[1].set_ylim(0, 1.1)

    # Singular values
    axes[2].plot(S, 'o-', color='#9C27B0', markersize=3)
    axes[2].axvline(x=energy_90, color='red', ls='--', alpha=0.7,
                    label=f'90% energy ({energy_90} dims)')
    axes[2].set_xlabel('Component', fontsize=12)
    axes[2].set_ylabel('Singular Value', fontsize=12)
    axes[2].set_title('W_compile Spectrum', fontsize=13, fontweight='bold')
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Phase 3: The Translation Matrix (Continuous Compiler)',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase3_translation_matrix.png'), dpi=150)
    plt.close()

    print(f"\nPhase 3 complete in {elapsed:.1f}s")
    return results


if __name__ == '__main__':
    main()
