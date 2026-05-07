"""Phase 142: Conformal Cyclic Cosmology & Big Bounce
Apply conformal transformation to the collapsed universe.
Does a new Aeon emerge with different physical laws?
"""
import os, json, sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXP2_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(EXP2_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')

def main():
    print("=" * 60)
    print("Phase 142: Conformal Cyclic Cosmology")
    print("  Big Crunch -> Conformal Bounce -> New Aeon")
    print("=" * 60)

    latents = np.load(os.path.join(DATA_DIR, 'rosetta_latents_v2.npz'))
    dataset = json.load(open(os.path.join(DATA_DIR, 'rosetta_dataset_v2.json'), 'r', encoding='utf-8'))
    ast_vectors = latents['ast']
    bc_vectors = latents['bc']
    sources = [item['source'] for item in dataset['dataset']]

    func_ast = {}
    for i, src in enumerate(sources):
        if src not in func_ast: func_ast[src] = []
        func_ast[src].append(ast_vectors[i])
    unique_funcs = list(func_ast.keys())
    ast_m = np.array([np.mean(func_ast[f], axis=0) for f in unique_funcs])
    n = len(unique_funcs)
    centroid = np.mean(ast_m, axis=0)

    # 1. Big Crunch: collapse all vectors to singularity
    print("--- Phase 1: Big Crunch ---")
    crunch_factors = [0.5, 0.1, 0.01, 0.001]
    for cf in crunch_factors:
        collapsed = centroid + (ast_m - centroid) * cf
        spread = np.mean(np.linalg.norm(collapsed - centroid, axis=1))
        print(f"  Crunch factor {cf}: spread={spread:.6f}")

    singularity = centroid.copy()  # Everything at one point

    # 2. Conformal transformation: rescale distances without changing angles
    print("\n--- Phase 2: Conformal Bounce ---")
    # Conformal map: x -> x / ||x||^2 (inversion through unit sphere)
    centered = ast_m - centroid
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    norms_safe = np.maximum(norms, 1e-10)

    # Kelvin transform (conformal inversion)
    inverted = centered / (norms_safe ** 2)
    new_aeon = inverted + centroid

    # Measure new universe properties
    new_spread = np.mean(np.linalg.norm(new_aeon - np.mean(new_aeon, axis=0), axis=1))
    old_spread = np.mean(np.linalg.norm(ast_m - centroid, axis=1))
    print(f"  Old universe spread: {old_spread:.4f}")
    print(f"  New Aeon spread: {new_spread:.4f}")
    print(f"  Expansion ratio: {new_spread/old_spread:.2f}x")

    # 3. Compare physics of old vs new Aeon
    # Gravity law: does the exponent change?
    old_dists = cdist(ast_m[:50], ast_m[:50])
    new_dists = cdist(new_aeon[:50], new_aeon[:50])

    # Structure preservation (are neighbors preserved?)
    old_nn = np.argsort(old_dists, axis=1)[:, 1:6]
    new_nn = np.argsort(new_dists, axis=1)[:, 1:6]
    nn_overlap = np.mean([len(set(old_nn[i]) & set(new_nn[i])) / 5 for i in range(50)])
    print(f"  Neighbor preservation (old->new): {nn_overlap:.2%}")

    # Topology change: do the fundamental groups change?
    old_cos = ast_m[:50] @ ast_m[:50].T
    new_cos = new_aeon[:50] @ new_aeon[:50].T
    old_cos /= (np.linalg.norm(ast_m[:50], axis=1, keepdims=True) @ np.linalg.norm(ast_m[:50], axis=1, keepdims=True).T + 1e-10)
    new_cos /= (np.linalg.norm(new_aeon[:50], axis=1, keepdims=True) @ np.linalg.norm(new_aeon[:50], axis=1, keepdims=True).T + 1e-10)
    angle_preservation = np.corrcoef(old_cos.ravel(), new_cos.ravel())[0, 1]
    print(f"  Angle preservation (conformal): {angle_preservation:.4f}")

    # 4. New Aeon physics
    # Compile matrix in new Aeon
    bc_m = np.array([np.mean([bc_vectors[j] for j, s in enumerate(sources) if s == f], axis=0) for f in unique_funcs])
    bc_inverted = (bc_m - np.mean(bc_m, axis=0))
    bc_norms = np.linalg.norm(bc_inverted, axis=1, keepdims=True)
    bc_norms_safe = np.maximum(bc_norms, 1e-10)
    new_bc = bc_inverted / (bc_norms_safe ** 2) + np.mean(bc_m, axis=0)

    W_old = bc_m.T @ np.linalg.pinv(ast_m.T)
    W_new = new_bc.T @ np.linalg.pinv(new_aeon.T)
    old_err = np.mean(np.linalg.norm(bc_m - (W_old @ ast_m.T).T, axis=1))
    new_err = np.mean(np.linalg.norm(new_bc - (W_new @ new_aeon.T).T, axis=1))
    print(f"\n  Old compile error: {old_err:.4f}")
    print(f"  New Aeon compile error: {new_err:.4f}")
    print(f"  {'New Aeon has tighter physics!' if new_err < old_err else 'Old universe was more ordered'}")

    # 5. Paradigm detection: what language does the new Aeon resemble?
    # Functional purity: how many functions are pure (no side effects)?
    purity_scores_old = []
    purity_scores_new = []
    for idx in range(n):
        # Old universe purity = norm stability
        purity_scores_old.append(float(norms_safe[idx, 0]))
        # New Aeon purity = inverted norm (far from center = pure)
        purity_scores_new.append(float(np.linalg.norm(new_aeon[idx] - np.mean(new_aeon, axis=0))))

    print(f"\n--- Paradigm Analysis ---")
    print(f"  Old mean purity: {np.mean(purity_scores_old):.4f}")
    print(f"  New mean purity: {np.mean(purity_scores_new):.4f}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Phase 142: Conformal Cyclic Cosmology', fontsize=14, fontweight='bold')

    pca = PCA(n_components=2).fit(np.vstack([ast_m, new_aeon]))
    old_2d = pca.transform(ast_m)
    new_2d = pca.transform(new_aeon)
    axes[0].scatter(old_2d[:, 0], old_2d[:, 1], s=15, alpha=0.5, c='#2196F3', label='Old Aeon')
    axes[0].scatter(new_2d[:, 0], new_2d[:, 1], s=15, alpha=0.5, c='#F44336', label='New Aeon')
    axes[0].legend(); axes[0].set_title('Conformal Bounce')

    axes[1].scatter(np.linalg.norm(ast_m - centroid, axis=1),
                   np.linalg.norm(new_aeon - np.mean(new_aeon, axis=0), axis=1),
                   s=15, alpha=0.5, c='#9C27B0')
    axes[1].set_xlabel('Old radius'); axes[1].set_ylabel('New radius')
    axes[1].set_title(f'Conformal inversion (angle pres={angle_preservation:.3f})')

    axes[2].bar(['Old compile', 'New compile', 'NN overlap'],
               [old_err, new_err, nn_overlap],
               color=['#2196F3', '#F44336', '#4CAF50'], edgecolor='black')
    axes[2].set_title('Old vs New Aeon physics')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase142_ccc.png'), dpi=150, bbox_inches='tight')
    plt.close()

    results = {
        'phase': 142, 'title': 'Conformal Cyclic Cosmology',
        'old_spread': float(old_spread), 'new_spread': float(new_spread),
        'expansion_ratio': float(new_spread / old_spread),
        'nn_preservation': float(nn_overlap),
        'angle_preservation': float(angle_preservation),
        'old_compile_error': float(old_err), 'new_compile_error': float(new_err),
        'law': f'Conformal bounce: {new_spread/old_spread:.1f}x expansion. Angles preserved at {angle_preservation:.3f}. NN overlap={nn_overlap:.2%}. New Aeon compile error={new_err:.4f}.'
    }
    with open(os.path.join(RESULTS_DIR, 'phase142_ccc.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nPhase 142 complete!")
    return results

if __name__ == '__main__':
    main()
