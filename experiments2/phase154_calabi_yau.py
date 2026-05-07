"""Phase 154: Calabi-Yau Bulk - Decode the 30.6 hidden bits
The 52 null dimensions (64 - 12 effective) contain the hidden variables.
Use persistent homology to extract their information content.
"""
import os, json, sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
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
    print("Phase 154: The Calabi-Yau Bulk")
    print("  Where are the 30.6 hidden bits?")
    print("=" * 60)

    latents = np.load(os.path.join(DATA_DIR, 'rosetta_latents_v2.npz'))
    ast_vectors = latents['ast']
    dataset = json.load(open(os.path.join(DATA_DIR, 'rosetta_dataset_v2.json'), 'r', encoding='utf-8'))
    sources = [item['source'] for item in dataset['dataset']]

    func_ast = {}
    for i, src in enumerate(sources):
        if src not in func_ast: func_ast[src] = []
        func_ast[src].append(ast_vectors[i])
    unique_funcs = list(func_ast.keys())
    ast_m = np.array([np.mean(func_ast[f], axis=0) for f in unique_funcs])
    n = len(unique_funcs)

    # PCA: separate visible (d_eff=12) from hidden (52) dimensions
    pca = PCA(n_components=64).fit(ast_m)
    ast_pca = pca.transform(ast_m)
    eigenvalues = pca.explained_variance_

    d_eff = int(np.sum(eigenvalues > 0.01 * np.max(eigenvalues)))
    visible = ast_pca[:, :d_eff]    # The 12 visible dimensions
    hidden = ast_pca[:, d_eff:]     # The 52 hidden dimensions (Calabi-Yau bulk)

    print(f"  Visible dimensions: {d_eff}")
    print(f"  Hidden dimensions: {64 - d_eff}")
    print(f"  Visible variance: {np.sum(eigenvalues[:d_eff])/np.sum(eigenvalues)*100:.2f}%")
    print(f"  Hidden variance: {np.sum(eigenvalues[d_eff:])/np.sum(eigenvalues)*100:.2f}%")

    # 1. Information content of hidden dimensions
    hidden_eigs = eigenvalues[d_eff:]
    hidden_eigs_pos = hidden_eigs[hidden_eigs > 1e-15]
    if len(hidden_eigs_pos) > 0:
        hidden_norm = hidden_eigs_pos / np.sum(hidden_eigs_pos)
        hidden_entropy = -np.sum(hidden_norm * np.log2(hidden_norm + 1e-15))
    else:
        hidden_entropy = 0

    print(f"\n--- Hidden Dimension Information ---")
    print(f"  Hidden entropy: {hidden_entropy:.4f} bits")
    print(f"  Target (from P152): 30.6 bits")
    print(f"  Match: {abs(hidden_entropy - 30.6)/30.6*100:.1f}% error")

    match_30_6 = abs(hidden_entropy - 30.6) < 30.6 * 0.5  # Within 50%

    # 2. Persistent homology approximation (without ripser)
    # Compute Betti numbers from distance matrix
    print(f"\n--- Topological Analysis (Persistent Homology) ---")
    hidden_subset = hidden[:50]  # Use subset for speed
    dist_mat = squareform(pdist(hidden_subset))

    # Vietoris-Rips approximation: count connected components at various scales
    scales = np.linspace(0, np.max(dist_mat), 30)
    betti_0 = []  # Connected components
    betti_1 = []  # Loops (approximated)

    for eps in scales:
        adj = (dist_mat < eps).astype(int)
        np.fill_diagonal(adj, 0)
        # Count connected components (BFS)
        visited = set()
        n_components = 0
        for i in range(len(adj)):
            if i not in visited:
                n_components += 1
                queue = [i]
                while queue:
                    node = queue.pop(0)
                    if node not in visited:
                        visited.add(node)
                        neighbors = np.where(adj[node] > 0)[0]
                        queue.extend([nb for nb in neighbors if nb not in visited])
        betti_0.append(n_components)
        # Approximate betti_1: edges - nodes + components (Euler characteristic)
        n_edges = np.sum(adj) // 2
        betti_1_approx = max(0, n_edges - len(adj) + n_components)
        betti_1.append(betti_1_approx)

    # Topological complexity = total persistence
    total_persistence = sum(abs(betti_0[i] - betti_0[i-1]) for i in range(1, len(betti_0)))
    topological_info = np.log2(total_persistence + 1)
    print(f"  Total topological persistence: {total_persistence}")
    print(f"  Topological information: {topological_info:.4f} bits")

    # 3. Do hidden dimensions encode structure not in visible?
    # Compile matrix using only visible vs hidden dimensions
    bc_m = np.array([np.mean([latents['bc'][j] for j, s in enumerate(sources) if s == f], axis=0) for f in unique_funcs])
    W_vis = bc_m.T @ np.linalg.pinv(visible.T)
    W_hid = bc_m.T @ np.linalg.pinv(hidden.T)
    W_full = bc_m.T @ np.linalg.pinv(ast_pca.T)

    err_vis = float(np.mean(np.linalg.norm(bc_m - (W_vis @ visible.T).T, axis=1)))
    err_hid = float(np.mean(np.linalg.norm(bc_m - (W_hid @ hidden.T).T, axis=1)))
    err_full = float(np.mean(np.linalg.norm(bc_m - (W_full @ ast_pca.T).T, axis=1)))

    print(f"\n--- Compile Matrix Decomposition ---")
    print(f"  Visible-only error: {err_vis:.4f}")
    print(f"  Hidden-only error: {err_hid:.4f}")
    print(f"  Full (vis+hid) error: {err_full:.4f}")
    print(f"  Hidden contribution: {(err_vis - err_full)/err_vis*100:.1f}%")

    hidden_contribution = (err_vis - err_full) / (err_vis + 1e-10) * 100

    # 4. Eigenvalue spectrum of hidden dims = Calabi-Yau shape
    print(f"\n--- Calabi-Yau Shape ---")
    n_cy_modes = int(np.sum(hidden_eigs_pos > 1e-10))
    print(f"  Active Calabi-Yau modes: {n_cy_modes}")
    if len(hidden_eigs_pos) > 0:
        cy_top3 = hidden_eigs_pos[:min(3, len(hidden_eigs_pos))]
        for i, e in enumerate(cy_top3):
            print(f"  Mode {i}: eigenvalue={e:.6f}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Phase 154: The Calabi-Yau Bulk', fontsize=14, fontweight='bold')

    axes[0].semilogy(range(64), eigenvalues, 'o-', color='#2196F3', markersize=4)
    axes[0].axvline(d_eff, color='red', linestyle='--', label=f'd_eff={d_eff}')
    axes[0].fill_between(range(d_eff, 64), eigenvalues[d_eff:], alpha=0.3, color='#F44336', label='Hidden (CY)')
    axes[0].set_xlabel('Dimension'); axes[0].set_ylabel('Eigenvalue')
    axes[0].set_title(f'Spectrum: {d_eff} visible + {64-d_eff} hidden')
    axes[0].legend()

    axes[1].plot(scales, betti_0, 'o-', color='#E91E63', markersize=3, label='beta_0 (components)')
    ax2 = axes[1].twinx()
    ax2.plot(scales, betti_1, 's-', color='#4CAF50', markersize=3, label='beta_1 (loops)')
    axes[1].set_xlabel('Scale'); axes[1].set_ylabel('beta_0', color='#E91E63')
    ax2.set_ylabel('beta_1', color='#4CAF50')
    axes[1].set_title(f'Persistent homology (info={topological_info:.1f} bits)')
    axes[1].legend(loc='upper left'); ax2.legend(loc='upper right')

    axes[2].bar(['Visible only', 'Hidden only', 'Full'], [err_vis, err_hid, err_full],
               color=['#2196F3', '#F44336', '#4CAF50'], edgecolor='black')
    axes[2].set_ylabel('Compile error')
    axes[2].set_title(f'Hidden contribution: {hidden_contribution:.1f}%')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase154_calabi_yau.png'), dpi=150, bbox_inches='tight')
    plt.close()

    results = {
        'phase': 154, 'title': 'The Calabi-Yau Bulk',
        'd_eff': d_eff, 'n_hidden': 64 - d_eff,
        'hidden_entropy_bits': float(hidden_entropy),
        'target_bits': 30.6,
        'match_30_6': bool(match_30_6),
        'topological_info': float(topological_info),
        'hidden_compile_contribution_pct': float(hidden_contribution),
        'n_cy_modes': n_cy_modes,
        'law': f'Hidden entropy={hidden_entropy:.1f} bits (target=30.6). {n_cy_modes} CY modes. Hidden contribute {hidden_contribution:.1f}% to compilation. 30.6-bit match: {match_30_6}.'
    }
    with open(os.path.join(RESULTS_DIR, 'phase154_calabi_yau.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nPhase 154 complete!")
    return results

if __name__ == '__main__':
    main()
