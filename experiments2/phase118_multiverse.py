"""Phase 118: The Language Multiverse - Do different encodings form parallel branes?
Deep Think proposal: different language paradigms as parallel D-branes.
Adapted: We test if AST vs BC vs NL form parallel 5D branes in the 64D bulk,
and measure inter-brane distance (the "brane gap").
"""
import os, json, sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from scipy.spatial.distance import pdist, squareform, cdist
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
    print("Phase 118: The Language Multiverse")
    print("  Are AST/BC/NL parallel D-branes in 64D bulk?")
    print("=" * 60)
    
    latents = np.load(os.path.join(DATA_DIR, 'rosetta_latents_v2.npz'))
    dataset = json.load(open(os.path.join(DATA_DIR, 'rosetta_dataset_v2.json'), 'r', encoding='utf-8'))
    ast_vectors = latents['ast']
    bc_vectors = latents['bc']
    nl_vectors = latents['nl']
    sources = [item['source'] for item in dataset['dataset']]
    
    # Compute per-function mean vectors in each modality
    func_ast, func_bc, func_nl = {}, {}, {}
    for i, src in enumerate(sources):
        if src not in func_ast:
            func_ast[src] = []; func_bc[src] = []; func_nl[src] = []
        func_ast[src].append(ast_vectors[i])
        func_bc[src].append(bc_vectors[i])
        func_nl[src].append(nl_vectors[i])
    
    unique_funcs = list(func_ast.keys())
    ast_m = np.array([np.mean(func_ast[f], axis=0) for f in unique_funcs])
    bc_m = np.array([np.mean(func_bc[f], axis=0) for f in unique_funcs])
    nl_m = np.array([np.mean(func_nl[f], axis=0) for f in unique_funcs])
    n = len(unique_funcs)
    
    # 1. PCA on each modality separately - do they have the SAME 5D structure?
    pca_ast = PCA(n_components=10).fit(ast_m)
    pca_bc = PCA(n_components=10).fit(bc_m)
    pca_nl = PCA(n_components=10).fit(nl_m)
    
    print("--- Variance explained (first 5 PCs) ---")
    for name, pca in [('AST', pca_ast), ('BC', pca_bc), ('NL', pca_nl)]:
        var5 = sum(pca.explained_variance_ratio_[:5]) * 100
        print(f"  {name}: {var5:.1f}%")
    
    # 2. Brane alignment: are the 5D subspaces parallel?
    # Compute principal angles between subspaces (Grassmann distance)
    def subspace_angles(A, B, k=5):
        """Principal angles between k-dim subspaces."""
        Qa = np.linalg.qr(A[:, :k])[0]
        Qb = np.linalg.qr(B[:, :k])[0]
        _, sigmas, _ = np.linalg.svd(Qa.T @ Qb)
        angles = np.arccos(np.clip(sigmas[:k], -1, 1))
        return angles
    
    angles_ast_bc = subspace_angles(pca_ast.components_.T, pca_bc.components_.T)
    angles_ast_nl = subspace_angles(pca_ast.components_.T, pca_nl.components_.T)
    angles_bc_nl = subspace_angles(pca_bc.components_.T, pca_nl.components_.T)
    
    print("\n--- Principal angles between branes (degrees) ---")
    for name, angs in [('AST-BC', angles_ast_bc), ('AST-NL', angles_ast_nl), ('BC-NL', angles_bc_nl)]:
        deg = np.degrees(angs)
        print(f"  {name}: {', '.join(f'{d:.1f}' for d in deg)}")
    
    # 3. Inter-brane distance: how far apart are the modalities?
    ast_bc_dist = np.mean(np.linalg.norm(ast_m - bc_m, axis=1))
    ast_nl_dist = np.mean(np.linalg.norm(ast_m - nl_m, axis=1))
    bc_nl_dist = np.mean(np.linalg.norm(bc_m - nl_m, axis=1))
    
    print("\n--- Inter-brane distances ---")
    print(f"  AST-BC: {ast_bc_dist:.4f}")
    print(f"  AST-NL: {ast_nl_dist:.4f}")
    print(f"  BC-NL:  {bc_nl_dist:.4f}")
    
    # 4. Wormholes: functions where BC and AST are very close despite being in different branes
    brane_gap = np.linalg.norm(ast_m - bc_m, axis=1)
    wormhole_idx = np.argsort(brane_gap)[:5]
    
    print("\n--- Wormholes (smallest brane gap) ---")
    for idx in wormhole_idx:
        f_short = unique_funcs[idx].split('return ')[-1].strip()[:20]
        print(f"  {f_short}: gap={brane_gap[idx]:.4f}")
    
    # 5. Brane tension: correlation of intra-brane distances
    ast_pdist = pdist(ast_m)
    bc_pdist = pdist(bc_m)
    nl_pdist = pdist(nl_m)
    corr_ast_bc = np.corrcoef(ast_pdist, bc_pdist)[0,1]
    corr_ast_nl = np.corrcoef(ast_pdist, nl_pdist)[0,1]
    
    print(f"\n--- Brane tension (distance correlation) ---")
    print(f"  AST-BC structure correlation: {corr_ast_bc:.4f}")
    print(f"  AST-NL structure correlation: {corr_ast_nl:.4f}")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Phase 118: The Language Multiverse', fontsize=14, fontweight='bold')
    
    ast_2d = PCA(n_components=2).fit_transform(ast_m)
    bc_2d = PCA(n_components=2).fit_transform(bc_m)
    nl_2d = PCA(n_components=2).fit_transform(nl_m)
    
    axes[0].scatter(ast_2d[:,0], ast_2d[:,1], s=15, alpha=0.5, c='#E91E63', label='AST brane')
    axes[0].scatter(bc_2d[:,0], bc_2d[:,1], s=15, alpha=0.5, c='#2196F3', label='BC brane')
    axes[0].scatter(nl_2d[:,0], nl_2d[:,1], s=15, alpha=0.5, c='#4CAF50', label='NL brane')
    axes[0].legend(fontsize=8); axes[0].set_title('Three branes in 2D projection')
    
    pairs = ['AST-BC', 'AST-NL', 'BC-NL']
    dists = [ast_bc_dist, ast_nl_dist, bc_nl_dist]
    axes[1].bar(pairs, dists, color=['#9C27B0','#FF9800','#00BCD4'], edgecolor='black')
    axes[1].set_ylabel('Mean Distance'); axes[1].set_title('Inter-brane distances')
    
    corrs = [corr_ast_bc, corr_ast_nl]
    axes[2].bar(['AST-BC', 'AST-NL'], corrs, color=['#E91E63','#4CAF50'], edgecolor='black')
    axes[2].set_ylabel('Correlation'); axes[2].set_title(f'Brane tension (structure corr)')
    axes[2].set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase118_multiverse.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    results = {
        'phase': 118, 'title': 'The Language Multiverse',
        'variance_5d': {'AST': float(sum(pca_ast.explained_variance_ratio_[:5])),
                        'BC': float(sum(pca_bc.explained_variance_ratio_[:5])),
                        'NL': float(sum(pca_nl.explained_variance_ratio_[:5]))},
        'brane_distances': {'AST_BC': float(ast_bc_dist), 'AST_NL': float(ast_nl_dist), 'BC_NL': float(bc_nl_dist)},
        'structure_correlation': {'AST_BC': float(corr_ast_bc), 'AST_NL': float(corr_ast_nl)},
        'principal_angles_deg': {'AST_BC': [float(d) for d in np.degrees(angles_ast_bc)]},
        'law': f'Three modalities form parallel branes: structure correlation AST-BC={corr_ast_bc:.3f}, AST-NL={corr_ast_nl:.3f}. Inter-brane distances: AST-BC={ast_bc_dist:.3f}, AST-NL={ast_nl_dist:.3f}.'
    }
    with open(os.path.join(RESULTS_DIR, 'phase118_multiverse.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nPhase 118 complete!")
    return results

if __name__ == '__main__':
    main()
