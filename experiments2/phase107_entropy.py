"""Phase 107: The Entropy of Code - Information theory meets software physics.
How much information does a program carry? Shannon entropy of the latent space.
Is there a connection between thermodynamic entropy (P88) and information entropy?
"""
import os, json, sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
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
    print("Phase 107: The Entropy of Code")
    print("  Information theory meets software physics")
    print("=" * 60)
    
    latents = np.load(os.path.join(DATA_DIR, 'rosetta_latents_v2.npz'))
    dataset = json.load(open(os.path.join(DATA_DIR, 'rosetta_dataset_v2.json'), 'r', encoding='utf-8'))
    ast_vectors = latents['ast']
    bc_vectors = latents['bc']
    sources = [item['source'] for item in dataset['dataset']]
    
    func_ast = {}; func_bc = {}
    for i, src in enumerate(sources):
        if src not in func_ast:
            func_ast[src] = []; func_bc[src] = []
        func_ast[src].append(ast_vectors[i])
        func_bc[src].append(bc_vectors[i])
    
    unique = list(func_ast.keys())
    ast_means = np.array([np.mean(func_ast[f], axis=0) for f in unique])
    bc_means = np.array([np.mean(func_bc[f], axis=0) for f in unique])
    
    # === 1. Differential entropy of the latent distribution ===
    # Estimate via KDE-like approach: entropy ~ 0.5 * log(det(2*pi*e*Cov))
    cov_ast = np.cov(ast_means.T)
    eigenvalues = np.linalg.eigvalsh(cov_ast)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    diff_entropy_ast = 0.5 * np.sum(np.log(2 * np.pi * np.e * eigenvalues))
    
    cov_bc = np.cov(bc_means.T)
    eigenvalues_bc = np.linalg.eigvalsh(cov_bc)
    eigenvalues_bc = eigenvalues_bc[eigenvalues_bc > 1e-10]
    diff_entropy_bc = 0.5 * np.sum(np.log(2 * np.pi * np.e * eigenvalues_bc))
    
    print(f"  Differential entropy (AST): {diff_entropy_ast:.2f} nats")
    print(f"  Differential entropy (BC):  {diff_entropy_bc:.2f} nats")
    print(f"  AST/BC entropy ratio:       {diff_entropy_ast/diff_entropy_bc:.3f}")
    
    # === 2. Discrete entropy via clustering ===
    entropy_by_k = []
    ks = [5, 10, 15, 20, 30, 50]
    for k in ks:
        if k >= len(unique): continue
        km = KMeans(n_clusters=k, random_state=42, n_init=5)
        labels = km.fit_predict(ast_means)
        counts = np.bincount(labels)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        H = -np.sum(probs * np.log2(probs))
        entropy_by_k.append((k, H))
        print(f"  K={k:3d}: H={H:.3f} bits (max={np.log2(k):.3f})")
    
    # === 3. Per-function entropy (source code entropy) ===
    src_entropies = []
    for f in unique:
        chars = list(f)
        char_counts = {}
        for c in chars:
            char_counts[c] = char_counts.get(c, 0) + 1
        total = len(chars)
        probs = np.array([v/total for v in char_counts.values()])
        H = -np.sum(probs * np.log2(probs + 1e-10))
        src_entropies.append(H)
    src_entropies = np.array(src_entropies)
    
    print(f"\n--- Source Code Entropy ---")
    print(f"  Mean: {np.mean(src_entropies):.3f} bits/char")
    print(f"  Std:  {np.std(src_entropies):.3f}")
    
    # === 4. Mutual information between AST and BC ===
    # Approximate via correlation
    from sklearn.cross_decomposition import CCA
    n_components = min(5, min(ast_means.shape[1], bc_means.shape[1]), len(unique) - 1)
    cca = CCA(n_components=n_components, max_iter=1000)
    try:
        X_c, Y_c = cca.fit_transform(ast_means, bc_means)
        correlations = [np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1] for i in range(n_components)]
        mean_cca_corr = np.mean(correlations)
    except:
        correlations = [0]
        mean_cca_corr = 0
    
    print(f"\n--- AST-BC Mutual Information (CCA) ---")
    for i, c in enumerate(correlations):
        print(f"  CCA component {i+1}: r={c:.4f}")
    print(f"  Mean CCA correlation: {mean_cca_corr:.4f}")
    
    # === 5. Kolmogorov complexity proxy (code length) ===
    code_lengths = [len(f) for f in unique]
    latent_norms = np.linalg.norm(ast_means, axis=1)
    kc_corr = np.corrcoef(code_lengths, latent_norms)[0, 1]
    kc_entropy_corr = np.corrcoef(code_lengths, src_entropies)[0, 1]
    
    print(f"\n--- Kolmogorov Complexity Proxy ---")
    print(f"  Length-Norm correlation:    {kc_corr:.3f}")
    print(f"  Length-Entropy correlation: {kc_entropy_corr:.3f}")
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Phase 107: The Entropy of Code', fontsize=14, fontweight='bold')
    
    if entropy_by_k:
        ks_plot = [e[0] for e in entropy_by_k]
        hs_plot = [e[1] for e in entropy_by_k]
        max_hs = [np.log2(k) for k in ks_plot]
        axes[0, 0].plot(ks_plot, hs_plot, 'o-', color='#E91E63', label='Actual H')
        axes[0, 0].plot(ks_plot, max_hs, 's--', color='gray', label='Max H')
        axes[0, 0].set_xlabel('K clusters'); axes[0, 0].set_ylabel('Entropy (bits)')
        axes[0, 0].set_title('Discrete Entropy vs Clustering')
        axes[0, 0].legend()
    
    axes[0, 1].hist(src_entropies, bins=30, color='#4CAF50', edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Source Code Entropy (bits/char)')
    axes[0, 1].set_title(f'Code Entropy Distribution (mean={np.mean(src_entropies):.2f})')
    
    axes[1, 0].bar(['AST', 'Bytecode'], [diff_entropy_ast, diff_entropy_bc],
                   color=['#2196F3', '#FF9800'], edgecolor='black')
    axes[1, 0].set_ylabel('Differential Entropy (nats)')
    axes[1, 0].set_title(f'Latent Space Entropy (ratio={diff_entropy_ast/diff_entropy_bc:.2f})')
    
    axes[1, 1].scatter(code_lengths, src_entropies, alpha=0.3, s=20, color='#9C27B0')
    axes[1, 1].set_xlabel('Code Length (chars)')
    axes[1, 1].set_ylabel('Source Entropy (bits/char)')
    axes[1, 1].set_title(f'Kolmogorov Proxy (r={kc_entropy_corr:.3f})')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase107_entropy.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    results = {
        'phase': 107, 'title': 'The Entropy of Code',
        'diff_entropy_ast': float(diff_entropy_ast),
        'diff_entropy_bc': float(diff_entropy_bc),
        'entropy_ratio': float(diff_entropy_ast/diff_entropy_bc),
        'mean_src_entropy': float(np.mean(src_entropies)),
        'cca_correlations': [float(c) for c in correlations],
        'mean_cca_corr': float(mean_cca_corr),
        'kc_length_norm_corr': float(kc_corr),
        'kc_length_entropy_corr': float(kc_entropy_corr),
        'discrete_entropy': {str(k): float(h) for k, h in entropy_by_k},
        'law': f'Code entropy: AST={diff_entropy_ast:.1f} nats, BC={diff_entropy_bc:.1f} nats (ratio={diff_entropy_ast/diff_entropy_bc:.2f}). CCA mutual info={mean_cca_corr:.3f}. Source entropy={np.mean(src_entropies):.2f} bits/char.'
    }
    with open(os.path.join(RESULTS_DIR, 'phase107_entropy.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nPhase 107 complete!")
    return results

if __name__ == '__main__':
    main()
