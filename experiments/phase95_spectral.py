"""Phase 95: The Spectral Gap - Phase transitions in program space.
Is there a critical dimensionality where semantics 'crystallizes'?
Study the eigenvalue spectrum of the function correlation matrix
to find spectral gaps that indicate phase transitions.
"""
import os, json, sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(EXPERIMENT_DIR)
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')

def main():
    print("=" * 60)
    print("Phase 95: The Spectral Gap")
    print("  Where does meaning crystallize?")
    print("=" * 60)
    
    latents = np.load(os.path.join(BASE_DIR, 'data', 'rosetta_latents_v2.npz'))
    dataset = json.load(open(os.path.join(BASE_DIR, 'data', 'rosetta_dataset_v2.json'), 'r', encoding='utf-8'))
    ast_vectors = latents['ast']
    sources = [item['source'] for item in dataset['dataset']]
    
    func_to_vecs = {}
    for i, src in enumerate(sources):
        if src not in func_to_vecs:
            func_to_vecs[src] = []
        func_to_vecs[src].append(ast_vectors[i])
    func_means = {src: np.mean(vecs, axis=0) for src, vecs in func_to_vecs.items()}
    unique_funcs = list(func_means.keys())
    all_vecs = np.array([func_means[f] for f in unique_funcs])
    print(f"Functions: {len(unique_funcs)}, Dimensions: {all_vecs.shape[1]}")
    
    # Full PCA to study eigenvalue spectrum
    pca_full = PCA(n_components=min(64, len(unique_funcs)))
    pca_full.fit(all_vecs)
    eigenvalues = pca_full.explained_variance_
    explained_ratio = pca_full.explained_variance_ratio_
    cumulative = np.cumsum(explained_ratio)
    
    print(f"\n--- Eigenvalue Spectrum ---")
    for i in range(min(20, len(eigenvalues))):
        gap = eigenvalues[i] / eigenvalues[i+1] if i+1 < len(eigenvalues) else 0
        print(f"  PC{i+1}: eigenvalue={eigenvalues[i]:.4f}, cumulative={cumulative[i]:.3f}, gap_ratio={gap:.2f}")
    
    # Find spectral gaps (ratios > 2.0 indicate phase transitions)
    gaps = []
    for i in range(len(eigenvalues)-1):
        ratio = eigenvalues[i] / eigenvalues[i+1]
        gaps.append({'pc': i+1, 'ratio': float(ratio), 'cumulative': float(cumulative[i])})
    
    significant_gaps = [g for g in gaps if g['ratio'] > 1.5]
    print(f"\n--- Significant Spectral Gaps (ratio > 1.5) ---")
    for g in significant_gaps:
        print(f"  Gap after PC{g['pc']}: ratio={g['ratio']:.2f}, cumulative={g['cumulative']:.3f}")
    
    # Find the critical dimension where 90%, 95%, 99% variance is captured
    dim_90 = np.argmax(cumulative >= 0.90) + 1
    dim_95 = np.argmax(cumulative >= 0.95) + 1
    dim_99 = np.argmax(cumulative >= 0.99) + 1
    print(f"\n--- Critical Dimensions ---")
    print(f"  90% variance: {dim_90}D")
    print(f"  95% variance: {dim_95}D")
    print(f"  99% variance: {dim_99}D")
    
    # Test: how does nearest-neighbor accuracy change with dimensionality?
    print(f"\n--- NN Classification vs Dimensionality ---")
    # Group functions by operation type
    op_labels = {}
    for f in unique_funcs:
        if 'return x + y' in f: op_labels[f] = 'add'
        elif 'return x - y' in f: op_labels[f] = 'sub'
        elif 'return x * y' in f: op_labels[f] = 'mul'
        elif 'return x > y' in f or 'return x < y' in f: op_labels[f] = 'cmp'
        elif 'return max' in f or 'return min' in f: op_labels[f] = 'minmax'
        elif 'return abs' in f: op_labels[f] = 'abs'
        else: op_labels[f] = 'other'
    
    labeled_funcs = [f for f in unique_funcs if op_labels.get(f, 'other') != 'other']
    labeled_labels = [op_labels[f] for f in labeled_funcs]
    labeled_vecs = np.array([func_means[f] for f in labeled_funcs])
    
    nn_acc_by_dim = []
    dims_to_test = list(range(1, min(40, len(unique_funcs)), 1))
    
    for d in dims_to_test:
        pca_d = PCA(n_components=d)
        vecs_d = pca_d.fit_transform(labeled_vecs)
        
        # Leave-one-out NN classification
        correct = 0
        for i in range(len(labeled_funcs)):
            dists = np.linalg.norm(vecs_d - vecs_d[i], axis=1)
            dists[i] = float('inf')
            nn_idx = np.argmin(dists)
            if labeled_labels[nn_idx] == labeled_labels[i]:
                correct += 1
        acc = correct / len(labeled_funcs)
        nn_acc_by_dim.append(acc)
    
    # Find the "crystallization point" - where accuracy plateaus
    max_acc = max(nn_acc_by_dim)
    crystal_dim = dims_to_test[nn_acc_by_dim.index(max_acc)]
    
    # Find where accuracy first reaches 95% of max
    threshold = max_acc * 0.95
    crystal_dim_95 = next((d for d, a in zip(dims_to_test, nn_acc_by_dim) if a >= threshold), dims_to_test[-1])
    
    print(f"  Max NN accuracy: {max_acc:.3f} at {crystal_dim}D")
    print(f"  95% of max at: {crystal_dim_95}D")
    print(f"  This is the MEANING crystallization point!")
    
    # Effective dimensionality (participation ratio)
    p_ratio = (np.sum(eigenvalues)**2) / np.sum(eigenvalues**2)
    print(f"\n  Participation ratio (effective dim): {p_ratio:.1f}")
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Phase 95: The Spectral Gap - Phase Transitions in Program Space',
                 fontsize=14, fontweight='bold')
    
    # 1. Eigenvalue spectrum (log scale)
    axes[0, 0].semilogy(range(1, len(eigenvalues)+1), eigenvalues, 'o-', color='#2196F3', markersize=4)
    for g in significant_gaps[:3]:
        axes[0, 0].axvline(g['pc'], color='red', linestyle='--', alpha=0.5, label=f"Gap at PC{g['pc']}")
    axes[0, 0].set_xlabel('Principal Component')
    axes[0, 0].set_ylabel('Eigenvalue (log)')
    axes[0, 0].set_title('Eigenvalue Spectrum')
    axes[0, 0].legend(fontsize=7)
    
    # 2. Cumulative variance
    axes[0, 1].plot(range(1, len(cumulative)+1), cumulative*100, 'o-', color='#4CAF50', markersize=3)
    axes[0, 1].axhline(90, color='orange', linestyle='--', alpha=0.5, label=f'90% at {dim_90}D')
    axes[0, 1].axhline(95, color='red', linestyle='--', alpha=0.5, label=f'95% at {dim_95}D')
    axes[0, 1].axhline(99, color='purple', linestyle='--', alpha=0.5, label=f'99% at {dim_99}D')
    axes[0, 1].set_xlabel('Dimensions')
    axes[0, 1].set_ylabel('Cumulative Variance (%)')
    axes[0, 1].set_title('Variance Captured vs Dimensionality')
    axes[0, 1].legend(fontsize=7)
    
    # 3. Gap ratios
    gap_ratios = [eigenvalues[i]/eigenvalues[i+1] for i in range(min(30, len(eigenvalues)-1))]
    axes[1, 0].bar(range(1, len(gap_ratios)+1), gap_ratios, color='#FF9800', edgecolor='black')
    axes[1, 0].axhline(1.5, color='red', linestyle='--', label='Significance threshold')
    axes[1, 0].set_xlabel('Gap Position')
    axes[1, 0].set_ylabel('Eigenvalue Ratio (i/i+1)')
    axes[1, 0].set_title('Spectral Gaps (Phase Transitions)')
    axes[1, 0].legend()
    
    # 4. NN accuracy vs dimensionality
    axes[1, 1].plot(dims_to_test, [a*100 for a in nn_acc_by_dim], 'o-', color='#E91E63', markersize=4)
    axes[1, 1].axvline(crystal_dim_95, color='green', linestyle='--',
                       label=f'Crystallization: {crystal_dim_95}D')
    axes[1, 1].axvline(5, color='blue', linestyle=':', alpha=0.5, label='5D (semantics)')
    axes[1, 1].set_xlabel('Dimensions')
    axes[1, 1].set_ylabel('NN Classification Accuracy (%)')
    axes[1, 1].set_title('Meaning Crystallization Point')
    axes[1, 1].legend(fontsize=7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase95_spectral.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    results = {
        'phase': 95, 'title': 'The Spectral Gap - Phase Transitions in Program Space',
        'n_functions': len(unique_funcs),
        'eigenvalues_top10': [float(e) for e in eigenvalues[:10]],
        'significant_gaps': significant_gaps[:5],
        'dim_90pct': int(dim_90), 'dim_95pct': int(dim_95), 'dim_99pct': int(dim_99),
        'crystallization_dim': int(crystal_dim_95),
        'max_nn_accuracy': float(max_acc),
        'participation_ratio': float(p_ratio),
        'nn_acc_by_dim': {str(d): float(a) for d, a in zip(dims_to_test, nn_acc_by_dim)},
        'law': f'Meaning crystallizes at {crystal_dim_95}D (95% max accuracy). Effective dim={p_ratio:.1f}. The spectral gap reveals phase transitions in program space.'
    }
    with open(os.path.join(RESULTS_DIR, 'phase95_spectral.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nPhase 95 complete!")
    return results

if __name__ == '__main__':
    main()
