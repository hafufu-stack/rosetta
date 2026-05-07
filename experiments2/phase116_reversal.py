"""Phase 116: Time Reversal Symmetry - Can we go from complex to simple?
P112 found PC2 = arrow of time. Is this reversible?
Can we decompose complex programs into simpler components?
"""
import os, json, sys, inspect
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ast as ast_mod

EXP2_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(EXP2_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')

def main():
    print("=" * 60)
    print("Phase 116: Time Reversal Symmetry")
    print("  Can we decompose complex into simple?")
    print("=" * 60)
    
    latents = np.load(os.path.join(DATA_DIR, 'rosetta_latents_v2.npz'))
    dataset = json.load(open(os.path.join(DATA_DIR, 'rosetta_dataset_v2.json'), 'r', encoding='utf-8'))
    ast_vectors = latents['ast']
    sources = [item['source'] for item in dataset['dataset']]
    
    func_to_vecs = {}
    for i, src in enumerate(sources):
        if src not in func_to_vecs: func_to_vecs[src] = []
        func_to_vecs[src].append(ast_vectors[i])
    func_means = {s: np.mean(v, axis=0) for s, v in func_to_vecs.items()}
    unique_funcs = list(func_means.keys())
    all_vecs = np.array([func_means[f] for f in unique_funcs])
    n = len(unique_funcs)
    
    # Complexity of each function
    complexity = {}
    for f in unique_funcs:
        try:
            tree = ast_mod.parse(f)
            complexity[f] = sum(1 for _ in ast_mod.walk(tree))
        except:
            complexity[f] = 1
    
    comp_arr = np.array([complexity[f] for f in unique_funcs])
    
    # PCA
    pca = PCA(n_components=10)
    vecs_pca = pca.fit_transform(all_vecs)
    
    # Test: vector addition decomposition
    # Can we express complex functions as vector sums of simpler ones?
    print("\n--- Vector Decomposition Test ---")
    decomp_results = []
    
    # Find "elementary" functions (simplest ones)
    elementary_threshold = np.percentile(comp_arr, 25)
    elementary_indices = [i for i in range(n) if comp_arr[i] <= elementary_threshold]
    complex_indices = [i for i in range(n) if comp_arr[i] > np.percentile(comp_arr, 75)]
    
    print(f"  Elementary functions: {len(elementary_indices)}")
    print(f"  Complex functions: {len(complex_indices)}")
    
    elementary_vecs = all_vecs[elementary_indices]
    
    for ci in complex_indices[:20]:
        target = all_vecs[ci]
        
        # Try to decompose as weighted sum of elementary functions
        # Solve: target = A @ weights (least squares)
        A = elementary_vecs.T  # 64 x n_elementary
        try:
            weights, residuals, _, _ = np.linalg.lstsq(A, target, rcond=None)
            reconstruction = A @ weights
            error = np.linalg.norm(target - reconstruction) / np.linalg.norm(target)
            
            # How many elementary functions contribute significantly?
            significant = np.sum(np.abs(weights) > 0.1 * np.max(np.abs(weights)))
            
            func_short = unique_funcs[ci].split('return ')[-1].strip()[:20]
            
            decomp_results.append({
                'function': func_short,
                'complexity': int(comp_arr[ci]),
                'reconstruction_error': float(error),
                'n_significant': int(significant),
                'total_elementary': len(elementary_indices)
            })
        except:
            pass
    
    mean_error = np.mean([d['reconstruction_error'] for d in decomp_results]) if decomp_results else 1.0
    mean_components = np.mean([d['n_significant'] for d in decomp_results]) if decomp_results else 0
    
    print(f"\n  Mean reconstruction error: {mean_error:.4f}")
    print(f"  Mean significant components: {mean_components:.1f}")
    print(f"  {'DECOMPOSITION WORKS!' if mean_error < 0.3 else 'Partial decomposition' if mean_error < 0.6 else 'Decomposition fails'}")
    
    # Time reversal: project complex functions BACK along PC2 (arrow of time)
    pc2_corr = np.corrcoef(comp_arr, vecs_pca[:, 1])[0, 1]
    pc2_sign = np.sign(pc2_corr)
    
    nn = NearestNeighbors(n_neighbors=3)
    nn.fit(all_vecs)
    
    reversal_results = []
    for ci in complex_indices[:15]:
        original = all_vecs[ci].copy()
        original_comp = comp_arr[ci]
        
        # Move BACKWARD along PC2 (reverse the arrow)
        pc2_direction = pca.components_[1]
        step_sizes = [0.1, 0.2, 0.5, 1.0, 2.0]
        
        for step in step_sizes:
            reversed_vec = original - pc2_sign * step * pc2_direction
            _, idx = nn.kneighbors(reversed_vec.reshape(1, -1))
            nearest = unique_funcs[idx[0, 0]]
            nearest_comp = complexity[nearest]
            
            if nearest_comp < original_comp:
                func_short = unique_funcs[ci].split('return ')[-1].strip()[:15]
                nearest_short = nearest.split('return ')[-1].strip()[:15]
                reversal_results.append({
                    'from': func_short,
                    'to': nearest_short,
                    'step': step,
                    'comp_from': int(original_comp),
                    'comp_to': int(nearest_comp),
                    'simplified': True
                })
                break
        else:
            func_short = unique_funcs[ci].split('return ')[-1].strip()[:15]
            reversal_results.append({
                'from': func_short, 'to': '(no simplification)',
                'step': 0, 'comp_from': int(original_comp), 'comp_to': 0, 'simplified': False
            })
    
    n_reversed = sum(1 for r in reversal_results if r['simplified'])
    print(f"\n--- Time Reversal Results ---")
    print(f"  Successfully simplified: {n_reversed}/{len(reversal_results)}")
    for r in reversal_results[:5]:
        if r['simplified']:
            print(f"    {r['from']} (c={r['comp_from']}) -> {r['to']} (c={r['comp_to']}) @ step={r['step']}")
    
    # CPT symmetry: is the space symmetric under complexity-reversal?
    # Check if distance between functions of equal complexity is LESS than different complexity
    same_comp_dists = []
    diff_comp_dists = []
    for i in range(min(100, n)):
        for j in range(i+1, min(100, n)):
            d = np.linalg.norm(all_vecs[i] - all_vecs[j])
            if abs(comp_arr[i] - comp_arr[j]) <= 1:
                same_comp_dists.append(d)
            else:
                diff_comp_dists.append(d)
    
    mean_same = np.mean(same_comp_dists) if same_comp_dists else 0
    mean_diff = np.mean(diff_comp_dists) if diff_comp_dists else 0
    
    print(f"\n--- CPT Symmetry ---")
    print(f"  Same-complexity mean dist: {mean_same:.4f}")
    print(f"  Diff-complexity mean dist: {mean_diff:.4f}")
    print(f"  Ratio: {mean_same/mean_diff:.3f}" if mean_diff > 0 else "  N/A")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Phase 116: Time Reversal Symmetry', fontsize=14, fontweight='bold')
    
    if decomp_results:
        errors = [d['reconstruction_error'] for d in decomp_results]
        comps = [d['complexity'] for d in decomp_results]
        axes[0].scatter(comps, errors, s=30, alpha=0.6, color='#E91E63')
        axes[0].set_xlabel('Complexity'); axes[0].set_ylabel('Reconstruction Error')
        axes[0].set_title(f'Decomposition (mean err={mean_error:.3f})')
    
    if reversal_results:
        cats = ['Simplified', 'Failed']
        vals = [n_reversed, len(reversal_results)-n_reversed]
        axes[1].bar(cats, vals, color=['#4CAF50','#F44336'], edgecolor='black')
        axes[1].set_title(f'Time Reversal: {n_reversed}/{len(reversal_results)}')
    
    axes[2].scatter(comp_arr, vecs_pca[:,1], alpha=0.3, s=20, c='#2196F3')
    axes[2].set_xlabel('Complexity')
    axes[2].set_ylabel('PC2 (Arrow of Time)')
    axes[2].set_title(f'PC2 vs Complexity (r={pc2_corr:.3f})')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase116_reversal.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    results = {
        'phase': 116, 'title': 'Time Reversal Symmetry',
        'mean_reconstruction_error': float(mean_error),
        'mean_significant_components': float(mean_components),
        'n_reversed': n_reversed,
        'n_total_reversal': len(reversal_results),
        'pc2_complexity_corr': float(pc2_corr),
        'same_comp_dist': float(mean_same),
        'diff_comp_dist': float(mean_diff),
        'law': f'Time reversal: {n_reversed}/{len(reversal_results)} simplified via PC2 reversal. Decomposition error={mean_error:.3f}. {"Reversible!" if n_reversed > len(reversal_results)//2 else "Partially reversible."}'
    }
    with open(os.path.join(RESULTS_DIR, 'phase116_reversal.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nPhase 116 complete!")
    return results

if __name__ == '__main__':
    main()
