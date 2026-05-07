"""Phase 99: Renormalization Group Flow - Scale invariance in program space.
Coarse-grain the space by merging nearby functions and study what structure survives.
In physics, RG flow reveals universal behavior independent of microscopic details.
"""
import os, json, sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from sklearn.cluster import AgglomerativeClustering
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
    print("Phase 99: Renormalization Group Flow")
    print("  What survives coarse-graining?")
    print("=" * 60)
    
    latents = np.load(os.path.join(BASE_DIR, 'data', 'rosetta_latents_v2.npz'))
    dataset = json.load(open(os.path.join(BASE_DIR, 'data', 'rosetta_dataset_v2.json'), 'r', encoding='utf-8'))
    ast_vectors = latents['ast']
    sources = [item['source'] for item in dataset['dataset']]
    
    func_to_vecs = {}
    for i, src in enumerate(sources):
        if src not in func_to_vecs: func_to_vecs[src] = []
        func_to_vecs[src].append(ast_vectors[i])
    func_means = {s: np.mean(v, axis=0) for s, v in func_to_vecs.items()}
    unique = list(func_means.keys())
    all_vecs = np.array([func_means[f] for f in unique])
    
    n_funcs = len(unique)
    print(f"  Starting with {n_funcs} functions")
    
    # RG flow: progressively merge nearest clusters
    scales = [n_funcs, n_funcs//2, n_funcs//4, n_funcs//8, n_funcs//16, max(5, n_funcs//32)]
    
    rg_results = []
    
    for n_clusters in scales:
        if n_clusters < 2: continue
        
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        labels = clustering.fit_predict(all_vecs)
        
        # Coarse-grained vectors (cluster centroids)
        centroids = np.array([all_vecs[labels == k].mean(axis=0) for k in range(n_clusters)])
        
        # Measure properties at this scale
        pca = PCA(n_components=min(5, n_clusters-1))
        pca.fit(centroids)
        
        # Effective dimensionality (participation ratio)
        if len(pca.explained_variance_) >= 2:
            evs = pca.explained_variance_
            p_ratio = (np.sum(evs)**2) / np.sum(evs**2)
        else:
            p_ratio = 1.0
        
        # Spectral ratio (PC1 dominance)
        spectral_ratio = pca.explained_variance_ratio_[0] if len(pca.explained_variance_ratio_) > 0 else 1.0
        
        # Packing fraction
        centroid_of_centroids = np.mean(centroids, axis=0)
        dists = np.linalg.norm(centroids - centroid_of_centroids, axis=1)
        packing = np.mean(dists) / (np.max(dists) + 1e-10)
        
        # Eigenvalue gap
        if len(pca.explained_variance_) >= 2:
            gap = pca.explained_variance_[0] / pca.explained_variance_[1]
        else:
            gap = 0
        
        result = {
            'n_clusters': int(n_clusters),
            'effective_dim': float(p_ratio),
            'spectral_ratio': float(spectral_ratio),
            'packing': float(packing),
            'eigenvalue_gap': float(gap)
        }
        rg_results.append(result)
        
        print(f"  Scale {n_clusters:3d}: eff_dim={p_ratio:.2f}, spectral={spectral_ratio:.3f}, "
              f"packing={packing:.3f}, gap={gap:.2f}")
    
    # Check for fixed points (values that don't change under RG flow)
    dims = [r['effective_dim'] for r in rg_results]
    packings = [r['packing'] for r in rg_results]
    spectral_ratios = [r['spectral_ratio'] for r in rg_results]
    
    dim_stability = np.std(dims) / np.mean(dims) if len(dims) > 1 else 0
    packing_stability = np.std(packings) / np.mean(packings) if len(packings) > 1 else 0
    spectral_stability = np.std(spectral_ratios) / np.mean(spectral_ratios) if len(spectral_ratios) > 1 else 0
    
    print(f"\n--- RG Fixed Points ---")
    print(f"  Effective dim CV:   {dim_stability:.3f} {'(FIXED POINT!)' if dim_stability < 0.2 else ''}")
    print(f"  Packing CV:         {packing_stability:.3f} {'(FIXED POINT!)' if packing_stability < 0.1 else ''}")
    print(f"  Spectral ratio CV:  {spectral_stability:.3f} {'(FIXED POINT!)' if spectral_stability < 0.2 else ''}")
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Phase 99: Renormalization Group Flow', fontsize=14, fontweight='bold')
    
    x_scale = [r['n_clusters'] for r in rg_results]
    
    axes[0, 0].plot(x_scale, dims, 'o-', color='#2196F3', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Scale (# clusters)')
    axes[0, 0].set_ylabel('Effective Dimensionality')
    axes[0, 0].set_title(f'Dim under RG flow (CV={dim_stability:.3f})')
    axes[0, 0].invert_xaxis()
    
    axes[0, 1].plot(x_scale, packings, 's-', color='#4CAF50', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Scale (# clusters)')
    axes[0, 1].set_ylabel('Packing Fraction')
    axes[0, 1].set_title(f'Packing under RG flow (CV={packing_stability:.3f})')
    axes[0, 1].invert_xaxis()
    
    axes[1, 0].plot(x_scale, spectral_ratios, 'D-', color='#E91E63', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Scale (# clusters)')
    axes[1, 0].set_ylabel('PC1 Dominance')
    axes[1, 0].set_title(f'Spectral ratio under RG flow (CV={spectral_stability:.3f})')
    axes[1, 0].invert_xaxis()
    
    axes[1, 1].axis('off')
    fixed_points = []
    if dim_stability < 0.2: fixed_points.append(f"eff_dim ~ {np.mean(dims):.1f}")
    if packing_stability < 0.1: fixed_points.append(f"packing ~ {np.mean(packings):.3f}")
    if spectral_stability < 0.2: fixed_points.append(f"spectral ~ {np.mean(spectral_ratios):.3f}")
    
    summary = f"""RENORMALIZATION GROUP FLOW

Scales tested: {len(rg_results)}
({' -> '.join(str(r['n_clusters']) for r in rg_results)})

Fixed points found: {len(fixed_points)}
{chr(10).join(f'  * {fp}' for fp in fixed_points) if fixed_points else '  (none)'}

Stability (CV):
  Effective dim:  {dim_stability:.3f}
  Packing:        {packing_stability:.3f}
  Spectral ratio: {spectral_stability:.3f}

Properties that survive coarse-graining
are the TRUE universal laws."""
    axes[1, 1].text(0.05, 0.5, summary, fontsize=10, fontfamily='monospace',
                    verticalalignment='center', transform=axes[1, 1].transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase99_renormalization.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    results = {
        'phase': 99, 'title': 'Renormalization Group Flow',
        'scales': rg_results,
        'dim_stability_cv': float(dim_stability),
        'packing_stability_cv': float(packing_stability),
        'spectral_stability_cv': float(spectral_stability),
        'fixed_points': fixed_points,
        'law': f'RG flow: {len(fixed_points)} fixed points. Packing CV={packing_stability:.3f}, dim CV={dim_stability:.3f}. Scale-invariant properties = true universal laws.'
    }
    with open(os.path.join(RESULTS_DIR, 'phase99_renormalization.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nPhase 99 complete!")
    return results

if __name__ == '__main__':
    main()
