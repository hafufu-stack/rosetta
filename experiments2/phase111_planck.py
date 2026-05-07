"""Phase 111: The Planck Length - Minimum meaningful distance in program space.
Below what distance do two programs become semantically indistinguishable?
"""
import os, json, sys, inspect
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
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
    print("Phase 111: The Planck Length")
    print("  Minimum meaningful distance in program space")
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
    
    # Compute ALL pairwise distances
    from scipy.spatial.distance import pdist
    all_dists = pdist(all_vecs)
    
    # For each pair, compute I/O difference
    g = {}
    test_inputs = [(1,2),(3,5),(0,0),(-1,4),(2,2)]
    func_outputs = {}
    
    for func_src in unique_funcs:
        try:
            exec(compile(func_src, '<string>', 'exec'), g)
            fn = g['f']
            try: n_args = len(inspect.signature(fn).parameters)
            except: n_args = 2
            outputs = []
            for x, y in test_inputs:
                try:
                    r = fn(x) if n_args == 1 else fn(x, y)
                    outputs.append(float(r) if isinstance(r, (int, float, bool)) else 0)
                except:
                    outputs.append(float('nan'))
            func_outputs[func_src] = outputs
        except:
            func_outputs[func_src] = [float('nan')] * len(test_inputs)
    
    # Compute behavioral distance for each pair
    pair_idx = 0
    lat_dists = []
    io_dists = []
    same_behavior = []
    
    for i in range(n):
        for j in range(i+1, n):
            d_lat = all_dists[pair_idx]
            pair_idx += 1
            
            o_i = func_outputs[unique_funcs[i]]
            o_j = func_outputs[unique_funcs[j]]
            
            if any(np.isnan(o_i)) or any(np.isnan(o_j)):
                continue
            
            d_io = np.sqrt(np.mean((np.array(o_i) - np.array(o_j))**2))
            is_same = d_io < 0.01
            
            lat_dists.append(d_lat)
            io_dists.append(d_io)
            same_behavior.append(is_same)
    
    lat_dists = np.array(lat_dists)
    io_dists = np.array(io_dists)
    same_behavior = np.array(same_behavior)
    
    # Find the Planck length: below what latent distance are ALL pairs same-behavior?
    thresholds = np.linspace(0, np.percentile(lat_dists, 50), 100)
    same_rates = []
    for t in thresholds:
        mask = lat_dists < t
        if mask.sum() > 0:
            rate = same_behavior[mask].mean()
        else:
            rate = 1.0
        same_rates.append(rate)
    
    # Planck length = largest distance where same_rate >= 0.95
    planck = 0
    for t, rate in zip(thresholds, same_rates):
        if rate >= 0.95 and t > planck:
            planck = t
    
    # Also find where same_rate drops below 0.5
    confusion_dist = 0
    for t, rate in zip(thresholds, same_rates):
        if rate < 0.5:
            confusion_dist = t
            break
    
    print(f"  Total pairs analyzed: {len(lat_dists)}")
    print(f"\n--- The Planck Length ---")
    print(f"  Planck length (95% same): {planck:.4f}")
    print(f"  Confusion dist (50% same): {confusion_dist:.4f}")
    print(f"  Min latent distance: {np.min(lat_dists):.4f}")
    print(f"  Mean latent distance: {np.mean(lat_dists):.4f}")
    
    # Below Planck length, programs are quantum-entangled (same behavior)
    below_planck = lat_dists < planck
    n_entangled = below_planck.sum()
    print(f"\n  Entangled pairs (d < planck): {n_entangled}")
    
    # Correlation between latent and behavioral distance
    valid = ~np.isinf(io_dists) & (io_dists < 1e4)
    corr = np.corrcoef(lat_dists[valid], io_dists[valid])[0, 1]
    print(f"  Latent-Behavioral correlation: {corr:.4f}")
    
    # Minimum discriminating distance per dimension
    from sklearn.decomposition import PCA
    pca = PCA(n_components=5)
    vecs_5d = pca.fit_transform(all_vecs)
    dists_5d = pdist(vecs_5d)
    
    # Planck length in 5D
    pair_idx = 0
    planck_5d_dists = []
    for i in range(n):
        for j in range(i+1, n):
            if pair_idx < len(dists_5d):
                planck_5d_dists.append(dists_5d[pair_idx])
            pair_idx += 1
    
    print(f"  Min 5D distance: {np.min(dists_5d):.4f}")
    print(f"  64D/5D ratio: {np.min(lat_dists)/np.min(dists_5d):.3f}" if np.min(dists_5d) > 0 else "  N/A")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Phase 111: The Planck Length = {planck:.4f}', fontsize=14, fontweight='bold')
    
    axes[0].scatter(lat_dists[::10], io_dists[::10], alpha=0.1, s=3, c='#2196F3')
    axes[0].axvline(planck, color='red', linestyle='--', label=f'Planck={planck:.3f}')
    axes[0].set_xlabel('Latent Distance')
    axes[0].set_ylabel('Behavioral Distance (I/O)')
    axes[0].set_title(f'Latent vs Behavioral (r={corr:.3f})')
    axes[0].legend()
    
    axes[1].plot(thresholds, same_rates, '-', color='#E91E63', linewidth=2)
    axes[1].axhline(0.95, color='green', linestyle='--', alpha=0.5, label='95%')
    axes[1].axhline(0.5, color='orange', linestyle='--', alpha=0.5, label='50%')
    axes[1].axvline(planck, color='red', linestyle='--', label=f'Planck={planck:.3f}')
    axes[1].set_xlabel('Latent Distance Threshold')
    axes[1].set_ylabel('Same-Behavior Rate')
    axes[1].set_title('Semantic Resolution Curve')
    axes[1].legend(fontsize=7)
    
    axes[2].hist(lat_dists, bins=50, color='#4CAF50', edgecolor='black', alpha=0.7, density=True)
    axes[2].axvline(planck, color='red', linewidth=2, linestyle='--', label=f'Planck={planck:.3f}')
    axes[2].set_xlabel('Pairwise Distance')
    axes[2].set_title(f'Distance Distribution ({n_entangled} entangled pairs)')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase111_planck.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    results = {
        'phase': 111, 'title': 'The Planck Length',
        'planck_length': float(planck),
        'confusion_distance': float(confusion_dist),
        'n_entangled_pairs': int(n_entangled),
        'latent_behavioral_corr': float(corr),
        'min_latent_dist': float(np.min(lat_dists)),
        'mean_latent_dist': float(np.mean(lat_dists)),
        'law': f'Planck length = {planck:.4f}. Below this, programs are semantically identical. Latent-behavioral correlation = {corr:.3f}. {n_entangled} entangled pairs.'
    }
    with open(os.path.join(RESULTS_DIR, 'phase111_planck.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nPhase 111 complete!")
    return results

if __name__ == '__main__':
    main()
