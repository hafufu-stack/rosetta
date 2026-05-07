"""Phase 128: Hawking Radiation - Does info escape from code singularities?
Drop function vectors into the gravitational singularity (x+y supernode),
decode the 'radiation' at the event horizon.
"""
import os, json, sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from scipy.spatial.distance import cdist
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
    print("Phase 128: Hawking Radiation")
    print("  Does information escape code singularities?")
    print("=" * 60)
    
    latents = np.load(os.path.join(DATA_DIR, 'rosetta_latents_v2.npz'))
    dataset = json.load(open(os.path.join(DATA_DIR, 'rosetta_dataset_v2.json'), 'r', encoding='utf-8'))
    ast_vectors = latents['ast']
    sources = [item['source'] for item in dataset['dataset']]
    
    func_ast = {}
    for i, src in enumerate(sources):
        if src not in func_ast: func_ast[src] = []
        func_ast[src].append(ast_vectors[i])
    unique_funcs = list(func_ast.keys())
    ast_m = np.array([np.mean(func_ast[f], axis=0) for f in unique_funcs])
    n = len(unique_funcs)
    
    # 1. Identify singularity (heaviest node = most neighbors)
    cos_sim = ast_m @ ast_m.T / (np.linalg.norm(ast_m, axis=1, keepdims=True) @ np.linalg.norm(ast_m, axis=1, keepdims=True).T + 1e-10)
    np.fill_diagonal(cos_sim, 0)
    masses = np.sum(cos_sim > 0.9, axis=1)
    singularity_idx = np.argmax(masses)
    singularity_func = unique_funcs[singularity_idx].split('return ')[-1].strip()[:20]
    singularity_v = ast_m[singularity_idx]
    print(f"  Singularity: '{singularity_func}' (mass={masses[singularity_idx]})")
    
    # 2. Drop 10 diverse functions toward the singularity
    distances = np.linalg.norm(ast_m - singularity_v, axis=1)
    # Select functions at medium distance (not too close, not too far)
    med_dist = np.median(distances)
    candidates = np.where((distances > med_dist * 0.5) & (distances < med_dist * 1.5))[0]
    np.random.seed(42)
    test_particles = np.random.choice(candidates, min(10, len(candidates)), replace=False)
    
    hawking_results = []
    
    for p_idx in test_particles:
        particle_v = ast_m[p_idx]
        particle_func = unique_funcs[p_idx].split('return ')[-1].strip()[:15]
        
        # Simulate infall: interpolate toward singularity
        event_horizon_radius = 0.3
        n_steps = 50
        
        trajectory = []
        radiation_vectors = []
        
        for t in range(n_steps):
            frac = t / n_steps
            current = (1 - frac) * particle_v + frac * singularity_v
            dist_to_singularity = np.linalg.norm(current - singularity_v)
            
            # At each step, the particle emits "radiation" = component orthogonal to infall direction
            if t > 0:
                infall_dir = singularity_v - current
                infall_dir /= np.linalg.norm(infall_dir) + 1e-10
                
                # Radiation = component perpendicular to infall
                proj = np.dot(current - trajectory[-1], infall_dir) * infall_dir
                radiation = (current - trajectory[-1]) - proj
                radiation_vectors.append(radiation)
            
            trajectory.append(current.copy())
            
            if dist_to_singularity < event_horizon_radius:
                break
        
        # 3. Decode radiation: can we recover the original function?
        if radiation_vectors:
            total_radiation = np.sum(radiation_vectors, axis=0)
            rad_norm = np.linalg.norm(total_radiation)
            
            # Try to reconstruct original from singularity + radiation
            reconstructed = singularity_v + total_radiation * 3  # amplify radiation
            
            # Find nearest function to reconstruction
            recon_dists = np.linalg.norm(ast_m - reconstructed.reshape(1,-1), axis=1)
            nearest_idx = np.argmin(recon_dists)
            nearest_func = unique_funcs[nearest_idx].split('return ')[-1].strip()[:15]
            recovered = nearest_idx == p_idx
            
            # Information retention: cosine similarity between original and reconstructed direction
            cos_retention = np.dot(particle_v - singularity_v, reconstructed - singularity_v)
            cos_retention /= (np.linalg.norm(particle_v - singularity_v) * np.linalg.norm(reconstructed - singularity_v) + 1e-10)
            
            hawking_results.append({
                'particle': particle_func,
                'radiation_magnitude': float(rad_norm),
                'reconstructed_to': nearest_func,
                'recovered': bool(recovered),
                'info_retention': float(cos_retention),
                'steps_to_horizon': len(trajectory),
            })
            
            status = "RECOVERED!" if recovered else f"-> {nearest_func}"
            print(f"  {particle_func}: radiation={rad_norm:.4f}, retention={cos_retention:.3f} [{status}]")
    
    # 4. Summary statistics
    n_recovered = sum(1 for r in hawking_results if r['recovered'])
    total = len(hawking_results)
    mean_retention = np.mean([r['info_retention'] for r in hawking_results])
    mean_radiation = np.mean([r['radiation_magnitude'] for r in hawking_results])
    
    print(f"\n--- Hawking Radiation Summary ---")
    print(f"  Information recovery: {n_recovered}/{total} ({n_recovered/total*100:.0f}%)")
    print(f"  Mean info retention: {mean_retention:.4f}")
    print(f"  Mean radiation magnitude: {mean_radiation:.4f}")
    
    paradox_resolved = mean_retention > 0.5
    print(f"  Information paradox: {'RESOLVED (info preserved)' if paradox_resolved else 'UNRESOLVED (info lost)'}")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Phase 128: Hawking Radiation', fontsize=14, fontweight='bold')
    
    labels = [r['particle'][:10] for r in hawking_results]
    retentions = [r['info_retention'] for r in hawking_results]
    colors = ['#4CAF50' if r['recovered'] else '#F44336' for r in hawking_results]
    axes[0].barh(labels, retentions, color=colors, edgecolor='black')
    axes[0].set_xlabel('Info Retention (cosine)')
    axes[0].set_title(f'Hawking Radiation: {n_recovered}/{total} recovered')
    
    rads = [r['radiation_magnitude'] for r in hawking_results]
    axes[1].scatter(rads, retentions, c=colors, s=80, edgecolor='black')
    axes[1].set_xlabel('Radiation magnitude'); axes[1].set_ylabel('Info retention')
    axes[1].set_title('Radiation vs Information')
    
    from sklearn.decomposition import PCA
    pca_2d = PCA(n_components=2).fit_transform(ast_m)
    axes[2].scatter(pca_2d[:,0], pca_2d[:,1], s=10, alpha=0.2, c='gray')
    axes[2].scatter(pca_2d[singularity_idx, 0], pca_2d[singularity_idx, 1], s=200, c='black', marker='*', zorder=10, label='Singularity')
    for p_idx in test_particles:
        axes[2].annotate('', xy=pca_2d[singularity_idx], xytext=pca_2d[p_idx],
                        arrowprops=dict(arrowstyle='->', color='red', alpha=0.5))
    axes[2].legend(); axes[2].set_title('Infall trajectories')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase128_hawking.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    results = {
        'phase': 128, 'title': 'Hawking Radiation',
        'singularity': singularity_func,
        'n_recovered': n_recovered, 'total': total,
        'recovery_rate_pct': float(n_recovered/total*100),
        'mean_info_retention': float(mean_retention),
        'mean_radiation_magnitude': float(mean_radiation),
        'paradox_resolved': bool(paradox_resolved),
        'details': hawking_results,
        'law': f'Singularity={singularity_func}. Recovery={n_recovered}/{total} ({n_recovered/total*100:.0f}%). Info retention={mean_retention:.3f}. Paradox {"resolved" if paradox_resolved else "unresolved"}.'
    }
    with open(os.path.join(RESULTS_DIR, 'phase128_hawking.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nPhase 128 complete!")
    return results

if __name__ == '__main__':
    main()
