"""Phase 137: False Vacuum Decay
Is the current Python universe in a false vacuum (L_min != 0)?
Inject quantum fluctuations and see if a phase transition occurs.
"""
import os, json, sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from sklearn.decomposition import PCA
from scipy.optimize import minimize
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
    print("Phase 137: False Vacuum Decay")
    print("  Is this universe in a metastable state?")
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
    
    # Current vacuum energy
    G, lam, mu = 1.1732, 0.7282, 1.0717
    
    def potential_energy(v):
        dists = np.linalg.norm(ast_m - v.reshape(1,-1), axis=1)
        V_grav = -G * np.mean(1.0 / (dists ** 2 + 0.01))
        V_holo = lam * np.sum(v ** 2)
        return V_grav + V_holo
    
    # Measure potential at current vacuum (centroid)
    centroid = np.mean(ast_m, axis=0)
    V_current = potential_energy(centroid)
    print(f"  Current vacuum energy: {V_current:.6f}")
    
    # 1. Scan the potential landscape
    np.random.seed(42)
    n_probes = 200
    bbox_min, bbox_max = ast_m.min(axis=0), ast_m.max(axis=0)
    probes = np.random.uniform(bbox_min * 1.5, bbox_max * 1.5, size=(n_probes, 64))
    
    potentials = [potential_energy(p) for p in probes]
    potentials = np.array(potentials)
    
    # Include function locations
    func_potentials = [potential_energy(ast_m[i]) for i in range(n)]
    func_potentials = np.array(func_potentials)
    
    global_min = min(np.min(potentials), np.min(func_potentials))
    global_min_idx = np.argmin(np.concatenate([potentials, func_potentials]))
    
    print(f"  Global minimum potential: {global_min:.6f}")
    print(f"  Vacuum gap (false - true): {V_current - global_min:.6f}")
    
    is_false_vacuum = V_current > global_min + 0.001
    print(f"  False vacuum: {'YES!' if is_false_vacuum else 'No (true vacuum)'}")
    
    # 2. Quantum tunneling: inject fluctuations
    fluctuation_sizes = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    decay_results = []
    
    print("\n--- Vacuum Decay Test ---")
    for sigma in fluctuation_sizes:
        n_trials = 50
        decayed = 0
        new_potentials = []
        
        for trial in range(n_trials):
            fluctuation = np.random.randn(64) * sigma
            perturbed = centroid + fluctuation
            V_new = potential_energy(perturbed)
            new_potentials.append(V_new)
            if V_new < V_current - 0.001:
                decayed += 1
        
        decay_rate = decayed / n_trials * 100
        mean_new_V = np.mean(new_potentials)
        decay_results.append({
            'sigma': float(sigma), 'decay_rate': float(decay_rate),
            'mean_V_after': float(mean_new_V)
        })
        print(f"  sigma={sigma:.2f}: decay rate={decay_rate:.0f}%, mean V after={mean_new_V:.4f}")
    
    # 3. Phase transition: what happens after decay?
    # Find the true vacuum location
    all_V = np.concatenate([potentials, func_potentials])
    all_locs = np.vstack([probes, ast_m])
    true_vacuum_loc = all_locs[np.argmin(all_V)]
    
    # What function is at the true vacuum?
    dist_to_true = np.linalg.norm(ast_m - true_vacuum_loc.reshape(1,-1), axis=1)
    nearest_true = np.argmin(dist_to_true)
    true_vacuum_func = unique_funcs[nearest_true].split('return ')[-1].strip()[:20]
    
    print(f"\n--- True Vacuum ---")
    print(f"  Location nearest function: {true_vacuum_func}")
    print(f"  Energy: {global_min:.6f}")
    
    # 4. Bubble nucleation: does the decay propagate?
    print("\n--- Bubble Nucleation ---")
    bubble_center = true_vacuum_loc
    bubble_radii = [0.1, 0.3, 0.5, 1.0, 2.0]
    
    for r in bubble_radii:
        inside = np.linalg.norm(ast_m - bubble_center.reshape(1,-1), axis=1) < r
        n_converted = int(np.sum(inside))
        if n_converted > 0:
            V_inside = np.mean([potential_energy(ast_m[i]) for i in range(n) if inside[i]])
        else:
            V_inside = V_current
        print(f"  Bubble r={r}: {n_converted} functions converted, V_avg={V_inside:.4f}")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Phase 137: False Vacuum Decay', fontsize=14, fontweight='bold')
    
    pca_2d = PCA(n_components=2).fit(ast_m)
    all_2d = pca_2d.transform(all_locs[:n_probes])
    func_2d = pca_2d.transform(ast_m)
    
    sc = axes[0].scatter(func_2d[:,0], func_2d[:,1], c=func_potentials, cmap='RdYlGn_r', s=20, alpha=0.7)
    plt.colorbar(sc, ax=axes[0], label='V (potential)')
    true_2d = pca_2d.transform(true_vacuum_loc.reshape(1,-1))
    axes[0].scatter(true_2d[0,0], true_2d[0,1], s=200, c='blue', marker='*', zorder=10, label='True vacuum')
    centroid_2d = pca_2d.transform(centroid.reshape(1,-1))
    axes[0].scatter(centroid_2d[0,0], centroid_2d[0,1], s=200, c='red', marker='D', zorder=10, label='False vacuum')
    axes[0].legend(fontsize=7); axes[0].set_title('Potential landscape')
    
    sigmas = [d['sigma'] for d in decay_results]
    rates = [d['decay_rate'] for d in decay_results]
    axes[1].plot(sigmas, rates, 'o-', color='#E91E63', linewidth=2, markersize=8)
    axes[1].set_xlabel('Fluctuation sigma'); axes[1].set_ylabel('Decay rate (%)')
    axes[1].set_title('Vacuum Decay Rate'); axes[1].set_xscale('log')
    
    axes[2].hist(func_potentials, bins=30, color='#2196F3', edgecolor='black', alpha=0.7)
    axes[2].axvline(V_current, color='red', linestyle='--', label=f'False vacuum V={V_current:.3f}')
    axes[2].axvline(global_min, color='blue', linestyle='--', label=f'True vacuum V={global_min:.3f}')
    axes[2].legend(); axes[2].set_xlabel('Potential V'); axes[2].set_title('Potential distribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase137_vacuum.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    results = {
        'phase': 137, 'title': 'False Vacuum Decay',
        'V_current': float(V_current), 'V_true': float(global_min),
        'vacuum_gap': float(V_current - global_min),
        'is_false_vacuum': bool(is_false_vacuum),
        'true_vacuum_func': true_vacuum_func,
        'decay_results': decay_results,
        'law': f'V_false={V_current:.4f}, V_true={global_min:.4f}, gap={V_current-global_min:.4f}. {"FALSE VACUUM confirmed" if is_false_vacuum else "True vacuum"}. True vacuum near: {true_vacuum_func}.'
    }
    with open(os.path.join(RESULTS_DIR, 'phase137_vacuum.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nPhase 137 complete!")
    return results

if __name__ == '__main__':
    main()
