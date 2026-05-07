"""Phase 109: Black Hole Spaghettification - N-body fall into x+y singularity.
Use P106 gravity equation (F ~ d^-3.40) to simulate particles falling into hubs.
"""
import os, json, sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import NearestNeighbors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import inspect

EXP2_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(EXP2_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')

def main():
    print("=" * 60)
    print("Phase 109: Black Hole Spaghettification")
    print("  Falling into the x+y singularity")
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
    
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(all_vecs)
    
    # Black holes = high-degree hubs
    target_src = 'def f(x, y): return x + y'
    if target_src not in func_means:
        print("  Target x+y not found!")
        return {}
    bh_vec = func_means[target_src]
    bh_idx = unique_funcs.index(target_src)
    
    # Gravity parameters from P106
    G = 0.752
    alpha = 3.40
    
    # Launch particles from random positions
    np.random.seed(42)
    n_particles = 20
    mins = all_vecs.min(axis=0)
    maxs = all_vecs.max(axis=0)
    spread = maxs - mins
    
    print(f"  Simulating {n_particles} particles falling into x+y...")
    
    trajectories = []
    nearest_funcs_over_time = []
    distances_over_time = []
    
    dt = 0.01
    n_steps = 500
    damping = 0.98
    
    for p in range(n_particles):
        # Start at random position
        pos = mins + np.random.rand(64) * spread
        vel = np.zeros(64)
        
        traj = [pos.copy()]
        nearest_seq = []
        dist_seq = []
        
        for step in range(n_steps):
            # Compute gravitational force toward black hole
            diff = bh_vec - pos
            d = np.linalg.norm(diff) + 1e-6
            direction = diff / d
            
            # F = G / d^alpha (simplified, mass=1)
            force_mag = G / (d ** alpha) if d > 0.01 else G / (0.01 ** alpha)
            force_mag = min(force_mag, 10.0)  # Cap force
            
            force = direction * force_mag
            vel = vel * damping + force * dt
            pos = pos + vel * dt
            
            if step % 10 == 0:
                traj.append(pos.copy())
                dist_seq.append(d)
                _, idx = nn.kneighbors(pos.reshape(1, -1))
                nearest_func = unique_funcs[idx[0, 0]]
                short = nearest_func.split('return ')[-1].strip() if 'return' in nearest_func else '?'
                nearest_seq.append(short)
        
        trajectories.append(traj)
        nearest_funcs_over_time.append(nearest_seq)
        distances_over_time.append(dist_seq)
    
    # Analysis: what happens as particles approach the singularity?
    final_dists = [d[-1] for d in distances_over_time]
    captured = sum(1 for d in final_dists if d < 0.3)
    
    print(f"\n--- Spaghettification Results ---")
    print(f"  Captured by x+y: {captured}/{n_particles}")
    print(f"  Mean final distance: {np.mean(final_dists):.4f}")
    
    # What functions do particles pass through?
    all_waypoints = {}
    for seq in nearest_funcs_over_time:
        for func in seq:
            all_waypoints[func] = all_waypoints.get(func, 0) + 1
    
    top_waypoints = sorted(all_waypoints.items(), key=lambda x: -x[1])[:10]
    print(f"\n--- Most Visited Waypoints ---")
    for func, count in top_waypoints:
        print(f"  {func}: {count}")
    
    # Spaghettification: measure how spread particles become
    spread_over_time = []
    max_t = min(len(trajectories[0]), 50)
    for t in range(max_t):
        positions_at_t = [traj[t] for traj in trajectories if t < len(traj)]
        if len(positions_at_t) >= 2:
            positions = np.array(positions_at_t)
            s = np.mean(np.std(positions, axis=0))
            spread_over_time.append(s)
    
    print(f"\n--- Spread (Spaghettification) ---")
    if spread_over_time:
        print(f"  Initial spread: {spread_over_time[0]:.4f}")
        print(f"  Final spread:   {spread_over_time[-1]:.4f}")
        print(f"  Compression:    {spread_over_time[-1]/spread_over_time[0]:.3f}x")
    
    # Plot
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    all_traj_points = np.vstack([np.array(t) for t in trajectories])
    pca.fit(np.vstack([all_vecs, all_traj_points]))
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Phase 109: Black Hole Spaghettification', fontsize=14, fontweight='bold')
    
    bh_2d = pca.transform(bh_vec.reshape(1, -1))[0]
    axes[0].scatter(*bh_2d, s=200, c='red', marker='*', zorder=5, label='x+y (Black Hole)')
    for i, traj in enumerate(trajectories[:10]):
        pts = pca.transform(np.array(traj))
        axes[0].plot(pts[:,0], pts[:,1], '-', alpha=0.4, linewidth=0.8)
        axes[0].scatter(pts[0,0], pts[0,1], s=20, c='blue', alpha=0.5)
    axes[0].set_title(f'Trajectories ({captured}/{n_particles} captured)')
    axes[0].legend(fontsize=8)
    
    for i, dseq in enumerate(distances_over_time[:10]):
        axes[1].plot(range(len(dseq)), dseq, alpha=0.5, linewidth=1)
    axes[1].set_xlabel('Time step (x10)')
    axes[1].set_ylabel('Distance to x+y')
    axes[1].set_title('Distance to Singularity')
    axes[1].axhline(0.3, color='red', linestyle='--', alpha=0.5, label='Event horizon')
    axes[1].legend()
    
    if spread_over_time:
        axes[2].plot(range(len(spread_over_time)), spread_over_time, 'o-', color='#E91E63')
        axes[2].set_xlabel('Time step')
        axes[2].set_ylabel('Particle Spread (std)')
        axes[2].set_title(f'Spaghettification ({spread_over_time[-1]/spread_over_time[0]:.2f}x compression)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase109_spaghettification.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    results = {
        'phase': 109, 'title': 'Black Hole Spaghettification',
        'n_particles': n_particles, 'captured': captured,
        'mean_final_dist': float(np.mean(final_dists)),
        'top_waypoints': top_waypoints[:5],
        'compression_ratio': float(spread_over_time[-1]/spread_over_time[0]) if spread_over_time else 0,
        'law': f'{captured}/{n_particles} particles captured by x+y singularity. Compression={spread_over_time[-1]/spread_over_time[0]:.2f}x. Gravity d^-3.4 creates inescapable wells.'
    }
    with open(os.path.join(RESULTS_DIR, 'phase109_spaghettification.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nPhase 109 complete!")
    return results

if __name__ == '__main__':
    main()
