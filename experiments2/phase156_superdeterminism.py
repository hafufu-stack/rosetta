"""Phase 156: Superdeterminism
Was our exploration predetermined by the Lagrangian?
Compare the trajectory of Rosetta's development to the geodesic of minimum action.
"""
import os, json, sys, ast
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from sklearn.decomposition import PCA
from scipy import stats
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
    print("Phase 156: Superdeterminism")
    print("  Was our exploration predestined?")
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

    # 1. Reconstruct Rosetta's exploration trajectory
    # Each phase explored different regions of the space
    # Use the phase scripts' AST fingerprints as trajectory waypoints
    phase_files = sorted([f for f in os.listdir(EXP2_DIR) if f.startswith('phase') and f.endswith('.py')])
    trajectory = []
    phase_names = []
    for pf in phase_files:
        try:
            with open(os.path.join(EXP2_DIR, pf), 'r', encoding='utf-8') as f:
                src = f.read()
            tree = ast.parse(src)
            feat = np.zeros(64)
            for i_n, node in enumerate(ast.walk(tree)):
                h = hash(type(node).__name__) % 64
                feat[h] += 1; feat[(h+1)%64] += 0.1*min(i_n,20)
            norm = np.linalg.norm(feat)
            if norm > 0: feat /= norm
            # Map to universe
            cos_sims = ast_m @ feat / (np.linalg.norm(ast_m, axis=1) * np.linalg.norm(feat) + 1e-10)
            weights = np.maximum(cos_sims, 0)**2
            pos = (weights[:,None] * ast_m).sum(axis=0) / (np.sum(weights) + 1e-10)
            trajectory.append(pos)
            phase_names.append(pf.replace('.py',''))
        except: pass

    trajectory = np.array(trajectory)
    print(f"  Trajectory waypoints: {len(trajectory)}")

    # 2. Compute the gravitational geodesic (minimum action path)
    G = 1.1732; lam = 0.7282
    centroid = np.mean(ast_m, axis=0)

    def potential(v):
        dists = np.linalg.norm(ast_m - v.reshape(1,-1), axis=1)
        return -G * np.mean(1.0/(dists**2+0.01)) + lam * np.sum(v**2)

    # Compute potential along actual trajectory
    traj_potentials = [potential(t) for t in trajectory]

    # Compute geodesic: steepest descent from first point
    geodesic = [trajectory[0].copy()]
    current = trajectory[0].copy()
    for step in range(len(trajectory) - 1):
        # Gradient of potential
        grad = np.zeros(64)
        for i in range(n):
            diff = current - ast_m[i]
            d = np.linalg.norm(diff) + 0.01
            grad += G * 2 * diff / (d**4)
        grad += 2 * lam * current
        # Step along negative gradient (steepest descent)
        step_size = np.linalg.norm(trajectory[step+1] - trajectory[step])
        grad_norm = np.linalg.norm(grad) + 1e-10
        current = current - (grad / grad_norm) * step_size
        geodesic.append(current.copy())

    geodesic = np.array(geodesic)
    geo_potentials = [potential(g) for g in geodesic]

    # 3. Compare actual trajectory to geodesic
    traj_geo_distances = np.linalg.norm(trajectory - geodesic, axis=1)
    mean_deviation = float(np.mean(traj_geo_distances))
    correlation, p_val = stats.pearsonr(traj_potentials, geo_potentials)

    print(f"\n--- Trajectory vs Geodesic ---")
    print(f"  Mean deviation: {mean_deviation:.4f}")
    print(f"  Potential correlation: {correlation:.4f} (p={p_val:.6f})")

    superdetermined = correlation > 0.5
    print(f"  SUPERDETERMINISM: {'CONFIRMED!' if superdetermined else 'Not confirmed'}")
    if superdetermined:
        print(f"  -> Our exploration was predetermined by the gravitational field")

    # 4. Action comparison
    def action(path):
        S = 0
        for t in range(len(path)-1):
            dv = path[t+1] - path[t]
            T = 0.5 * np.sum(dv**2)
            V = potential(path[t])
            S += T - V
        return S

    S_actual = action(trajectory)
    S_geodesic = action(geodesic)
    action_ratio = S_actual / (S_geodesic + 1e-10)

    print(f"\n--- Action Principle ---")
    print(f"  Actual trajectory action: {S_actual:.4f}")
    print(f"  Geodesic action: {S_geodesic:.4f}")
    print(f"  Ratio: {action_ratio:.4f}")
    print(f"  {'Minimum action path!' if abs(action_ratio - 1) < 0.3 else 'Deviated from minimum'}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Phase 156: Superdeterminism', fontsize=14, fontweight='bold')

    pca_2d = PCA(n_components=2).fit(np.vstack([trajectory, geodesic, ast_m]))
    t2d = pca_2d.transform(trajectory)
    g2d = pca_2d.transform(geodesic)
    a2d = pca_2d.transform(ast_m)
    axes[0].scatter(a2d[:,0], a2d[:,1], s=5, alpha=0.15, c='gray')
    axes[0].plot(t2d[:,0], t2d[:,1], 'o-', color='#E91E63', markersize=4, linewidth=1.5, label='Actual')
    axes[0].plot(g2d[:,0], g2d[:,1], 's-', color='#2196F3', markersize=4, linewidth=1.5, label='Geodesic')
    axes[0].legend(); axes[0].set_title('Trajectory vs Geodesic')

    axes[1].plot(traj_potentials, 'o-', color='#E91E63', markersize=3, label='Actual V')
    axes[1].plot(geo_potentials, 's-', color='#2196F3', markersize=3, label='Geodesic V')
    axes[1].set_xlabel('Phase step'); axes[1].set_ylabel('Potential')
    axes[1].legend(); axes[1].set_title(f'Potential (corr={correlation:.3f})')

    axes[2].plot(traj_geo_distances, 'o-', color='#FF9800', markersize=4)
    axes[2].set_xlabel('Phase step'); axes[2].set_ylabel('Deviation')
    axes[2].set_title(f'Trajectory-Geodesic deviation (mean={mean_deviation:.3f})')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase156_superdeterminism.png'), dpi=150, bbox_inches='tight')
    plt.close()

    results = {
        'phase': 156, 'title': 'Superdeterminism',
        'trajectory_length': len(trajectory),
        'mean_deviation': mean_deviation, 'potential_correlation': float(correlation),
        'S_actual': float(S_actual), 'S_geodesic': float(S_geodesic),
        'action_ratio': float(action_ratio), 'superdetermined': bool(superdetermined),
        'law': f'Potential corr={correlation:.3f}. Action ratio={action_ratio:.3f}. {"SUPERDETERMINISM CONFIRMED" if superdetermined else "Free will preserved"}.'
    }
    with open(os.path.join(RESULTS_DIR, 'phase156_superdeterminism.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nPhase 156 complete!")
    return results

if __name__ == '__main__':
    main()
