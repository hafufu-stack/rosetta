"""
Phase 80: The Semantic Gradient Field
========================================
Compute the "force field" of program space.

Where does semantic gravity pull? Where are the
attractors (stable programs) and repellers (unstable)?

The gradient of density reveals the topology of
the program universe — its mountains and valleys.
"""
import os, json, time
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')


def main():
    print("=" * 60)
    print("Phase 80: The Semantic Gradient Field")
    print("Mapping the force field of program space")
    print("=" * 60)
    t0 = time.time()
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    lat_file = os.path.join(DATA_DIR, 'rosetta_latents_v2.npz')
    if not os.path.exists(lat_file):
        lat_file = os.path.join(DATA_DIR, 'rosetta_latents.npz')
    latents = np.load(lat_file)
    z_ast = latents['ast']

    ds_file = os.path.join(DATA_DIR, 'rosetta_dataset_v2.json')
    if not os.path.exists(ds_file):
        ds_file = os.path.join(DATA_DIR, 'rosetta_dataset.json')
    with open(ds_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)['dataset']
    sources = [d['source'] for d in dataset]

    from sklearn.decomposition import PCA
    from sklearn.neighbors import KernelDensity
    pca = PCA(n_components=10).fit(z_ast)
    z_5d = pca.transform(z_ast)[:, :5]

    unique = {}
    for i, src in enumerate(sources):
        if src not in unique:
            unique[src] = z_5d[i]
    all_srcs = list(unique.keys())
    all_z5 = np.array([unique[s] for s in all_srcs])

    # Fit KDE
    kde = KernelDensity(bandwidth=0.3, kernel='gaussian').fit(all_z5)

    # Compute gradient field on a 2D slice (PC1-PC2)
    print("\n--- Computing Gradient Field (PC1-PC2 slice) ---")
    grid_size = 30
    pc1_range = np.linspace(all_z5[:, 0].min() - 0.5, all_z5[:, 0].max() + 0.5, grid_size)
    pc2_range = np.linspace(all_z5[:, 1].min() - 0.5, all_z5[:, 1].max() + 0.5, grid_size)

    density_grid = np.zeros((grid_size, grid_size))
    grad_x = np.zeros((grid_size, grid_size))
    grad_y = np.zeros((grid_size, grid_size))
    mean_345 = all_z5[:, 2:].mean(0)

    for i, x in enumerate(pc1_range):
        for j, y in enumerate(pc2_range):
            point = np.array([x, y] + list(mean_345)).reshape(1, -1)
            density_grid[j, i] = np.exp(kde.score_samples(point))[0]

    # Compute numerical gradient
    eps = 0.01
    for i, x in enumerate(pc1_range):
        for j, y in enumerate(pc2_range):
            point_x_plus = np.array([x+eps, y] + list(mean_345)).reshape(1, -1)
            point_x_minus = np.array([x-eps, y] + list(mean_345)).reshape(1, -1)
            point_y_plus = np.array([x, y+eps] + list(mean_345)).reshape(1, -1)
            point_y_minus = np.array([x, y-eps] + list(mean_345)).reshape(1, -1)

            dx = (np.exp(kde.score_samples(point_x_plus))[0] -
                  np.exp(kde.score_samples(point_x_minus))[0]) / (2 * eps)
            dy = (np.exp(kde.score_samples(point_y_plus))[0] -
                  np.exp(kde.score_samples(point_y_minus))[0]) / (2 * eps)
            grad_x[j, i] = dx
            grad_y[j, i] = dy

    # Find local maxima (attractors) and minima (repellers)
    print("\n--- Finding Attractors (Density Peaks) ---")
    from scipy.ndimage import maximum_filter, label
    local_max = maximum_filter(density_grid, size=3) == density_grid
    local_max &= density_grid > 0.05

    labeled, n_attractors = label(local_max)
    print(f"  Found {n_attractors} attractors")

    attractor_info = []
    for a in range(1, n_attractors + 1):
        coords = np.argwhere(labeled == a)
        if len(coords) > 0:
            j, i = coords[0]
            x, y = pc1_range[i], pc2_range[j]
            d = density_grid[j, i]

            # Find nearest function
            point_5d = np.array([x, y] + list(mean_345))
            dists = np.linalg.norm(all_z5 - point_5d, axis=1)
            nn_idx = np.argmin(dists)
            nn_src = all_srcs[nn_idx]
            short = nn_src.split('return ')[1][:20] if 'return' in nn_src else '?'

            print(f"  Attractor {a}: ({x:.2f}, {y:.2f}), density={d:.4f}, "
                  f"fn={short}")
            attractor_info.append({
                'pc1': float(x), 'pc2': float(y), 'density': float(d),
                'function': nn_src,
            })

    # Compute field statistics
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    avg_grad = float(np.mean(grad_magnitude))
    max_grad = float(np.max(grad_magnitude))
    print(f"\n  Average gradient magnitude: {avg_grad:.4f}")
    print(f"  Maximum gradient magnitude: {max_grad:.4f}")

    # Flow simulation: drop a particle and follow the gradient
    print("\n--- Gradient Flow Simulation ---")
    start_points = [
        np.array([0.0, 0.0] + list(mean_345)),
        np.array([0.5, 0.5] + list(mean_345)),
        np.array([-0.5, -0.5] + list(mean_345)),
    ]

    flows = []
    for start in start_points:
        pos = start.copy()
        trajectory = [pos[:2].tolist()]
        for step in range(50):
            point = pos.reshape(1, -1)
            # Compute gradient at this point
            dx = (np.exp(kde.score_samples(pos.reshape(1,-1) + np.array([eps,0,0,0,0])))[0] -
                  np.exp(kde.score_samples(pos.reshape(1,-1) - np.array([eps,0,0,0,0])))[0]) / (2*eps)
            dy = (np.exp(kde.score_samples(pos.reshape(1,-1) + np.array([0,eps,0,0,0])))[0] -
                  np.exp(kde.score_samples(pos.reshape(1,-1) - np.array([0,eps,0,0,0])))[0]) / (2*eps)
            grad = np.array([dx, dy, 0, 0, 0])
            grad_norm = np.linalg.norm(grad) + 1e-8
            pos = pos + grad / grad_norm * 0.05
            trajectory.append(pos[:2].tolist())

        # Final function
        dists = np.linalg.norm(all_z5 - pos, axis=1)
        nn_src = all_srcs[np.argmin(dists)]
        short = nn_src.split('return ')[1][:20] if 'return' in nn_src else '?'
        print(f"  Start ({start[0]:.1f},{start[1]:.1f}) -> converges to: {short}")
        flows.append({
            'start': start[:2].tolist(),
            'end': pos[:2].tolist(),
            'function': nn_src,
            'trajectory': trajectory,
        })

    elapsed = time.time() - t0
    results = {
        'phase': 80, 'name': 'The Semantic Gradient Field',
        'n_attractors': n_attractors,
        'attractors': attractor_info,
        'avg_gradient': avg_grad, 'max_gradient': max_grad,
        'flows': flows,
        'elapsed': elapsed, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase80_gradient.json'), 'w',
              encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    # Figure
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Density landscape
    im = axes[0].contourf(pc1_range, pc2_range, density_grid, levels=20,
                          cmap='viridis')
    axes[0].scatter(all_z5[:, 0], all_z5[:, 1], c='white', s=5, alpha=0.3)
    for a in attractor_info:
        axes[0].plot(a['pc1'], a['pc2'], 'r*', markersize=15)
    axes[0].set_xlabel('PC1'); axes[0].set_ylabel('PC2')
    axes[0].set_title('Density Landscape\n(stars = attractors)', fontweight='bold')
    plt.colorbar(im, ax=axes[0], shrink=0.8)

    # 2. Gradient field (quiver plot)
    skip = 3
    axes[1].quiver(pc1_range[::skip], pc2_range[::skip],
                  grad_x[::skip, ::skip], grad_y[::skip, ::skip],
                  grad_magnitude[::skip, ::skip], cmap='hot', alpha=0.8)
    axes[1].scatter(all_z5[:, 0], all_z5[:, 1], c='cyan', s=5, alpha=0.3)
    axes[1].set_xlabel('PC1'); axes[1].set_ylabel('PC2')
    axes[1].set_title('Gradient Force Field\n(arrows = semantic gravity)',
                     fontweight='bold')

    # 3. Flow trajectories
    for flow in flows:
        traj = np.array(flow['trajectory'])
        axes[2].plot(traj[:, 0], traj[:, 1], '-', linewidth=2, alpha=0.8)
        axes[2].plot(traj[0, 0], traj[0, 1], 'go', markersize=8)
        axes[2].plot(traj[-1, 0], traj[-1, 1], 'r*', markersize=12)
    axes[2].scatter(all_z5[:, 0], all_z5[:, 1], c='gray', s=5, alpha=0.2)
    axes[2].set_xlabel('PC1'); axes[2].set_ylabel('PC2')
    axes[2].set_title('Gradient Flow\n(green=start, red=attractor)',
                     fontweight='bold')

    plt.suptitle('Phase 80: The Semantic Gradient Field\n'
                 'Force Field of Program Space',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase80_gradient.png'), dpi=150)
    plt.close()
    print(f"\nPhase 80 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
