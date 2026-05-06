"""
Phase 71: The Fractal Hypothesis
==================================
Is the 5D program space SELF-SIMILAR?

Zoom into local neighborhoods and check if the local
structure mirrors the global structure. If so, programs
form a fractal — meaning the same patterns repeat at
every scale, like coastlines or fern leaves.
"""
import os, json, time
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')


def main():
    print("=" * 60)
    print("Phase 71: The Fractal Hypothesis")
    print("Is the 5D space self-similar at all scales?")
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
    from sklearn.cluster import KMeans
    from scipy.spatial.distance import pdist

    pca = PCA(n_components=10).fit(z_ast)
    z_5d = pca.transform(z_ast)[:, :5]

    # Deduplicate
    unique_z5 = []
    seen = set()
    for i, src in enumerate(sources):
        if src not in seen:
            seen.add(src)
            unique_z5.append(z_5d[i])
    unique_z5 = np.array(unique_z5)
    N = len(unique_z5)
    print(f"  Unique functions: {N}")

    # Global statistics
    global_dists = pdist(unique_z5)
    global_stats = {
        'mean': float(np.mean(global_dists)),
        'std': float(np.std(global_dists)),
        'skew': float(((global_dists - np.mean(global_dists))**3).mean() /
                      (np.std(global_dists)**3 + 1e-8)),
        'kurtosis': float(((global_dists - np.mean(global_dists))**4).mean() /
                         (np.std(global_dists)**4 + 1e-8) - 3),
    }
    print(f"\n  Global distance stats:")
    for k, v in global_stats.items():
        print(f"    {k}: {v:.4f}")

    # Local neighborhoods
    print("\n--- Analyzing Local Neighborhoods ---")
    # For each point, compute stats of its k-nearest neighbors
    from sklearn.neighbors import NearestNeighbors

    local_results = []
    for k in [5, 10, 20]:
        nn = NearestNeighbors(n_neighbors=min(k+1, N)).fit(unique_z5)
        distances, indices = nn.kneighbors(unique_z5)

        # For each neighborhood, compute local distance distribution
        local_skews = []
        local_kurtoses = []
        local_vars_explained = []

        for i in range(N):
            neighbors = unique_z5[indices[i][1:]]  # Exclude self
            if len(neighbors) < 3:
                continue

            # Local PCA: how many dims does the local patch need?
            local_pca = PCA(n_components=min(5, len(neighbors)-1)).fit(neighbors)
            var_1d = local_pca.explained_variance_ratio_[0] * 100
            local_vars_explained.append(var_1d)

            # Local distance distribution
            local_dists = pdist(neighbors)
            if len(local_dists) > 2:
                std = np.std(local_dists) + 1e-8
                skew = ((local_dists - np.mean(local_dists))**3).mean() / (std**3)
                kurt = ((local_dists - np.mean(local_dists))**4).mean() / (std**4) - 3
                local_skews.append(float(skew))
                local_kurtoses.append(float(kurt))

        avg_var_1d = np.mean(local_vars_explained)
        avg_skew = np.mean(local_skews) if local_skews else 0
        avg_kurt = np.mean(local_kurtoses) if local_kurtoses else 0

        # Self-similarity: compare local stats to global stats
        skew_similarity = 1 - abs(avg_skew - global_stats['skew']) / (abs(global_stats['skew']) + 1e-8)
        kurt_similarity = 1 - abs(avg_kurt - global_stats['kurtosis']) / (abs(global_stats['kurtosis']) + 1e-8)

        print(f"\n  K={k} neighborhood:")
        print(f"    Local 1D variance: {avg_var_1d:.1f}%")
        print(f"    Local skewness:    {avg_skew:.4f} (global: {global_stats['skew']:.4f})")
        print(f"    Local kurtosis:    {avg_kurt:.4f} (global: {global_stats['kurtosis']:.4f})")
        print(f"    Skew similarity:   {skew_similarity:.4f}")

        local_results.append({
            'k': k, 'avg_var_1d': float(avg_var_1d),
            'avg_skew': float(avg_skew), 'avg_kurtosis': float(avg_kurt),
            'skew_similarity': float(skew_similarity),
            'kurt_similarity': float(kurt_similarity),
        })

    # Box-counting dimension estimate
    print("\n--- Box-Counting Fractal Dimension ---")
    # Normalize to [0, 1] range
    z_norm = (unique_z5 - unique_z5.min(0)) / (unique_z5.max(0) - unique_z5.min(0) + 1e-8)

    box_counts = []
    epsilons = [0.5, 0.3, 0.2, 0.15, 0.1, 0.08, 0.05]
    for eps in epsilons:
        # Count non-empty boxes
        grid = set()
        for z in z_norm:
            cell = tuple((z / eps).astype(int))
            grid.add(cell)
        box_counts.append(len(grid))
        print(f"  eps={eps:.2f}: {len(grid)} non-empty boxes")

    # Fit log-log line for fractal dimension
    log_eps = np.log(epsilons)
    log_count = np.log(box_counts)
    coeffs = np.polyfit(log_eps, log_count, 1)
    fractal_dim = -coeffs[0]
    print(f"\n  Estimated fractal dimension: {fractal_dim:.2f}")
    print(f"  (Euclidean 5D would be 5.0)")
    print(f"  (Fractal if < 5.0 and > integer)")

    is_fractal = 1.0 < fractal_dim < 4.5
    print(f"  Verdict: {'FRACTAL STRUCTURE DETECTED' if is_fractal else 'NOT FRACTAL'}")

    elapsed = time.time() - t0
    results = {
        'phase': 71, 'name': 'The Fractal Hypothesis',
        'global_stats': global_stats,
        'local_results': local_results,
        'fractal_dimension': float(fractal_dim),
        'box_counts': list(zip(epsilons, box_counts)),
        'is_fractal': bool(is_fractal),
        'elapsed': elapsed, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase71_fractal.json'), 'w',
              encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    # Figure
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Box-counting plot
    axes[0].plot(-np.array(log_eps), log_count, 'o-', color='#9C27B0', markersize=8)
    fit_x = np.linspace(min(-np.array(log_eps)), max(-np.array(log_eps)), 50)
    fit_y = -coeffs[0] * (-fit_x) + coeffs[1]
    axes[0].plot(fit_x, fit_y, '--', color='red', label=f'D={fractal_dim:.2f}')
    axes[0].set_xlabel('-log(epsilon)')
    axes[0].set_ylabel('log(N(epsilon))')
    axes[0].set_title(f'Box-Counting Dimension\nD = {fractal_dim:.2f}', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Local vs Global structure comparison
    ks = [r['k'] for r in local_results]
    local_sk = [r['avg_skew'] for r in local_results]
    axes[1].bar(ks, local_sk, color='#2196F3', edgecolor='black', width=3)
    axes[1].axhline(global_stats['skew'], color='red', linestyle='--',
                   label=f'Global skew={global_stats["skew"]:.3f}')
    axes[1].set_xlabel('K (neighborhood size)')
    axes[1].set_ylabel('Average Local Skewness')
    axes[1].set_title('Self-Similarity Test\n(Local should match Global)',
                     fontweight='bold')
    axes[1].legend()

    # 3. Local dimensionality
    local_vars = [r['avg_var_1d'] for r in local_results]
    axes[2].bar(ks, local_vars, color='#FF9800', edgecolor='black', width=3)
    axes[2].set_xlabel('K (neighborhood size)')
    axes[2].set_ylabel('Local 1D Variance (%)')
    axes[2].set_title('Local Dimensionality\n(High = locally 1D)', fontweight='bold')

    plt.suptitle('Phase 71: The Fractal Hypothesis\n'
                 'Is the Rosetta Space Self-Similar?',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase71_fractal.png'), dpi=150)
    plt.close()
    print(f"\nPhase 71 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
