"""
Phase 73: Program Dreaming
=============================
What does the 5D space "dream" about?

Sample random points in the Rosetta manifold.
Decode them to the nearest real function.
Analyze: does the space dream of arithmetic?
Of logic? Of strings? What regions are "dense"?

This is the program space equivalent of
neural network "deep dream" visualizations.
"""
import os, json, time
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')


def main():
    print("=" * 60)
    print("Phase 73: Program Dreaming")
    print("What does the 5D space dream about?")
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
    pca = PCA(n_components=10).fit(z_ast)
    z_5d = pca.transform(z_ast)[:, :5]

    unique = {}
    for i, src in enumerate(sources):
        if src not in unique:
            unique[src] = z_5d[i]
    all_srcs = list(unique.keys())
    all_z5 = np.array([unique[s] for s in all_srcs])

    # Space bounds
    z_min = all_z5.min(0)
    z_max = all_z5.max(0)
    z_mean = all_z5.mean(0)
    z_std = all_z5.std(0)
    print(f"\n  Space bounds:")
    for dim in range(5):
        print(f"    PC{dim+1}: [{z_min[dim]:.3f}, {z_max[dim]:.3f}], "
              f"mean={z_mean[dim]:.3f}, std={z_std[dim]:.3f}")

    # ==============================
    # Dream 1: Random uniform sampling
    # ==============================
    print("\n--- Dream 1: Random Points in 5D ---")
    np.random.seed(42)
    n_dreams = 200

    # Sample from the bounding box
    dream_points = np.random.uniform(z_min - z_std, z_max + z_std,
                                     size=(n_dreams, 5))

    dreams = []
    dream_ops = {}
    for i, z in enumerate(dream_points):
        dists = np.linalg.norm(all_z5 - z, axis=1)
        nn_idx = np.argmin(dists)
        nn_src = all_srcs[nn_idx]
        nn_dist = float(dists[nn_idx])

        # Classify the dream
        if 'return' in nn_src:
            op = nn_src.split('return ')[1].strip()
            # Categorize
            if '+' in op:
                cat = 'Addition'
            elif '-' in op:
                cat = 'Subtraction'
            elif '*' in op and '**' not in op:
                cat = 'Multiplication'
            elif '/' in op:
                cat = 'Division'
            elif '>' in op or '<' in op:
                cat = 'Comparison'
            elif '==' in op or '!=' in op:
                cat = 'Equality'
            elif 'abs' in op:
                cat = 'Absolute'
            elif 'max' in op or 'min' in op:
                cat = 'MinMax'
            elif 'and' in op or 'or' in op:
                cat = 'Logic'
            else:
                cat = 'Other'
        else:
            cat = 'Other'

        dream_ops[cat] = dream_ops.get(cat, 0) + 1
        dreams.append({
            'z': z.tolist(), 'nn_src': nn_src,
            'nn_dist': nn_dist, 'category': cat,
        })

    print(f"  Dream categories:")
    for cat, count in sorted(dream_ops.items(), key=lambda x: x[1], reverse=True):
        pct = count / n_dreams * 100
        bar = '#' * int(pct / 2)
        print(f"    {cat:15s}: {count:3d} ({pct:5.1f}%) {bar}")

    # Most frequent dream function
    dream_fn_counts = {}
    for d in dreams:
        dream_fn_counts[d['nn_src']] = dream_fn_counts.get(d['nn_src'], 0) + 1

    top_dreams = sorted(dream_fn_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"\n  Most dreamed functions:")
    for src, count in top_dreams:
        print(f"    {count:3d}x: {src}")

    # ==============================
    # Dream 2: Walk along each PC axis
    # ==============================
    print("\n--- Dream 2: Walking the Principal Components ---")
    axis_walks = {}
    for dim in range(5):
        walk = []
        for t in np.linspace(z_min[dim] - z_std[dim],
                            z_max[dim] + z_std[dim], 20):
            point = z_mean.copy()
            point[dim] = t
            dists = np.linalg.norm(all_z5 - point, axis=1)
            nn_idx = np.argmin(dists)
            walk.append({
                't': float(t),
                'src': all_srcs[nn_idx],
                'dist': float(dists[nn_idx]),
            })

        # Show the walk
        print(f"\n  PC{dim+1} axis walk:")
        prev_src = None
        for w in walk:
            if w['src'] != prev_src:
                short = w['src'].split('return ')[1][:25] if 'return' in w['src'] else '?'
                print(f"    t={w['t']:+.2f}: {short} (d={w['dist']:.3f})")
                prev_src = w['src']

        axis_walks[f'PC{dim+1}'] = walk

    # ==============================
    # Dream 3: Density map - where are the programs?
    # ==============================
    print("\n--- Dream 3: Density Analysis ---")
    from sklearn.neighbors import KernelDensity

    # Fit KDE
    kde = KernelDensity(bandwidth=0.3, kernel='gaussian').fit(all_z5)

    # Evaluate density at dream points
    dream_density = np.exp(kde.score_samples(dream_points))
    avg_density = float(np.mean(dream_density))
    max_density = float(np.max(dream_density))
    min_density = float(np.min(dream_density))

    # Also at data points
    data_density = np.exp(kde.score_samples(all_z5))
    avg_data_density = float(np.mean(data_density))

    print(f"  Density at random points: avg={avg_density:.4f}")
    print(f"  Density at data points:   avg={avg_data_density:.4f}")
    print(f"  Ratio: {avg_density / (avg_data_density + 1e-8):.4f}")

    # Find the densest region (mode of the space)
    mode_idx = np.argmax(data_density)
    mode_src = all_srcs[mode_idx]
    print(f"\n  Densest point (mode): {mode_src}")
    print(f"  Mode density: {data_density[mode_idx]:.4f}")

    # Find voids (empty regions)
    n_void = sum(1 for d in dream_density if d < 0.001)
    print(f"  Void regions (<0.001 density): {n_void}/{n_dreams} "
          f"({n_void/n_dreams*100:.0f}%)")

    elapsed = time.time() - t0
    results = {
        'phase': 73, 'name': 'Program Dreaming',
        'n_dreams': n_dreams,
        'dream_categories': dream_ops,
        'top_dreams': [(src, count) for src, count in top_dreams],
        'avg_density_random': avg_density,
        'avg_density_data': avg_data_density,
        'density_ratio': float(avg_density / (avg_data_density + 1e-8)),
        'n_voids': n_void,
        'mode_function': mode_src,
        'elapsed': elapsed, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase73_dreaming.json'), 'w',
              encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    # Figure
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Dream categories pie chart
    cats = list(dream_ops.keys())
    vals = [dream_ops[c] for c in cats]
    colors_pie = ['#4CAF50', '#2196F3', '#FF9800', '#F44336',
                 '#9C27B0', '#009688', '#795548', '#607D8B',
                 '#E91E63', '#CDDC39'][:len(cats)]
    axes[0].pie(vals, labels=cats, colors=colors_pie, autopct='%1.0f%%',
               startangle=90, textprops={'fontsize': 8})
    axes[0].set_title('What Does the Space Dream?', fontweight='bold')

    # 2. Density comparison
    axes[1].hist(np.log10(dream_density + 1e-10), bins=30, alpha=0.6,
                color='#2196F3', label='Random points', edgecolor='black')
    axes[1].hist(np.log10(data_density + 1e-10), bins=30, alpha=0.6,
                color='#4CAF50', label='Real programs', edgecolor='black')
    axes[1].set_xlabel('log10(Density)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Density Distribution\n(Real vs Random)', fontweight='bold')
    axes[1].legend()

    # 3. PC1 walk visualization
    if 'PC1' in axis_walks:
        walk = axis_walks['PC1']
        t_vals = [w['t'] for w in walk]
        d_vals = [w['dist'] for w in walk]
        axes[2].plot(t_vals, d_vals, 'o-', color='#9C27B0', markersize=4)
        axes[2].set_xlabel('PC1 Value')
        axes[2].set_ylabel('Distance to Nearest Function')
        axes[2].set_title('PC1 Axis Walk\n(Low = dense region)', fontweight='bold')
        axes[2].grid(True, alpha=0.3)

    plt.suptitle('Phase 73: Program Dreaming\n'
                 'Exploring the Void Between Functions',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase73_dreaming.png'), dpi=150)
    plt.close()
    print(f"\nPhase 73 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
