"""
Phase 75: Program Interpolation (Morphing)
=============================================
Can we SMOOTHLY morph between two programs in 5D space?

Walk from addition to multiplication along a straight line.
At each step, decode to the nearest function.
Does the path make semantic sense?

This is the "movie" version of code transformation.
"""
import os, json, time
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')


def main():
    print("=" * 60)
    print("Phase 75: Program Interpolation")
    print("Morphing between programs in 5D space")
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

    src_to_z = {}
    for i, src in enumerate(sources):
        if src not in src_to_z:
            src_to_z[src] = z_5d[i]
    all_srcs = list(src_to_z.keys())
    all_z5 = np.array([src_to_z[s] for s in all_srcs])

    # Morphing paths to test
    morph_paths = [
        ('def f(x, y): return x + y', 'def f(x, y): return x * y',
         'Addition -> Multiplication'),
        ('def f(x, y): return x + y', 'def f(x, y): return x - y',
         'Addition -> Subtraction'),
        ('def f(x): return abs(x)', 'def f(x): return -x',
         'Absolute -> Negation'),
        ('def f(x, y): return x > y', 'def f(x, y): return x == y',
         'Comparison -> Equality'),
        ('def f(x, y): return max(x, y)', 'def f(x, y): return min(x, y)',
         'Max -> Min'),
        ('def f(x, y): return x + y', 'def f(x, y): return x ** y',
         'Addition -> Power'),
    ]

    morph_results = []
    N_STEPS = 20

    for src_a, src_b, desc in morph_paths:
        z_a = src_to_z.get(src_a)
        z_b = src_to_z.get(src_b)
        if z_a is None or z_b is None:
            continue

        print(f"\n  --- {desc} ---")
        path_funcs = []
        path_dists = []
        prev_src = None

        for step in range(N_STEPS + 1):
            t = step / N_STEPS
            z_interp = (1 - t) * z_a + t * z_b  # Linear interpolation

            # Find nearest function
            dists = np.linalg.norm(all_z5 - z_interp, axis=1)
            nn_idx = np.argmin(dists)
            nn_src = all_srcs[nn_idx]
            nn_dist = float(dists[nn_idx])

            path_funcs.append(nn_src)
            path_dists.append(nn_dist)

            if nn_src != prev_src:
                short = nn_src.split('return ')[1][:25] if 'return' in nn_src else '?'
                print(f"    t={t:.2f}: {short} (d={nn_dist:.3f})")
                prev_src = nn_src

        # Count unique functions visited
        unique_visited = len(set(path_funcs))
        # Count transitions
        transitions = sum(1 for i in range(len(path_funcs)-1)
                        if path_funcs[i] != path_funcs[i+1])

        morph_results.append({
            'desc': desc, 'src_a': src_a, 'src_b': src_b,
            'unique_visited': unique_visited,
            'transitions': transitions,
            'avg_dist': float(np.mean(path_dists)),
            'max_dist': float(np.max(path_dists)),
            'path': [{'t': i/N_STEPS, 'fn': path_funcs[i],
                      'dist': path_dists[i]} for i in range(0, len(path_funcs), 4)],
        })

    # Summary
    print(f"\n  === INTERPOLATION SUMMARY ===")
    for r in morph_results:
        print(f"  {r['desc']:30s}: {r['unique_visited']} funcs, "
              f"{r['transitions']} transitions, max_d={r['max_dist']:.3f}")

    avg_unique = np.mean([r['unique_visited'] for r in morph_results])
    avg_transitions = np.mean([r['transitions'] for r in morph_results])

    elapsed = time.time() - t0
    results = {
        'phase': 75, 'name': 'Program Interpolation',
        'n_paths': len(morph_results),
        'avg_unique_visited': float(avg_unique),
        'avg_transitions': float(avg_transitions),
        'morphs': morph_results,
        'elapsed': elapsed, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase75_interpolation.json'), 'w',
              encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    # Figure
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Distance along path (first morph)
    if morph_results:
        r0 = morph_results[0]
        ts = [p['t'] for p in r0['path']]
        ds = [p['dist'] for p in r0['path']]
        axes[0].plot(ts, ds, 'o-', color='#2196F3', markersize=6)
        axes[0].set_xlabel('Interpolation t')
        axes[0].set_ylabel('Distance to Nearest Function')
        axes[0].set_title(f'{r0["desc"]}\n(distance along path)', fontweight='bold')
        axes[0].grid(True, alpha=0.3)

    # 2. Number of transitions per path
    names = [r['desc'][:15] for r in morph_results]
    trans = [r['transitions'] for r in morph_results]
    axes[1].barh(names, trans, color='#FF9800', edgecolor='black')
    axes[1].set_xlabel('Number of Transitions')
    axes[1].set_title('Path Smoothness\n(fewer = smoother)', fontweight='bold')

    # 3. Unique functions visited
    uniq = [r['unique_visited'] for r in morph_results]
    axes[2].barh(names, uniq, color='#4CAF50', edgecolor='black')
    axes[2].set_xlabel('Unique Functions')
    axes[2].set_title('Functions Discovered\nDuring Interpolation', fontweight='bold')

    plt.suptitle('Phase 75: Program Interpolation\nSmooth Morphing in 5D Space',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase75_interpolation.png'), dpi=150)
    plt.close()
    print(f"\nPhase 75 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
