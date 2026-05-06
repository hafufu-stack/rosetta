"""
Phase 74: The Symmetry Groups
================================
What symmetries does the 5D program space have?

In physics, symmetries reveal conservation laws (Noether's theorem).
In program space, symmetries = transformations that preserve meaning.

We already know: variable renaming is a symmetry (cos=1.0).
What others exist? Argument order? Return type? Nesting?
"""
import os, json, time
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')


def main():
    print("=" * 60)
    print("Phase 74: The Symmetry Groups")
    print("What transformations preserve meaning?")
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
    from sklearn.metrics.pairwise import cosine_similarity
    pca = PCA(n_components=10).fit(z_ast)
    z_5d = pca.transform(z_ast)[:, :5]

    src_to_z = {}
    for i, src in enumerate(sources):
        if src not in src_to_z:
            src_to_z[src] = z_5d[i]

    # Define symmetry groups to test
    symmetry_groups = {
        'Variable Renaming': [
            ('def f(x, y): return x + y', 'def f(a, b): return a + b'),
            ('def f(x, y): return x - y', 'def f(a, b): return a - b'),
            ('def f(x, y): return x * y', 'def f(a, b): return a * b'),
            ('def f(x, y): return x > y', 'def f(a, b): return a > b'),
            ('def f(x): return abs(x)', 'def f(a): return abs(a)'),
            ('def f(x): return -x', 'def f(a): return -a'),
        ],
        'Argument Naming (4 vars)': [
            ('def f(x, y): return x + y', 'def f(m, n): return m + n'),
            ('def f(x, y): return x + y', 'def f(p, q): return p + q'),
            ('def f(x, y): return x * y', 'def f(m, n): return m * n'),
            ('def f(x, y): return x * y', 'def f(p, q): return p * q'),
        ],
        'Commutativity': [
            ('def f(x, y): return x + y', 'def f(x, y): return y + x'),
            ('def f(x, y): return x * y', 'def f(x, y): return y * x'),
            ('def f(x, y): return max(x, y)', 'def f(x, y): return max(y, x)'),
            ('def f(x, y): return min(x, y)', 'def f(x, y): return min(y, x)'),
        ],
        'Anti-Symmetry (Inverse)': [
            ('def f(x, y): return x + y', 'def f(x, y): return x - y'),
            ('def f(x, y): return x * y', 'def f(x, y): return x / y'),
            ('def f(x, y): return x > y', 'def f(x, y): return x < y'),
            ('def f(x): return -x', 'def f(x): return abs(x)'),
        ],
        'Double Negation': [
            ('def f(x): return x', 'def f(x): return -(-x)'),
            ('def f(x): return abs(x)', 'def f(x): return abs(-x)'),
        ],
        'Absorption (Idempotent)': [
            ('def f(x): return abs(x)', 'def f(x): return abs(abs(x))'),
            ('def f(x): return int(x)', 'def f(x): return int(int(x))'),
        ],
    }

    results_by_group = {}
    print()
    for group_name, pairs in symmetry_groups.items():
        cos_vals = []
        dist_vals = []
        for src1, src2 in pairs:
            z1 = src_to_z.get(src1)
            z2 = src_to_z.get(src2)
            if z1 is None or z2 is None:
                continue
            cos = float(cosine_similarity(z1.reshape(1,-1), z2.reshape(1,-1))[0,0])
            dist = float(np.linalg.norm(z1 - z2))
            cos_vals.append(cos)
            dist_vals.append(dist)

        if cos_vals:
            avg_cos = float(np.mean(cos_vals))
            avg_dist = float(np.mean(dist_vals))
            is_symmetry = avg_cos > 0.9
            print(f"  {group_name:30s}: cos={avg_cos:.4f}, dist={avg_dist:.4f} "
                  f"[{'SYMMETRY' if is_symmetry else 'BROKEN'}]")
            results_by_group[group_name] = {
                'avg_cos': avg_cos, 'avg_dist': avg_dist,
                'is_symmetry': is_symmetry, 'n_pairs': len(cos_vals),
            }

    # Compute the symmetry group structure
    print("\n--- Symmetry Group Analysis ---")
    n_sym = sum(1 for r in results_by_group.values() if r['is_symmetry'])
    n_total = len(results_by_group)
    print(f"  Symmetries found: {n_sym}/{n_total}")

    # Compute the "symmetry breaking" spectrum
    print("\n  Symmetry spectrum (cos similarity):")
    for name, r in sorted(results_by_group.items(), key=lambda x: x[1]['avg_cos'], reverse=True):
        bar = '#' * int(r['avg_cos'] * 50)
        sym = 'S' if r['is_symmetry'] else ' '
        print(f"    [{sym}] {name:30s}: {r['avg_cos']:.4f} {bar}")

    elapsed = time.time() - t0
    results = {
        'phase': 74, 'name': 'The Symmetry Groups',
        'groups': results_by_group,
        'n_symmetries': n_sym, 'n_total': n_total,
        'elapsed': elapsed, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase74_symmetry.json'), 'w',
              encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    # Figure
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    names = list(results_by_group.keys())
    cos_vals = [results_by_group[n]['avg_cos'] for n in names]
    colors = ['#4CAF50' if results_by_group[n]['is_symmetry'] else '#F44336' for n in names]

    axes[0].barh(names, cos_vals, color=colors, edgecolor='black')
    axes[0].axvline(0.9, color='orange', linestyle='--', label='Symmetry threshold')
    axes[0].set_xlabel('Cosine Similarity')
    axes[0].set_title('Symmetry Groups of Program Space', fontweight='bold')
    axes[0].legend()
    axes[0].set_xlim(-1.1, 1.1)

    dist_vals = [results_by_group[n]['avg_dist'] for n in names]
    axes[1].barh(names, dist_vals, color=colors, edgecolor='black')
    axes[1].set_xlabel('5D Distance')
    axes[1].set_title('Symmetry Breaking Distance', fontweight='bold')

    plt.suptitle('Phase 74: The Symmetry Groups\nNoether\'s Theorem for Software',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase74_symmetry.png'), dpi=150)
    plt.close()
    print(f"\nPhase 74 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
