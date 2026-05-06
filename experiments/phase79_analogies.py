"""
Phase 79: Program Analogies (The King-Queen Test)
===================================================
Word2Vec proved: king - man + woman = queen.
Can we do the same with programs?

add - commutative + noncommutative = subtract?
abs - unary + binary = max?

If vector arithmetic produces meaningful program analogies,
5D space has the SAME algebraic structure as language!
"""
import os, json, time
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')


def main():
    print("=" * 60)
    print("Phase 79: Program Analogies (King-Queen Test)")
    print("A is to B as C is to ???")
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

    def find_nearest(z_point, exclude=None):
        """Find nearest function, optionally excluding some."""
        dists = np.linalg.norm(all_z5 - z_point, axis=1)
        if exclude:
            for ex in exclude:
                idx = all_srcs.index(ex) if ex in all_srcs else -1
                if idx >= 0:
                    dists[idx] = float('inf')
        nn_idx = np.argmin(dists)
        return all_srcs[nn_idx], float(dists[nn_idx])

    # Analogies: A - B + C = ?
    # Expected: the result should be semantically meaningful
    analogies = [
        # (A, B, C, expected_description)
        ('def f(x, y): return x + y', 'def f(x, y): return x - y',
         'def f(x, y): return x * y', 'x / y (inverse of mult)'),

        ('def f(x, y): return x + y', 'def f(x): return abs(x)',
         'def f(x, y): return x * y', 'something binary from unary'),

        ('def f(x, y): return x + y', 'def f(a, b): return a + b',
         'def f(x, y): return x * y', 'a * b (rename transfer)'),

        ('def f(x, y): return x > y', 'def f(x, y): return x < y',
         'def f(x, y): return x >= y', 'x <= y (flip transfer)'),

        ('def f(x, y): return max(x, y)', 'def f(x, y): return min(x, y)',
         'def f(x, y): return x + y', 'x - y (dual pair)'),

        ('def f(x): return abs(x)', 'def f(x): return -x',
         'def f(x, y): return max(x, y)', 'min(x,y) (abs:neg as max:?)'),

        ('def f(x, y): return x + y', 'def f(x, y): return x * y',
         'def f(x, y): return x - y', 'x / y (add:mult as sub:?)'),

        ('def f(x): return x * 2', 'def f(x): return x + x',
         'def f(x): return x * 3', 'x + x + x?'),

        ('def f(x, y): return x + y', 'def f(x, y): return x + y + 1',
         'def f(x, y): return x * y', 'x * y + 1?'),

        ('def f(x): return -x', 'def f(x): return x',
         'def f(x, y): return x - y', 'x + y? (negate the negation)'),
    ]

    results_list = []
    n_meaningful = 0
    print()

    for A, B, C, expected in analogies:
        z_a = src_to_z.get(A)
        z_b = src_to_z.get(B)
        z_c = src_to_z.get(C)
        if z_a is None or z_b is None or z_c is None:
            continue

        # A - B + C = ?
        z_result = z_a - z_b + z_c
        found_src, found_dist = find_nearest(z_result, exclude=[A, B, C])

        a_short = A.split('return ')[1][:12] if 'return' in A else '?'
        b_short = B.split('return ')[1][:12] if 'return' in B else '?'
        c_short = C.split('return ')[1][:12] if 'return' in C else '?'
        f_short = found_src.split('return ')[1][:15] if 'return' in found_src else '?'

        # Check if result is meaningful
        meaningful = found_dist < 0.8
        if meaningful:
            n_meaningful += 1

        print(f"  {a_short:12s} - {b_short:12s} + {c_short:12s} = {f_short:15s} "
              f"(d={found_dist:.3f}) {'OK' if meaningful else '  '}")
        print(f"    Expected: {expected}")

        results_list.append({
            'A': A, 'B': B, 'C': C, 'result': found_src,
            'expected': expected, 'distance': found_dist,
            'meaningful': meaningful,
        })

    pct = n_meaningful / max(len(results_list), 1) * 100
    print(f"\n  Meaningful analogies: {n_meaningful}/{len(results_list)} ({pct:.0f}%)")

    elapsed = time.time() - t0
    results = {
        'phase': 79, 'name': 'Program Analogies',
        'n_analogies': len(results_list),
        'n_meaningful': n_meaningful,
        'pct_meaningful': float(pct),
        'analogies': results_list,
        'elapsed': elapsed, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase79_analogies.json'), 'w',
              encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    # Figure
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Analogy results
    if results_list:
        labels = [f"{r['A'].split('return ')[1][:8]}-\n{r['B'].split('return ')[1][:8]}+\n{r['C'].split('return ')[1][:8]}"
                 if 'return' in r['A'] and 'return' in r['B'] and 'return' in r['C']
                 else '?' for r in results_list]
        dists = [r['distance'] for r in results_list]
        colors = ['#4CAF50' if r['meaningful'] else '#F44336' for r in results_list]
        axes[0].barh(range(len(labels)), dists, color=colors, edgecolor='black')
        axes[0].set_yticks(range(len(labels)))
        axes[0].set_yticklabels(labels, fontsize=6)
        axes[0].axvline(0.8, color='orange', linestyle='--')
        axes[0].set_xlabel('Distance to Result')
        axes[0].set_title(f'Program Analogies\n{n_meaningful}/{len(results_list)} meaningful',
                         fontweight='bold')

    # 2. Summary
    summary = (f"PROGRAM ANALOGIES\n"
              f"(King-Queen Test)\n\n"
              f"Tested: {len(results_list)}\n"
              f"Meaningful: {n_meaningful}\n"
              f"Rate: {pct:.0f}%\n\n")
    if pct > 50:
        summary += "5D space supports\nvector arithmetic\nfor code semantics!"
    elif pct > 20:
        summary += "Partial analogy\nsupport detected."
    else:
        summary += "Analogies need\nnon-linear operators."
    axes[1].text(0.5, 0.5, summary, ha='center', va='center',
                fontsize=13, fontweight='bold', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#E3F2FD', alpha=0.8),
                transform=axes[1].transAxes)
    axes[1].axis('off')

    plt.suptitle('Phase 79: Program Analogies\nA - B + C = ? in 5D Space',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase79_analogies.png'), dpi=150)
    plt.close()
    print(f"\nPhase 79 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
