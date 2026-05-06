"""
Phase 85: Latent Debugging
=============================
Given a BUGGY program, find the FIX by navigating 5D space.

Strategy: compute the "bug vector" (direction from buggy to correct)
and use it to repair other bugs. If bugs have a consistent
direction in 5D, debugging becomes GEOMETRIC.
"""
import os, json, time
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')


def main():
    print("=" * 60)
    print("Phase 85: Latent Debugging")
    print("Fixing bugs by navigating 5D space")
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
    all_srcs = list(src_to_z.keys())
    all_z5 = np.array([src_to_z[s] for s in all_srcs])

    # Bug-fix pairs: (buggy, correct, description)
    bug_pairs = [
        # Off-by-one errors
        ('def f(x, y): return x + y + 1', 'def f(x, y): return x + y',
         'off-by-one in addition'),
        ('def f(x, y): return x * y + 1', 'def f(x, y): return x * y',
         'off-by-one in multiplication'),
        # Wrong operator
        ('def f(x, y): return x - y', 'def f(x, y): return x + y',
         'wrong operator (sub->add)'),
        ('def f(x, y): return x + y', 'def f(x, y): return x * y',
         'wrong operator (add->mul)'),
        # Argument swap
        ('def f(x, y): return y - x', 'def f(x, y): return x - y',
         'argument swap'),
        # Sign error
        ('def f(x): return -abs(x)', 'def f(x): return abs(x)',
         'sign error in abs'),
        # Logic inversion
        ('def f(x, y): return x < y', 'def f(x, y): return x > y',
         'logic inversion'),
        ('def f(x, y): return x != y', 'def f(x, y): return x == y',
         'equality inversion'),
    ]

    # Compute bug vectors
    print("\n--- Computing Bug Vectors ---")
    bug_vectors = []
    for buggy, correct, desc in bug_pairs:
        z_bug = src_to_z.get(buggy)
        z_fix = src_to_z.get(correct)
        if z_bug is None or z_fix is None:
            continue

        bug_vec = z_fix - z_bug
        dist = float(np.linalg.norm(bug_vec))
        print(f"  {desc:30s}: dist={dist:.4f}, vec=[{', '.join(f'{x:.3f}' for x in bug_vec)}]")
        bug_vectors.append({
            'buggy': buggy, 'correct': correct, 'desc': desc,
            'vector': bug_vec, 'distance': dist,
        })

    if not bug_vectors:
        print("  No bug pairs found in dataset")
        elapsed = time.time() - t0
        results = {'phase': 85, 'name': 'Latent Debugging',
                   'error': 'no_pairs', 'elapsed': elapsed}
        with open(os.path.join(RESULTS_DIR, 'phase85_debugging.json'), 'w',
                  encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        return results

    # Are bug vectors consistent? (do bugs have a "direction"?)
    print("\n--- Bug Vector Consistency ---")
    if len(bug_vectors) >= 2:
        vecs = np.array([bv['vector'] for bv in bug_vectors])
        # Pairwise cosine between bug vectors
        cos_matrix = cosine_similarity(vecs)
        avg_cos = float(np.mean(cos_matrix[np.triu_indices(len(vecs), k=1)]))
        print(f"  Average cosine between bug vectors: {avg_cos:.4f}")
        print(f"  Bug direction {'CONSISTENT' if avg_cos > 0.3 else 'VARIED'}")
    else:
        avg_cos = 0.0

    # Test: can we fix a bug by applying the average bug vector?
    print("\n--- Bug Repair Test ---")
    avg_bug_vec = np.mean([bv['vector'] for bv in bug_vectors], axis=0)
    print(f"  Average bug vector: [{', '.join(f'{x:.3f}' for x in avg_bug_vec)}]")

    repair_results = []
    for bv in bug_vectors:
        z_bug = src_to_z[bv['buggy']]
        z_correct = src_to_z[bv['correct']]

        # Method 1: Apply average bug vector
        z_repaired = z_bug + avg_bug_vec
        dists = np.linalg.norm(all_z5 - z_repaired, axis=1)
        nn_idx = np.argmin(dists)
        found_src = all_srcs[nn_idx]
        nn_dist = float(dists[nn_idx])

        # Method 2: Nearest neighbor (find closest correct function)
        correct_short = bv['correct'].split('return ')[1][:15] if 'return' in bv['correct'] else '?'
        found_short = found_src.split('return ')[1][:15] if 'return' in found_src else '?'
        fixed = found_src == bv['correct']

        print(f"  Bug: {bv['desc'][:25]:25s}")
        print(f"    Buggy:    {bv['buggy'].split('return ')[1][:20] if 'return' in bv['buggy'] else '?'}")
        print(f"    Expected: {correct_short}")
        print(f"    Repaired: {found_short} (d={nn_dist:.3f}) {'FIXED!' if fixed else ''}")

        repair_results.append({
            'desc': bv['desc'], 'buggy': bv['buggy'],
            'correct': bv['correct'], 'repaired': found_src,
            'fixed': fixed, 'distance': nn_dist,
        })

    n_fixed = sum(1 for r in repair_results if r['fixed'])
    print(f"\n  Bugs fixed: {n_fixed}/{len(repair_results)}")

    elapsed = time.time() - t0
    results = {
        'phase': 85, 'name': 'Latent Debugging',
        'n_pairs': len(bug_vectors),
        'avg_bug_cos': float(avg_cos),
        'avg_bug_vector': avg_bug_vec.tolist(),
        'n_fixed': n_fixed, 'n_total': len(repair_results),
        'repairs': repair_results,
        'elapsed': elapsed, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase85_debugging.json'), 'w',
              encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if bug_vectors:
        vecs = np.array([bv['vector'] for bv in bug_vectors])
        labels = [bv['desc'][:15] for bv in bug_vectors]
        im = axes[0].imshow(vecs, aspect='auto', cmap='RdBu', vmin=-1, vmax=1)
        axes[0].set_yticks(range(len(labels)))
        axes[0].set_yticklabels(labels, fontsize=7)
        axes[0].set_xticks(range(5))
        axes[0].set_xticklabels([f'PC{i+1}' for i in range(5)])
        axes[0].set_title('Bug Vectors in 5D', fontweight='bold')
        plt.colorbar(im, ax=axes[0], shrink=0.8)

    repair_names = [r['desc'][:15] for r in repair_results]
    repair_dists = [r['distance'] for r in repair_results]
    colors = ['#4CAF50' if r['fixed'] else '#F44336' for r in repair_results]
    axes[1].barh(repair_names, repair_dists, color=colors, edgecolor='black')
    axes[1].set_xlabel('Distance to Repair')
    axes[1].set_title(f'Bug Repair Results\n{n_fixed}/{len(repair_results)} fixed',
                     fontweight='bold')

    plt.suptitle('Phase 85: Latent Debugging\nFixing Bugs by Navigating 5D Space',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase85_debugging.png'), dpi=150)
    plt.close()
    print(f"\nPhase 85 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
