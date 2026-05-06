"""
Phase 82: The Conservation Laws
==================================
Noether's theorem: every symmetry has a conservation law.

In P74 we found that variable renaming is a perfect symmetry.
What QUANTITY is conserved under this transformation?

Also test: what quantities are conserved when we
transform f(x,y) -> f(y,x) for commutative functions?
"""
import os, json, time, inspect
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')


def main():
    print("=" * 60)
    print("Phase 82: The Conservation Laws")
    print("Noether's theorem for software")
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

    # Compute "charges" — scalar quantities from 5D
    print("\n--- Computing Program Charges ---")

    def compute_charges(z):
        """Compute various scalar invariants from a 5D vector."""
        norm = float(np.linalg.norm(z))
        energy = float(np.sum(z**2))  # "kinetic energy"
        angle_12 = float(np.arctan2(z[1], z[0]) if (z[0]**2 + z[1]**2) > 1e-10 else 0)
        angle_34 = float(np.arctan2(z[3], z[2]) if (z[2]**2 + z[3]**2) > 1e-10 else 0)
        parity = float(np.prod(np.sign(z + 1e-10)))
        moment = float(np.sum(z * np.arange(5)))  # angular momentum analog
        return {
            'norm': norm, 'energy': energy,
            'angle_12': angle_12, 'angle_34': angle_34,
            'parity': parity, 'moment': moment,
        }

    # Test conservation under variable renaming
    print("\n--- Conservation under Variable Renaming ---")
    rename_pairs = [
        ('def f(x, y): return x + y', 'def f(a, b): return a + b'),
        ('def f(x, y): return x - y', 'def f(a, b): return a - b'),
        ('def f(x, y): return x * y', 'def f(a, b): return a * b'),
        ('def f(x, y): return x > y', 'def f(a, b): return a > b'),
        ('def f(x): return abs(x)', 'def f(a): return abs(a)'),
        ('def f(x): return -x', 'def f(a): return -a'),
        ('def f(x, y): return x ** y', 'def f(a, b): return a ** b'),
        ('def f(x, y): return x % y', 'def f(a, b): return a % b'),
    ]

    charge_names = ['norm', 'energy', 'angle_12', 'angle_34', 'parity', 'moment']
    conservation_scores = {c: [] for c in charge_names}

    for src1, src2 in rename_pairs:
        z1 = src_to_z.get(src1)
        z2 = src_to_z.get(src2)
        if z1 is None or z2 is None:
            continue

        c1 = compute_charges(z1)
        c2 = compute_charges(z2)

        for cn in charge_names:
            diff = abs(c1[cn] - c2[cn])
            conservation_scores[cn].append(diff)

    print(f"\n  Charge conservation (lower = more conserved):")
    conserved_charges = []
    for cn in charge_names:
        diffs = conservation_scores[cn]
        if diffs:
            avg_diff = np.mean(diffs)
            is_conserved = avg_diff < 0.01
            print(f"    {cn:12s}: avg_diff={avg_diff:.6f} "
                  f"[{'CONSERVED' if is_conserved else 'NOT conserved'}]")
            conserved_charges.append({
                'name': cn, 'avg_diff': float(avg_diff),
                'is_conserved': bool(is_conserved),
            })

    # Test conservation under semantic-preserving transforms
    print("\n--- Conservation under Semantic Transforms ---")
    semantic_pairs = [
        ('def f(x): return x', 'def f(x): return x + 0', 'identity+0'),
        ('def f(x): return x', 'def f(x): return x * 1', 'identity*1'),
        ('def f(x): return x * 2', 'def f(x): return x + x', 'double'),
    ]

    semantic_conservation = []
    for src1, src2, desc in semantic_pairs:
        z1 = src_to_z.get(src1)
        z2 = src_to_z.get(src2)
        if z1 is None or z2 is None:
            continue

        c1 = compute_charges(z1)
        c2 = compute_charges(z2)

        diffs = {cn: abs(c1[cn] - c2[cn]) for cn in charge_names}
        conserved = [cn for cn, d in diffs.items() if d < 0.1]
        print(f"  {desc:15s}: conserved={conserved}")
        semantic_conservation.append({
            'desc': desc, 'diffs': diffs,
            'conserved': conserved,
        })

    # Compute charges for ALL functions and look for universal patterns
    print("\n--- Universal Charge Distribution ---")
    all_charges = {cn: [] for cn in charge_names}
    for src, z in src_to_z.items():
        charges = compute_charges(z)
        for cn in charge_names:
            all_charges[cn].append(charges[cn])

    for cn in charge_names:
        vals = np.array(all_charges[cn])
        print(f"  {cn:12s}: mean={vals.mean():.4f}, std={vals.std():.4f}, "
              f"range=[{vals.min():.4f}, {vals.max():.4f}]")

    elapsed = time.time() - t0
    results = {
        'phase': 82, 'name': 'The Conservation Laws',
        'conserved_charges': conserved_charges,
        'semantic_conservation': semantic_conservation,
        'charge_stats': {cn: {'mean': float(np.mean(all_charges[cn])),
                             'std': float(np.std(all_charges[cn]))}
                        for cn in charge_names},
        'elapsed': elapsed, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase82_conservation.json'), 'w',
              encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    # Figure
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Conservation scores
    names = [c['name'] for c in conserved_charges]
    diffs = [c['avg_diff'] for c in conserved_charges]
    colors = ['#4CAF50' if c['is_conserved'] else '#F44336' for c in conserved_charges]
    axes[0].barh(names, diffs, color=colors, edgecolor='black')
    axes[0].axvline(0.01, color='orange', linestyle='--')
    axes[0].set_xlabel('Average Change')
    axes[0].set_title('Conservation under\nVariable Renaming', fontweight='bold')
    axes[0].set_xscale('log')

    # 2. Charge distributions
    for i, cn in enumerate(charge_names[:4]):
        axes[1].hist(all_charges[cn], bins=30, alpha=0.4, label=cn)
    axes[1].set_xlabel('Charge Value')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Charge Distributions', fontweight='bold')
    axes[1].legend(fontsize=7)

    # 3. Summary
    n_conserved = sum(1 for c in conserved_charges if c['is_conserved'])
    summary = (f"NOETHER'S THEOREM\nFOR SOFTWARE\n\n"
              f"Conserved charges:\n{n_conserved}/{len(conserved_charges)}\n\n")
    for c in conserved_charges:
        if c['is_conserved']:
            summary += f"  {c['name']}\n"
    axes[2].text(0.5, 0.5, summary, ha='center', va='center',
                fontsize=12, fontweight='bold', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#E8F5E9', alpha=0.8),
                transform=axes[2].transAxes)
    axes[2].axis('off')

    plt.suptitle("Phase 82: The Conservation Laws\n"
                 "Noether's Theorem for Software",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase82_conservation.png'), dpi=150)
    plt.close()
    print(f"\nPhase 82 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
