"""
Phase 81: The Periodic Table of Programs
==========================================
Just as elements are organized by atomic number and electron
configuration, programs can be organized by their 5D coordinates.

Build the "Periodic Table" — a 2D projection that reveals
natural groups, periods, and relationships.
"""
import os, json, time
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')


def main():
    print("=" * 60)
    print("Phase 81: The Periodic Table of Programs")
    print("Organizing code by its fundamental properties")
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
    pca = PCA(n_components=10).fit(z_ast)
    z_5d = pca.transform(z_ast)[:, :5]

    unique = {}
    for i, src in enumerate(sources):
        if src not in unique:
            unique[src] = z_5d[i]
    all_srcs = list(unique.keys())
    all_z5 = np.array([unique[s] for s in all_srcs])
    N = len(all_z5)

    # Assign "atomic number" by distance from origin
    distances = np.linalg.norm(all_z5, axis=1)
    atomic_numbers = np.argsort(distances) + 1

    # Assign "periods" (rows) based on PC1 quantiles
    n_periods = 7
    pc1_quantiles = np.quantile(all_z5[:, 0], np.linspace(0, 1, n_periods + 1))
    periods = np.digitize(all_z5[:, 0], pc1_quantiles[1:-1]) + 1

    # Assign "groups" (columns) based on PC2 quantiles
    n_groups = 10
    pc2_quantiles = np.quantile(all_z5[:, 1], np.linspace(0, 1, n_groups + 1))
    groups = np.digitize(all_z5[:, 1], pc2_quantiles[1:-1]) + 1

    # Classify by operation type (element name)
    def classify_op(src):
        if 'return' not in src:
            return 'Other', 'Ot'
        expr = src.split('return ')[1].strip()
        if '+' in expr and '-' not in expr:
            return 'Addition', 'Ad'
        elif '-' in expr and '+' not in expr:
            return 'Subtraction', 'Su'
        elif '**' in expr:
            return 'Power', 'Pw'
        elif '*' in expr and '**' not in expr:
            return 'Multiplication', 'Mu'
        elif '//' in expr:
            return 'IntDivision', 'ID'
        elif '/' in expr:
            return 'Division', 'Dv'
        elif '%' in expr:
            return 'Modulo', 'Mo'
        elif '>=' in expr or '<=' in expr:
            return 'OrderCompare', 'OC'
        elif '>' in expr:
            return 'GreaterThan', 'Gt'
        elif '<' in expr:
            return 'LessThan', 'Lt'
        elif '==' in expr:
            return 'Equal', 'Eq'
        elif '!=' in expr:
            return 'NotEqual', 'NE'
        elif 'abs(' in expr:
            return 'Absolute', 'Ab'
        elif 'max(' in expr:
            return 'Maximum', 'Mx'
        elif 'min(' in expr:
            return 'Minimum', 'Mn'
        elif 'not ' in expr:
            return 'Negation', 'Ng'
        elif ' and ' in expr:
            return 'LogicAnd', 'An'
        elif ' or ' in expr:
            return 'LogicOr', 'Or'
        elif 'int(' in expr:
            return 'IntCast', 'In'
        elif 'len(' in expr:
            return 'Length', 'Ln'
        else:
            return 'Other', 'Ot'

    # Build the periodic table
    print("\n--- The Periodic Table of Programs ---")
    elements = []
    for i, src in enumerate(all_srcs):
        op_name, symbol = classify_op(src)
        elements.append({
            'src': src, 'symbol': symbol, 'op_name': op_name,
            'atomic_number': int(atomic_numbers[i]),
            'period': int(periods[i]), 'group': int(groups[i]),
            'z_5d': all_z5[i].tolist(),
            'distance': float(distances[i]),
        })

    # Print the table by period and group
    table = {}
    for elem in elements:
        key = (elem['period'], elem['group'])
        if key not in table:
            table[key] = elem
        elif elem['distance'] < table[key]['distance']:
            table[key] = elem

    print(f"\n  {'':4s}", end='')
    for g in range(1, n_groups + 1):
        print(f"  G{g:2d}", end='')
    print()
    print(f"  {'':4s}" + "-" * (n_groups * 5))

    for p in range(1, n_periods + 1):
        print(f"  P{p}: ", end='')
        for g in range(1, n_groups + 1):
            elem = table.get((p, g))
            if elem:
                print(f"  {elem['symbol']:2s} ", end='')
            else:
                print(f"  -- ", end='')
        print()

    # Count operation types
    op_counts = {}
    for elem in elements:
        op_counts[elem['op_name']] = op_counts.get(elem['op_name'], 0) + 1

    print(f"\n  Operation types found: {len(op_counts)}")
    for op, count in sorted(op_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"    {op:15s}: {count}")

    elapsed = time.time() - t0
    results = {
        'phase': 81, 'name': 'The Periodic Table of Programs',
        'n_elements': len(elements),
        'n_periods': n_periods, 'n_groups': n_groups,
        'op_counts': op_counts,
        'table': {f'{k[0]},{k[1]}': v['symbol'] for k, v in table.items()},
        'elapsed': elapsed, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase81_periodic_table.json'), 'w',
              encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    # Figure
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Color mapping for operations
    op_colors = {
        'Addition': '#FF5252', 'Subtraction': '#FF4081', 'Multiplication': '#E040FB',
        'Division': '#7C4DFF', 'IntDivision': '#536DFE', 'Modulo': '#448AFF',
        'Power': '#40C4FF', 'GreaterThan': '#18FFFF', 'LessThan': '#64FFDA',
        'OrderCompare': '#69F0AE', 'Equal': '#B2FF59', 'NotEqual': '#EEFF41',
        'Absolute': '#FFD740', 'Maximum': '#FFAB40', 'Minimum': '#FF6E40',
        'Negation': '#795548', 'LogicAnd': '#9E9E9E', 'LogicOr': '#607D8B',
        'IntCast': '#FF8A80', 'Length': '#82B1FF', 'Other': '#CFD8DC',
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Periodic table visualization
    for elem in elements:
        c = op_colors.get(elem['op_name'], '#999999')
        axes[0].scatter(elem['group'], -elem['period'], c=c, s=100, alpha=0.6,
                       edgecolors='black', linewidth=0.5)

    # Add representative labels
    for key, elem in table.items():
        axes[0].text(elem['group'], -elem['period'], elem['symbol'],
                    ha='center', va='center', fontsize=5, fontweight='bold')

    axes[0].set_xlabel('Group (PC2 quantile)')
    axes[0].set_ylabel('Period (PC1 quantile)')
    axes[0].set_title('The Periodic Table\nof Programs', fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # 2. Operation type distribution
    sorted_ops = sorted(op_counts.items(), key=lambda x: x[1], reverse=True)[:15]
    op_names = [x[0] for x in sorted_ops]
    op_vals = [x[1] for x in sorted_ops]
    op_cols = [op_colors.get(n, '#999') for n in op_names]
    axes[1].barh(op_names, op_vals, color=op_cols, edgecolor='black')
    axes[1].set_xlabel('Count')
    axes[1].set_title('Element Types', fontweight='bold')

    plt.suptitle('Phase 81: The Periodic Table of Programs\n'
                 f'{len(elements)} Elements, {len(op_counts)} Types',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase81_periodic_table.png'), dpi=150)
    plt.close()
    print(f"\nPhase 81 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
