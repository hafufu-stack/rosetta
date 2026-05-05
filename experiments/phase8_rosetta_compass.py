"""
Phase 8: The Rosetta Compass (Bonus)
=====================================
Visualize the latent space with t-SNE. Do NL/AST/Bytecode vectors
form meaningful clusters by semantic category?
"""
import os, json, time
import numpy as np
from sklearn.manifold import TSNE

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')


def categorize(source):
    s = source.lower()
    if any(op in s for op in ['.upper', '.lower', '.strip', '.title',
                               '.swapcase', '[::-1]', '.split']):
        return 'String'
    if any(op in s for op in ['sum(', 'sorted(', 'reversed(', 'max(l', 'min(l']):
        return 'List'
    if any(op in s for op in ['> ', '< ', '==', '!=', '>=', '<=']):
        return 'Compare'
    if ' and ' in s or ' or ' in s or 'not ' in s:
        return 'Boolean'
    if ' if ' in s:
        return 'Conditional'
    if 'abs(' in s:
        return 'Abs/Unary'
    return 'Arithmetic'


def main():
    print("=" * 60)
    print("Phase 8: The Rosetta Compass")
    print("=" * 60)
    t0 = time.time()

    ds_file = os.path.join(DATA_DIR, 'rosetta_dataset_v2.json')
    if not os.path.exists(ds_file):
        ds_file = os.path.join(DATA_DIR, 'rosetta_dataset.json')
    with open(ds_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)['dataset']

    lat_file = os.path.join(DATA_DIR, 'rosetta_latents_v2.npz')
    if not os.path.exists(lat_file):
        lat_file = os.path.join(DATA_DIR, 'rosetta_latents.npz')
    latents = np.load(lat_file)
    z_nl, z_ast, z_bc = latents['nl'], latents['ast'], latents['bc']
    N = len(z_nl)

    # Subsample for t-SNE speed
    max_n = min(N, 2000)
    idx = np.random.RandomState(42).choice(N, max_n, replace=False)
    categories = [categorize(dataset[i]['source']) for i in idx]

    cat_set = sorted(set(categories))
    cat2id = {c: i for i, c in enumerate(cat_set)}
    cat_ids = [cat2id[c] for c in categories]

    print(f"Samples: {max_n}, Categories: {cat_set}")
    for c in cat_set:
        print(f"  {c}: {categories.count(c)}")

    # t-SNE on combined NL+AST+BC space
    print("\nRunning t-SNE...")
    combined = np.concatenate([z_nl[idx], z_ast[idx], z_bc[idx]], axis=1)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    coords = tsne.fit_transform(combined)

    # Also per-modality t-SNE
    tsne_nl = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(z_nl[idx])
    tsne_ast = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(z_ast[idx])

    elapsed = time.time() - t0

    # Cluster purity (how well categories cluster)
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=5)
    split = int(max_n * 0.8)
    knn.fit(combined[:split], cat_ids[:split])
    purity = knn.score(combined[split:], cat_ids[split:])
    print(f"\nCluster purity (5-NN): {purity:.1%}")

    results = {
        'phase': 8, 'name': 'The Rosetta Compass',
        'n_samples': max_n, 'categories': cat_set,
        'cluster_purity': float(purity),
        'elapsed': elapsed,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase8_rosetta_compass.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Figure
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    colors_map = {
        'Arithmetic': '#E91E63', 'Compare': '#2196F3', 'String': '#4CAF50',
        'List': '#FF9800', 'Boolean': '#9C27B0', 'Conditional': '#00BCD4',
        'Abs/Unary': '#795548',
    }

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for ax, data, title in [
        (axes[0], coords, 'Combined (NL+AST+BC)'),
        (axes[1], tsne_nl, 'NL Space Only'),
        (axes[2], tsne_ast, 'AST Space Only'),
    ]:
        for c in cat_set:
            mask = [i for i, cat in enumerate(categories) if cat == c]
            ax.scatter(data[mask, 0], data[mask, 1], s=15, alpha=0.6,
                      color=colors_map.get(c, '#999'), label=c)
        ax.set_title(title, fontweight='bold', fontsize=13)
        ax.legend(fontsize=8, loc='best', markerscale=2)
        ax.set_xticks([]); ax.set_yticks([])

    plt.suptitle(f'Phase 8: The Rosetta Compass (Purity={purity:.0%})',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase8_rosetta_compass.png'), dpi=150)
    plt.close()

    print(f"\nPhase 8 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
