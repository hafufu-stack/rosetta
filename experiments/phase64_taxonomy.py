"""
Phase 64: The Rosetta Taxonomy
================================
BONUS PHASE (Opus's idea)

Do programs naturally form "species" in 5D space?
Like biology has kingdoms/phyla/species, do programs
cluster into functional families?

Use HDBSCAN/KMeans to discover natural groupings,
then analyze what each "species" means semantically.
"""
import os, json, time, sys, inspect
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')


def main():
    print("=" * 60)
    print("Phase 64: The Rosetta Taxonomy")
    print("Do programs form natural 'species'?")
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
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.metrics import silhouette_score

    pca = PCA(n_components=10).fit(z_ast)
    z_5d = pca.transform(z_ast)[:, :5]

    # Deduplicate
    unique = {}
    for i, src in enumerate(sources):
        if src not in unique:
            unique[src] = {'idx': i, 'z_5d': z_5d[i], 'src': src}
    func_list = list(unique.values())
    X = np.array([f['z_5d'] for f in func_list])
    N = len(func_list)
    print(f"  Unique functions: {N}")

    # Find optimal K using silhouette score
    print("\n--- Optimal Cluster Count ---")
    sil_scores = {}
    for k in range(3, 15):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels)
        sil_scores[k] = sil
        print(f"  K={k:2d}: silhouette={sil:.4f}")

    best_k = max(sil_scores, key=sil_scores.get)
    print(f"\n  Best K = {best_k} (silhouette = {sil_scores[best_k]:.4f})")

    # Final clustering with best K
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    # Analyze each cluster = "species"
    print(f"\n--- The {best_k} Species of Programs ---")
    species = {}
    for i, f in enumerate(func_list):
        c = int(labels[i])
        if c not in species:
            species[c] = []
        species[c].append(f['src'])

    species_info = []
    for c in sorted(species.keys()):
        members = species[c]
        # Determine what operations dominate this cluster
        ops = {'+': 0, '-': 0, '*': 0, '/': 0, '%': 0, '**': 0,
               '>': 0, '<': 0, '==': 0, '!=': 0,
               'abs': 0, 'max': 0, 'min': 0, 'if': 0,
               'not': 0, 'len': 0, 'int': 0, 'float': 0, 'bool': 0,
               'upper': 0, 'lower': 0, 'strip': 0}
        for src in members:
            for op in ops:
                if op in src:
                    ops[op] += 1

        # Find dominant operation
        dominant = max(ops, key=ops.get) if max(ops.values()) > 0 else 'unknown'
        # Get top-3 ops
        top_ops = sorted(ops.items(), key=lambda x: x[1], reverse=True)[:3]
        top_ops_str = ', '.join(f'{k}({v})' for k, v in top_ops if v > 0)

        # Compute centroid
        centroid = km.cluster_centers_[c]
        centroid_str = ', '.join(f'{x:.2f}' for x in centroid)

        # Classify species name
        if ops['+'] > len(members) * 0.3:
            sp_name = "Additive"
        elif ops['-'] > len(members) * 0.3:
            sp_name = "Subtractive"
        elif ops['*'] > len(members) * 0.3:
            sp_name = "Multiplicative"
        elif ops['>'] + ops['<'] > len(members) * 0.3:
            sp_name = "Comparative"
        elif ops['=='] + ops['!='] > len(members) * 0.3:
            sp_name = "Equality"
        elif ops['abs'] > len(members) * 0.2:
            sp_name = "Absolute"
        elif ops['if'] > len(members) * 0.2:
            sp_name = "Conditional"
        elif ops['upper'] + ops['lower'] + ops['strip'] > 0:
            sp_name = "String"
        elif ops['len'] > 0:
            sp_name = "Collection"
        else:
            sp_name = f"Species-{c}"

        print(f"\n  === Species {c}: {sp_name} ({len(members)} members) ===")
        print(f"  Centroid: [{centroid_str}]")
        print(f"  Top ops: {top_ops_str}")
        print(f"  Examples:")
        for m in members[:5]:
            print(f"    {m}")

        species_info.append({
            'id': c, 'name': sp_name, 'n_members': len(members),
            'centroid': centroid.tolist(),
            'top_ops': [(k, v) for k, v in top_ops if v > 0],
            'examples': members[:5],
        })

    # Inter-species distances
    print("\n--- Inter-Species Distances ---")
    centroids = km.cluster_centers_
    for i in range(best_k):
        for j in range(i+1, best_k):
            d = np.linalg.norm(centroids[i] - centroids[j])
            print(f"  {species_info[i]['name']:15s} <-> "
                  f"{species_info[j]['name']:15s}: {d:.4f}")

    # DBSCAN for noise detection (are there "outlier" programs?)
    print("\n--- DBSCAN Outlier Detection ---")
    db = DBSCAN(eps=0.5, min_samples=3).fit(X)
    n_clusters_db = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    n_noise = list(db.labels_).count(-1)
    print(f"  DBSCAN clusters: {n_clusters_db}")
    print(f"  Noise points (outliers): {n_noise} ({n_noise/N*100:.0f}%)")

    if n_noise > 0:
        print("  Outlier programs:")
        for i, f in enumerate(func_list):
            if db.labels_[i] == -1:
                print(f"    {f['src']}")
                if i > 10:
                    print(f"    ... ({n_noise - 10} more)")
                    break

    elapsed = time.time() - t0
    results = {
        'phase': 64, 'name': 'The Rosetta Taxonomy',
        'best_k': best_k,
        'silhouette': float(sil_scores[best_k]),
        'silhouette_scores': {str(k): float(v) for k, v in sil_scores.items()},
        'species': species_info,
        'n_dbscan_clusters': n_clusters_db,
        'n_outliers': n_noise,
        'elapsed': elapsed, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase64_taxonomy.json'), 'w',
              encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    # Figure
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. 2D projection with species coloring
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(20, N-1))
    z_2d = tsne.fit_transform(X)

    cmap = plt.cm.Set1
    for c in range(best_k):
        mask = labels == c
        axes[0].scatter(z_2d[mask, 0], z_2d[mask, 1],
                       c=[cmap(c/best_k)], s=30, alpha=0.7,
                       label=species_info[c]['name'])
    axes[0].set_title(f'The {best_k} Species of Programs\n(t-SNE)',
                     fontweight='bold')
    axes[0].legend(fontsize=7, loc='upper left')

    # 2. Silhouette scores
    ks = sorted(sil_scores.keys())
    axes[1].plot(ks, [sil_scores[k] for k in ks], 'o-', color='#2196F3')
    axes[1].axvline(best_k, color='red', linestyle='--',
                   label=f'Best K={best_k}')
    axes[1].set_xlabel('Number of Clusters (K)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('Optimal Species Count', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3. Species sizes
    names = [s['name'] for s in species_info]
    sizes = [s['n_members'] for s in species_info]
    colors = [cmap(i/best_k) for i in range(best_k)]
    axes[2].barh(names, sizes, color=colors, edgecolor='black')
    axes[2].set_xlabel('Number of Functions')
    axes[2].set_title('Species Population', fontweight='bold')

    plt.suptitle('Phase 64: The Rosetta Taxonomy\n'
                 'Natural Species of Programs in 5D Space',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase64_taxonomy.png'), dpi=150)
    plt.close()
    print(f"\nPhase 64 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
