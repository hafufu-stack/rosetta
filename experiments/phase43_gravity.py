"""
Phase 43: Semantic Gravity
=============================
Some functions are "heavy" (many programs cluster around them).
Others are "light" (isolated in the manifold).
Map the density field to find the gravitational wells of software.
Which functions are the black holes?
"""
import os, json, time, sys
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    print("=" * 60)
    print("Phase 43: Semantic Gravity")
    print("The gravitational wells of software")
    print("=" * 60)
    t0 = time.time()

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

    src_to_idx = {}
    for i, d in enumerate(dataset):
        if d['source'] not in src_to_idx:
            src_to_idx[d['source']] = i
    unique_srcs = list(src_to_idx.keys())
    unique_vecs = np.array([z_ast[src_to_idx[s]] for s in unique_srcs])

    # === Local density (KDE-like) ===
    print("\n--- Computing gravitational mass of each function ---")
    from sklearn.metrics.pairwise import cosine_similarity
    cos_matrix = cosine_similarity(unique_vecs)

    # "Mass" = number of neighbors within cos > 0.9
    THRESHOLD = 0.9
    masses = []
    for i in range(len(unique_srcs)):
        n_neighbors = int(np.sum(cos_matrix[i] > THRESHOLD)) - 1  # Exclude self
        masses.append(n_neighbors)

    masses = np.array(masses)
    sorted_idx = np.argsort(masses)[::-1]

    print("\n  Heaviest functions (most neighbors):")
    heavy_funcs = []
    for rank, idx in enumerate(sorted_idx[:10]):
        print(f"    #{rank+1}: mass={masses[idx]:3d} | {unique_srcs[idx][:45]}")
        heavy_funcs.append({'rank': rank+1, 'mass': int(masses[idx]),
                           'src': unique_srcs[idx]})

    print("\n  Lightest functions (most isolated):")
    light_funcs = []
    for rank, idx in enumerate(sorted_idx[-5:]):
        print(f"    mass={masses[idx]:3d} | {unique_srcs[idx][:45]}")
        light_funcs.append({'mass': int(masses[idx]), 'src': unique_srcs[idx]})

    # === Gravitational center ===
    center = unique_vecs.mean(axis=0)
    dists_to_center = np.array([
        float(np.linalg.norm(unique_vecs[i] - center))
        for i in range(len(unique_srcs))
    ])
    nearest_center = unique_srcs[np.argmin(dists_to_center)]
    print(f"\n  Gravitational center nearest function: {nearest_center[:45]}")

    # === Cluster analysis ===
    print("\n--- Semantic Galaxy Clusters ---")
    from sklearn.cluster import KMeans
    n_clusters = 5
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(unique_vecs)

    sys.path.insert(0, BASE_DIR)
    with open(os.path.join(DATA_DIR, 'char_vocab.json'), 'r') as f:
        vd = json.load(f)
    idx2char = {int(i): c for c, i in vd['char2idx'].items()}
    V = len(vd['char2idx'])
    from experiments.phase9_generative_decompiler import CodeDecoder, decode_tokens
    decoder = CodeDecoder(64, V, hidden=128, max_len=80).to(DEVICE)
    dec_path = os.path.join(DATA_DIR, 'decoder_robust.pt')
    if not os.path.exists(dec_path):
        dec_path = os.path.join(DATA_DIR, 'decoder.pt')
    decoder.load_state_dict(torch.load(dec_path, map_location=DEVICE, weights_only=True))
    decoder.eval()

    def gen(z):
        with torch.no_grad():
            z_t = torch.tensor(z.astype(np.float32)).unsqueeze(0).to(DEVICE)
            tokens = decoder(z_t)
            return decode_tokens(tokens[0].cpu().numpy(), idx2char)

    clusters = []
    for c in range(n_clusters):
        members = [unique_srcs[i] for i in range(len(unique_srcs)) if labels[i] == c]
        centroid_code = gen(km.cluster_centers_[c])
        print(f"\n  Cluster {c} ({len(members)} members): {centroid_code[:40]}")
        for m in members[:3]:
            print(f"    - {m[:45]}")
        clusters.append({
            'id': c, 'size': len(members), 'centroid_code': centroid_code,
            'examples': [m[:50] for m in members[:5]],
        })

    # === Escape velocity ===
    print("\n--- Escape Velocity ---")
    # How far must you move from a function before you reach a different one?
    escape_vels = []
    for i in range(min(20, len(unique_srcs))):
        v = unique_vecs[i]
        orig_code = gen(v)
        for r in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
            noise = np.random.randn(64).astype(np.float32)
            noise = noise / np.linalg.norm(noise) * r
            new_code = gen(v + noise)
            if new_code.strip() != orig_code.strip():
                escape_vels.append(r)
                break
        else:
            escape_vels.append(1.0)

    avg_escape = float(np.mean(escape_vels))
    print(f"  Average escape velocity: {avg_escape:.3f}")

    elapsed = time.time() - t0
    results = {
        'phase': 43, 'name': 'Semantic Gravity',
        'heaviest': heavy_funcs, 'lightest': light_funcs,
        'center_func': nearest_center,
        'clusters': clusters,
        'avg_escape_velocity': avg_escape,
        'mass_distribution': {'mean': float(masses.mean()),
                             'max': int(masses.max()), 'min': int(masses.min())},
        'elapsed': elapsed, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase43_gravity.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Mass distribution
    axes[0].hist(masses, bins=20, color='#E91E63', edgecolor='black', alpha=0.8)
    axes[0].set_xlabel('Gravitational Mass (# neighbors)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Mass Distribution\n(heaviest = most common pattern)',
                     fontweight='bold')

    # 2. 2D galaxy map with mass as size
    pca2 = PCA(n_components=2)
    z2d = pca2.fit_transform(unique_vecs)
    sizes = (masses / max(masses.max(), 1) * 100 + 5)
    scatter = axes[1].scatter(z2d[:, 0], z2d[:, 1], c=labels, cmap='Set1',
                             s=sizes, alpha=0.6, edgecolor='black', linewidth=0.3)
    axes[1].set_title('The Software Galaxy\n(size = gravitational mass)',
                     fontweight='bold')

    # 3. Cluster sizes
    cluster_sizes = [c['size'] for c in clusters]
    cluster_names = [c['centroid_code'][:20] for c in clusters]
    axes[2].barh(cluster_names, cluster_sizes, color='#2196F3', edgecolor='black')
    axes[2].set_xlabel('Number of Functions')
    axes[2].set_title('Galaxy Clusters\n(semantic neighborhoods)', fontweight='bold')

    plt.suptitle('Phase 43: Semantic Gravity\n'
                 'The gravitational wells of the code manifold',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase43_gravity.png'), dpi=150)
    plt.close()
    print(f"\nPhase 43 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
