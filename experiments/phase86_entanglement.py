"""
Phase 86: The Entanglement Map
================================
Which programs are "entangled"?
Two programs are entangled if knowing one's position
tells you EXACTLY where the other is.

Like quantum entanglement: measuring one particle
instantly determines the state of its partner.
"""
import os, json, time
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')


def main():
    print("=" * 60)
    print("Phase 86: The Entanglement Map")
    print("Which programs are quantum entangled?")
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

    unique = {}
    for i, src in enumerate(sources):
        if src not in unique:
            unique[src] = z_5d[i]
    all_srcs = list(unique.keys())
    all_z5 = np.array([unique[s] for s in all_srcs])
    N = len(all_z5)

    # Compute entanglement: pairs with CONSTANT vector offset
    # (one always predicts the other)
    print(f"\n  Analyzing {N} unique functions")

    # Find "twin" pairs: programs related by a SIMPLE transformation
    print("\n--- Finding Entangled Pairs ---")
    
    # Method: for each pair, compute the offset vector.
    # If two different pairs share the SAME offset, they're "entangled"
    # through a common transformation.
    
    # Compute all pairwise offset vectors (subsample)
    np.random.seed(42)
    sample_idx = np.random.choice(N, min(100, N), replace=False)
    
    offset_clusters = {}
    entangled_pairs = []
    
    for i in range(len(sample_idx)):
        for j in range(i+1, len(sample_idx)):
            ii, jj = sample_idx[i], sample_idx[j]
            offset = all_z5[jj] - all_z5[ii]
            dist = float(np.linalg.norm(offset))
            
            if dist < 0.01:  # Same point (renaming symmetry)
                cos = 1.0
                entangled_pairs.append({
                    'src1': all_srcs[ii], 'src2': all_srcs[jj],
                    'type': 'identity',
                    'distance': dist,
                    'cos': cos,
                })

    # Find reflection pairs: f and -f
    print("\n--- Reflection Pairs (f and mirror-f) ---")
    for i in range(N):
        # Check if -z_i exists
        dists_to_mirror = np.linalg.norm(all_z5 + all_z5[i], axis=1)
        nn_idx = np.argmin(dists_to_mirror)
        nn_dist = dists_to_mirror[nn_idx]
        if nn_dist < 0.3 and nn_idx != i:
            s1 = all_srcs[i].split('return ')[1][:15] if 'return' in all_srcs[i] else '?'
            s2 = all_srcs[nn_idx].split('return ')[1][:15] if 'return' in all_srcs[nn_idx] else '?'
            if s1 != s2:
                cos = float(cosine_similarity(
                    all_z5[i].reshape(1,-1), all_z5[nn_idx].reshape(1,-1))[0,0])
                print(f"  {s1:15s} <-> {s2:15s}: cos={cos:.4f}, mirror_dist={nn_dist:.4f}")
                entangled_pairs.append({
                    'src1': all_srcs[i], 'src2': all_srcs[nn_idx],
                    'type': 'reflection',
                    'distance': float(np.linalg.norm(all_z5[i] - all_z5[nn_idx])),
                    'cos': cos,
                })

    # Find closest pairs (nearest neighbors)
    print("\n--- Most Entangled Pairs (Nearest Neighbors) ---")
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=2).fit(all_z5)
    distances, indices = nn.kneighbors(all_z5)
    
    nn_dists = distances[:, 1]
    closest_pairs = np.argsort(nn_dists)
    
    seen = set()
    shown = 0
    for idx in closest_pairs:
        nn_idx = indices[idx, 1]
        pair = tuple(sorted([idx, nn_idx]))
        if pair in seen:
            continue
        seen.add(pair)
        
        s1 = all_srcs[idx].split('return ')[1][:15] if 'return' in all_srcs[idx] else '?'
        s2 = all_srcs[nn_idx].split('return ')[1][:15] if 'return' in all_srcs[nn_idx] else '?'
        d = float(nn_dists[idx])
        
        if s1 != s2 and d > 0.001:  # Not just variable renaming
            print(f"  d={d:.4f}: {s1:15s} <-> {s2}")
            shown += 1
            if shown >= 10:
                break

    # Compute entanglement entropy
    print("\n--- Entanglement Entropy ---")
    # Use nearest neighbor distances as a proxy
    nn_log = np.log(nn_dists + 1e-10)
    entanglement_entropy = -np.mean(nn_log)
    print(f"  Entanglement entropy: {entanglement_entropy:.4f}")
    print(f"  Mean NN distance: {np.mean(nn_dists):.4f}")
    print(f"  Std NN distance: {np.std(nn_dists):.4f}")
    
    # What fraction are "strongly entangled" (d < 0.1)?
    n_entangled = sum(1 for d in nn_dists if d < 0.1)
    pct = n_entangled / N * 100
    print(f"  Strongly entangled (d<0.1): {n_entangled}/{N} ({pct:.0f}%)")

    elapsed = time.time() - t0
    results = {
        'phase': 86, 'name': 'The Entanglement Map',
        'n_functions': N,
        'n_identity_pairs': sum(1 for p in entangled_pairs if p['type'] == 'identity'),
        'n_reflection_pairs': sum(1 for p in entangled_pairs if p['type'] == 'reflection'),
        'entanglement_entropy': float(entanglement_entropy),
        'mean_nn_dist': float(np.mean(nn_dists)),
        'n_strongly_entangled': n_entangled,
        'pct_entangled': float(pct),
        'elapsed': elapsed, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase86_entanglement.json'), 'w',
              encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(nn_dists, bins=30, color='#9C27B0', edgecolor='black', alpha=0.8)
    axes[0].axvline(0.1, color='red', linestyle='--', label='Entanglement threshold')
    axes[0].set_xlabel('NN Distance')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'Nearest Neighbor Distances\n{pct:.0f}% entangled', fontweight='bold')
    axes[0].legend()

    # Entanglement network
    for pair in entangled_pairs[:50]:
        z1 = unique.get(pair['src1'])
        z2 = unique.get(pair['src2'])
        if z1 is not None and z2 is not None:
            c = '#4CAF50' if pair['type'] == 'identity' else '#FF9800'
            axes[1].plot([z1[0], z2[0]], [z1[1], z2[1]], '-', color=c, alpha=0.3)
    axes[1].scatter(all_z5[:, 0], all_z5[:, 1], c='gray', s=10, alpha=0.3)
    axes[1].set_xlabel('PC1'); axes[1].set_ylabel('PC2')
    axes[1].set_title('Entanglement Network', fontweight='bold')

    summary = (f"ENTANGLEMENT MAP\n\n"
              f"Functions: {N}\n"
              f"Identity pairs: {sum(1 for p in entangled_pairs if p['type'] == 'identity')}\n"
              f"Reflection pairs: {sum(1 for p in entangled_pairs if p['type'] == 'reflection')}\n"
              f"Entropy: {entanglement_entropy:.3f}\n"
              f"Entangled: {pct:.0f}%")
    axes[2].text(0.5, 0.5, summary, ha='center', va='center',
                fontsize=12, fontweight='bold', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#F3E5F5', alpha=0.8),
                transform=axes[2].transAxes)
    axes[2].axis('off')

    plt.suptitle('Phase 86: The Entanglement Map\nQuantum Correlations in Program Space',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase86_entanglement.png'), dpi=150)
    plt.close()
    print(f"\nPhase 86 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
