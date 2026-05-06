"""
Phase 76: The Information Bottleneck
======================================
How much information does the 5D representation retain?

Compute the mutual information between:
1. Source code -> 5D representation
2. 5D representation -> I/O behavior

If 5D captures most of the behavioral information,
it's an optimal lossy compression of programs.
"""
import os, json, time, inspect
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')


def main():
    print("=" * 60)
    print("Phase 76: The Information Bottleneck")
    print("How compressible is program behavior?")
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
    z_all = pca.transform(z_ast)

    # Compute information at different dimensionalities
    print("\n--- Reconstruction Quality vs Dimensions ---")
    src_to_z64 = {}
    for i, src in enumerate(sources):
        if src not in src_to_z64:
            src_to_z64[src] = z_ast[i]
    unique_srcs = list(src_to_z64.keys())
    unique_z64 = np.array([src_to_z64[s] for s in unique_srcs])

    # For each dimensionality, measure reconstruction error
    dim_results = []
    for ndim in [1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 32, 64]:
        pca_n = PCA(n_components=min(ndim, unique_z64.shape[1])).fit(unique_z64)
        z_compressed = pca_n.transform(unique_z64)
        z_reconstructed = pca_n.inverse_transform(z_compressed)
        recon_error = np.mean(np.sum((unique_z64 - z_reconstructed)**2, axis=1))
        var_explained = sum(pca_n.explained_variance_ratio_) * 100

        dim_results.append({
            'ndim': ndim, 'recon_error': float(recon_error),
            'var_explained': float(var_explained),
        })
        print(f"  {ndim:2d}D: var={var_explained:5.1f}%, recon_err={recon_error:.4f}")

    # I/O prediction accuracy at different dimensionalities
    print("\n--- I/O Prediction vs Dimensions ---")
    test_vals = [1, 2, -1, 3, 5]

    # Build behavioral fingerprints
    behaviors = {}
    for src in unique_srcs:
        try:
            ns = {}
            exec(compile(src, '<string>', 'exec'), ns)
            fn = [v for k, v in ns.items()
                  if callable(v) and not k.startswith('_')][0]
            sig = inspect.signature(fn)
            n_params = len(sig.parameters)

            fingerprint = []
            if n_params == 1:
                for v in test_vals:
                    try:
                        r = fn(v)
                        if isinstance(r, (int, float)):
                            fingerprint.append(float(r))
                    except Exception:
                        fingerprint.append(None)
            elif n_params == 2:
                pairs = [(1,2), (2,3), (-1,1), (3,5), (5,7)]
                for a, b in pairs:
                    try:
                        r = fn(a, b)
                        if isinstance(r, (int, float)):
                            fingerprint.append(float(r))
                    except Exception:
                        fingerprint.append(None)

            if fingerprint and all(x is not None for x in fingerprint):
                behaviors[src] = np.array(fingerprint)
        except Exception:
            pass

    print(f"  Functions with valid fingerprints: {len(behaviors)}")

    # For each dimension, train a predictor from z -> behavior
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score

    behav_srcs = list(behaviors.keys())
    Y_behav = np.array([behaviors[s] for s in behav_srcs])

    io_results = []
    for ndim in [1, 2, 3, 5, 10, 20, 64]:
        pca_n = PCA(n_components=min(ndim, unique_z64.shape[1])).fit(unique_z64)
        X_ndim = np.array([pca_n.transform(src_to_z64[s].reshape(1,-1))[0]
                          for s in behav_srcs])

        if len(X_ndim) > 10:
            ridge = Ridge(alpha=1.0)
            scores = cross_val_score(ridge, X_ndim, Y_behav,
                                    cv=min(5, len(X_ndim)//2),
                                    scoring='r2')
            r2 = float(np.mean(scores))
        else:
            r2 = 0.0

        io_results.append({'ndim': ndim, 'r2': r2})
        print(f"  {ndim:2d}D -> I/O prediction R2: {r2:.4f}")

    # Information content (entropy of 5D representation)
    print("\n--- Information Content ---")
    z_5d = pca.transform(unique_z64)[:, :5]

    # Discretize and compute entropy
    n_bins = 20
    entropy_per_dim = []
    for dim in range(5):
        vals = z_5d[:, dim]
        hist, _ = np.histogram(vals, bins=n_bins, density=True)
        hist = hist[hist > 0]
        bin_width = (vals.max() - vals.min()) / n_bins
        entropy = -np.sum(hist * bin_width * np.log2(hist * bin_width + 1e-10))
        entropy_per_dim.append(float(entropy))
        print(f"  PC{dim+1} entropy: {entropy:.3f} bits")

    total_entropy = sum(entropy_per_dim)
    print(f"  Total 5D entropy: {total_entropy:.3f} bits")
    print(f"  Bits per program: {total_entropy:.1f}")

    # Source code length (for comparison)
    avg_src_len = np.mean([len(s) for s in unique_srcs])
    avg_src_bits = avg_src_len * 7  # ASCII = 7 bits per char
    compression_ratio = avg_src_bits / max(total_entropy, 1e-8)
    print(f"  Avg source length: {avg_src_len:.0f} chars ({avg_src_bits:.0f} bits)")
    print(f"  Compression ratio: {compression_ratio:.1f}x")

    elapsed = time.time() - t0
    results = {
        'phase': 76, 'name': 'The Information Bottleneck',
        'dim_results': dim_results,
        'io_results': io_results,
        'entropy_per_dim': entropy_per_dim,
        'total_entropy': total_entropy,
        'compression_ratio': float(compression_ratio),
        'elapsed': elapsed, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase76_information.json'), 'w',
              encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    # Figure
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Variance vs dimensions
    dims = [r['ndim'] for r in dim_results]
    vars_exp = [r['var_explained'] for r in dim_results]
    axes[0].plot(dims, vars_exp, 'o-', color='#4CAF50', markersize=6)
    axes[0].axvline(5, color='red', linestyle='--', label='5D')
    axes[0].set_xlabel('Dimensions')
    axes[0].set_ylabel('Variance Explained (%)')
    axes[0].set_title('Information vs Compression', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xscale('log')

    # 2. I/O prediction
    io_dims = [r['ndim'] for r in io_results]
    io_r2 = [r['r2'] for r in io_results]
    axes[1].plot(io_dims, io_r2, 's-', color='#2196F3', markersize=8)
    axes[1].axvline(5, color='red', linestyle='--', label='5D')
    axes[1].set_xlabel('Dimensions')
    axes[1].set_ylabel('I/O Prediction R2')
    axes[1].set_title('Behavioral Prediction\nvs Compression', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xscale('log')

    # 3. Entropy per dimension
    axes[2].bar([f'PC{i+1}' for i in range(5)], entropy_per_dim,
               color='#FF9800', edgecolor='black')
    axes[2].set_ylabel('Entropy (bits)')
    axes[2].set_title(f'Information per Dimension\nTotal: {total_entropy:.1f} bits',
                     fontweight='bold')

    plt.suptitle('Phase 76: The Information Bottleneck\n'
                 f'Compression: {compression_ratio:.0f}x (source -> 5D)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase76_information.png'), dpi=150)
    plt.close()
    print(f"\nPhase 76 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
