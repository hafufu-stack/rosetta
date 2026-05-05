"""
Phase 42: The Holographic Principle of Compilation
=====================================================
In physics, the holographic principle says 3D information
is encoded on a 2D surface. Can we project programs to 2D
and still reconstruct them? The ultimate compression test.
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
    print("Phase 42: The Holographic Principle of Compilation")
    print("Can 2 numbers encode an entire program?")
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

    src_to_idx = {}
    for i, d in enumerate(dataset):
        if d['source'] not in src_to_idx:
            src_to_idx[d['source']] = i
    unique_srcs = list(src_to_idx.keys())

    # === Compress to N dimensions, reconstruct, evaluate ===
    from sklearn.decomposition import PCA

    dims_to_test = [1, 2, 3, 4, 5, 6, 8, 10, 16, 32, 64]
    compression_results = []

    for n_dim in dims_to_test:
        if n_dim > z_ast.shape[1]:
            continue
        pca = PCA(n_components=n_dim)
        z_compressed = pca.fit_transform(z_ast)
        z_reconstructed = pca.inverse_transform(z_compressed)

        # Reconstruction quality
        cos_sims = []
        code_matches = 0
        n_test = min(50, len(unique_srcs))

        for si, src in enumerate(unique_srcs[:n_test]):
            idx = src_to_idx[src]
            orig = z_ast[idx]
            recon = z_reconstructed[idx]

            cos = float(np.dot(orig, recon) /
                       (np.linalg.norm(orig) * np.linalg.norm(recon) + 1e-8))
            cos_sims.append(cos)

            # Decode and compare
            code_orig = gen(orig)
            code_recon = gen(recon)
            if code_orig.strip() == code_recon.strip():
                code_matches += 1

        avg_cos = float(np.mean(cos_sims))
        code_acc = code_matches / n_test
        var_explained = float(sum(pca.explained_variance_ratio_))

        print(f"  {n_dim:2d}D: cos={avg_cos:.4f}, code_match={code_acc:.0%}, "
              f"var={var_explained:.3f}")

        compression_results.append({
            'dims': n_dim, 'avg_cos': avg_cos, 'code_accuracy': code_acc,
            'var_explained': var_explained,
        })

    # === The Holographic Test: 2D encoding ===
    print("\n--- The 2D Hologram ---")
    pca2 = PCA(n_components=2)
    z_2d = pca2.fit_transform(z_ast)
    z_recon_2d = pca2.inverse_transform(z_2d)

    print("  Decoding from just 2 numbers:")
    holo_examples = []
    for src in unique_srcs[:10]:
        idx = src_to_idx[src]
        coords = z_2d[idx]
        code_orig = gen(z_ast[idx])
        code_2d = gen(z_recon_2d[idx])
        match = code_orig.strip() == code_2d.strip()
        print(f"    ({coords[0]:+.2f}, {coords[1]:+.2f}) -> {code_2d[:35]} "
              f"{'== MATCH' if match else '!= ' + code_orig[:20]}")
        holo_examples.append({
            'coords': [float(coords[0]), float(coords[1])],
            'original': code_orig, 'from_2d': code_2d, 'match': match,
        })

    # === Minimum viable dimensions ===
    min_dim_90 = None
    for cr in compression_results:
        if cr['code_accuracy'] >= 0.9:
            min_dim_90 = cr['dims']
            break

    print(f"\n  Minimum dims for 90% code accuracy: {min_dim_90}")

    elapsed = time.time() - t0
    results = {
        'phase': 42, 'name': 'Holographic Principle of Compilation',
        'compression_results': compression_results,
        'min_dims_90pct': min_dim_90,
        'holographic_examples': holo_examples,
        'elapsed': elapsed, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase42_holographic.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Compression curve
    dims = [cr['dims'] for cr in compression_results]
    accs = [cr['code_accuracy'] for cr in compression_results]
    cos_vals = [cr['avg_cos'] for cr in compression_results]
    axes[0].plot(dims, accs, 'b-o', linewidth=2, markersize=6, label='Code accuracy')
    axes[0].plot(dims, cos_vals, 'r-s', linewidth=2, markersize=6, label='Cosine sim')
    axes[0].axhline(0.9, color='green', ls='--', alpha=0.5, label='90% threshold')
    if min_dim_90:
        axes[0].axvline(min_dim_90, color='purple', ls=':', alpha=0.5)
    axes[0].set_xlabel('Number of Dimensions')
    axes[0].set_ylabel('Reconstruction Quality')
    axes[0].set_title('Holographic Compression\n(how few dims needed?)', fontweight='bold')
    axes[0].legend()
    axes[0].set_xscale('log', base=2)

    # 2. 2D hologram: scatter of programs
    # Color by category
    cat_colors = []
    for i, d in enumerate(dataset):
        src = d['source']
        if any(op in src for op in ['.upper', '.lower', '.strip']):
            cat_colors.append('#4CAF50')
        elif any(op in src for op in ['>', '<', '==']):
            cat_colors.append('#2196F3')
        else:
            cat_colors.append('#F44336')

    axes[1].scatter(z_2d[:, 0], z_2d[:, 1], c=cat_colors, alpha=0.3, s=5)
    axes[1].set_xlabel('PC0'); axes[1].set_ylabel('PC1')
    axes[1].set_title('The 2D Hologram of All Programs\n(red=arith, blue=compare, green=string)',
                     fontweight='bold')

    # 3. Variance explained curve
    pca_full = PCA(n_components=min(64, z_ast.shape[1]))
    pca_full.fit(z_ast)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    axes[2].plot(range(1, len(cumvar)+1), cumvar, 'b-', linewidth=2)
    axes[2].axhline(0.9, color='red', ls='--', label='90%')
    axes[2].axhline(0.95, color='orange', ls='--', label='95%')
    axes[2].set_xlabel('Number of Dimensions')
    axes[2].set_ylabel('Cumulative Variance Explained')
    axes[2].set_title('Information Content per Dimension', fontweight='bold')
    axes[2].legend()

    plt.suptitle('Phase 42: The Holographic Principle of Compilation\n'
                 'How few numbers can encode an entire program?',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase42_holographic.png'), dpi=150)
    plt.close()
    print(f"\nPhase 42 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
