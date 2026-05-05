"""
Phase 36: Semantic Precision Trade-off (Software Uncertainty Principle)
=========================================================================
Is there a trade-off between knowing a program's exact syntax
and the range of semantic manipulations available?
Like Heisenberg's uncertainty principle, but for software.
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
    print("Phase 36: Software Uncertainty Principle")
    print("Position (syntax) vs Momentum (semantic range)")
    print("=" * 60)
    t0 = time.time()

    lat_file = os.path.join(DATA_DIR, 'rosetta_latents_v2.npz')
    if not os.path.exists(lat_file):
        lat_file = os.path.join(DATA_DIR, 'rosetta_latents.npz')
    latents = np.load(lat_file)
    z_ast, z_nl = latents['ast'], latents['nl']

    ds_file = os.path.join(DATA_DIR, 'rosetta_dataset_v2.json')
    if not os.path.exists(ds_file):
        ds_file = os.path.join(DATA_DIR, 'rosetta_dataset.json')
    with open(ds_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)['dataset']

    # Load decoder
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

    # === Experiment 1: Reconstruction precision vs perturbation sensitivity ===
    print("\n--- Position Precision vs Semantic Sensitivity ---")

    precision_data = []
    for src in unique_srcs[:30]:
        idx = src_to_idx[src]
        v = z_ast[idx]

        # Measure "position precision": can we reconstruct the exact source?
        decoded = gen(v)
        exact_match = 1.0 if decoded.strip() == src.strip() else 0.0

        # Measure "semantic range": how far can we perturb before code changes?
        radii = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
        change_radius = None
        for r in radii:
            n_changed = 0
            N_SAMPLES = 10
            for _ in range(N_SAMPLES):
                noise = np.random.randn(64).astype(np.float32)
                noise = noise / np.linalg.norm(noise) * r
                perturbed = gen(v + noise)
                if perturbed.strip() != decoded.strip():
                    n_changed += 1
            if n_changed >= N_SAMPLES // 2:
                change_radius = r
                break

        if change_radius is None:
            change_radius = 1.0

        # "Uncertainty product" = precision * range
        # High precision + small range OR low precision + large range
        precision_data.append({
            'src': src[:40], 'exact_match': exact_match,
            'change_radius': float(change_radius),
            'uncertainty_product': exact_match * change_radius,
        })

    # === Experiment 2: Dimension-wise uncertainty ===
    print("\n--- Per-Dimension Uncertainty ---")
    # For each AST dimension: how much does perturbing it change semantics?
    dim_sensitivity = np.zeros(64)
    dim_precision = np.zeros(64)

    test_vecs = z_ast[list(src_to_idx.values())[:20]]
    for d in range(64):
        changes, precisions = 0, 0
        for v in test_vecs:
            decoded_orig = gen(v)
            # Perturb dimension d
            v_pert = v.copy()
            v_pert[d] += 0.3 * z_ast[:, d].std()
            decoded_pert = gen(v_pert)
            if decoded_pert.strip() != decoded_orig.strip():
                changes += 1
            # Reconstruction from just this dim
            v_single = np.zeros(64, dtype=np.float32)
            v_single[d] = v[d]
            cos = abs(np.dot(v_single, v) / (np.linalg.norm(v_single) *
                     np.linalg.norm(v) + 1e-8))
            precisions += cos

        dim_sensitivity[d] = changes / len(test_vecs)
        dim_precision[d] = precisions / len(test_vecs)

    # Uncertainty relation: sensitivity * precision
    uncertainty_product = dim_sensitivity * dim_precision
    print(f"  Mean dim sensitivity: {dim_sensitivity.mean():.3f}")
    print(f"  Mean dim precision:   {dim_precision.mean():.3f}")
    print(f"  Mean uncertainty product: {uncertainty_product.mean():.4f}")

    # Most sensitive dims (high semantic impact)
    top_sensitive = np.argsort(dim_sensitivity)[::-1][:5]
    print(f"\n  Most semantically sensitive dims: {top_sensitive.tolist()}")
    print(f"  Their sensitivity: {[f'{dim_sensitivity[d]:.2f}' for d in top_sensitive]}")

    # === Experiment 3: Semantic manifold curvature ===
    print("\n--- Manifold Curvature (local vs global linearity) ---")
    curvature_data = []
    for src in unique_srcs[:15]:
        idx = src_to_idx[src]
        v = z_ast[idx]

        # Sample nearby points
        local_cos = []
        for _ in range(20):
            noise = np.random.randn(64).astype(np.float32) * 0.1
            v2 = v + noise
            code1 = gen(v)
            code2 = gen(v2)
            same = code1.strip() == code2.strip()
            cos = float(np.dot(v, v2) / (np.linalg.norm(v) * np.linalg.norm(v2) + 1e-8))
            local_cos.append({'cos': cos, 'same_code': same})

        # "Curvature" = how quickly code changes as we move
        n_same = sum(1 for lc in local_cos if lc['same_code'])
        stability = n_same / len(local_cos)
        curvature_data.append({
            'src': src[:35], 'local_stability': float(stability),
        })
        print(f"  {src[:35]}: stability={stability:.2f}")

    elapsed = time.time() - t0
    results = {
        'phase': 36, 'name': 'Software Uncertainty Principle',
        'precision_data': precision_data[:15],
        'dim_sensitivity': dim_sensitivity.tolist(),
        'dim_precision': dim_precision.tolist(),
        'mean_uncertainty': float(uncertainty_product.mean()),
        'top_sensitive_dims': top_sensitive.tolist(),
        'curvature': curvature_data,
        'elapsed': elapsed, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase36_uncertainty.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Position (reconstruction) vs Range (change radius)
    exacts = [p['exact_match'] for p in precision_data]
    radii = [p['change_radius'] for p in precision_data]
    colors = ['#4CAF50' if e > 0.5 else '#F44336' for e in exacts]
    axes[0].scatter(exacts, radii, c=colors, s=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Position Precision (exact reconstruction)')
    axes[0].set_ylabel('Semantic Range (change radius)')
    axes[0].set_title('Position vs Momentum\n(Software Uncertainty)', fontweight='bold')

    # 2. Per-dimension sensitivity vs precision
    axes[1].scatter(dim_precision, dim_sensitivity, c=uncertainty_product,
                   cmap='hot', s=40, edgecolor='black')
    axes[1].set_xlabel('Dimension Precision')
    axes[1].set_ylabel('Dimension Sensitivity')
    axes[1].set_title('Per-Dimension Uncertainty\n(precision x sensitivity)',
                     fontweight='bold')
    plt.colorbar(axes[1].collections[0], ax=axes[1], label='Uncertainty product')

    # 3. Manifold stability
    if curvature_data:
        stabs = [c['local_stability'] for c in curvature_data]
        srcs = [c['src'][:15] for c in curvature_data]
        axes[2].barh(srcs, stabs, color='#9C27B0', edgecolor='black', alpha=0.8)
        axes[2].set_xlabel('Local Stability')
        axes[2].set_title('Manifold Curvature\n(how stable is the neighborhood?)',
                         fontweight='bold')

    plt.suptitle('Phase 36: The Software Uncertainty Principle\n'
                 'Position (syntax) x Momentum (semantics) >= const?',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase36_uncertainty.png'), dpi=150)
    plt.close()
    print(f"\nPhase 36 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
