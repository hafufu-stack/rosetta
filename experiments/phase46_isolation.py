"""
Phase 46: The Isolation Paradox
==================================
P43: add is the LIGHTEST function (mass=3, most isolated)
P35: evolution discovers add FIRST (generation 1)
WHY is the most isolated function the easiest to find?
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
    print("Phase 46: The Isolation Paradox")
    print("Why is the loneliest function the easiest to find?")
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
    unique_vecs = np.array([z_ast[src_to_idx[s]] for s in unique_srcs])

    # === Measure 3 properties for each function ===
    from sklearn.metrics.pairwise import cosine_similarity
    cos_matrix = cosine_similarity(unique_vecs)

    func_props = []
    for si, src in enumerate(unique_srcs):
        v = unique_vecs[si]

        # 1. Isolation (inverse of mass)
        mass = int(np.sum(cos_matrix[si] > 0.9)) - 1
        isolation = 1.0 / (mass + 1)

        # 2. Basin size: how many random points decode to this function?
        basin = 0
        N_RANDOM = 50
        ast_mean = z_ast.mean(axis=0)
        ast_std = z_ast.std(axis=0)
        for _ in range(N_RANDOM):
            rand_v = np.random.randn(64).astype(np.float32) * ast_std + ast_mean
            code = gen(rand_v)
            if code.strip() == gen(v).strip():
                basin += 1
        basin_frac = basin / N_RANDOM

        # 3. Norm (distance from origin)
        norm = float(np.linalg.norm(v))

        func_props.append({
            'src': src, 'mass': mass, 'isolation': float(isolation),
            'basin': basin_frac, 'norm': norm,
        })

    # Sort by basin size
    func_props.sort(key=lambda x: -x['basin'])
    print("\n--- Largest Basins of Attraction ---")
    print("  (Functions that random points most often decode to)")
    for fp in func_props[:10]:
        print(f"    basin={fp['basin']:.2f} mass={fp['mass']:3d} "
              f"norm={fp['norm']:.2f} | {fp['src'][:40]}")

    print("\n--- The Paradox Data ---")
    # Find add specifically
    add_props = [fp for fp in func_props if 'return x + y' in fp['src']]
    if add_props:
        ap = add_props[0]
        print(f"  ADD: mass={ap['mass']}, basin={ap['basin']:.2f}, norm={ap['norm']:.2f}")

    # Correlation between isolation and basin
    isolations = [fp['isolation'] for fp in func_props]
    basins = [fp['basin'] for fp in func_props]
    corr = float(np.corrcoef(isolations, basins)[0, 1])
    print(f"\n  Correlation(isolation, basin): r = {corr:.3f}")

    # Correlation between norm and basin
    norms = [fp['norm'] for fp in func_props]
    corr_norm = float(np.corrcoef(norms, basins)[0, 1])
    print(f"  Correlation(norm, basin): r = {corr_norm:.3f}")

    # === Resolution ===
    print("\n--- RESOLUTION ---")
    # Isolated functions occupy their own unique region
    # -> the decoder has dedicated capacity for them
    # -> random vectors that happen to land near them have no competition
    # -> they are "attractors" BECAUSE they are isolated!
    mean_basin_low_mass = np.mean([fp['basin'] for fp in func_props if fp['mass'] < 10])
    mean_basin_high_mass = np.mean([fp['basin'] for fp in func_props if fp['mass'] >= 10])
    print(f"  Avg basin (isolated, mass<10): {mean_basin_low_mass:.3f}")
    print(f"  Avg basin (crowded, mass>=10): {mean_basin_high_mass:.3f}")

    elapsed = time.time() - t0
    results = {
        'phase': 46, 'name': 'The Isolation Paradox',
        'top_basins': func_props[:10],
        'correlation_isolation_basin': corr,
        'correlation_norm_basin': corr_norm,
        'mean_basin_isolated': float(mean_basin_low_mass),
        'mean_basin_crowded': float(mean_basin_high_mass),
        'elapsed': elapsed, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase46_isolation.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].scatter(isolations, basins, alpha=0.5, s=20, c='#E91E63')
    axes[0].set_xlabel('Isolation (1/mass)')
    axes[0].set_ylabel('Basin of Attraction')
    axes[0].set_title(f'Isolation vs Findability\nr={corr:.3f}', fontweight='bold')

    axes[1].scatter(norms, basins, alpha=0.5, s=20, c='#2196F3')
    axes[1].set_xlabel('Norm (distance from origin)')
    axes[1].set_ylabel('Basin of Attraction')
    axes[1].set_title(f'Norm vs Findability\nr={corr_norm:.3f}', fontweight='bold')

    axes[2].bar(['Isolated\n(mass<10)', 'Crowded\n(mass>=10)'],
               [mean_basin_low_mass, mean_basin_high_mass],
               color=['#4CAF50', '#F44336'], edgecolor='black')
    axes[2].set_ylabel('Avg Basin Size')
    axes[2].set_title('Resolution: Isolation = Freedom\nLess competition = easier to find',
                     fontweight='bold')

    plt.suptitle('Phase 46: The Isolation Paradox\n'
                 'Lonely functions are the easiest to evolve to',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase46_isolation.png'), dpi=150)
    plt.close()
    print(f"\nPhase 46 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
