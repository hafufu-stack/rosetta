"""
Phase 40: The 5-Dimensional Theory of Code
=============================================
P38 revealed programs live in ~5 effective dimensions.
WHAT ARE those 5 dimensions? Name them. Decode them.
The Five Elements of Software.
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
    print("Phase 40: The 5-Dimensional Theory of Code")
    print("What ARE the 5 elements of software?")
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

    # PCA to find the 5 principal axes
    from sklearn.decomposition import PCA
    pca = PCA(n_components=10)
    z_pca = pca.fit_transform(z_ast)

    print(f"  Explained variance ratios:")
    for i, v in enumerate(pca.explained_variance_ratio_[:10]):
        print(f"    PC{i}: {v:.4f} ({sum(pca.explained_variance_ratio_[:i+1])*100:.1f}% cumul)")

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

    # === Decode each principal axis ===
    print("\n--- The Five Elements ---")
    mean_vec = pca.mean_
    elements = []

    for pc in range(5):
        axis = pca.components_[pc]  # Direction in 64-dim space
        # Walk along this axis: -3sigma to +3sigma
        std = np.sqrt(pca.explained_variance_[pc])

        print(f"\n  === PC{pc} (var={pca.explained_variance_ratio_[pc]:.3f}) ===")
        codes_along = []
        for t in [-2, -1, 0, 1, 2]:
            v = mean_vec + t * std * axis
            code = gen(v)
            codes_along.append({'t': t, 'code': code})
            print(f"    t={t:+d}: {code[:45]}")

        # Classify what this axis controls
        # Check which semantic categories correlate with this PC
        src_to_idx = {}
        for i, d in enumerate(dataset):
            if d['source'] not in src_to_idx:
                src_to_idx[d['source']] = i

        # Correlate PC score with function properties
        categories = {'arithmetic': [], 'comparison': [], 'string': [],
                     'list': [], 'boolean': [], 'unary': [], 'binary_op': []}
        for i, d in enumerate(dataset):
            src = d['source']
            if any(op in src for op in ['+', '-', '*', '/', '%', '**']):
                categories['arithmetic'].append(i)
            if any(op in src for op in ['>', '<', '==', '!=']):
                categories['comparison'].append(i)
            if any(op in src for op in ['.upper', '.lower', '.strip', 'len(']):
                categories['string'].append(i)
            if any(op in src for op in ['sorted', 'reversed', 'max(', 'min(']):
                categories['list'].append(i)
            # Arity
            if 'x, y' in src or 'a, b' in src or 'm, n' in src or 'p, q' in src:
                categories['binary_op'].append(i)
            else:
                categories['unary'].append(i)

        cat_means = {}
        for cat, idxs in categories.items():
            if len(idxs) > 5:
                cat_means[cat] = float(np.mean(z_pca[idxs, pc]))

        # Sort by mean PC score
        sorted_cats = sorted(cat_means.items(), key=lambda x: x[1])
        axis_desc = f"PC{pc}: "
        if sorted_cats:
            axis_desc += f"'{sorted_cats[0][0]}' (low={sorted_cats[0][1]:.2f}) <-> "
            axis_desc += f"'{sorted_cats[-1][0]}' (high={sorted_cats[-1][1]:.2f})"
        print(f"    Interpretation: {axis_desc}")

        elements.append({
            'pc': pc, 'variance': float(pca.explained_variance_ratio_[pc]),
            'codes': codes_along, 'category_means': cat_means,
            'interpretation': axis_desc,
        })

    # === 5D coordinates of known functions ===
    print("\n--- 5D Coordinates of Functions ---")
    example_funcs = [
        "def f(x, y): return x + y",
        "def f(x, y): return x > y",
        "def f(s): return s.upper()",
        "def f(x): return abs(x)",
        "def f(x): return -x",
    ]
    coords = []
    for src in example_funcs:
        if src in src_to_idx:
            idx = src_to_idx[src]
            coord = z_pca[idx, :5].tolist()
            print(f"  {src[:30]}: [{', '.join(f'{c:.2f}' for c in coord)}]")
            coords.append({'src': src, 'coords': coord})

    elapsed = time.time() - t0
    results = {
        'phase': 40, 'name': '5-Dimensional Theory of Code',
        'variance_ratios': pca.explained_variance_ratio_[:10].tolist(),
        'elements': elements,
        'coordinates': coords,
        'elapsed': elapsed, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase40_five_elements.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Variance explained
    axes[0,0].bar(range(10), pca.explained_variance_ratio_[:10]*100,
                 color='#E91E63', edgecolor='black')
    axes[0,0].set_xlabel('PC'); axes[0,0].set_ylabel('% Variance')
    axes[0,0].set_title('PCA Spectrum\n(how many dims matter?)', fontweight='bold')

    # 2-6. Each of the 5 elements as a scatter of categories
    for pc in range(5):
        ax = axes[(pc+1)//3, (pc+1)%3]
        el = elements[pc]
        cats = list(el['category_means'].keys())
        vals = [el['category_means'][c] for c in cats]
        colors_map = {'arithmetic':'#F44336','comparison':'#2196F3','string':'#4CAF50',
                     'list':'#FF9800','boolean':'#9C27B0','unary':'#607D8B',
                     'binary_op':'#795548'}
        colors = [colors_map.get(c, '#999') for c in cats]
        ax.barh(cats, vals, color=colors, edgecolor='black')
        ax.axvline(0, color='black', lw=0.5)
        ax.set_title(f'PC{pc} ({el["variance"]*100:.1f}%)', fontweight='bold', fontsize=10)

    plt.suptitle('Phase 40: The 5-Dimensional Theory of Code\n'
                 'The fundamental axes of software meaning',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase40_five_elements.png'), dpi=150)
    plt.close()
    print(f"\nPhase 40 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
