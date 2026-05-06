"""
Phase 77: The Feature Attribution Map
=======================================
Which AST features contribute most to each PC dimension?

This answers: WHAT does each of the 5 dimensions MEAN?
Is PC1 "arithmetic complexity"? Is PC2 "comparison vs computation"?

Use PCA loadings to decode the meaning of each dimension.
"""
import os, json, time, ast
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')


def main():
    print("=" * 60)
    print("Phase 77: The Feature Attribution Map")
    print("What does each of the 5 dimensions mean?")
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
    z_5d = pca.transform(z_ast)[:, :5]

    # PCA loadings (components_) tell us which features matter for each PC
    loadings = pca.components_[:5]  # 5 x 64 matrix
    print(f"\n  Loadings shape: {loadings.shape}")

    # Feature names (from the AST encoder)
    feature_names = [
        'FunctionDef', 'Return', 'BinOp', 'UnaryOp', 'Call', 'Name',
        'Constant', 'Compare', 'Add', 'Sub', 'Mult', 'Div', 'Mod', 'Pow',
        'USub', 'Gt', 'Lt', 'Eq', 'NotEq', 'BoolOp', 'IfExp', 'Attribute',
        'Subscript', 'arguments', 'arg', 'Store', 'Load', 'Expr', 'Module',
        'And', 'Or', 'Not', 'GtE', 'LtE', 'AugAssign', 'Assign', 'For',
        'While', 'If', 'Break', 'Continue', 'Pass', 'Lambda', 'List',
        'Tuple', 'Dict', 'Set', 'Slice', 'Index', 'Starred',
        'FormattedValue', 'JoinedStr', 'Num', 'Str', 'NameConstant',
        'Bytes', 'Ellipsis', 'In', 'NotIn', 'Is', 'IsNot', 'BitOr',
        'BitAnd', 'BitXor'
    ]
    # Pad if needed
    while len(feature_names) < loadings.shape[1]:
        feature_names.append(f'feat_{len(feature_names)}')

    # Analyze each PC
    pc_meanings = []
    for pc in range(5):
        loading = loadings[pc]
        # Top positive contributors
        top_pos_idx = np.argsort(loading)[-5:][::-1]
        top_neg_idx = np.argsort(loading)[:5]

        print(f"\n  === PC{pc+1} (var={pca.explained_variance_ratio_[pc]*100:.1f}%) ===")
        print(f"  Positive direction (high PC{pc+1}):")
        pos_feats = []
        for idx in top_pos_idx:
            if abs(loading[idx]) > 0.01:
                name = feature_names[idx] if idx < len(feature_names) else f'feat_{idx}'
                print(f"    +{loading[idx]:.4f}: {name}")
                pos_feats.append(name)

        print(f"  Negative direction (low PC{pc+1}):")
        neg_feats = []
        for idx in top_neg_idx:
            if abs(loading[idx]) > 0.01:
                name = feature_names[idx] if idx < len(feature_names) else f'feat_{idx}'
                print(f"    {loading[idx]:.4f}: {name}")
                neg_feats.append(name)

        # Interpret the meaning
        if 'BinOp' in pos_feats or 'Add' in pos_feats or 'Mult' in pos_feats:
            meaning = 'Arithmetic Complexity'
        elif 'Compare' in pos_feats or 'Gt' in pos_feats or 'Lt' in pos_feats:
            meaning = 'Comparison Operations'
        elif 'Call' in pos_feats:
            meaning = 'Function Call Depth'
        elif 'Name' in pos_feats or 'Load' in pos_feats:
            meaning = 'Variable Usage'
        elif 'UnaryOp' in pos_feats or 'USub' in pos_feats:
            meaning = 'Unary Operations'
        else:
            meaning = f'Feature Mix ({", ".join(pos_feats[:2])})'

        print(f"  Interpretation: {meaning}")
        pc_meanings.append({
            'pc': pc + 1, 'meaning': meaning,
            'var_explained': float(pca.explained_variance_ratio_[pc] * 100),
            'top_positive': [(feature_names[i] if i < len(feature_names) else f'feat_{i}',
                            float(loading[i])) for i in top_pos_idx],
            'top_negative': [(feature_names[i] if i < len(feature_names) else f'feat_{i}',
                            float(loading[i])) for i in top_neg_idx],
        })

    # Correlate dimensions with semantic categories
    print("\n--- Semantic Category Correlations ---")
    src_to_z5 = {}
    for i, src in enumerate(sources):
        if src not in src_to_z5:
            src_to_z5[src] = z_5d[i]

    categories = {
        'arithmetic': ['+', '-', '*', '/'],
        'comparison': ['>', '<', '>=', '<='],
        'equality': ['==', '!='],
        'unary': ['abs(', '-x', '-a'],
        'logical': [' and ', ' or ', 'not '],
    }

    cat_correlations = {}
    for cat_name, keywords in categories.items():
        labels = []
        z_vals = []
        for src, z in src_to_z5.items():
            has_cat = any(kw in src for kw in keywords)
            labels.append(1 if has_cat else 0)
            z_vals.append(z)

        labels = np.array(labels)
        z_vals = np.array(z_vals)

        if labels.sum() > 0 and labels.sum() < len(labels):
            corrs = []
            for dim in range(5):
                corr = float(np.corrcoef(labels, z_vals[:, dim])[0, 1])
                corrs.append(corr)
            best_dim = np.argmax(np.abs(corrs))
            cat_correlations[cat_name] = {
                'correlations': corrs,
                'best_dim': int(best_dim + 1),
                'best_corr': float(corrs[best_dim]),
            }
            print(f"  {cat_name:12s}: best=PC{best_dim+1} "
                  f"(corr={corrs[best_dim]:+.4f})")

    elapsed = time.time() - t0
    results = {
        'phase': 77, 'name': 'The Feature Attribution Map',
        'pc_meanings': pc_meanings,
        'category_correlations': cat_correlations,
        'elapsed': elapsed, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase77_attribution.json'), 'w',
              encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    # Figure
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Top features per PC (heatmap style)
    heatmap = np.abs(loadings[:5, :20])  # First 20 features
    im = axes[0].imshow(heatmap, aspect='auto', cmap='YlOrRd')
    axes[0].set_yticks(range(5))
    axes[0].set_yticklabels([f'PC{i+1}' for i in range(5)])
    axes[0].set_xticks(range(min(20, len(feature_names))))
    axes[0].set_xticklabels(feature_names[:20], rotation=90, fontsize=6)
    axes[0].set_title('Feature Attribution Heatmap', fontweight='bold')
    plt.colorbar(im, ax=axes[0], shrink=0.8)

    # 2. Category correlations
    if cat_correlations:
        cat_names = list(cat_correlations.keys())
        best_corrs = [abs(cat_correlations[c]['best_corr']) for c in cat_names]
        best_dims = [f'PC{cat_correlations[c]["best_dim"]}' for c in cat_names]
        bars = axes[1].barh(cat_names, best_corrs, color='#9C27B0', edgecolor='black')
        for i, (bar, dim) in enumerate(zip(bars, best_dims)):
            axes[1].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                        dim, va='center', fontsize=9)
        axes[1].set_xlabel('|Correlation|')
        axes[1].set_title('Category -> Best PC\nCorrelation', fontweight='bold')

    # 3. PC meaning summary
    meaning_text = "THE 5 DIMENSIONS OF CODE\n\n"
    for m in pc_meanings:
        meaning_text += f"PC{m['pc']}: {m['meaning']}\n"
        meaning_text += f"     ({m['var_explained']:.1f}%)\n"
    axes[2].text(0.5, 0.5, meaning_text, ha='center', va='center',
                fontsize=11, fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#E3F2FD', alpha=0.8),
                transform=axes[2].transAxes)
    axes[2].axis('off')

    plt.suptitle('Phase 77: The Feature Attribution Map\n'
                 'Decoding the Meaning of 5 Dimensions',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase77_attribution.png'), dpi=150)
    plt.close()
    print(f"\nPhase 77 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
