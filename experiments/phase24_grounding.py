"""
Phase 24: Semantic-Binary Grounding
======================================
Visualize HOW natural language maps to specific bytecode instructions.
Which NL concept activates which binary instruction?
The micro-level "Rosetta Stone" decipherment.
"""
import os, json, time
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')


def main():
    print("=" * 60)
    print("Phase 24: Semantic-Binary Grounding")
    print("How does language become binary?")
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

    # Load the NL-to-Binary translation matrix
    W_file = os.path.join(DATA_DIR, 'W_compile.npy')
    if os.path.exists(W_file):
        W_compile = np.load(W_file)
    else:
        # Fit it
        from sklearn.linear_model import Ridge
        reg = Ridge(alpha=1.0).fit(z_ast, z_bc)
        W_compile = reg.coef_.T  # (64, 64)

    # Also fit NL->Binary
    from sklearn.linear_model import Ridge
    reg_nl = Ridge(alpha=1.0).fit(z_nl, z_bc)
    W_nl_bin = reg_nl.coef_.T  # (64, 64)

    # === Analysis 1: Which NL dimensions drive which Binary dimensions? ===
    print("\n--- NL-to-Binary Grounding Matrix ---")

    # Group functions by semantic category
    categories = {}
    for i, d in enumerate(dataset):
        src = d['source']
        if any(op in src for op in ['+', '- ', 'abs(', '* ', '**', '% ']):
            cat = 'Arithmetic'
        elif any(op in src for op in ['>', '<', '==', '!=', '>=']):
            cat = 'Comparison'
        elif any(op in src for op in ['.upper', '.lower', '.strip', 'len(']):
            cat = 'String'
        elif any(op in src for op in ['sorted', 'reversed', 'sum(', 'max(', 'min(']):
            cat = 'List'
        elif any(op in src for op in [' and ', ' or ', 'not ']):
            cat = 'Boolean'
        else:
            cat = 'Other'
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(i)

    # Compute category centroids in NL and Binary space
    cat_names = sorted(categories.keys())
    nl_centroids = {}
    bc_centroids = {}
    for cat in cat_names:
        idxs = categories[cat]
        nl_centroids[cat] = z_nl[idxs].mean(axis=0)
        bc_centroids[cat] = z_bc[idxs].mean(axis=0)

    # === Analysis 2: Per-category NL->Binary activation patterns ===
    print("\n--- Category-specific activation patterns ---")
    activation_matrix = np.zeros((len(cat_names), 64))
    for ci, cat in enumerate(cat_names):
        # Project NL centroid through W_nl_bin
        nl_c = nl_centroids[cat]
        bc_pred = W_nl_bin @ nl_c
        activation_matrix[ci] = bc_pred

    # Normalize per row
    act_norm = activation_matrix / (np.abs(activation_matrix).max(axis=1, keepdims=True) + 1e-8)

    # === Analysis 3: Differential grounding ===
    # What changes in binary when NL shifts from "add" to "subtract"?
    print("\n--- Differential Grounding ---")
    diff_pairs = [
        ('Arithmetic', 'Comparison', 'Arith vs Compare'),
        ('String', 'List', 'String vs List'),
        ('Boolean', 'Arithmetic', 'Bool vs Arith'),
    ]

    diff_results = []
    for cat_a, cat_b, desc in diff_pairs:
        if cat_a not in nl_centroids or cat_b not in nl_centroids:
            continue
        nl_diff = nl_centroids[cat_a] - nl_centroids[cat_b]
        bc_diff = W_nl_bin @ nl_diff
        # Which binary dimensions change most?
        top_dims = np.argsort(np.abs(bc_diff))[::-1][:5]
        print(f"\n  {desc}:")
        print(f"    Top binary dims affected: {top_dims.tolist()}")
        print(f"    Magnitudes: {[f'{bc_diff[d]:.3f}' for d in top_dims]}")
        diff_results.append({
            'desc': desc, 'top_dims': top_dims.tolist(),
            'magnitudes': [float(bc_diff[d]) for d in top_dims],
        })

    # === Analysis 4: SVD of NL-to-Binary mapping ===
    U, S, Vt = np.linalg.svd(W_nl_bin, full_matrices=False)
    print(f"\n--- SVD of NL->Binary matrix ---")
    print(f"  Top 10 singular values: {[f'{s:.3f}' for s in S[:10]]}")
    energy = np.cumsum(S**2) / np.sum(S**2)
    n90 = int(np.searchsorted(energy, 0.9)) + 1
    n95 = int(np.searchsorted(energy, 0.95)) + 1
    print(f"  90% energy in {n90} dims, 95% in {n95} dims")

    # === Analysis 5: Per-word grounding ===
    # Find which NL words correlate with specific binary patterns
    print("\n--- Word-Level Grounding ---")
    word_groups = {
        'add/plus': [], 'subtract/minus': [], 'multiply/times': [],
        'compare/greater': [], 'string/upper': [], 'list/sort': [],
    }
    for i, d in enumerate(dataset):
        nl = d['nl'].lower()
        if 'add' in nl or 'plus' in nl or 'sum' in nl:
            word_groups['add/plus'].append(i)
        if 'subtract' in nl or 'minus' in nl:
            word_groups['subtract/minus'].append(i)
        if 'multiply' in nl or 'times' in nl or 'product' in nl:
            word_groups['multiply/times'].append(i)
        if 'greater' in nl or 'compare' in nl or 'less' in nl:
            word_groups['compare/greater'].append(i)
        if 'upper' in nl or 'lower' in nl or 'string' in nl:
            word_groups['string/upper'].append(i)
        if 'sort' in nl or 'reverse' in nl or 'list' in nl:
            word_groups['list/sort'].append(i)

    word_activations = {}
    for word, idxs in word_groups.items():
        if len(idxs) < 3:
            continue
        mean_nl = z_nl[idxs].mean(axis=0)
        mean_bc = z_bc[idxs].mean(axis=0)
        predicted_bc = W_nl_bin @ mean_nl
        cos = float(np.dot(predicted_bc, mean_bc) /
                    (np.linalg.norm(predicted_bc) * np.linalg.norm(mean_bc) + 1e-8))
        top3 = np.argsort(np.abs(predicted_bc))[::-1][:3]
        print(f"  '{word}': cos={cos:.3f}, top binary dims={top3.tolist()}")
        word_activations[word] = {
            'cos': cos, 'top_dims': top3.tolist(), 'n_samples': len(idxs),
        }

    elapsed = time.time() - t0
    results = {
        'phase': 24, 'name': 'Semantic-Binary Grounding',
        'categories': {k: len(v) for k, v in categories.items()},
        'svd_90pct': n90, 'svd_95pct': n95,
        'top_singular_values': [float(s) for s in S[:10]],
        'differential': diff_results,
        'word_activations': word_activations,
        'elapsed': elapsed, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase24_grounding.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    # Visualization
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Category activation heatmap
    im = axes[0,0].imshow(act_norm[:, :20], aspect='auto', cmap='RdBu_r',
                          vmin=-1, vmax=1)
    axes[0,0].set_yticks(range(len(cat_names)))
    axes[0,0].set_yticklabels(cat_names, fontsize=10)
    axes[0,0].set_xlabel('Binary Dimension (top 20)')
    axes[0,0].set_title('NL Category -> Binary Activation', fontweight='bold')
    plt.colorbar(im, ax=axes[0,0], shrink=0.8)

    # 2. SVD spectrum
    axes[0,1].bar(range(len(S)), S, color='#E91E63', alpha=0.8)
    axes[0,1].axvline(n90-1, color='blue', ls='--', label=f'90% energy ({n90} dims)')
    axes[0,1].set_xlabel('Singular Value Index')
    axes[0,1].set_ylabel('Magnitude')
    axes[0,1].set_title('SVD of NL->Binary Matrix', fontweight='bold')
    axes[0,1].legend()

    # 3. Differential grounding
    if diff_results:
        n_diffs = len(diff_results)
        for di, dr in enumerate(diff_results):
            axes[1,0].barh([f"dim{d}" for d in dr['top_dims']],
                          dr['magnitudes'],
                          alpha=0.7, label=dr['desc'])
        axes[1,0].set_xlabel('Activation Difference')
        axes[1,0].set_title('Differential Grounding\n(which binary dims change?)',
                           fontweight='bold')
        axes[1,0].legend(fontsize=8)

    # 4. Word-level grounding
    if word_activations:
        words = list(word_activations.keys())
        cosines = [word_activations[w]['cos'] for w in words]
        colors = ['#4CAF50' if c > 0.5 else '#FF9800' if c > 0 else '#F44336' for c in cosines]
        axes[1,1].barh(words, cosines, color=colors, edgecolor='black')
        axes[1,1].set_xlabel('Cosine Similarity (predicted vs actual)')
        axes[1,1].set_title('Word-Level Grounding Accuracy', fontweight='bold')
        axes[1,1].axvline(0, color='black', lw=0.5)

    plt.suptitle('Phase 24: Semantic-Binary Grounding\n'
                 'How Natural Language Maps to Binary Instructions',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase24_grounding.png'), dpi=150)
    plt.close()
    print(f"\nPhase 24 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
