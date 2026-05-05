"""
Phase 7: Mechanistic Compiler Anatomy
=======================================
What do the top 4 SVD components of W_compile actually mean?
Use linear probes to decode semantic properties, then align with SVD axes.
"""
import os, json, time, ast
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')


def extract_code_properties(source):
    """Extract semantic properties from source code."""
    props = {}
    try:
        tree = ast.parse(source)
        src = source.lower()
        props['has_loop'] = 'for ' in src or 'while ' in src
        props['has_conditional'] = ' if ' in src
        props['has_comparison'] = any(op in src for op in ['>', '<', '==', '!=', '>=', '<='])
        props['is_arithmetic'] = any(op in src for op in [' + ', ' - ', ' * ', ' / ', ' ** ', ' // ', ' % '])
        props['is_string_op'] = any(op in src for op in ['.upper', '.lower', '.strip', '.title',
                                                          '.swapcase', '[::-1]', '.split'])
        props['is_list_op'] = any(op in src for op in ['sum(', 'len(', 'max(', 'min(',
                                                        'sorted(', 'reversed('])
        props['is_boolean'] = ' and ' in src or ' or ' in src or 'not ' in src
        props['n_args'] = src.count(',') + 1 if 'def f(' in src else 1
        props['has_abs'] = 'abs(' in src
        props['is_unary'] = src.count(',') == 0 and 'def f(' in src
        # Bytecode complexity proxy
        props['code_length'] = len(src)
    except:
        pass
    return props


def main():
    print("=" * 60)
    print("Phase 7: Mechanistic Compiler Anatomy")
    print("=" * 60)
    t0 = time.time()

    # Load v2 data
    ds_file = os.path.join(DATA_DIR, 'rosetta_dataset_v2.json')
    if not os.path.exists(ds_file):
        ds_file = os.path.join(DATA_DIR, 'rosetta_dataset.json')
    with open(ds_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)['dataset']

    lat_file = os.path.join(DATA_DIR, 'rosetta_latents_v2.npz')
    if not os.path.exists(lat_file):
        lat_file = os.path.join(DATA_DIR, 'rosetta_latents.npz')
    latents = np.load(lat_file)
    z_ast = latents['ast']
    z_bc = latents['bc']
    N, D = z_ast.shape
    print(f"Loaded {N} vectors of dim {D}")

    # === 1. Re-compute W_compile and SVD ===
    print("\n--- SVD of W_compile ---")
    from sklearn.linear_model import Ridge
    n_train = int(N * 0.8)
    perm = np.random.RandomState(42).permutation(N)
    train_idx = perm[:n_train]

    reg = Ridge(alpha=1.0).fit(z_ast[train_idx], z_bc[train_idx])
    W = reg.coef_  # (D, D)
    U, S, Vt = np.linalg.svd(W)

    print(f"  Top 10 singular values: {S[:10].round(3)}")
    cumulative_energy = np.cumsum(S**2) / np.sum(S**2)
    for pct in [0.5, 0.8, 0.9, 0.95, 0.99]:
        n_comp = np.searchsorted(cumulative_energy, pct) + 1
        print(f"  {pct:.0%} energy in {n_comp} components")

    # Top 4 right singular vectors (the "compiler axes")
    compiler_axes = Vt[:4]  # (4, D) - these are the 4 main "directions" of compilation

    # === 2. Extract semantic properties for all samples ===
    print("\n--- Extracting code properties ---")
    property_names = ['has_loop', 'has_conditional', 'has_comparison',
                      'is_arithmetic', 'is_string_op', 'is_list_op',
                      'is_boolean', 'is_unary', 'has_abs']
    prop_matrix = np.zeros((N, len(property_names)), dtype=np.float32)

    for i, d in enumerate(dataset):
        props = extract_code_properties(d['source'])
        for j, pname in enumerate(property_names):
            prop_matrix[i, j] = float(props.get(pname, False))

    # Show property distribution
    print("  Property distribution:")
    for j, pname in enumerate(property_names):
        count = int(prop_matrix[:, j].sum())
        print(f"    {pname}: {count}/{N} ({count/N:.1%})")

    # === 3. Train linear probes on AST latent vectors ===
    print("\n--- Training linear probes ---")
    test_idx = perm[n_train:]
    probe_results = {}
    probe_weights = {}

    for j, pname in enumerate(property_names):
        y = prop_matrix[:, j]
        if y.sum() < 10 or (N - y.sum()) < 10:
            print(f"  {pname}: SKIP (too few positives)")
            continue

        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(z_ast[train_idx], y[train_idx])
        y_pred = clf.predict(z_ast[test_idx])
        acc = accuracy_score(y[test_idx], y_pred)
        probe_results[pname] = float(acc)
        probe_weights[pname] = clf.coef_[0]  # (D,) weight vector
        print(f"  {pname}: accuracy={acc:.1%}")

    # === 4. Align compiler axes with probe directions ===
    print("\n--- Aligning compiler axes with semantic probes ---")
    alignment_matrix = np.zeros((4, len(probe_results)))
    probe_names = list(probe_results.keys())

    for i in range(4):
        for j, pname in enumerate(probe_names):
            # Cosine similarity between SVD axis and probe weight
            w_probe = probe_weights[pname]
            cos = float(np.dot(compiler_axes[i], w_probe) /
                       (np.linalg.norm(compiler_axes[i]) *
                        np.linalg.norm(w_probe) + 1e-8))
            alignment_matrix[i, j] = cos

    print("\n  Alignment Matrix (SVD axis vs Semantic Probe):")
    print(f"  {'':20s}", end='')
    for pname in probe_names:
        print(f" {pname[:12]:>12s}", end='')
    print()
    for i in range(4):
        print(f"  SVD axis {i} (s={S[i]:.2f})", end='')
        for j in range(len(probe_names)):
            val = alignment_matrix[i, j]
            print(f" {val:>12.3f}", end='')
        print()

    # Identify strongest alignment for each axis
    print("\n  Strongest alignments:")
    for i in range(4):
        best_j = np.argmax(np.abs(alignment_matrix[i]))
        best_cos = alignment_matrix[i, best_j]
        print(f"    SVD axis {i} (s={S[i]:.2f}) <-> {probe_names[best_j]} "
              f"(cos={best_cos:.3f})")

    # === 5. Project data onto compiler axes and visualize ===
    projections = z_ast @ compiler_axes.T  # (N, 4)

    elapsed = time.time() - t0
    results = {
        'phase': 7, 'name': 'Mechanistic Compiler Anatomy',
        'singular_values': S[:10].tolist(),
        'energy_90pct': int(np.searchsorted(cumulative_energy, 0.9) + 1),
        'probe_accuracies': probe_results,
        'alignment_matrix': alignment_matrix.tolist(),
        'axis_interpretations': [],
        'elapsed': elapsed,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    for i in range(4):
        best_j = int(np.argmax(np.abs(alignment_matrix[i])))
        results['axis_interpretations'].append({
            'axis': i, 'singular_value': float(S[i]),
            'best_probe': probe_names[best_j],
            'cosine': float(alignment_matrix[i, best_j]),
        })

    with open(os.path.join(RESULTS_DIR, 'phase7_compiler_anatomy.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Figure
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Probe accuracies
    pnames_short = [p.replace('has_','').replace('is_','')[:10] for p in probe_names]
    paccs = [probe_results[p] for p in probe_names]
    bars = axes[0].barh(pnames_short, paccs, color='#4CAF50', edgecolor='black')
    for b, v in zip(bars, paccs):
        axes[0].text(v+0.01, b.get_y()+b.get_height()/2, f'{v:.0%}',
                     va='center', fontweight='bold')
    axes[0].set_xlabel('Accuracy')
    axes[0].set_title('Linear Probes on AST Latent', fontweight='bold')
    axes[0].set_xlim(0, 1.1)

    # Alignment heatmap
    im = axes[1].imshow(np.abs(alignment_matrix), cmap='YlOrRd', aspect='auto',
                        vmin=0, vmax=0.5)
    axes[1].set_xticks(range(len(probe_names)))
    axes[1].set_xticklabels(pnames_short, rotation=45, ha='right', fontsize=9)
    axes[1].set_yticks(range(4))
    axes[1].set_yticklabels([f'Axis {i}\n(s={S[i]:.2f})' for i in range(4)])
    axes[1].set_title('|Cosine| SVD Axis vs Probe', fontweight='bold')
    plt.colorbar(im, ax=axes[1], shrink=0.8)

    # Scatter: projection onto axis 0 vs axis 1, colored by property
    # Color by most discriminative property
    best_prop = max(probe_results, key=probe_results.get)
    best_j = list(probe_results.keys()).index(best_prop)
    colors = prop_matrix[:, property_names.index(best_prop)]
    axes[2].scatter(projections[:, 0], projections[:, 1], c=colors,
                    cmap='coolwarm', alpha=0.5, s=10)
    axes[2].set_xlabel(f'Compiler Axis 0 (s={S[0]:.2f})')
    axes[2].set_ylabel(f'Compiler Axis 1 (s={S[1]:.2f})')
    axes[2].set_title(f'Projection (color={best_prop})', fontweight='bold')

    plt.suptitle('Phase 7: Mechanistic Compiler Anatomy',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase7_compiler_anatomy.png'), dpi=150)
    plt.close()

    print(f"\nPhase 7 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
