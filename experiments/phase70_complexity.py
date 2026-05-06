"""
Phase 70: The Complexity Compass
==================================
Is computational complexity encoded in 5D space?

Map O(1), O(n), O(n^2), O(n*log(n)) algorithms to
the latent space and check if there's a "complexity axis."

If complexity has a direction, we can PREDICT
a function's runtime from its embedding alone!
"""
import os, json, time, sys, ast, inspect
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')


def main():
    print("=" * 60)
    print("Phase 70: The Complexity Compass")
    print("Is Big-O encoded in the 5D manifold?")
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

    src_to_z = {}
    for i, src in enumerate(sources):
        if src not in src_to_z:
            src_to_z[src] = z_5d[i]

    # Classify functions by computational complexity
    print("\n--- Classifying Functions by Complexity ---")

    # Heuristic complexity classification based on AST analysis
    def classify_complexity(src):
        """Estimate Big-O from source code structure."""
        try:
            tree = ast.parse(src)
            has_loop = any(isinstance(n, (ast.For, ast.While))
                         for n in ast.walk(tree))
            has_nested = False
            for node in ast.walk(tree):
                if isinstance(node, (ast.For, ast.While)):
                    for child in ast.walk(node):
                        if child is not node and isinstance(child, (ast.For, ast.While)):
                            has_nested = True

            has_recursion = False
            # Check for function calls to the same function name
            func_names = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_names.add(node.name)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id in func_names:
                        has_recursion = True

            n_ops = sum(1 for n in ast.walk(tree)
                       if isinstance(n, (ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp)))

            if has_nested:
                return 'O(n^2)'
            elif has_loop or has_recursion:
                return 'O(n)'
            elif n_ops > 3:
                return 'O(k)'  # constant but complex
            else:
                return 'O(1)'
        except Exception:
            return 'O(?)'

    # Also measure ACTUAL runtime
    def measure_runtime(src, n_trials=5):
        """Measure actual execution time."""
        try:
            ns = {}
            exec(compile(src, '<string>', 'exec'), ns)
            fn = [v for k, v in ns.items()
                  if callable(v) and not k.startswith('_')][0]
            sig = inspect.signature(fn)
            n_params = len(sig.parameters)

            if n_params == 1:
                test_args = [(42,)]
            elif n_params == 2:
                test_args = [(42, 7)]
            else:
                return None

            # Warmup
            for args in test_args:
                try:
                    fn(*args)
                except Exception:
                    return None

            # Measure
            import timeit
            times = []
            for _ in range(n_trials):
                start = time.perf_counter()
                for args in test_args:
                    for _ in range(1000):
                        fn(*args)
                elapsed = time.perf_counter() - start
                times.append(elapsed)

            return float(np.median(times))
        except Exception:
            return None

    # Classify and measure all functions
    complexity_data = {}
    for src, z5 in src_to_z.items():
        cx = classify_complexity(src)
        rt = measure_runtime(src)
        if rt is not None:
            if cx not in complexity_data:
                complexity_data[cx] = []
            complexity_data[cx].append({
                'src': src, 'z_5d': z5, 'runtime': rt, 'complexity': cx,
            })

    print(f"\n  Complexity distribution:")
    for cx in sorted(complexity_data.keys()):
        items = complexity_data[cx]
        avg_rt = np.mean([d['runtime'] for d in items])
        print(f"    {cx:6s}: {len(items):3d} functions, avg runtime: {avg_rt*1000:.3f}ms")

    # Analyze: Is there a "complexity direction" in 5D?
    print("\n--- Searching for Complexity Direction ---")

    # Compute centroids per complexity class
    centroids = {}
    for cx, items in complexity_data.items():
        z_arr = np.array([d['z_5d'] for d in items])
        centroids[cx] = z_arr.mean(axis=0)

    print(f"\n  Centroids in 5D:")
    for cx in sorted(centroids.keys()):
        c = centroids[cx]
        print(f"    {cx:6s}: [{', '.join(f'{x:.3f}' for x in c)}]")

    # Compute inter-class distances
    print(f"\n  Inter-class distances:")
    cx_list = sorted(centroids.keys())
    for i in range(len(cx_list)):
        for j in range(i+1, len(cx_list)):
            d = np.linalg.norm(centroids[cx_list[i]] - centroids[cx_list[j]])
            print(f"    {cx_list[i]:6s} <-> {cx_list[j]:6s}: {d:.4f}")

    # Train a classifier: can we predict complexity from 5D?
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    all_items = []
    for cx, items in complexity_data.items():
        all_items.extend(items)

    X = np.array([d['z_5d'] for d in all_items])
    y = np.array([d['complexity'] for d in all_items])

    if len(set(y)) > 1:
        clf = LogisticRegression(max_iter=1000)
        scores = cross_val_score(clf, X, y, cv=min(5, len(X)//2), scoring='accuracy')
        cx_accuracy = float(scores.mean())
        print(f"\n  Complexity prediction accuracy (5-fold CV): {cx_accuracy*100:.1f}%")
    else:
        cx_accuracy = 0.0
        print("\n  Only one complexity class found")

    # Runtime correlation with 5D coordinates
    runtimes = np.array([d['runtime'] for d in all_items])
    correlations = []
    for dim in range(5):
        vals = X[:, dim]
        corr = float(np.corrcoef(vals, runtimes)[0, 1])
        correlations.append(corr)
        print(f"  PC{dim+1} <-> runtime correlation: {corr:+.4f}")

    best_dim = np.argmax(np.abs(correlations))
    print(f"\n  Best complexity axis: PC{best_dim+1} "
          f"(corr={correlations[best_dim]:+.4f})")

    elapsed = time.time() - t0
    results = {
        'phase': 70, 'name': 'The Complexity Compass',
        'complexity_distribution': {cx: len(items) for cx, items in complexity_data.items()},
        'centroids': {cx: c.tolist() for cx, c in centroids.items()},
        'cx_accuracy': cx_accuracy,
        'pc_runtime_correlations': correlations,
        'best_complexity_axis': int(best_dim + 1),
        'elapsed': elapsed, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase70_complexity.json'), 'w',
              encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    # Figure
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Functions per complexity class
    cx_names = sorted(complexity_data.keys())
    cx_counts = [len(complexity_data[cx]) for cx in cx_names]
    axes[0].bar(cx_names, cx_counts, color='#2196F3', edgecolor='black')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Complexity Distribution', fontweight='bold')

    # 2. 5D projection colored by complexity
    colors_map = {'O(1)': '#4CAF50', 'O(k)': '#FF9800', 'O(n)': '#F44336', 'O(n^2)': '#9C27B0'}
    for cx in cx_names:
        items = complexity_data[cx]
        z_arr = np.array([d['z_5d'] for d in items])
        c = colors_map.get(cx, '#999999')
        axes[1].scatter(z_arr[:, 0], z_arr[:, 1], c=c, label=cx, s=20, alpha=0.6)
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')
    axes[1].set_title('Complexity in 5D Space\n(PC1 vs PC2)', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3. Runtime correlations
    dims = [f'PC{i+1}' for i in range(5)]
    colors_c = ['#F44336' if abs(c) > 0.3 else '#2196F3' for c in correlations]
    axes[2].bar(dims, [abs(c) for c in correlations], color=colors_c, edgecolor='black')
    axes[2].set_ylabel('|Correlation with Runtime|')
    axes[2].set_title(f'Complexity Axis\n(Best: PC{best_dim+1})', fontweight='bold')

    plt.suptitle('Phase 70: The Complexity Compass\n'
                 'Is Big-O Encoded in the 5D Manifold?',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase70_complexity.png'), dpi=150)
    plt.close()
    print(f"\nPhase 70 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
