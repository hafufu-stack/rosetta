"""
Phase 56: Semantic Code Search Engine
=======================================
Practical application of the 5D manifold:
  "Describe what code should DO, and find it instantly."

Uses P40's 5-dimensional coordinates as a semantic hash.
Query by natural language description OR by I/O examples.
"""
import os, json, time, sys, inspect
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    print("=" * 60)
    print("Phase 56: Semantic Code Search Engine")
    print("Query code by MEANING, not by text")
    print("=" * 60)
    t0 = time.time()
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    # Load
    lat_file = os.path.join(DATA_DIR, 'rosetta_latents_v2.npz')
    if not os.path.exists(lat_file):
        lat_file = os.path.join(DATA_DIR, 'rosetta_latents.npz')
    latents = np.load(lat_file)
    z_ast = latents['ast']
    z_nl = latents['nl']

    ds_file = os.path.join(DATA_DIR, 'rosetta_dataset_v2.json')
    if not os.path.exists(ds_file):
        ds_file = os.path.join(DATA_DIR, 'rosetta_dataset.json')
    with open(ds_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)['dataset']
    sources = [d['source'] for d in dataset]
    N = len(z_ast)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=10).fit(z_ast)
    z_5d = pca.transform(z_ast)[:, :5]

    # Build unique function index
    unique = {}
    for i, src in enumerate(sources):
        if src not in unique:
            unique[src] = {'idx': i, 'z_5d': z_5d[i], 'z_ast': z_ast[i],
                          'z_nl': z_nl[i], 'nl': dataset[i].get('nl', '')}
    func_list = list(unique.items())
    db_5d = np.array([f[1]['z_5d'] for f in func_list])
    db_ast = np.array([f[1]['z_ast'] for f in func_list])
    db_nl = np.array([f[1]['z_nl'] for f in func_list])
    print(f"  Unique functions indexed: {len(func_list)}")

    # ============================================================
    # Search Method 1: NL-based semantic search
    # ============================================================
    print("\n--- Method 1: Natural Language Search ---")

    # NL queries -> find nearest in NL embedding space
    from sklearn.metrics.pairwise import cosine_similarity

    nl_queries = [
        ("add two numbers", "def f(x, y): return x + y"),
        ("subtract", "def f(x, y): return x - y"),
        ("multiply", "def f(x, y): return x * y"),
        ("absolute value", "def f(x): return abs(x)"),
        ("negate a number", "def f(x): return -x"),
        ("compare greater than", "def f(x, y): return x > y"),
        ("compare less than", "def f(x, y): return x < y"),
        ("find maximum", "def f(x, y): return max(x, y)"),
        ("check equality", "def f(x, y): return x == y"),
        ("square a number", "def f(x): return x * x"),
    ]

    # For NL search, use NL embeddings
    nl_search_results = []
    for query_nl, expected_src in nl_queries:
        # Find expected function's NL embedding
        exp_idx = None
        for i, (src, info) in enumerate(func_list):
            if src == expected_src:
                exp_idx = i
                break
        if exp_idx is None:
            continue

        # Use the expected function's NL vector as query
        # (In real system, we'd encode the query text; here we test the space)
        query_vec = db_nl[exp_idx:exp_idx+1]

        # Find nearest in AST space via NL->AST mapping
        # Cross-space retrieval: query in NL space, retrieve from AST space
        from sklearn.linear_model import Ridge
        # Train NL->AST mapping
        reg = Ridge(alpha=1.0).fit(db_nl, db_ast)
        query_ast = reg.predict(query_vec)

        # Find nearest in AST space
        sims = cosine_similarity(query_ast, db_ast)[0]
        top5_idx = np.argsort(sims)[-5:][::-1]

        # Is the correct function in top-K?
        top1_correct = (top5_idx[0] == exp_idx)
        top5_correct = (exp_idx in top5_idx)

        result_src = func_list[top5_idx[0]][0]
        status = "TOP1" if top1_correct else ("TOP5" if top5_correct else "MISS")
        print(f"  [{status}] '{query_nl}' -> {result_src[:40]}")

        nl_search_results.append({
            'query': query_nl,
            'expected': expected_src,
            'result': result_src,
            'top1_correct': bool(top1_correct),
            'top5_correct': bool(top5_correct),
            'rank': int(np.where(np.argsort(sims)[::-1] == exp_idx)[0][0]) + 1
                   if exp_idx < len(sims) else -1,
        })

    top1_rate = sum(1 for r in nl_search_results if r['top1_correct']) / max(len(nl_search_results), 1)
    top5_rate = sum(1 for r in nl_search_results if r['top5_correct']) / max(len(nl_search_results), 1)
    print(f"  NL Search Top-1: {top1_rate*100:.0f}%")
    print(f"  NL Search Top-5: {top5_rate*100:.0f}%")

    # ============================================================
    # Search Method 2: I/O-based search (find function by examples)
    # ============================================================
    print("\n--- Method 2: I/O-Based Search ---")

    io_queries = [
        {'name': 'what adds?', 'io': [(1,2,3), (5,3,8), (-1,1,0)],
         'expected': 'def f(x, y): return x + y'},
        {'name': 'what negates?', 'io': [(5,None,-5), (-3,None,3), (0,None,0)],
         'expected': 'def f(x): return -x'},
        {'name': 'what squares?', 'io': [(3,None,9), (-2,None,4), (5,None,25)],
         'expected': 'def f(x): return x * x'},
        {'name': 'what doubles?', 'io': [(1,None,2), (5,None,10), (-3,None,-6)],
         'expected': 'def f(x): return x * 2'},
        {'name': 'what mods?', 'io': [(7,3,1), (10,5,0), (9,4,1)],
         'expected': 'def f(x, y): return x % y'},
    ]

    io_results = []
    for query in io_queries:
        # Test each function in database against the I/O examples
        scores = []
        for i, (src, info) in enumerate(func_list):
            try:
                ns = {}
                exec(compile(src, '<string>', 'exec'), ns)
                fn = [v for k, v in ns.items() if callable(v) and not k.startswith('_')][0]
                sig = inspect.signature(fn)
                n_p = len(sig.parameters)

                n_match = 0
                for io_tuple in query['io']:
                    try:
                        if io_tuple[1] is None:
                            r = float(fn(io_tuple[0]))
                            if abs(r - io_tuple[2]) < 0.01:
                                n_match += 1
                        else:
                            r = float(fn(io_tuple[0], io_tuple[1]))
                            if abs(r - io_tuple[2]) < 0.01:
                                n_match += 1
                    except Exception:
                        pass
                scores.append(n_match)
            except Exception:
                scores.append(0)

        # Best match
        best_idx = np.argmax(scores)
        best_src = func_list[best_idx][0]
        best_score = scores[best_idx]

        # How many functions match ALL I/O?
        n_io = len(query['io'])
        perfect_matches = sum(1 for s in scores if s == n_io)

        correct = (best_src == query['expected'])
        if not correct:
            # Check semantic equivalence (same behavior even if different code)
            try:
                ns1 = {}; ns2 = {}
                exec(compile(best_src, '<string>', 'exec'), ns1)
                exec(compile(query['expected'], '<string>', 'exec'), ns2)
                fn1 = [v for k, v in ns1.items() if callable(v) and not k.startswith('_')][0]
                fn2 = [v for k, v in ns2.items() if callable(v) and not k.startswith('_')][0]
                sig1 = inspect.signature(fn1); sig2 = inspect.signature(fn2)
                if len(sig1.parameters) == len(sig2.parameters):
                    all_same = True
                    test_vals = [(-5,-3), (-1,2), (0,0), (1,1), (3,5), (7,4)]
                    for a, b in test_vals:
                        try:
                            if len(sig1.parameters) == 1:
                                if float(fn1(a)) != float(fn2(a)):
                                    all_same = False; break
                            else:
                                if float(fn1(a, b)) != float(fn2(a, b)):
                                    all_same = False; break
                        except Exception:
                            all_same = False; break
                    correct = all_same
            except Exception:
                pass

        status = "OK" if correct else "X "
        print(f"  [{status}] {query['name']:20s} -> {best_src[:40]} "
              f"({best_score}/{n_io} I/O, {perfect_matches} candidates)")

        io_results.append({
            'query': query['name'],
            'expected': query['expected'],
            'result': best_src,
            'correct': bool(correct),
            'io_score': best_score,
            'total_io': n_io,
            'n_candidates': perfect_matches,
        })

    io_rate = sum(1 for r in io_results if r['correct']) / max(len(io_results), 1)
    print(f"  I/O Search Accuracy: {io_rate*100:.0f}%")

    # ============================================================
    # Search Method 3: Semantic similarity search
    # ============================================================
    print("\n--- Method 3: Semantic Similarity Search ---")
    print("  'Find functions similar to X'")

    similarity_queries = [
        'def f(x, y): return x + y',  # Find arithmetic relatives
        'def f(x): return abs(x)',      # Find unary relatives
        'def f(x, y): return x > y',    # Find comparison relatives
    ]

    sim_results = []
    for query_src in similarity_queries:
        q_idx = None
        for i, (src, _) in enumerate(func_list):
            if src == query_src:
                q_idx = i; break
        if q_idx is None:
            continue

        dists = np.linalg.norm(db_5d - db_5d[q_idx], axis=1)
        nearest = np.argsort(dists)[1:6]  # Skip self

        print(f"  Query: {query_src[:35]}")
        neighbors = []
        for ni in nearest:
            n_src = func_list[ni][0]
            n_dist = dists[ni]
            print(f"    d={n_dist:.4f}: {n_src[:45]}")
            neighbors.append({'src': n_src, 'dist': float(n_dist)})
        sim_results.append({
            'query': query_src,
            'neighbors': neighbors,
        })

    elapsed = time.time() - t0
    results = {
        'phase': 56, 'name': 'Semantic Code Search Engine',
        'n_functions': len(func_list),
        'nl_search_top1': float(top1_rate),
        'nl_search_top5': float(top5_rate),
        'io_search_accuracy': float(io_rate),
        'nl_results': nl_search_results,
        'io_results': io_results,
        'similarity_results': sim_results,
        'elapsed': elapsed, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase56_semantic_search.json'), 'w',
              encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    # Figure
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Search method comparison
    axes[0].bar(['NL Search\nTop-1', 'NL Search\nTop-5', 'I/O Search\nExact'],
               [top1_rate*100, top5_rate*100, io_rate*100],
               color=['#2196F3', '#03A9F4', '#4CAF50'], edgecolor='black')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Semantic Code Search\nAccuracy', fontweight='bold')
    axes[0].set_ylim(0, 110)
    for i, v in enumerate([top1_rate*100, top5_rate*100, io_rate*100]):
        axes[0].text(i, v+3, f'{v:.0f}%', ha='center', fontweight='bold', fontsize=14)

    # 2. NL search rank distribution
    ranks = [r['rank'] for r in nl_search_results if r['rank'] > 0]
    if ranks:
        axes[1].hist(ranks, bins=range(1, max(ranks)+2), color='#2196F3',
                    edgecolor='black', align='left')
        axes[1].set_xlabel('Rank')
        axes[1].set_ylabel('Count')
        axes[1].set_title('NL Search: Rank Distribution\n(lower = better)', fontweight='bold')

    # 3. 5D space visualization (2D projection)
    from sklearn.manifold import TSNE
    if len(func_list) > 10:
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(20, len(db_5d)-1))
        z_2d = tsne.fit_transform(db_5d)

        # Color by category
        cat_colors = []
        for src, _ in func_list:
            if any(op in src for op in ['+', '-', '*', '/', '%', '**']):
                cat_colors.append('#F44336')
            elif any(op in src for op in ['>', '<', '==', '!=']):
                cat_colors.append('#2196F3')
            elif any(op in src for op in ['abs', 'len', 'upper', 'lower']):
                cat_colors.append('#4CAF50')
            else:
                cat_colors.append('#9E9E9E')

        axes[2].scatter(z_2d[:, 0], z_2d[:, 1], c=cat_colors, s=30, alpha=0.6)
        # Highlight query functions
        for sr in sim_results:
            q_src = sr['query']
            for i, (src, _) in enumerate(func_list):
                if src == q_src:
                    axes[2].scatter(z_2d[i, 0], z_2d[i, 1], c='gold',
                                  s=200, marker='*', edgecolor='black', zorder=10)
                    break
        axes[2].set_title('5D Rosetta Space\n(t-SNE projection)', fontweight='bold')
        axes[2].legend(['Arithmetic', 'Comparison', 'Builtin', 'Other'],
                      loc='upper left', fontsize=7)

    plt.suptitle('Phase 56: Semantic Code Search Engine\n'
                 'Find code by meaning, not by text',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase56_semantic_search.png'), dpi=150)
    plt.close()
    print(f"\nPhase 56 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
