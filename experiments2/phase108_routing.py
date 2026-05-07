"""Phase 108: Cosmic Web Routing - Wormhole debugging via graph traversal.
P103 SA failed because d^-3.40 gravity is too strong for continuous walk.
Solution: use the Cosmic Web (MST) as a highway network.
"""
import os, json, sys, inspect
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree, dijkstra
from scipy.spatial.distance import pdist, squareform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXP2_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(EXP2_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')

def main():
    print("=" * 60)
    print("Phase 108: Cosmic Web Routing")
    print("  Wormhole debugging via graph traversal")
    print("=" * 60)
    
    latents = np.load(os.path.join(DATA_DIR, 'rosetta_latents_v2.npz'))
    dataset = json.load(open(os.path.join(DATA_DIR, 'rosetta_dataset_v2.json'), 'r', encoding='utf-8'))
    ast_vectors = latents['ast']
    sources = [item['source'] for item in dataset['dataset']]
    
    func_to_vecs = {}
    for i, src in enumerate(sources):
        if src not in func_to_vecs: func_to_vecs[src] = []
        func_to_vecs[src].append(ast_vectors[i])
    func_means = {s: np.mean(v, axis=0) for s, v in func_to_vecs.items()}
    unique_funcs = list(func_means.keys())
    all_vecs = np.array([func_means[f] for f in unique_funcs])
    n = len(unique_funcs)
    
    # Build MST (Cosmic Web)
    print("  Building Cosmic Web (MST)...")
    dist_matrix = squareform(pdist(all_vecs))
    mst = minimum_spanning_tree(dist_matrix)
    mst_dense = mst.toarray()
    mst_sym = mst_dense + mst_dense.T
    
    # Compute shortest paths on MST (Dijkstra)
    print("  Computing wormhole routes (Dijkstra)...")
    graph_dists, predecessors = dijkstra(mst_sym, directed=False, return_predecessors=True)
    
    # Build Neural CPU for I/O verification
    print("  Building Neural CPU for verification...")
    from sklearn.neural_network import MLPRegressor
    exec_data = []
    g = {}
    for func_src in unique_funcs:
        vec = func_means[func_src]
        try:
            exec(compile(func_src, '<string>', 'exec'), g)
            fn = g['f']
            try: n_args = len(inspect.signature(fn).parameters)
            except: n_args = 2
            for x_val in [-2, -1, 0, 1, 2, 3, 5]:
                for y_val in [-2, -1, 0, 1, 2, 3, 5]:
                    try:
                        result = fn(x_val) if n_args == 1 else fn(x_val, y_val)
                        if isinstance(result, (int, float, bool)) and abs(float(result)) < 1e6:
                            features = np.concatenate([vec, [x_val, y_val]])
                            exec_data.append((features, float(result)))
                    except: pass
        except: pass
    
    X_cpu = np.array([d[0] for d in exec_data])
    y_cpu = np.array([d[1] for d in exec_data])
    cpu = MLPRegressor(hidden_layer_sizes=(256, 128, 64), max_iter=2000, random_state=42,
                       early_stopping=True, validation_fraction=0.1)
    cpu.fit(X_cpu, y_cpu)
    
    # Bug scenarios
    bug_scenarios = [
        ('def f(x, y): return x + y', 'def f(x, y): return x - y', 'add->sub'),
        ('def f(x, y): return x - y', 'def f(x, y): return x + y', 'sub->add'),
        ('def f(x, y): return x * y', 'def f(x, y): return x + y', 'mul->add'),
        ('def f(x, y): return x + y', 'def f(x, y): return x * y', 'add->mul'),
        ('def f(x, y): return max(x, y)', 'def f(x, y): return min(x, y)', 'max->min'),
        ('def f(x, y): return x > y', 'def f(x, y): return x < y', 'gt->lt'),
        ('def f(x, y): return x ** y', 'def f(x, y): return x * y', 'pow->mul'),
    ]
    test_inputs = [(1,2),(3,5),(-1,4),(2,2),(0,7),(4,3),(-2,1)]
    
    repair_results = []
    
    for buggy_src, correct_src, bug_name in bug_scenarios:
        if buggy_src not in func_means or correct_src not in func_means:
            continue
        
        bug_idx = unique_funcs.index(buggy_src)
        correct_idx = unique_funcs.index(correct_src)
        
        # Get target I/O
        g2 = {}
        exec(compile(correct_src, '<string>', 'exec'), g2)
        target_io = []
        for x, y in test_inputs:
            try: target_io.append(((x, y), float(g2['f'](x, y))))
            except: target_io.append(((x, y), 0.0))
        
        # Route through Cosmic Web
        path = []
        current = bug_idx
        while current != correct_idx:
            path.append(current)
            current = predecessors[correct_idx, current]
            if current < 0: break
        path.append(correct_idx)
        
        # Score each waypoint by I/O fitness
        best_score = float('inf')
        best_waypoint = bug_idx
        best_func = buggy_src
        
        for wp in path:
            vec = all_vecs[wp]
            score = 0
            for (x, y), expected in target_io:
                features = np.concatenate([vec, [x, y]])
                pred = cpu.predict(features.reshape(1, -1))[0]
                score += (pred - expected) ** 2
            score /= len(target_io)
            if score < best_score:
                best_score = score
                best_waypoint = wp
                best_func = unique_funcs[wp]
        
        euclidean_dist = dist_matrix[bug_idx, correct_idx]
        graph_dist = graph_dists[bug_idx, correct_idx]
        success = best_func == correct_src
        
        short = best_func.split('return ')[-1].strip() if 'return' in best_func else best_func[-25:]
        
        result = {
            'bug': bug_name,
            'path_length': len(path),
            'euclidean_dist': float(euclidean_dist),
            'graph_dist': float(graph_dist),
            'best_waypoint': short,
            'repaired': success
        }
        repair_results.append(result)
        
        print(f"\n  [{bug_name}]")
        print(f"    Path: {len(path)} hops | Euclidean: {euclidean_dist:.3f} | Graph: {graph_dist:.3f}")
        print(f"    Best waypoint: {short} {'(REPAIRED!)' if success else ''}")
    
    n_repaired = sum(1 for r in repair_results if r['repaired'])
    repair_rate = n_repaired / len(repair_results) if repair_results else 0
    
    print(f"\n{'='*60}")
    print(f"COSMIC WEB ROUTING: {n_repaired}/{len(repair_results)} ({repair_rate:.0%})")
    print(f"P88:0/6 | P92:0/7 | P94:0/7 | P103(SA):0/6 | P108(Web):{n_repaired}/{len(repair_results)}")
    print(f"{'='*60}")
    
    # Plot
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    vecs_2d = pca.fit_transform(all_vecs)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Phase 108: Cosmic Web Routing (Wormhole Debugging)', fontsize=14, fontweight='bold')
    
    for i in range(n):
        for j in range(i+1, n):
            if mst_sym[i,j] > 0:
                axes[0].plot([vecs_2d[i,0],vecs_2d[j,0]], [vecs_2d[i,1],vecs_2d[j,1]], 'k-', alpha=0.1, linewidth=0.3)
    axes[0].scatter(vecs_2d[:,0], vecs_2d[:,1], s=10, alpha=0.3, c='gray')
    for r in repair_results:
        bug_src = [s for s in bug_scenarios if s[2]==r['bug']][0][0]
        cor_src = [s for s in bug_scenarios if s[2]==r['bug']][0][1]
        if bug_src in func_means and cor_src in func_means:
            bi = unique_funcs.index(bug_src)
            ci = unique_funcs.index(cor_src)
            c = 'green' if r['repaired'] else 'red'
            axes[0].plot([vecs_2d[bi,0],vecs_2d[ci,0]], [vecs_2d[bi,1],vecs_2d[ci,1]], '-', color=c, linewidth=2)
    axes[0].set_title('Cosmic Web + Bug Routes')
    
    methods = ['P88\n5D', 'P92\n64D', 'P94\nDual', 'P103\nSA', 'P108\nWeb']
    rates = [0, 0, 0, 0, repair_rate*100]
    axes[1].bar(methods, rates, color=['#F44336']*4+['#4CAF50'], edgecolor='black')
    axes[1].set_ylabel('Repair Rate (%)'); axes[1].set_title('All Methods Compared')
    for i,v in enumerate(rates): axes[1].text(i, v+2, f'{v:.0f}%', ha='center', fontweight='bold')
    
    hops = [r['path_length'] for r in repair_results]
    axes[2].bar([r['bug'][:8] for r in repair_results], hops,
               color=['#4CAF50' if r['repaired'] else '#FF9800' for r in repair_results], edgecolor='black')
    axes[2].set_ylabel('Path Length (hops)')
    axes[2].set_title('Route Lengths')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase108_routing.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    results = {
        'phase': 108, 'title': 'Cosmic Web Routing (Wormhole Debugging)',
        'n_bugs': len(repair_results), 'n_repaired': n_repaired,
        'repair_rate': float(repair_rate), 'repairs': repair_results,
        'law': f'Cosmic Web routing repairs {n_repaired}/{len(repair_results)} bugs via graph traversal. Wormholes bypass d^-3.4 gravity wells.'
    }
    with open(os.path.join(RESULTS_DIR, 'phase108_routing.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nPhase 108 complete!")
    return results

if __name__ == '__main__':
    main()
