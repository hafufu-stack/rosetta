"""Phase 117: The Dual-Space Wormhole - Combine AST+BC for ultimate routing.
P107 showed CCA=1.000 (perfect duality). Can we exploit BOTH spaces simultaneously
to find wormholes invisible in either space alone?
"""
import os, json, sys, inspect
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import dijkstra
from sklearn.neural_network import MLPRegressor
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
    print("Phase 117: The Dual-Space Wormhole")
    print("  AST + BC combined routing")
    print("=" * 60)
    
    latents = np.load(os.path.join(DATA_DIR, 'rosetta_latents_v2.npz'))
    dataset = json.load(open(os.path.join(DATA_DIR, 'rosetta_dataset_v2.json'), 'r', encoding='utf-8'))
    ast_vectors = latents['ast']
    bc_vectors = latents['bc']
    sources = [item['source'] for item in dataset['dataset']]
    
    func_ast = {}; func_bc = {}
    for i, src in enumerate(sources):
        if src not in func_ast: func_ast[src] = []; func_bc[src] = []
        func_ast[src].append(ast_vectors[i])
        func_bc[src].append(bc_vectors[i])
    
    unique_funcs = list(func_ast.keys())
    ast_means = np.array([np.mean(func_ast[f], axis=0) for f in unique_funcs])
    bc_means = np.array([np.mean(func_bc[f], axis=0) for f in unique_funcs])
    n = len(unique_funcs)
    
    # Build 3 different distance matrices
    ast_dists = squareform(pdist(ast_means))
    bc_dists = squareform(pdist(bc_means))
    
    # Combined: 128D concatenation
    combined = np.hstack([ast_means, bc_means])
    comb_dists = squareform(pdist(combined))
    
    # Minimum of AST and BC distances (wormhole = shortcut in EITHER space)
    wormhole_dists = np.minimum(ast_dists, bc_dists)
    
    # Build Neural CPU (AST-based, as before)
    print("  Building Neural CPU...")
    exec_data = []
    g = {}
    for func_src in unique_funcs:
        vec = ast_means[unique_funcs.index(func_src)]
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
    
    def io_energy(vec, target_io):
        total = 0.0
        for (x, y), expected in target_io:
            features = np.concatenate([vec, [x, y]])
            pred = cpu.predict(features.reshape(1, -1))[0]
            total += (pred - expected) ** 2
        return total / len(target_io)
    
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
    
    k = 10  # From P113's best result
    space_results = {}
    
    for space_name, dist_mat, vecs in [('AST', ast_dists, ast_means), ('BC', bc_dists, ast_means),
                                         ('Combined', comb_dists, ast_means), ('Wormhole', wormhole_dists, ast_means)]:
        # Build k-NN graph
        knn = np.zeros((n, n))
        for i in range(n):
            d_i = dist_mat[i].copy(); d_i[i] = float('inf')
            for j in np.argsort(d_i)[:k]:
                knn[i,j] = dist_mat[i,j]; knn[j,i] = dist_mat[j,i]
        
        _, preds = dijkstra(knn, directed=False, return_predecessors=True)
        
        n_repaired = 0
        for buggy_src, correct_src, bug_name in bug_scenarios:
            if buggy_src not in unique_funcs or correct_src not in unique_funcs: continue
            bug_idx = unique_funcs.index(buggy_src)
            correct_idx = unique_funcs.index(correct_src)
            
            g2 = {}; exec(compile(correct_src, '<string>', 'exec'), g2)
            target_io = [((x,y), float(g2['f'](x,y))) for x,y in test_inputs]
            
            # I/O guided search over all reachable
            graph_d, _ = dijkstra(knn, directed=False, return_predecessors=True, indices=bug_idx)
            reachable = [j for j in range(n) if graph_d[j] < float('inf')]
            
            best_func = buggy_src
            best_score = float('inf')
            for wp in reachable:
                score = io_energy(ast_means[wp], target_io)
                if score < best_score:
                    best_score = score; best_func = unique_funcs[wp]
            
            if best_func == correct_src: n_repaired += 1
        
        rate = n_repaired / len([s for s in bug_scenarios if s[0] in unique_funcs and s[1] in unique_funcs])
        space_results[space_name] = {'n_repaired': n_repaired, 'rate': rate}
        print(f"  {space_name:10s}: {n_repaired} repaired ({rate:.0%})")
    
    # Find wormholes: pairs where BC distance << AST distance
    wormhole_pairs = []
    for i in range(n):
        for j in range(i+1, n):
            ratio = bc_dists[i,j] / (ast_dists[i,j] + 1e-10)
            if ratio < 0.5:
                f_i = unique_funcs[i].split('return ')[-1].strip()[:15]
                f_j = unique_funcs[j].split('return ')[-1].strip()[:15]
                wormhole_pairs.append((f_i, f_j, float(ratio), float(ast_dists[i,j]), float(bc_dists[i,j])))
    
    wormhole_pairs.sort(key=lambda x: x[2])
    print(f"\n--- Wormholes (BC << AST) ---")
    for f1, f2, ratio, ad, bd in wormhole_pairs[:5]:
        print(f"  {f1} <-> {f2}: BC/AST={ratio:.3f} (AST={ad:.3f}, BC={bd:.3f})")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Phase 117: The Dual-Space Wormhole', fontsize=14, fontweight='bold')
    
    spaces = list(space_results.keys())
    rates = [space_results[s]['rate']*100 for s in spaces]
    colors = ['#2196F3','#FF9800','#9C27B0','#4CAF50']
    axes[0].bar(spaces, rates, color=colors, edgecolor='black')
    axes[0].set_ylabel('Repair Rate (%)')
    axes[0].set_title('Routing in Different Spaces')
    for i,v in enumerate(rates): axes[0].text(i, v+2, f'{v:.0f}%', ha='center', fontweight='bold')
    
    axes[1].scatter(ast_dists.flatten()[::100], bc_dists.flatten()[::100], alpha=0.1, s=3, c='#2196F3')
    axes[1].plot([0,3],[0,3], 'r--', label='AST=BC')
    axes[1].set_xlabel('AST Distance'); axes[1].set_ylabel('BC Distance')
    axes[1].set_title('AST vs BC Distances')
    axes[1].legend()
    
    if wormhole_pairs:
        ratios = [w[2] for w in wormhole_pairs[:30]]
        axes[2].hist(ratios, bins=20, color='#4CAF50', edgecolor='black', alpha=0.7)
        axes[2].set_xlabel('BC/AST Ratio')
        axes[2].set_title(f'Wormhole Distribution ({len(wormhole_pairs)} found)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase117_dual_wormhole.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    results = {
        'phase': 117, 'title': 'The Dual-Space Wormhole',
        'space_results': {k: v for k,v in space_results.items()},
        'n_wormholes': len(wormhole_pairs),
        'top_wormholes': wormhole_pairs[:5],
        'law': f'Dual-space routing: AST={space_results["AST"]["rate"]:.0%}, BC={space_results["BC"]["rate"]:.0%}, Combined={space_results["Combined"]["rate"]:.0%}, Wormhole={space_results["Wormhole"]["rate"]:.0%}. {len(wormhole_pairs)} wormholes found.'
    }
    with open(os.path.join(RESULTS_DIR, 'phase117_dual_wormhole.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nPhase 117 complete!")
    return results

if __name__ == '__main__':
    main()
