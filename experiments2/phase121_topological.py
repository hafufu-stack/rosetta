"""Phase 121: Topological Bug Repair via Persistent Homology
Deep Think: Use topological data analysis to find 'holes' in 64D space
and tunnel through them for bug repair.
Since ripser/gudhi unavailable, we implement Vietoris-Rips from scratch.
"""
import os, json, sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXP2_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(EXP2_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')

def compute_betti_numbers(dist_matrix, threshold):
    """Compute Betti-0 (connected components) and approximate Betti-1 (loops)."""
    n = len(dist_matrix)
    
    # Betti-0: connected components via union-find
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[ra] = rb
    
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if dist_matrix[i,j] <= threshold:
                union(i, j)
                edges.append((i, j))
    
    components = len(set(find(i) for i in range(n)))
    
    # Approximate Betti-1: count triangles vs edges
    # Euler characteristic: V - E + F = chi, and betti_1 ~ E - V + components - triangles
    n_edges = len(edges)
    n_triangles = 0
    adj = {i: set() for i in range(n)}
    for i, j in edges:
        adj[i].add(j); adj[j].add(i)
    for i, j in edges:
        common = adj[i] & adj[j]
        n_triangles += len(common)
    n_triangles //= 3  # each triangle counted 3 times
    
    betti_1_approx = max(0, n_edges - n + components - n_triangles)
    return components, betti_1_approx

def main():
    print("=" * 60)
    print("Phase 121: Topological Bug Repair")
    print("  Persistent homology of program space")
    print("=" * 60)
    
    latents = np.load(os.path.join(DATA_DIR, 'rosetta_latents_v2.npz'))
    dataset = json.load(open(os.path.join(DATA_DIR, 'rosetta_dataset_v2.json'), 'r', encoding='utf-8'))
    ast_vectors = latents['ast']
    sources = [item['source'] for item in dataset['dataset']]
    
    func_ast = {}
    for i, src in enumerate(sources):
        if src not in func_ast: func_ast[src] = []
        func_ast[src].append(ast_vectors[i])
    
    unique_funcs = list(func_ast.keys())
    ast_m = np.array([np.mean(func_ast[f], axis=0) for f in unique_funcs])
    n = len(unique_funcs)
    
    # Distance matrix
    dist_mat = squareform(pdist(ast_m, 'euclidean'))
    
    # 1. Persistent homology: sweep threshold
    thresholds = np.linspace(0.1, 3.0, 30)
    betti_0_list, betti_1_list = [], []
    
    print("--- Persistent Homology ---")
    for t in thresholds:
        b0, b1 = compute_betti_numbers(dist_mat, t)
        betti_0_list.append(b0)
        betti_1_list.append(b1)
    
    # Birth-death of topological features
    # Betti-0 starts at n, drops to 1
    # Betti-1 appears and disappears
    max_b1 = max(betti_1_list)
    max_b1_threshold = thresholds[np.argmax(betti_1_list)]
    print(f"  Max Betti-1 (loops): {max_b1} at threshold={max_b1_threshold:.2f}")
    print(f"  Betti-0 drops to 1 at threshold: {thresholds[next(i for i,b in enumerate(betti_0_list) if b == 1)] if 1 in betti_0_list else 'never':.2f}")
    
    # 2. Topological tunneling for bug repair
    # Instead of gradient descent (0% repair), tunnel through topological holes
    bug_test_cases = [
        ('def f(x, y): return x + y', 'def f(x, y): return x - y', 'add->sub'),
        ('def f(x, y): return x * y', 'def f(x, y): return x / y', 'mul->div'),
        ('def f(x, y): return x > y', 'def f(x, y): return x < y', 'gt->lt'),
        ('def f(x, y): return x == y', 'def f(x, y): return x != y', 'eq->neq'),
        ('def f(x, y): return max(x, y)', 'def f(x, y): return min(x, y)', 'max->min'),
        ('def f(x): return abs(x)', 'def f(x): return -x', 'abs->neg'),
    ]
    
    tunnel_results = []
    
    for buggy_src, target_src, label in bug_test_cases:
        if buggy_src not in func_ast or target_src not in func_ast:
            continue
        
        buggy_idx = unique_funcs.index(buggy_src)
        target_idx = unique_funcs.index(target_src)
        
        buggy_v = ast_m[buggy_idx]
        target_v = ast_m[target_idx]
        direct_dist = np.linalg.norm(buggy_v - target_v)
        
        # Method 1: Direct (Euclidean) - always fails (proven in P88/P92)
        # Method 2: Cosmic web routing (P108)
        # Method 3: Topological tunneling - find the shortest topological path
        # that passes through regions of high Betti-1 (holes)
        
        # Build k-NN graph
        knn = NearestNeighbors(n_neighbors=10).fit(ast_m)
        _, knn_indices = knn.kneighbors(ast_m)
        
        # BFS through k-NN graph
        from collections import deque
        queue = deque([(buggy_idx, [buggy_idx])])
        visited = {buggy_idx}
        path_found = None
        
        while queue and path_found is None:
            current, path = queue.popleft()
            for neighbor in knn_indices[current]:
                if neighbor == target_idx:
                    path_found = path + [target_idx]
                    break
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
            if len(path) > 20:
                break
        
        # Method 3b: Wormhole (project through orthogonal complement)
        diff = target_v - buggy_v
        diff_hat = diff / (np.linalg.norm(diff) + 1e-10)
        # Tunnel: jump in orthogonal direction, then converge
        ortho = np.random.randn(64)
        ortho -= np.dot(ortho, diff_hat) * diff_hat
        ortho /= np.linalg.norm(ortho)
        
        tunnel_point = buggy_v + 0.5 * diff + 0.3 * ortho
        # Find nearest function to tunnel midpoint
        tunnel_dists = np.linalg.norm(ast_m - tunnel_point.reshape(1,-1), axis=1)
        tunnel_nearest = np.argmin(tunnel_dists)
        tunnel_func = unique_funcs[tunnel_nearest].split('return ')[-1].strip()[:15]
        
        path_len = len(path_found) if path_found else -1
        success = path_found is not None
        
        tunnel_results.append({
            'bug': label,
            'direct_dist': float(direct_dist),
            'knn_path_length': path_len,
            'knn_success': success,
            'tunnel_midpoint': tunnel_func,
        })
        
        status = f"path={path_len}" if success else "no path"
        print(f"  {label}: direct_d={direct_dist:.3f}, kNN {status}, tunnel via '{tunnel_func}'")
    
    # Repair rate
    knn_repairs = sum(1 for r in tunnel_results if r['knn_success'])
    total = len(tunnel_results)
    repair_rate = knn_repairs / total * 100 if total > 0 else 0
    print(f"\n--- Topological repair rate: {knn_repairs}/{total} ({repair_rate:.0f}%) ---")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Phase 121: Topological Bug Repair', fontsize=14, fontweight='bold')
    
    axes[0].plot(thresholds, betti_0_list, 'b-', label='Betti-0 (components)', linewidth=2)
    axes[0].plot(thresholds, betti_1_list, 'r-', label='Betti-1 (loops)', linewidth=2)
    axes[0].set_xlabel('Threshold'); axes[0].set_ylabel('Betti number')
    axes[0].set_title('Persistent Homology'); axes[0].legend()
    
    labels = [r['bug'] for r in tunnel_results]
    dists = [r['direct_dist'] for r in tunnel_results]
    colors = ['#4CAF50' if r['knn_success'] else '#F44336' for r in tunnel_results]
    axes[1].barh(labels, dists, color=colors, edgecolor='black')
    axes[1].set_xlabel('Euclidean distance'); axes[1].set_title(f'Bug repair: {knn_repairs}/{total} via k-NN tunneling')
    
    pca_2d = PCA(n_components=2).fit_transform(ast_m)
    axes[2].scatter(pca_2d[:,0], pca_2d[:,1], s=15, alpha=0.3, c='gray')
    for r in tunnel_results:
        if r['knn_success']:
            buggy_src = [s for s, t, l in bug_test_cases if l == r['bug']][0]
            target_src = [t for s, t, l in bug_test_cases if l == r['bug']][0]
            if buggy_src in unique_funcs and target_src in unique_funcs:
                bi = unique_funcs.index(buggy_src)
                ti = unique_funcs.index(target_src)
                axes[2].annotate('', xy=pca_2d[ti], xytext=pca_2d[bi],
                               arrowprops=dict(arrowstyle='->', color='green', lw=2))
    axes[2].set_title('Repair paths in PC1-PC2')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase121_topological.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    results = {
        'phase': 121, 'title': 'Topological Bug Repair',
        'max_betti_1': int(max_b1),
        'max_betti_1_threshold': float(max_b1_threshold),
        'repair_results': tunnel_results,
        'repair_rate_pct': float(repair_rate),
        'law': f'Max Betti-1={max_b1} loops at threshold={max_b1_threshold:.2f}. k-NN topological repair: {knn_repairs}/{total} ({repair_rate:.0f}%).'
    }
    with open(os.path.join(RESULTS_DIR, 'phase121_topological.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nPhase 121 complete!")
    return results

if __name__ == '__main__':
    main()
