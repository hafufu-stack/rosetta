"""Phase 105: The Cosmic Web - Large-scale structure of program space.
Map the topology using minimum spanning tree and find filaments.
Like the cosmic web in our universe, is there structure at the largest scales?
"""
import os, json, sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

EXP2_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(EXP2_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(EXP2_DIR, 'results')
FIGURES_DIR = os.path.join(EXP2_DIR, 'figures')

def main():
    print("=" * 60)
    print("Phase 105: The Cosmic Web")
    print("  Large-scale structure of program space")
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
    
    print(f"  Functions: {n}")
    
    # Compute distance matrix
    print("  Computing distance matrix...")
    dist_matrix = squareform(pdist(all_vecs))
    
    # Minimum Spanning Tree
    print("  Building MST...")
    mst = minimum_spanning_tree(dist_matrix)
    mst_dense = mst.toarray()
    mst_sym = mst_dense + mst_dense.T
    
    # MST statistics
    edge_weights = mst_dense[mst_dense > 0]
    print(f"  MST edges: {len(edge_weights)}")
    print(f"  Mean edge weight: {np.mean(edge_weights):.4f}")
    print(f"  Std edge weight:  {np.std(edge_weights):.4f}")
    
    # Degree distribution (filament detection)
    degrees = np.array([(mst_sym[i] > 0).sum() for i in range(n)])
    degree_counts = np.bincount(degrees)
    
    leaves = np.sum(degrees == 1)   # endpoints
    chains = np.sum(degrees == 2)   # filaments
    hubs = np.sum(degrees >= 3)     # cluster centers
    
    print(f"\n--- Cosmic Web Topology ---")
    print(f"  Leaves (degree 1):    {leaves} ({leaves/n:.1%})")
    print(f"  Filaments (degree 2): {chains} ({chains/n:.1%})")
    print(f"  Hubs (degree 3+):     {hubs} ({hubs/n:.1%})")
    
    # Hub analysis
    hub_indices = np.where(degrees >= 3)[0]
    print(f"\n--- Hub Programs (Cosmic Nodes) ---")
    for idx in hub_indices[:10]:
        func = unique_funcs[idx]
        short = func.split('return ')[-1].strip() if 'return' in func else func[-30:]
        print(f"  Hub (deg={degrees[idx]}): {short}")
    
    # Longest path in MST (diameter = cosmic scale)
    from collections import deque
    def bfs_farthest(start, adj, n):
        visited = [-1] * n
        visited[start] = 0
        queue = deque([start])
        farthest = start
        max_dist = 0
        while queue:
            node = queue.popleft()
            for neighbor in range(n):
                if adj[node, neighbor] > 0 and visited[neighbor] < 0:
                    visited[neighbor] = visited[node] + 1
                    queue.append(neighbor)
                    if visited[neighbor] > max_dist:
                        max_dist = visited[neighbor]
                        farthest = neighbor
        return farthest, max_dist
    
    far1, _ = bfs_farthest(0, mst_sym, n)
    far2, diameter = bfs_farthest(far1, mst_sym, n)
    
    print(f"\n--- Cosmic Scale ---")
    print(f"  MST diameter (longest path): {diameter} edges")
    f1_short = unique_funcs[far1].split('return ')[-1].strip() if 'return' in unique_funcs[far1] else unique_funcs[far1][-25:]
    f2_short = unique_funcs[far2].split('return ')[-1].strip() if 'return' in unique_funcs[far2] else unique_funcs[far2][-25:]
    print(f"  Endpoints: '{f1_short}' <-> '{f2_short}'")
    
    # Clustering coefficient
    total_triangles = 0
    total_triplets = 0
    for i in range(n):
        neighbors = [j for j in range(n) if mst_sym[i, j] > 0]
        k = len(neighbors)
        if k >= 2:
            total_triplets += k * (k - 1) / 2
            for a in range(len(neighbors)):
                for b in range(a+1, len(neighbors)):
                    if dist_matrix[neighbors[a], neighbors[b]] < np.mean(edge_weights):
                        total_triangles += 1
    clustering = total_triangles / total_triplets if total_triplets > 0 else 0
    
    print(f"  Clustering coefficient: {clustering:.4f}")
    
    # PCA for visualization
    pca = PCA(n_components=2)
    vecs_2d = pca.fit_transform(all_vecs)
    
    # Operation labels for coloring
    op_colors = {}
    for f in unique_funcs:
        if 'x + y' in f: op_colors[f] = '#F44336'
        elif 'x - y' in f: op_colors[f] = '#2196F3'
        elif 'x * y' in f: op_colors[f] = '#4CAF50'
        elif 'max(' in f or 'min(' in f: op_colors[f] = '#FF9800'
        else: op_colors[f] = '#CCCCCC'
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Phase 105: The Cosmic Web of Programs', fontsize=14, fontweight='bold')
    
    # MST visualization
    for i in range(n):
        for j in range(i+1, n):
            if mst_sym[i, j] > 0:
                axes[0].plot([vecs_2d[i,0], vecs_2d[j,0]], [vecs_2d[i,1], vecs_2d[j,1]],
                           'k-', alpha=0.15, linewidth=0.5)
    
    colors = [op_colors[f] for f in unique_funcs]
    sizes = [20 + degrees[i] * 30 for i in range(n)]
    axes[0].scatter(vecs_2d[:,0], vecs_2d[:,1], c=colors, s=sizes, alpha=0.7, edgecolors='black', linewidth=0.5)
    axes[0].set_xlabel('PC1'); axes[0].set_ylabel('PC2')
    axes[0].set_title('Cosmic Web (MST in 2D projection)')
    
    # Degree distribution
    max_deg = min(8, max(degrees))
    axes[1].bar(range(max_deg+1), degree_counts[:max_deg+1], color='#2196F3', edgecolor='black')
    axes[1].set_xlabel('Degree (MST connections)')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'Topology: {leaves} leaves, {chains} filaments, {hubs} hubs')
    
    # Edge weight distribution
    axes[2].hist(edge_weights, bins=30, color='#FF9800', edgecolor='black', alpha=0.7)
    axes[2].axvline(np.mean(edge_weights), color='red', linestyle='--', label=f'mean={np.mean(edge_weights):.3f}')
    axes[2].set_xlabel('Edge Weight (distance)')
    axes[2].set_title(f'MST Edge Distribution (diameter={diameter})')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase105_cosmic_web.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    results = {
        'phase': 105, 'title': 'The Cosmic Web of Programs',
        'n_functions': n, 'mst_edges': len(edge_weights),
        'mean_edge': float(np.mean(edge_weights)),
        'leaves': int(leaves), 'filaments': int(chains), 'hubs': int(hubs),
        'diameter': int(diameter),
        'clustering_coefficient': float(clustering),
        'hub_programs': [unique_funcs[i].split('return ')[-1].strip()[:30] for i in hub_indices[:5]],
        'law': f'Cosmic web: {leaves} leaves ({leaves/n:.0%}), {chains} filaments ({chains/n:.0%}), {hubs} hubs ({hubs/n:.0%}). Diameter={diameter}. Clustering={clustering:.3f}.'
    }
    with open(os.path.join(RESULTS_DIR, 'phase105_cosmic_web.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nPhase 105 complete!")
    return results

if __name__ == '__main__':
    main()
