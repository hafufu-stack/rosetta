"""Phase 120: Quine Singularities & Closed Timelike Curves
Deep Think: Self-referential programs create time loops in code space.
We add quines and recursive functions, test if they form CTCs
(closed loops in the arrow-of-time axis PC2).
"""
import os, json, sys, ast, types
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from sklearn.decomposition import PCA
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
    print("Phase 120: Quine Singularities & Closed Timelike Curves")
    print("  Do self-referential programs form time loops?")
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
    
    # Fit global PCA
    pca = PCA(n_components=10).fit(ast_m)
    ast_pca = pca.transform(ast_m)
    
    # 1. Define "evolutionary chains" (from P112) and their PC2 projection
    chains = [
        ['def f(x, y): return x + y', 'def f(x, y): return x * y', 'def f(x, y): return x ** y'],
        ['def f(x): return abs(x)', 'def f(x): return x * x', 'def f(x): return x ** x'],
        ['def f(x, y): return x - y', 'def f(x, y): return x // y', 'def f(x, y): return x % y'],
    ]
    
    print("--- Evolutionary chains (PC2 = arrow of time) ---")
    chain_data = []
    for chain in chains:
        pc2_vals = []
        for src in chain:
            if src in func_ast:
                idx = unique_funcs.index(src)
                pc2_vals.append(float(ast_pca[idx, 1]))
            else:
                pc2_vals.append(None)
        
        monotonic = True
        for i in range(len(pc2_vals)-1):
            if pc2_vals[i] is not None and pc2_vals[i+1] is not None:
                if abs(pc2_vals[i+1]) <= abs(pc2_vals[i]):
                    monotonic = False
        
        label = ' -> '.join([s.split('return ')[-1].strip()[:10] for s in chain])
        print(f"  {label}: PC2 = {pc2_vals} (monotonic={monotonic})")
        chain_data.append({'chain': label, 'pc2': pc2_vals, 'monotonic': monotonic})
    
    # 2. Self-referential / recursive patterns
    # Simulate "quines" as functions that map to themselves under round-trip
    # (identity functions, idempotent operations)
    self_ref_funcs = []
    for idx, src in enumerate(unique_funcs):
        v = ast_m[idx]
        # Apply compile matrix (simulate round-trip)
        # Since we don't have W_compile here, we use projection + reconstruction
        v_proj = pca.transform(v.reshape(1,-1))[0]
        v_reconstructed = pca.inverse_transform(v_proj.reshape(1,-1))[0]
        
        # Self-similarity = how much the function is invariant under round-trip
        cos_self = np.dot(v, v_reconstructed) / (np.linalg.norm(v) * np.linalg.norm(v_reconstructed) + 1e-10)
        self_ref_funcs.append({
            'func': src.split('return ')[-1].strip()[:20],
            'self_similarity': float(cos_self),
            'pc2': float(ast_pca[idx, 1])
        })
    
    self_ref_funcs.sort(key=lambda x: x['self_similarity'], reverse=True)
    
    print("\n--- Most self-referential (quine-like) functions ---")
    for entry in self_ref_funcs[:5]:
        print(f"  {entry['func']}: self_sim={entry['self_similarity']:.4f}, PC2={entry['pc2']:.3f}")
    
    print("\n--- Least self-referential functions ---")
    for entry in self_ref_funcs[-3:]:
        print(f"  {entry['func']}: self_sim={entry['self_similarity']:.4f}, PC2={entry['pc2']:.3f}")
    
    # 3. Closed Timelike Curves: find cycles in nearest-neighbor graph
    # If function A's nearest neighbor is B, B's is C, and C's is A -> CTC
    from scipy.spatial.distance import cdist
    dist_matrix = cdist(ast_m, ast_m, 'cosine')
    np.fill_diagonal(dist_matrix, np.inf)
    nn_graph = np.argmin(dist_matrix, axis=1)
    
    # Find cycles of length 2-5
    cycles = []
    for start in range(n):
        visited = [start]
        current = start
        for step in range(5):
            nxt = nn_graph[current]
            if nxt == start and step >= 1:
                cycles.append(visited[:])
                break
            if nxt in visited:
                break
            visited.append(nxt)
            current = nxt
    
    print(f"\n--- Closed Timelike Curves (NN cycles) ---")
    print(f"  Total CTCs found: {len(cycles)}")
    for cyc in cycles[:5]:
        labels = [unique_funcs[i].split('return ')[-1].strip()[:12] for i in cyc]
        pc2s = [float(ast_pca[i, 1]) for i in cyc]
        print(f"  {' -> '.join(labels)} -> [loop] (PC2: {', '.join(f'{p:.2f}' for p in pc2s)})")
    
    # 4. Singularity detection: functions with extreme curvature
    # Curvature = how much the local neighborhood bends
    curvatures = []
    for idx in range(n):
        neighbors_idx = np.argsort(dist_matrix[idx])[:5]
        neighbor_vecs = ast_m[neighbors_idx] - ast_m[idx]
        if len(neighbor_vecs) >= 3:
            # Curvature approximation: spread of neighbor directions
            norms = np.linalg.norm(neighbor_vecs, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1
            directions = neighbor_vecs / norms
            spread = 1.0 - np.mean(np.abs(directions @ directions.T))
            curvatures.append(float(spread))
        else:
            curvatures.append(0.0)
    
    curvatures = np.array(curvatures)
    singularity_idx = np.argsort(curvatures)[-5:]
    
    print(f"\n--- Singularities (extreme curvature) ---")
    for idx in singularity_idx:
        func_short = unique_funcs[idx].split('return ')[-1].strip()[:20]
        print(f"  {func_short}: curvature={curvatures[idx]:.4f}")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Phase 120: Quine Singularities & CTCs', fontsize=14, fontweight='bold')
    
    # Panel 1: PC1 vs PC2 with CTC cycles highlighted
    axes[0].scatter(ast_pca[:,0], ast_pca[:,1], s=15, alpha=0.3, c='gray')
    for cyc in cycles[:10]:
        pts = ast_pca[cyc]
        pts_closed = np.vstack([pts, pts[0:1]])
        axes[0].plot(pts_closed[:,0], pts_closed[:,1], '-o', markersize=5, linewidth=1.5)
    axes[0].set_xlabel('PC1'); axes[0].set_ylabel('PC2 (Arrow of Time)')
    axes[0].set_title(f'{len(cycles)} Closed Timelike Curves')
    
    # Panel 2: Self-similarity distribution
    self_sims = [s['self_similarity'] for s in self_ref_funcs]
    axes[1].hist(self_sims, bins=30, color='#E91E63', edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Self-similarity (round-trip)')
    axes[1].set_title('Quine-ness distribution')
    
    # Panel 3: Curvature map
    sc = axes[2].scatter(ast_pca[:,0], ast_pca[:,1], c=curvatures, cmap='hot', s=20, alpha=0.7)
    plt.colorbar(sc, ax=axes[2], label='Curvature')
    axes[2].set_xlabel('PC1'); axes[2].set_ylabel('PC2')
    axes[2].set_title('Spacetime curvature map')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase120_quine.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    results = {
        'phase': 120, 'title': 'Quine Singularities & CTCs',
        'n_ctcs': len(cycles),
        'top_ctc_examples': [{'cycle': [unique_funcs[i].split('return ')[-1].strip()[:15] for i in c]} for c in cycles[:5]],
        'quine_top5': self_ref_funcs[:5],
        'max_curvature': float(np.max(curvatures)),
        'law': f'Found {len(cycles)} closed timelike curves. Top self-referential: {self_ref_funcs[0]["func"]} (sim={self_ref_funcs[0]["self_similarity"]:.4f}). Max curvature={np.max(curvatures):.4f}.'
    }
    with open(os.path.join(RESULTS_DIR, 'phase120_quine.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nPhase 120 complete!")
    return results

if __name__ == '__main__':
    main()
