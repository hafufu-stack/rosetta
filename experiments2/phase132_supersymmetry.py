"""Phase 132: Supersymmetry - Do functions have 'superpartners'?
Opus original: For every function f, does there exist a unique 'superpartner'
f~ that satisfies mirror symmetry in the latent space?
If SUSY holds, every program has a dual that reflects it across a hidden axis.
"""
import os, json, sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
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
    print("Phase 132: Supersymmetry")
    print("  Do all functions have mirror superpartners?")
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
    
    centroid = np.mean(ast_m, axis=0)
    centered = ast_m - centroid
    
    # 1. For each function, find its "superpartner" = mirror image through centroid
    mirrors = -centered  # Reflected through origin (after centering)
    
    # Find the nearest real function to each mirror
    mirror_dists = cdist(mirrors + centroid, ast_m)
    np.fill_diagonal(mirror_dists, np.inf)  # Can't be partner to yourself
    
    susy_pairs = []
    for i in range(n):
        partner_idx = np.argmin(mirror_dists[i])
        partner_dist = mirror_dists[i, partner_idx]
        
        func_name = unique_funcs[i].split('return ')[-1].strip()[:12]
        partner_name = unique_funcs[partner_idx].split('return ')[-1].strip()[:12]
        
        # Bidirectional check: is i also the partner of partner_idx?
        reverse_partner = np.argmin(mirror_dists[partner_idx])
        bidirectional = reverse_partner == i
        
        susy_pairs.append({
            'func': func_name, 'partner': partner_name,
            'distance': float(partner_dist),
            'bidirectional': bool(bidirectional),
        })
    
    n_bidirectional = sum(1 for p in susy_pairs if p['bidirectional'])
    
    print(f"--- Superpartner results ---")
    print(f"  Bidirectional SUSY pairs: {n_bidirectional}/{n} ({n_bidirectional/n*100:.1f}%)")
    
    # Top 5 best SUSY pairs (smallest distance)
    susy_pairs_sorted = sorted(susy_pairs, key=lambda x: x['distance'])
    print(f"\n--- Best superpartner pairs ---")
    for p in susy_pairs_sorted[:5]:
        bi = "bi" if p['bidirectional'] else "uni"
        print(f"  {p['func']} <-> {p['partner']}: d={p['distance']:.4f} ({bi})")
    
    # Worst pairs
    print(f"\n--- Worst superpartner pairs ---")
    for p in susy_pairs_sorted[-3:]:
        print(f"  {p['func']} <-> {p['partner']}: d={p['distance']:.4f}")
    
    # 2. SUSY breaking energy: how much does the mirror symmetry deviate?
    partner_dists = np.array([p['distance'] for p in susy_pairs])
    susy_breaking_energy = np.mean(partner_dists)
    
    # Compare with random pairing
    np.random.seed(42)
    random_dists = []
    for _ in range(100):
        perm = np.random.permutation(n)
        rd = np.mean(np.linalg.norm(ast_m - ast_m[perm], axis=1))
        random_dists.append(rd)
    mean_random = np.mean(random_dists)
    
    print(f"\n--- SUSY Breaking ---")
    print(f"  Mean partner distance: {susy_breaking_energy:.4f}")
    print(f"  Mean random pairing distance: {mean_random:.4f}")
    print(f"  SUSY strength ratio: {susy_breaking_energy/mean_random:.4f}")
    print(f"  {'SUSY approximately holds!' if susy_breaking_energy < mean_random * 0.7 else 'SUSY broken'}")
    
    # 3. Optimal matching: find the best global SUSY pairing
    # Use Hungarian algorithm on subset for computational feasibility
    subset_n = min(50, n)
    cost_matrix = mirror_dists[:subset_n, :subset_n]
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    optimal_cost = cost_matrix[row_ind, col_ind].mean()
    
    print(f"\n--- Optimal SUSY matching ({subset_n} functions) ---")
    print(f"  Mean optimal pair distance: {optimal_cost:.4f}")
    print(f"  Improvement over greedy: {(susy_breaking_energy - optimal_cost) / susy_breaking_energy * 100:.1f}%")
    
    # 4. SUSY algebra: do superpartners satisfy f(x) = -partner(x)?
    semantic_susy = []
    for i in range(min(30, n)):
        src = unique_funcs[i]
        partner_idx = np.argmin(mirror_dists[i])
        partner_src = unique_funcs[partner_idx]
        
        try:
            env1, env2 = {}, {}
            exec(src, env1); exec(partner_src, env2)
            f1 = [v for v in env1.values() if callable(v)][0]
            f2 = [v for v in env2.values() if callable(v)][0]
            
            import inspect
            n1 = len(inspect.signature(f1).parameters)
            n2 = len(inspect.signature(f2).parameters)
            
            if n1 == n2:
                tests = [(2,3),(5,7)] if n1 == 2 else [(3,),(7,)]
                anti_corr = 0
                for args in tests:
                    try:
                        r1 = f1(*args[:n1]); r2 = f2(*args[:n2])
                        if isinstance(r1, (int,float)) and isinstance(r2, (int,float)):
                            if abs(r1 + r2) < 0.01:  # f(x) = -partner(x)
                                anti_corr += 1
                    except Exception: pass
                if anti_corr > 0:
                    semantic_susy.append({
                        'f': src.split('return ')[-1].strip()[:12],
                        'partner': partner_src.split('return ')[-1].strip()[:12],
                        'anti_correlated_tests': anti_corr
                    })
        except Exception: pass
    
    print(f"\n--- Semantic SUSY (f + partner = 0) ---")
    print(f"  Anti-correlated pairs found: {len(semantic_susy)}")
    for s in semantic_susy[:3]:
        print(f"  {s['f']} + {s['partner']} = 0 ({s['anti_correlated_tests']} tests)")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Phase 132: Supersymmetry', fontsize=14, fontweight='bold')
    
    pca_2d = PCA(n_components=2).fit_transform(ast_m)
    axes[0].scatter(pca_2d[:,0], pca_2d[:,1], s=15, alpha=0.3, c='#2196F3', label='Functions')
    # Draw lines between top 10 SUSY pairs
    for p in susy_pairs_sorted[:10]:
        idx_f = next(i for i, u in enumerate(unique_funcs) if u.split('return ')[-1].strip()[:12] == p['func'])
        idx_p = next((i for i, u in enumerate(unique_funcs) if u.split('return ')[-1].strip()[:12] == p['partner']), None)
        if idx_p is not None:
            axes[0].plot([pca_2d[idx_f,0], pca_2d[idx_p,0]], [pca_2d[idx_f,1], pca_2d[idx_p,1]], 'r-', alpha=0.5, linewidth=1)
    axes[0].legend(); axes[0].set_title('SUSY pairs in PC1-PC2')
    
    axes[1].hist(partner_dists, bins=30, color='#E91E63', edgecolor='black', alpha=0.7, label='SUSY pairs')
    axes[1].axvline(mean_random, color='gray', linestyle='--', label=f'Random: {mean_random:.3f}')
    axes[1].axvline(susy_breaking_energy, color='red', linestyle='-', label=f'SUSY: {susy_breaking_energy:.3f}')
    axes[1].legend(); axes[1].set_xlabel('Partner distance'); axes[1].set_title('SUSY breaking energy')
    
    bi_counts = [n_bidirectional, n - n_bidirectional]
    axes[2].pie(bi_counts, labels=['Bidirectional', 'Unidirectional'],
               colors=['#4CAF50', '#FF9800'], autopct='%1.0f%%', startangle=90)
    axes[2].set_title(f'SUSY pair types')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase132_supersymmetry.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    results = {
        'phase': 132, 'title': 'Supersymmetry',
        'n_bidirectional': n_bidirectional, 'total': n,
        'susy_breaking_energy': float(susy_breaking_energy),
        'random_baseline': float(mean_random),
        'susy_ratio': float(susy_breaking_energy / mean_random),
        'optimal_cost': float(optimal_cost),
        'semantic_anti_pairs': len(semantic_susy),
        'top_pairs': susy_pairs_sorted[:5],
        'law': f'SUSY pairs: {n_bidirectional}/{n} bidirectional. Breaking energy={susy_breaking_energy:.3f} vs random={mean_random:.3f} (ratio={susy_breaking_energy/mean_random:.3f}). Semantic anti-pairs: {len(semantic_susy)}.'
    }
    with open(os.path.join(RESULTS_DIR, 'phase132_supersymmetry.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nPhase 132 complete!")
    return results

if __name__ == '__main__':
    main()
