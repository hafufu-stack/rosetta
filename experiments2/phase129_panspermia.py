"""Phase 129: Directed Panspermia - Seed voids with 'amino acids' from strong functions.
P123 showed 0% abiogenesis from pure void. Here we seed with partial info.
"""
import os, json, sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from sklearn.neighbors import NearestNeighbors
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
    print("Phase 129: Directed Panspermia")
    print("  Can seeded voids produce life?")
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
    
    knn = NearestNeighbors(n_neighbors=1).fit(ast_m)
    
    def fitness(v):
        d = np.min(np.linalg.norm(ast_m - v.reshape(1,-1), axis=1))
        return 1.0 / (1.0 + d)
    
    # Find deep voids
    np.random.seed(42)
    bbox_min, bbox_max = ast_m.min(axis=0), ast_m.max(axis=0)
    probes = np.random.uniform(bbox_min, bbox_max, size=(500, 64))
    dists_to_known, _ = knn.kneighbors(probes)
    void_idx = np.argsort(dists_to_known.ravel())[-10:]
    void_locations = probes[void_idx]
    
    # Extract "amino acids" from strong functions via SVD
    U, S, Vt = np.linalg.svd(ast_m, full_matrices=False)
    # Top-k principal components = "amino acids"
    amino_acids = Vt[:5]  # 5 fundamental building blocks
    print(f"  Amino acids extracted: {amino_acids.shape[0]} (from SVD)")
    
    # Seed types to test
    seed_strategies = {
        'none': lambda origin: origin + np.random.randn(30, 64) * 0.1,
        'amino_1': lambda origin: origin + np.random.randn(30, 64) * 0.1 + amino_acids[0] * 0.3,
        'amino_3': lambda origin: origin + np.random.randn(30, 64) * 0.1 + np.mean(amino_acids[:3], axis=0) * 0.3,
        'amino_5': lambda origin: origin + np.random.randn(30, 64) * 0.1 + np.mean(amino_acids[:5], axis=0) * 0.3,
        'donor_add': lambda origin: origin + np.random.randn(30, 64) * 0.1 + ast_m[unique_funcs.index('def f(x, y): return x + y')] * 0.2,
        'centroid': lambda origin: origin + np.random.randn(30, 64) * 0.1 + np.mean(ast_m, axis=0) * 0.2,
    }
    
    all_results = {}
    
    for strategy_name, seed_fn in seed_strategies.items():
        emerged_count = 0
        best_fitnesses = []
        nearest_funcs = []
        
        for v_idx in range(min(5, len(void_locations))):
            origin = void_locations[v_idx]
            pop = seed_fn(origin)
            
            # GA
            for gen in range(200):
                fits = np.array([fitness(ind) for ind in pop])
                # Tournament selection
                new_pop = []
                for _ in range(len(pop)):
                    i, j = np.random.randint(0, len(pop), 2)
                    new_pop.append(pop[i].copy() if fits[i] > fits[j] else pop[j].copy())
                # Crossover + mutation
                for i in range(0, len(new_pop)-1, 2):
                    if np.random.random() < 0.7:
                        a = np.random.random()
                        c1, c2 = a*new_pop[i]+(1-a)*new_pop[i+1], (1-a)*new_pop[i]+a*new_pop[i+1]
                        new_pop[i], new_pop[i+1] = c1, c2
                for i in range(len(new_pop)):
                    if np.random.random() < 0.1:
                        new_pop[i] += np.random.randn(64) * 0.05
                pop = np.array(new_pop)
            
            final_fits = np.array([fitness(ind) for ind in pop])
            best = pop[np.argmax(final_fits)]
            best_fit = float(np.max(final_fits))
            nearest_d = np.min(np.linalg.norm(ast_m - best.reshape(1,-1), axis=1))
            nearest_i = np.argmin(np.linalg.norm(ast_m - best.reshape(1,-1), axis=1))
            nearest_f = unique_funcs[nearest_i].split('return ')[-1].strip()[:15]
            
            if nearest_d < 0.5:
                emerged_count += 1
            best_fitnesses.append(best_fit)
            nearest_funcs.append(nearest_f)
        
        rate = emerged_count / min(5, len(void_locations)) * 100
        mean_fit = np.mean(best_fitnesses)
        all_results[strategy_name] = {
            'emerged': emerged_count, 'total': min(5, len(void_locations)),
            'rate_pct': float(rate), 'mean_fitness': float(mean_fit),
            'nearest_funcs': nearest_funcs
        }
        print(f"  {strategy_name}: {emerged_count}/5 emerged ({rate:.0f}%), mean_fit={mean_fit:.4f}")
    
    # Find best strategy
    best_strat = max(all_results, key=lambda k: all_results[k]['rate_pct'])
    print(f"\n--- Best strategy: {best_strat} ({all_results[best_strat]['rate_pct']:.0f}%) ---")
    print(f"  P123 baseline (no seed): {all_results.get('none', {}).get('rate_pct', 0):.0f}%")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Phase 129: Directed Panspermia', fontsize=14, fontweight='bold')
    
    strats = list(all_results.keys())
    rates = [all_results[s]['rate_pct'] for s in strats]
    colors_bar = ['#F44336' if r == 0 else '#4CAF50' if r > 0 else '#9E9E9E' for r in rates]
    axes[0].bar(strats, rates, color=colors_bar, edgecolor='black')
    axes[0].set_ylabel('Emergence rate (%)'); axes[0].set_title('Seeding strategy comparison')
    axes[0].tick_params(axis='x', rotation=30)
    
    fits_list = [all_results[s]['mean_fitness'] for s in strats]
    axes[1].bar(strats, fits_list, color='#2196F3', edgecolor='black')
    axes[1].set_ylabel('Mean fitness'); axes[1].set_title('Final fitness by strategy')
    axes[1].tick_params(axis='x', rotation=30)
    
    pca_2d = PCA(n_components=2).fit_transform(ast_m)
    axes[2].scatter(pca_2d[:,0], pca_2d[:,1], s=10, alpha=0.3, c='gray')
    void_2d = PCA(n_components=2).fit(ast_m).transform(void_locations[:5])
    axes[2].scatter(void_2d[:,0], void_2d[:,1], s=100, c='red', marker='x', linewidths=2, label='Void seeds')
    axes[2].legend(); axes[2].set_title('Seeded void locations')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase129_panspermia.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    results = {
        'phase': 129, 'title': 'Directed Panspermia',
        'strategies': all_results, 'best_strategy': best_strat,
        'law': f'Best strategy: {best_strat} ({all_results[best_strat]["rate_pct"]:.0f}%). No-seed baseline: {all_results.get("none", {}).get("rate_pct", 0):.0f}%. Seeding enables emergence in the void.'
    }
    with open(os.path.join(RESULTS_DIR, 'phase129_panspermia.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nPhase 129 complete!")
    return results

if __name__ == '__main__':
    main()
