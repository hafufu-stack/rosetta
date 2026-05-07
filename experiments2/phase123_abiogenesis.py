"""Phase 123: Abiogenesis in the Void - Can life emerge from dark matter?
Deep Think: Run GA in the deepest voids to see if structured programs emerge.
Extended from P110 with long-term evolution.
"""
import os, json, sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXP2_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(EXP2_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')

sys.path.insert(0, os.path.join(BASE_DIR, 'experiments'))
try:
    from phase9_generative_decompiler import RosettaDecoder
except ImportError:
    RosettaDecoder = None

def main():
    print("=" * 60)
    print("Phase 123: Abiogenesis in the Void")
    print("  Can structured programs emerge from dark matter?")
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
    
    # Find the deepest void (most isolated point in 64D)
    knn = NearestNeighbors(n_neighbors=1).fit(ast_m)
    
    # Random probes to find voids
    np.random.seed(42)
    n_probes = 500
    bbox_min = ast_m.min(axis=0)
    bbox_max = ast_m.max(axis=0)
    
    probes = np.random.uniform(bbox_min, bbox_max, size=(n_probes, 64))
    distances, _ = knn.kneighbors(probes)
    distances = distances.ravel()
    
    # Select the 20 deepest void locations
    void_indices = np.argsort(distances)[-20:]
    void_locations = probes[void_indices]
    void_depths = distances[void_indices]
    
    print(f"  Deepest void depth: {void_depths[-1]:.4f}")
    print(f"  Mean void depth (top 20): {np.mean(void_depths):.4f}")
    
    # Genetic Algorithm in the void
    pop_size = 30
    n_generations = 200
    mutation_rate = 0.1
    
    # Fitness = how close the evolved vector is to any known program
    def fitness(v):
        dist = np.min(np.linalg.norm(ast_m - v.reshape(1,-1), axis=1))
        return 1.0 / (1.0 + dist)
    
    # Run evolution from each void
    genesis_results = []
    all_fitness_histories = []
    
    for void_id in range(min(10, len(void_locations))):
        origin = void_locations[void_id]
        initial_depth = void_depths[void_id]
        
        # Initialize population around the void center
        pop = origin + np.random.randn(pop_size, 64) * 0.1
        best_fitness_history = []
        
        for gen in range(n_generations):
            fitnesses = np.array([fitness(ind) for ind in pop])
            best_idx = np.argmax(fitnesses)
            best_fitness_history.append(float(fitnesses[best_idx]))
            
            # Selection (tournament)
            new_pop = []
            for _ in range(pop_size):
                i, j = np.random.randint(0, pop_size, 2)
                winner = pop[i] if fitnesses[i] > fitnesses[j] else pop[j]
                new_pop.append(winner.copy())
            
            # Crossover
            for i in range(0, pop_size-1, 2):
                if np.random.random() < 0.7:
                    alpha = np.random.random()
                    child1 = alpha * new_pop[i] + (1-alpha) * new_pop[i+1]
                    child2 = (1-alpha) * new_pop[i] + alpha * new_pop[i+1]
                    new_pop[i] = child1
                    new_pop[i+1] = child2
            
            # Mutation
            for i in range(pop_size):
                if np.random.random() < mutation_rate:
                    new_pop[i] += np.random.randn(64) * 0.05
            
            pop = np.array(new_pop)
        
        # Final result
        final_fitnesses = np.array([fitness(ind) for ind in pop])
        best_final = pop[np.argmax(final_fitnesses)]
        best_fit = float(np.max(final_fitnesses))
        
        nearest_dist = np.min(np.linalg.norm(ast_m - best_final.reshape(1,-1), axis=1))
        nearest_idx = np.argmin(np.linalg.norm(ast_m - best_final.reshape(1,-1), axis=1))
        nearest_func = unique_funcs[nearest_idx].split('return ')[-1].strip()[:20]
        
        emerged = nearest_dist < 0.5  # Close enough to be a "real" program
        
        genesis_results.append({
            'void_id': void_id,
            'initial_depth': float(initial_depth),
            'final_fitness': best_fit,
            'nearest_distance': float(nearest_dist),
            'nearest_function': nearest_func,
            'emerged': bool(emerged),
            'generations': n_generations,
        })
        
        all_fitness_histories.append(best_fitness_history)
        
        status = "EMERGED!" if emerged else "still in void"
        print(f"  Void {void_id}: depth={initial_depth:.3f} -> fitness={best_fit:.4f}, nearest='{nearest_func}' (d={nearest_dist:.3f}) [{status}]")
    
    # Summary
    n_emerged = sum(1 for r in genesis_results if r['emerged'])
    total = len(genesis_results)
    print(f"\n--- Abiogenesis Rate: {n_emerged}/{total} ({n_emerged/total*100:.0f}%) ---")
    
    # Self-replication test: can the emerged programs "reproduce"?
    # (evolve copies that are close to the original)
    replication_scores = []
    for result in genesis_results:
        if result['emerged']:
            target_idx = unique_funcs.index([f for f in unique_funcs 
                if f.split('return ')[-1].strip()[:20] == result['nearest_function']][0])
            target_v = ast_m[target_idx]
            
            # Can evolution find the same program again from a nearby void?
            offspring = target_v + np.random.randn(10, 64) * 0.3
            offspring_fits = [fitness(o) for o in offspring]
            replication_scores.append(float(np.mean(offspring_fits)))
    
    mean_replication = np.mean(replication_scores) if replication_scores else 0
    print(f"  Mean replication fitness: {mean_replication:.4f}")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Phase 123: Abiogenesis in the Void', fontsize=14, fontweight='bold')
    
    for i, hist in enumerate(all_fitness_histories[:10]):
        color = '#4CAF50' if genesis_results[i]['emerged'] else '#9E9E9E'
        axes[0].plot(hist, color=color, alpha=0.7, linewidth=1.5)
    axes[0].set_xlabel('Generation'); axes[0].set_ylabel('Best Fitness')
    axes[0].set_title(f'Evolution from void ({n_emerged}/{total} emerged)')
    
    depths = [r['initial_depth'] for r in genesis_results]
    fits = [r['final_fitness'] for r in genesis_results]
    colors = ['#4CAF50' if r['emerged'] else '#F44336' for r in genesis_results]
    axes[1].scatter(depths, fits, c=colors, s=80, edgecolor='black', zorder=5)
    axes[1].set_xlabel('Void depth'); axes[1].set_ylabel('Final fitness')
    axes[1].set_title('Depth vs Emergence')
    
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2).fit(ast_m)
    ast_2d = pca.transform(ast_m)
    axes[2].scatter(ast_2d[:,0], ast_2d[:,1], s=10, alpha=0.3, c='gray', label='Known')
    
    void_2d = pca.transform(void_locations[:10])
    axes[2].scatter(void_2d[:,0], void_2d[:,1], s=100, c='red', marker='x',
                   linewidths=2, label='Void origins', zorder=5)
    axes[2].legend(fontsize=8)
    axes[2].set_title('Void locations in PC1-PC2')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase123_abiogenesis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    results = {
        'phase': 123, 'title': 'Abiogenesis in the Void',
        'n_voids_tested': total,
        'n_emerged': n_emerged,
        'abiogenesis_rate_pct': float(n_emerged/total*100),
        'mean_replication_fitness': float(mean_replication),
        'genesis_details': genesis_results,
        'law': f'Abiogenesis rate: {n_emerged}/{total} ({n_emerged/total*100:.0f}%). Mean replication fitness: {mean_replication:.4f}.'
    }
    with open(os.path.join(RESULTS_DIR, 'phase123_abiogenesis.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nPhase 123 complete!")
    return results

if __name__ == '__main__':
    main()
