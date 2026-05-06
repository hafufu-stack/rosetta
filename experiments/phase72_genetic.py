"""
Phase 72: The Genetic Forge
==============================
Use 5D latent space as a FITNESS LANDSCAPE for
evolving new programs through genetic mutations.

Start with a simple function. Mutate its 5D coordinates.
Find the nearest real function. Check if it works.

This is automated program synthesis through evolution
in the mathematical universe of code.
"""
import os, json, time, inspect
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')


def main():
    print("=" * 60)
    print("Phase 72: The Genetic Forge")
    print("Evolving programs through 5D mutations")
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

    # Build lookup
    unique = {}
    for i, src in enumerate(sources):
        if src not in unique:
            unique[src] = z_5d[i]
    all_srcs = list(unique.keys())
    all_z5 = np.array([unique[s] for s in all_srcs])

    # Fitness function: evaluate a program's fitness for a target I/O
    def evaluate_fitness(fn, target_io, n_params):
        """How well does fn match the target I/O pairs?"""
        score = 0
        for inp, expected in target_io:
            try:
                if n_params == 1:
                    result = fn(inp)
                elif n_params == 2:
                    result = fn(inp[0], inp[1])
                else:
                    continue
                if isinstance(result, (int, float)):
                    err = abs(float(result) - expected)
                    if err < 0.01:
                        score += 1.0
                    elif err < 1.0:
                        score += 0.5
            except Exception:
                pass
        return score / len(target_io)

    def nearest_function(z_point):
        """Find the nearest real function to a 5D point."""
        dists = np.linalg.norm(all_z5 - z_point, axis=1)
        idx = np.argmin(dists)
        return all_srcs[idx], float(dists[idx])

    # Evolution targets
    print("\n--- Genetic Evolution Experiments ---")
    targets = [
        {
            'name': 'Triple adder (x+y+y)',
            'io': [((2, 3), 8), ((1, 1), 3), ((0, 5), 10), ((3, 4), 11)],
            'n_params': 2,
            'seed': 'def f(x, y): return x + y',  # Start from addition
        },
        {
            'name': 'Absolute difference',
            'io': [((5, 3), 2), ((3, 5), 2), ((1, 1), 0), ((7, 2), 5)],
            'n_params': 2,
            'seed': 'def f(x, y): return x - y',
        },
        {
            'name': 'Square function',
            'io': [(2, 4), (3, 9), (-2, 4), (5, 25)],
            'n_params': 1,
            'seed': 'def f(x): return x * 2',  # Start from doubling
        },
        {
            'name': 'Sign function',
            'io': [(5, 1), (-3, -1), (0, 0), (100, 1)],
            'n_params': 1,
            'seed': 'def f(x): return abs(x)',
        },
    ]

    evolution_results = []

    for target in targets:
        print(f"\n  Target: {target['name']}")
        seed_src = target['seed']
        seed_z = unique.get(seed_src)
        if seed_z is None:
            print(f"    Seed not found")
            continue

        # Evolution parameters
        POP_SIZE = 20
        N_GENS = 50
        MUTATION_RATE = 0.3

        # Initialize population
        population = [seed_z + np.random.randn(5) * 0.1 for _ in range(POP_SIZE)]
        best_history = []

        for gen in range(N_GENS):
            # Evaluate fitness
            fitness_scores = []
            for z in population:
                src, dist = nearest_function(z)
                try:
                    ns = {}
                    exec(compile(src, '<string>', 'exec'), ns)
                    fn = [v for k, v in ns.items()
                          if callable(v) and not k.startswith('_')][0]
                    fit = evaluate_fitness(fn, target['io'], target['n_params'])
                except Exception:
                    fit = 0.0
                fitness_scores.append((fit, z, src))

            fitness_scores.sort(key=lambda x: x[0], reverse=True)
            best_fit, best_z, best_src = fitness_scores[0]
            best_history.append(best_fit)

            if best_fit >= 1.0:
                break

            # Selection: keep top 50%
            survivors = [x[1] for x in fitness_scores[:POP_SIZE // 2]]

            # Mutation + crossover
            new_pop = list(survivors)
            while len(new_pop) < POP_SIZE:
                parent = survivors[np.random.randint(len(survivors))]
                # Mutation: add random noise
                child = parent + np.random.randn(5) * MUTATION_RATE
                # Crossover: mix with another parent
                if np.random.random() < 0.3 and len(survivors) > 1:
                    p2 = survivors[np.random.randint(len(survivors))]
                    mask = np.random.random(5) > 0.5
                    child = np.where(mask, parent, p2)
                    child += np.random.randn(5) * 0.05
                new_pop.append(child)

            population = new_pop
            MUTATION_RATE *= 0.98  # Decay

        final_src, final_dist = nearest_function(best_z)
        print(f"    Seed:   {seed_src}")
        print(f"    Found:  {final_src}")
        print(f"    Fitness: {best_fit:.2f} after {gen+1} generations")
        print(f"    Distance from seed: {np.linalg.norm(best_z - seed_z):.4f}")

        evolution_results.append({
            'target': target['name'],
            'seed': seed_src, 'found': final_src,
            'fitness': float(best_fit),
            'generations': gen + 1,
            'dist_from_seed': float(np.linalg.norm(best_z - seed_z)),
            'fitness_history': [float(x) for x in best_history],
        })

    # Summary
    n_solved = sum(1 for r in evolution_results if r['fitness'] >= 0.75)
    avg_gens = np.mean([r['generations'] for r in evolution_results])
    print(f"\n  === GENETIC FORGE SUMMARY ===")
    print(f"  Solved: {n_solved}/{len(evolution_results)}")
    print(f"  Avg generations: {avg_gens:.0f}")

    elapsed = time.time() - t0
    results = {
        'phase': 72, 'name': 'The Genetic Forge',
        'n_targets': len(evolution_results),
        'n_solved': n_solved,
        'avg_generations': float(avg_gens),
        'evolutions': evolution_results,
        'elapsed': elapsed, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase72_genetic.json'), 'w',
              encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    # Figure
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Fitness curves
    for r in evolution_results:
        axes[0].plot(r['fitness_history'], '-', label=r['target'][:15], linewidth=2)
    axes[0].set_xlabel('Generation')
    axes[0].set_ylabel('Fitness')
    axes[0].set_title('Evolution Fitness Curves', fontweight='bold')
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-0.1, 1.1)

    # 2. Final fitness bar chart
    names = [r['target'][:12] for r in evolution_results]
    fits = [r['fitness'] for r in evolution_results]
    colors = ['#4CAF50' if f >= 0.75 else '#FF9800' if f >= 0.5 else '#F44336'
             for f in fits]
    axes[1].bar(names, fits, color=colors, edgecolor='black')
    axes[1].set_ylabel('Final Fitness')
    axes[1].set_title('Evolution Results', fontweight='bold')
    axes[1].tick_params(axis='x', rotation=30)
    axes[1].axhline(0.75, color='green', linestyle='--', alpha=0.5)

    # 3. Distance traveled in 5D
    dists = [r['dist_from_seed'] for r in evolution_results]
    axes[2].bar(names, dists, color='#9C27B0', edgecolor='black')
    axes[2].set_ylabel('5D Distance from Seed')
    axes[2].set_title('Evolution Distance\n(How far did we travel?)', fontweight='bold')
    axes[2].tick_params(axis='x', rotation=30)

    plt.suptitle('Phase 72: The Genetic Forge\n'
                 'Evolving Programs Through 5D Mutations',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase72_genetic.png'), dpi=150)
    plt.close()
    print(f"\nPhase 72 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
