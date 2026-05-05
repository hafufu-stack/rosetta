"""
Phase 35: Evolutionary Programming in Latent Space
=====================================================
Evolve programs WITHOUT writing text.
Start from random vectors, mutate, select by Neural CPU fitness.
Can evolution discover "multiply by 2" from scratch?
"""
import os, json, time, sys
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    print("=" * 60)
    print("Phase 35: Evolutionary Programming in Latent Space")
    print("Programs that evolve without being written")
    print("=" * 60)
    t0 = time.time()

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

    # Load decoder
    sys.path.insert(0, BASE_DIR)
    with open(os.path.join(DATA_DIR, 'char_vocab.json'), 'r') as f:
        vd = json.load(f)
    idx2char = {int(i): c for c, i in vd['char2idx'].items()}
    V = len(vd['char2idx'])
    from experiments.phase9_generative_decompiler import CodeDecoder, decode_tokens

    decoder = CodeDecoder(64, V, hidden=128, max_len=80).to(DEVICE)
    dec_path = os.path.join(DATA_DIR, 'decoder_robust.pt')
    if not os.path.exists(dec_path):
        dec_path = os.path.join(DATA_DIR, 'decoder.pt')
    decoder.load_state_dict(torch.load(dec_path, map_location=DEVICE, weights_only=True))
    decoder.eval()

    def gen(z):
        with torch.no_grad():
            z_t = torch.tensor(z.astype(np.float32)).unsqueeze(0).to(DEVICE)
            tokens = decoder(z_t)
            return decode_tokens(tokens[0].cpu().numpy(), idx2char)

    def evaluate_func(code_str, test_cases):
        """Run code and return fitness (higher = better)."""
        try:
            ns = {}
            exec(code_str, ns)
            if 'f' not in ns:
                return 0.0
            f = ns['f']
            score = 0
            for args, expected in test_cases:
                try:
                    result = f(*args)
                    if result == expected:
                        score += 1.0
                    elif isinstance(expected, (int, float)):
                        # Partial credit for closeness
                        score += max(0, 1.0 - abs(result - expected) / (abs(expected) + 1))
                except:
                    pass
            return score / len(test_cases)
        except:
            return 0.0

    # === Evolution targets ===
    targets = [
        {
            'name': 'double(x) = x * 2',
            'tests': [((3,), 6), ((5,), 10), ((0,), 0), ((-2,), -4), ((10,), 20)],
        },
        {
            'name': 'square(x) = x * x',
            'tests': [((3,), 9), ((5,), 25), ((0,), 0), ((-2,), 4), ((4,), 16)],
        },
        {
            'name': 'add(x, y) = x + y',
            'tests': [((3, 5), 8), ((0, 0), 0), ((-1, 1), 0), ((10, 7), 17), ((2, 3), 5)],
        },
        {
            'name': 'negate(x) = -x',
            'tests': [((5,), -5), ((-3,), 3), ((0,), 0), ((1,), -1), ((100,), -100)],
        },
    ]

    # Evolutionary parameters
    POP_SIZE = 50
    N_GENS = 100
    MUTATION_RATE = 0.3
    ELITE = 5

    # Distribution statistics of real programs (for initialization)
    ast_mean = z_ast.mean(axis=0)
    ast_std = z_ast.std(axis=0)

    evolution_results = []

    for target in targets:
        print(f"\n--- Evolving: {target['name']} ---")
        tests = target['tests']

        # Initialize population near the manifold
        population = np.random.randn(POP_SIZE, 64).astype(np.float32) * ast_std + ast_mean

        best_fitness_history = []
        best_code_history = []
        best_ever_fitness = 0
        best_ever_code = ""

        for gen_i in range(N_GENS):
            # Evaluate fitness
            fitness = np.zeros(POP_SIZE)
            codes = []
            for pi in range(POP_SIZE):
                code = gen(population[pi])
                codes.append(code)
                fitness[pi] = evaluate_func(code, tests)

            # Track best
            best_idx = np.argmax(fitness)
            if fitness[best_idx] > best_ever_fitness:
                best_ever_fitness = fitness[best_idx]
                best_ever_code = codes[best_idx]

            best_fitness_history.append(float(fitness[best_idx]))

            if (gen_i+1) % 25 == 0 or fitness[best_idx] >= 1.0:
                print(f"  Gen {gen_i+1}: best={fitness[best_idx]:.2f} "
                      f"avg={fitness.mean():.2f} -> {codes[best_idx][:40]}")

            if fitness[best_idx] >= 1.0:
                print(f"  ** PERFECT SOLUTION FOUND at gen {gen_i+1}! **")
                break

            # Selection (tournament)
            new_pop = []
            # Keep elite
            elite_idx = np.argsort(fitness)[::-1][:ELITE]
            for ei in elite_idx:
                new_pop.append(population[ei].copy())

            # Fill rest via tournament + mutation
            while len(new_pop) < POP_SIZE:
                # Tournament selection
                t1, t2 = np.random.randint(0, POP_SIZE, 2)
                parent = population[t1] if fitness[t1] > fitness[t2] else population[t2]

                # Crossover with another parent
                t3, t4 = np.random.randint(0, POP_SIZE, 2)
                parent2 = population[t3] if fitness[t3] > fitness[t4] else population[t4]
                mask = np.random.rand(64) > 0.5
                child = np.where(mask, parent, parent2)

                # Mutation
                if np.random.rand() < MUTATION_RATE:
                    mutation = np.random.randn(64).astype(np.float32) * ast_std * 0.1
                    child = child + mutation

                new_pop.append(child)

            population = np.array(new_pop[:POP_SIZE])

        print(f"  Best ever: fitness={best_ever_fitness:.2f}, code={best_ever_code[:50]}")

        evolution_results.append({
            'target': target['name'],
            'best_fitness': float(best_ever_fitness),
            'best_code': best_ever_code,
            'n_gens': len(best_fitness_history),
            'fitness_history': best_fitness_history,
            'solved': bool(best_ever_fitness >= 1.0),
        })

    n_solved = sum(1 for r in evolution_results if r['solved'])
    print(f"\n  Solved: {n_solved}/{len(targets)}")

    elapsed = time.time() - t0
    results = {
        'phase': 35, 'name': 'Evolutionary Programming in Latent Space',
        'n_solved': n_solved, 'n_targets': len(targets),
        'pop_size': POP_SIZE, 'n_gens': N_GENS,
        'evolution': evolution_results,
        'elapsed': elapsed, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase35_evolution.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n_t = len(evolution_results)
    fig, axes = plt.subplots(1, n_t, figsize=(5*n_t, 4))
    if n_t == 1:
        axes = [axes]
    for ti, er in enumerate(evolution_results):
        ax = axes[ti]
        ax.plot(er['fitness_history'], 'b-', linewidth=2)
        ax.axhline(1.0, color='green', ls='--', alpha=0.5, label='Perfect')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Best Fitness')
        status = 'SOLVED!' if er['solved'] else f"best={er['best_fitness']:.2f}"
        ax.set_title(f"{er['target'][:20]}\n{status}", fontweight='bold',
                    color='green' if er['solved'] else 'red')
        ax.set_ylim(-0.1, 1.1)
        ax.legend(fontsize=8)

    plt.suptitle('Phase 35: Evolutionary Programming in Latent Space\n'
                 f'Programs evolved WITHOUT text: {n_solved}/{n_t} solved',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase35_evolution.png'), dpi=150)
    plt.close()
    print(f"\nPhase 35 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
