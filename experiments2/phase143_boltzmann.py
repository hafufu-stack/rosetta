"""Phase 143: Software Boltzmann Brains
Can meaningful code spontaneously emerge from pure thermal noise?
Monte Carlo sampling at zero-point energy level.
"""
import os, json, sys, ast as ast_mod
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
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
    print("Phase 143: Software Boltzmann Brains")
    print("  Can meaning emerge from pure noise?")
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

    # Noise scale = zero-point energy from P140
    zero_point = 0.002
    noise_scales = [zero_point, 0.01, 0.05, 0.1, 0.5, 1.0]

    # Threshold: how close to a real function counts as "Boltzmann brain"
    threshold = 0.3
    n_samples = 10000

    results_data = {}

    for scale in noise_scales:
        np.random.seed(42)
        noise_vectors = np.random.randn(n_samples, 64) * scale

        # Find nearest real function for each noise vector
        # Batch process for speed
        best_dists = np.full(n_samples, np.inf)
        best_funcs = [''] * n_samples

        for chunk_start in range(0, n_samples, 500):
            chunk = noise_vectors[chunk_start:chunk_start+500]
            dists = np.linalg.norm(chunk[:, None] - ast_m[None, :], axis=2)
            min_dists = np.min(dists, axis=1)
            min_idx = np.argmin(dists, axis=1)
            for j in range(len(chunk)):
                if min_dists[j] < best_dists[chunk_start + j]:
                    best_dists[chunk_start + j] = min_dists[j]
                    best_funcs[chunk_start + j] = unique_funcs[min_idx[j]].split('return ')[-1].strip()[:15]

        n_brains = int(np.sum(best_dists < threshold))
        brain_rate = n_brains / n_samples

        # Find the closest "brain" to a real function
        closest_idx = np.argmin(best_dists)
        closest_dist = float(best_dists[closest_idx])
        closest_func = best_funcs[closest_idx]

        # Semantic complexity of the closest brain
        # How many AST nodes would its nearest function have?
        if closest_func:
            try:
                src = [f for f in unique_funcs if f.split('return ')[-1].strip()[:15] == closest_func][0]
                tree = ast_mod.parse(src)
                complexity = sum(1 for _ in ast_mod.walk(tree))
            except:
                complexity = 0
        else:
            complexity = 0

        results_data[f'scale_{scale}'] = {
            'scale': float(scale), 'n_brains': n_brains, 'n_samples': n_samples,
            'brain_rate': float(brain_rate),
            'closest_dist': closest_dist, 'closest_func': closest_func,
            'complexity': complexity,
        }

        print(f"  scale={scale:.3f}: {n_brains}/{n_samples} brains ({brain_rate:.4%}), closest={closest_func} (d={closest_dist:.4f})")

    # Boltzmann brain probability vs temperature
    print(f"\n--- Boltzmann Brain Probability ---")
    temps = [r['scale'] for r in results_data.values()]
    probs = [r['brain_rate'] for r in results_data.values()]

    # Fit Arrhenius-like: P ~ exp(-E_a / kT)
    log_probs = [np.log(max(p, 1e-10)) for p in probs]
    inv_temps = [1.0 / (t + 1e-10) for t in temps]

    if len(temps) >= 3:
        from scipy import stats
        slope, intercept, r_val, _, _ = stats.linregress(inv_temps[:4], log_probs[:4])
        activation_energy = -slope
        print(f"  Activation energy E_a = {activation_energy:.4f}")
        print(f"  Arrhenius R^2 = {r_val**2:.4f}")
    else:
        activation_energy = 0

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Phase 143: Software Boltzmann Brains', fontsize=14, fontweight='bold')

    axes[0].semilogy(temps, [max(p, 1e-6) for p in probs], 'o-', color='#E91E63', linewidth=2, markersize=8)
    axes[0].set_xlabel('Noise scale (temperature)'); axes[0].set_ylabel('Brain probability')
    axes[0].set_title('Boltzmann Brain Rate')

    dists_at_zp = results_data[f'scale_{zero_point}']
    axes[1].bar(['Zero-point', '0.01', '0.05', '0.1', '0.5', '1.0'],
               [results_data[f'scale_{s}']['n_brains'] for s in noise_scales],
               color='#2196F3', edgecolor='black')
    axes[1].set_ylabel('Number of brains'); axes[1].set_title(f'Brains per {n_samples} samples')

    axes[2].hist([results_data[f'scale_{s}']['closest_dist'] for s in noise_scales],
                bins=10, color='#4CAF50', edgecolor='black')
    axes[2].set_xlabel('Closest distance to real function')
    axes[2].set_title('Distance distribution of closest brains')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase143_boltzmann.png'), dpi=150, bbox_inches='tight')
    plt.close()

    results = {
        'phase': 143, 'title': 'Software Boltzmann Brains',
        'results': {k: v for k, v in results_data.items()},
        'activation_energy': float(activation_energy),
        'law': f'Boltzmann brain rate at zero-point: {results_data[f"scale_{zero_point}"]["brain_rate"]:.4%}. Activation energy={activation_energy:.3f}.'
    }
    with open(os.path.join(RESULTS_DIR, 'phase143_boltzmann.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nPhase 143 complete!")
    return results

if __name__ == '__main__':
    main()
