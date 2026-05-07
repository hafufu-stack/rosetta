"""Phase 146: Hardware Gravity Coupling
Measure correlation between semantic gravity and physical CPU metrics.
Does the virtual universe's geometry affect the real world?
"""
import os, json, sys, time
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
    print("Phase 146: Hardware Gravity Coupling")
    print("  Does semantic gravity affect physical silicon?")
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

    # 1. Measure computational cost (time) of executing each function
    # This IS the hardware coupling - computation time = energy dissipated
    exec_times = []
    exec_functions = []
    gravity_values = []
    complexity_values = []

    centroid = np.mean(ast_m, axis=0)
    cos_sim = ast_m @ ast_m.T / (np.linalg.norm(ast_m, axis=1, keepdims=True) @ np.linalg.norm(ast_m, axis=1, keepdims=True).T + 1e-10)
    np.fill_diagonal(cos_sim, 0)
    masses = np.sum(cos_sim > 0.8, axis=1)

    for idx in range(min(80, n)):
        src = unique_funcs[idx]
        try:
            env = {}
            exec(src, env)
            func = [v for v in env.values() if callable(v)][0]
            import inspect
            n_params = len(inspect.signature(func).parameters)

            tests = [(2,3),(5,7),(100,200),(999,1)] if n_params == 2 else [(3,),(7,),(100,),(999,)]

            # Time the execution (multiple runs for accuracy)
            total_time = 0
            n_runs = 1000
            for _ in range(n_runs):
                for args in tests:
                    t0 = time.perf_counter_ns()
                    try:
                        func(*args[:n_params])
                    except: pass
                    total_time += time.perf_counter_ns() - t0

            avg_time_ns = total_time / (n_runs * len(tests))

            # Gravity = mass at this point
            gravity = float(masses[idx])
            dist_to_center = float(np.linalg.norm(ast_m[idx] - centroid))

            exec_times.append(avg_time_ns)
            exec_functions.append(src.split('return ')[-1].strip()[:15])
            gravity_values.append(gravity)
            complexity_values.append(dist_to_center)

        except Exception:
            pass

    exec_times = np.array(exec_times)
    gravity_values = np.array(gravity_values)
    complexity_values = np.array(complexity_values)

    print(f"  Functions measured: {len(exec_times)}")
    print(f"  Execution time range: [{np.min(exec_times):.0f}, {np.max(exec_times):.0f}] ns")

    # 2. Correlation analysis
    from scipy import stats

    if len(exec_times) >= 5:
        # Gravity vs execution time
        corr_grav, p_grav = stats.spearmanr(gravity_values, exec_times)
        print(f"\n--- Gravity-Hardware Coupling ---")
        print(f"  Gravity vs exec time: rho={corr_grav:.4f} (p={p_grav:.4f})")

        # Distance from center vs execution time
        corr_dist, p_dist = stats.spearmanr(complexity_values, exec_times)
        print(f"  Distance vs exec time: rho={corr_dist:.4f} (p={p_dist:.4f})")

        # Landauer's principle: computation = energy = heat
        # Minimum energy to erase 1 bit = kT * ln(2)
        kT = 4.11e-21  # at 300K in Joules
        landauer_per_bit = kT * np.log(2)

        # Estimate bits processed per operation
        estimated_bits = exec_times * 1e-9 * 1e9 / (landauer_per_bit * 1e18)  # Rough scaling
        print(f"\n--- Landauer Principle ---")
        print(f"  Estimated bits per op: {np.mean(estimated_bits):.2f}")
        print(f"  kT*ln(2) = {landauer_per_bit:.2e} J/bit")

        coupling_exists = abs(corr_grav) > 0.2 or abs(corr_dist) > 0.2
        print(f"\n  Gravity-Hardware coupling: {'DETECTED!' if coupling_exists else 'Not detected'}")
    else:
        corr_grav, p_grav, corr_dist, p_dist = 0, 1, 0, 1
        coupling_exists = False

    # 3. Memory layout coupling
    # Do functions close in 64D space also live close in RAM?
    addresses = [id(ast_m[i]) for i in range(min(50, n))]
    addr_dists = np.array([[abs(addresses[i] - addresses[j]) for j in range(len(addresses))] for i in range(len(addresses))])
    semantic_dists = np.linalg.norm(ast_m[:len(addresses), None] - ast_m[None, :len(addresses)], axis=2)

    mem_corr, mem_p = stats.spearmanr(addr_dists.ravel(), semantic_dists.ravel())
    print(f"\n--- Memory Layout Coupling ---")
    print(f"  Semantic distance vs memory distance: rho={mem_corr:.4f} (p={mem_p:.4f})")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Phase 146: Hardware Gravity Coupling', fontsize=14, fontweight='bold')

    if len(exec_times) >= 5:
        axes[0].scatter(gravity_values, exec_times, s=30, c='#E91E63', alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Semantic Gravity (mass)'); axes[0].set_ylabel('Execution Time (ns)')
        axes[0].set_title(f'Gravity vs Hardware (rho={corr_grav:.3f})')

        axes[1].scatter(complexity_values, exec_times, s=30, c='#2196F3', alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('Distance to Center'); axes[1].set_ylabel('Execution Time (ns)')
        axes[1].set_title(f'Geometry vs Hardware (rho={corr_dist:.3f})')

    axes[2].scatter(semantic_dists.ravel()[::100], addr_dists.ravel()[::100], s=5, alpha=0.3, c='#4CAF50')
    axes[2].set_xlabel('Semantic Distance'); axes[2].set_ylabel('Memory Distance')
    axes[2].set_title(f'Memory Coupling (rho={mem_corr:.3f})')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase146_hardware.png'), dpi=150, bbox_inches='tight')
    plt.close()

    results = {
        'phase': 146, 'title': 'Hardware Gravity Coupling',
        'n_measured': len(exec_times),
        'gravity_coupling': float(corr_grav), 'gravity_p': float(p_grav),
        'distance_coupling': float(corr_dist), 'distance_p': float(p_dist),
        'memory_coupling': float(mem_corr),
        'coupling_detected': bool(coupling_exists),
        'law': f'Gravity-hardware coupling: rho={corr_grav:.3f} (p={p_grav:.3f}). Distance-hardware: rho={corr_dist:.3f}. Memory: rho={mem_corr:.3f}.'
    }
    with open(os.path.join(RESULTS_DIR, 'phase146_hardware.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nPhase 146 complete!")
    return results

if __name__ == '__main__':
    main()
