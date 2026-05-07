"""Phase 103: Simulated Annealing Code Synthesis - Classical thermodynamics for bug repair.
P97 proved the space is classical. P94 failed because GD gets stuck in local minima.
Solution: Simulated Annealing (thermal fluctuations to escape local minima).
"""
import os, json, sys, inspect, math
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from sklearn.neural_network import MLPRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXP2_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(EXP2_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(EXP2_DIR, 'results')
FIGURES_DIR = os.path.join(EXP2_DIR, 'figures')

def main():
    print("=" * 60)
    print("Phase 103: Simulated Annealing Code Synthesis")
    print("  Classical thermodynamics for bug repair")
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
    
    # Build 64D Neural CPU
    print("Building 64D Neural CPU...")
    exec_data = []
    g = {}
    for func_src in unique_funcs:
        vec = func_means[func_src]
        try:
            exec(compile(func_src, '<string>', 'exec'), g)
            fn = g['f']
            try: n_args = len(inspect.signature(fn).parameters)
            except: n_args = 2
            for x_val in [-2, -1, 0, 1, 2, 3, 5]:
                for y_val in [-2, -1, 0, 1, 2, 3, 5]:
                    try:
                        result = fn(x_val) if n_args == 1 else fn(x_val, y_val)
                        if isinstance(result, (int, float, bool)) and abs(float(result)) < 1e6:
                            features = np.concatenate([vec, [x_val, y_val]])
                            exec_data.append((features, float(result)))
                    except: pass
        except: pass
    
    X_cpu = np.array([d[0] for d in exec_data])
    y_cpu = np.array([d[1] for d in exec_data])
    cpu = MLPRegressor(hidden_layer_sizes=(256, 128, 64), max_iter=2000, random_state=42,
                       early_stopping=True, validation_fraction=0.1, learning_rate_init=0.001)
    cpu.fit(X_cpu, y_cpu)
    cpu_r2 = cpu.score(X_cpu, y_cpu)
    print(f"  Neural CPU R2: {cpu_r2:.4f}")
    
    mins = all_vecs.min(axis=0)
    maxs = all_vecs.max(axis=0)
    spread = maxs - mins
    
    def energy(vec, target_io):
        """Energy = mean squared error of Neural CPU predictions."""
        total = 0.0
        for (x, y), expected in target_io:
            features = np.concatenate([vec, [x, y]])
            pred = cpu.predict(features.reshape(1, -1))[0]
            total += (pred - expected) ** 2
        return total / len(target_io)
    
    def simulated_annealing(start_vec, target_io, T_init=2.0, T_min=0.001,
                            cooling=0.995, max_steps=3000):
        """Classical SA: accept worse states with probability exp(-dE/T)."""
        current = start_vec.copy()
        current_e = energy(current, target_io)
        best = current.copy()
        best_e = current_e
        
        T = T_init
        history = [current_e]
        temp_history = [T]
        accept_count = 0
        
        for step in range(max_steps):
            # Generate neighbor (random perturbation)
            perturbation = np.random.randn(64) * spread * 0.02 * (T / T_init)
            candidate = current + perturbation
            candidate = np.clip(candidate, mins - spread * 0.2, maxs + spread * 0.2)
            
            candidate_e = energy(candidate, target_io)
            dE = candidate_e - current_e
            
            # Metropolis criterion
            if dE < 0:
                # Better: always accept
                current = candidate
                current_e = candidate_e
                accept_count += 1
            elif T > 0 and np.random.rand() < math.exp(-dE / T):
                # Worse but accept with thermal probability
                current = candidate
                current_e = candidate_e
                accept_count += 1
            
            if current_e < best_e:
                best = current.copy()
                best_e = current_e
            
            T *= cooling
            T = max(T, T_min)
            history.append(best_e)
            temp_history.append(T)
        
        return best, best_e, history, temp_history, accept_count
    
    # Bug scenarios
    bug_scenarios = [
        ('def f(x, y): return x + y', 'def f(x, y): return x - y', 'add->sub'),
        ('def f(x, y): return x - y', 'def f(x, y): return x + y', 'sub->add'),
        ('def f(x, y): return x * y', 'def f(x, y): return x + y', 'mul->add'),
        ('def f(x, y): return x + y', 'def f(x, y): return x * y', 'add->mul'),
        ('def f(x, y): return max(x, y)', 'def f(x, y): return min(x, y)', 'max->min'),
        ('def f(x, y): return x > y', 'def f(x, y): return x < y', 'gt->lt'),
        ('def f(x, y): return x ** y', 'def f(x, y): return x * y', 'pow->mul'),
    ]
    
    test_inputs = [(1,2),(3,5),(-1,4),(2,2),(0,7),(4,3),(-2,1)]
    
    repair_results = []
    all_curves = []
    
    for buggy_src, correct_src, bug_name in bug_scenarios:
        if buggy_src not in func_means or correct_src not in func_means:
            continue
        
        buggy_vec = func_means[buggy_src].copy()
        correct_vec = func_means[correct_src].copy()
        
        g2 = {}
        exec(compile(correct_src, '<string>', 'exec'), g2)
        expected = []
        for x, y in test_inputs:
            try: expected.append(float(g2['f'](x, y)))
            except: expected.append(0.0)
        target_io = [((x,y), e) for (x,y), e in zip(test_inputs, expected)]
        
        # Run SA
        best_vec, best_e, history, temps, accepts = simulated_annealing(
            buggy_vec, target_io, T_init=2.0, cooling=0.995, max_steps=3000)
        
        # Find nearest function
        dists = np.linalg.norm(all_vecs - best_vec.reshape(1, -1), axis=1)
        nearest_idx = np.argmin(dists)
        nearest_func = unique_funcs[nearest_idx]
        
        orig_dist = np.linalg.norm(buggy_vec - correct_vec)
        final_dist = np.linalg.norm(best_vec - correct_vec)
        improvement = 1.0 - (final_dist / orig_dist) if orig_dist > 0 else 0
        success = nearest_func == correct_src
        
        initial_e = history[0]
        nearest_short = nearest_func.split('return ')[-1].strip() if 'return' in nearest_func else nearest_func[-25:]
        
        result = {
            'bug': bug_name,
            'initial_energy': float(initial_e),
            'final_energy': float(best_e),
            'energy_reduction': float(1 - best_e / initial_e) if initial_e > 0 else 0,
            'dist_improvement': float(improvement),
            'accepts': accepts,
            'nearest': nearest_short,
            'repaired': success
        }
        repair_results.append(result)
        all_curves.append(history)
        
        print(f"\n  [{bug_name}]")
        print(f"    Energy: {initial_e:.2f} -> {best_e:.2f} ({result['energy_reduction']:.1%} reduction)")
        print(f"    Dist: {orig_dist:.4f} -> {final_dist:.4f} ({improvement:.1%} closer)")
        print(f"    Accepts: {accepts}/3000")
        print(f"    Nearest: {nearest_short} {'(REPAIRED!)' if success else ''}")
    
    n_repaired = sum(1 for r in repair_results if r['repaired'])
    repair_rate = n_repaired / len(repair_results) if repair_results else 0
    mean_e_red = np.mean([r['energy_reduction'] for r in repair_results])
    
    print(f"\n{'='*60}")
    print(f"SIMULATED ANNEALING: {n_repaired}/{len(repair_results)} ({repair_rate:.0%})")
    print(f"P88(5D GD):0/6 | P92(64D GD):0/7 | P94(Dual):0/7 | P103(SA):{n_repaired}/{len(repair_results)}")
    print(f"Mean energy reduction: {mean_e_red:.1%}")
    print(f"{'='*60}")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Phase 103: Simulated Annealing Code Synthesis', fontsize=14, fontweight='bold')
    
    colors = ['#E91E63','#2196F3','#4CAF50','#FF9800','#9C27B0','#00BCD4','#795548']
    for i, (curve, r) in enumerate(zip(all_curves, repair_results)):
        label = r['bug'][:10]
        c = colors[i % len(colors)]
        ls = '-' if r['repaired'] else '--'
        axes[0].plot(curve[:500], color=c, label=label, linewidth=1.5, linestyle=ls)
    axes[0].set_xlabel('SA Step'); axes[0].set_ylabel('Best Energy')
    axes[0].set_title('SA Energy Curves'); axes[0].legend(fontsize=7)
    
    methods = ['P88\n5D GD', 'P92\n64D GD', 'P94\nDual', 'P103\nSA']
    rates = [0, 0, 0, repair_rate*100]
    axes[1].bar(methods, rates, color=['#F44336','#F44336','#FF9800','#4CAF50'], edgecolor='black')
    axes[1].set_ylabel('Repair Rate (%)'); axes[1].set_title('SA vs Previous Methods')
    for i, v in enumerate(rates):
        axes[1].text(i, v+2, f'{v:.0f}%', ha='center', fontweight='bold')
    
    e_reds = [r['energy_reduction']*100 for r in repair_results]
    bug_names = [r['bug'][:10] for r in repair_results]
    bc = ['#4CAF50' if r['repaired'] else '#FF9800' for r in repair_results]
    axes[2].bar(range(len(e_reds)), e_reds, color=bc, edgecolor='black')
    axes[2].set_xticks(range(len(e_reds)))
    axes[2].set_xticklabels(bug_names, rotation=45, ha='right', fontsize=7)
    axes[2].set_ylabel('Energy Reduction (%)'); axes[2].set_title(f'Per-Bug: {n_repaired}/{len(repair_results)}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase103_annealing.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    results = {
        'phase': 103, 'title': 'Simulated Annealing Code Synthesis',
        'cpu_r2': float(cpu_r2),
        'n_bugs': len(repair_results), 'n_repaired': n_repaired,
        'repair_rate': float(repair_rate),
        'mean_energy_reduction': float(mean_e_red),
        'repairs': repair_results,
        'law': f'SA repairs {n_repaired}/{len(repair_results)} bugs. Classical thermodynamics (P97) enables thermal escape from local minima. Mean energy reduction: {mean_e_red:.1%}.'
    }
    with open(os.path.join(RESULTS_DIR, 'phase103_annealing.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nPhase 103 complete!")
    return results

if __name__ == '__main__':
    main()
