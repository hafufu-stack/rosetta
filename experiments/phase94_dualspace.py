"""Phase 94: Dual-Space Navigation - 5D macro warp + 64D micro landing.
The Dimensionality Hierarchy says: meaning=5D, computation=64D.
Use 5D for coarse navigation, then 64D GD for precise I/O matching.
"""
import os, json, sys, inspect
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import NearestNeighbors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(EXPERIMENT_DIR)
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')

def main():
    print("=" * 60)
    print("Phase 94: Dual-Space Navigation")
    print("  5D Macro Warp + 64D Micro Landing")
    print("=" * 60)
    
    latents = np.load(os.path.join(BASE_DIR, 'data', 'rosetta_latents_v2.npz'))
    dataset = json.load(open(os.path.join(BASE_DIR, 'data', 'rosetta_dataset_v2.json'), 'r', encoding='utf-8'))
    ast_vectors = latents['ast']
    sources = [item['source'] for item in dataset['dataset']]
    
    func_to_vecs = {}
    for i, src in enumerate(sources):
        if src not in func_to_vecs:
            func_to_vecs[src] = []
        func_to_vecs[src].append(ast_vectors[i])
    func_means = {src: np.mean(vecs, axis=0) for src, vecs in func_to_vecs.items()}
    unique_funcs = list(func_means.keys())
    
    all_vecs_64d = np.array([func_means[f] for f in unique_funcs])
    
    # PCA to 5D
    pca = PCA(n_components=5)
    all_vecs_5d = pca.fit_transform(all_vecs_64d)
    func_5d = {f: all_vecs_5d[i] for i, f in enumerate(unique_funcs)}
    
    # Build 5D NN index for macro warp
    nn_5d = NearestNeighbors(n_neighbors=10, metric='euclidean')
    nn_5d.fit(all_vecs_5d)
    
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
                            exec_data.append((features, float(result), func_src))
                    except: pass
        except: pass
    
    X_cpu = np.array([d[0] for d in exec_data])
    y_cpu = np.array([d[1] for d in exec_data])
    cpu = MLPRegressor(hidden_layer_sizes=(256, 128, 64), max_iter=2000, random_state=42,
                       early_stopping=True, validation_fraction=0.1, learning_rate_init=0.001)
    cpu.fit(X_cpu, y_cpu)
    cpu_r2 = cpu.score(X_cpu, y_cpu)
    print(f"  64D Neural CPU R2: {cpu_r2:.4f}")
    
    mins = all_vecs_64d.min(axis=0)
    maxs = all_vecs_64d.max(axis=0)
    spread = maxs - mins
    
    def io_fitness_64d(vec, target_io):
        total = 0.0
        for (x, y), expected in target_io:
            features = np.concatenate([vec, [x, y]])
            pred = cpu.predict(features.reshape(1, -1))[0]
            total += (pred - expected) ** 2
        return total / len(target_io)
    
    def gd_64d(vec, target_io, steps=100, lr=0.0005):
        current = vec.copy()
        energies = []
        for _ in range(steps):
            grad = np.zeros(64)
            e = 0.0
            for (x_val, y_val), expected in target_io:
                features = np.concatenate([current, [x_val, y_val]])
                pred = cpu.predict(features.reshape(1, -1))[0]
                error = pred - expected
                e += error**2
                for d in range(64):
                    p = current.copy()
                    eps = max(1e-4, abs(current[d]) * 1e-4)
                    p[d] += eps
                    fp = np.concatenate([p, [x_val, y_val]])
                    pp = cpu.predict(fp.reshape(1, -1))[0]
                    grad[d] += 2 * error * (pp - pred) / eps
            energies.append(e / len(target_io))
            gn = np.linalg.norm(grad)
            if gn > 5.0: grad = grad * 5.0 / gn
            current -= lr * grad / len(target_io)
            current = np.clip(current, mins - spread * 0.2, maxs + spread * 0.2)
            if _ % 30 == 29: lr *= 0.7
        return current, energies
    
    # Bug scenarios
    bug_scenarios = [
        ('def f(x, y): return x + y', 'def f(x, y): return x - y', 'add->sub'),
        ('def f(x, y): return x - y', 'def f(x, y): return x + y', 'sub->add'),
        ('def f(x, y): return x * y', 'def f(x, y): return x + y', 'mul->add'),
        ('def f(x, y): return x + y', 'def f(x, y): return x * y', 'add->mul'),
        ('def f(x, y): return max(x, y)', 'def f(x, y): return min(x, y)', 'max->min'),
        ('def f(x, y): return x > y', 'def f(x, y): return x < y', 'gt->lt'),
        ('def f(x, y): return x ** y', 'def f(x, y): return x * y', 'pow->mul'),
        ('def f(x, y): return x // y', 'def f(x, y): return x % y', 'div->mod'),
    ]
    
    test_inputs = [(1,2),(3,5),(-1,4),(2,2),(0,7),(4,3),(-2,1)]
    
    repair_results = []
    
    for buggy_src, correct_src, bug_name in bug_scenarios:
        if buggy_src not in func_means or correct_src not in func_means:
            continue
        
        buggy_64d = func_means[buggy_src].copy()
        correct_64d = func_means[correct_src].copy()
        buggy_5d = func_5d[buggy_src].copy()
        correct_5d = func_5d[correct_src].copy()
        
        # Get expected I/O
        g2 = {}
        exec(compile(correct_src, '<string>', 'exec'), g2)
        expected = []
        for x, y in test_inputs:
            try: expected.append(float(g2['f'](x, y)))
            except: expected.append(0.0)
        target_io = [((x,y), e) for (x,y), e in zip(test_inputs, expected)]
        
        # === STEP 1: 5D MACRO WARP ===
        # Find nearest semantic neighbors in 5D
        dists_5d, indices_5d = nn_5d.kneighbors(buggy_5d.reshape(1, -1))
        
        # Score each 5D neighbor by I/O fitness in 64D
        best_warp_score = float('inf')
        best_warp_vec = buggy_64d.copy()
        
        for idx in indices_5d[0]:
            candidate_func = unique_funcs[idx]
            candidate_64d = func_means[candidate_func]
            score = io_fitness_64d(candidate_64d, target_io)
            if score < best_warp_score:
                best_warp_score = score
                best_warp_vec = candidate_64d.copy()
        
        warp_dist = np.linalg.norm(best_warp_vec - correct_64d)
        orig_dist = np.linalg.norm(buggy_64d - correct_64d)
        
        # === STEP 2: 64D MICRO LANDING (GD) ===
        landed_vec, energy_curve = gd_64d(best_warp_vec, target_io, steps=200)
        
        # Find nearest function
        all_dists = np.linalg.norm(all_vecs_64d - landed_vec.reshape(1, -1), axis=1)
        nearest_idx = np.argmin(all_dists)
        nearest_func = unique_funcs[nearest_idx]
        
        final_dist = np.linalg.norm(landed_vec - correct_64d)
        improvement = 1.0 - (final_dist / orig_dist) if orig_dist > 0 else 0
        success = nearest_func == correct_src
        
        nearest_short = nearest_func.split('return ')[-1].strip() if 'return' in nearest_func else nearest_func[-25:]
        
        result = {
            'bug': bug_name,
            'initial_dist': float(orig_dist),
            'after_warp_dist': float(warp_dist),
            'final_dist': float(final_dist),
            'improvement': float(improvement),
            'initial_energy': float(energy_curve[0]) if energy_curve else 0,
            'final_energy': float(energy_curve[-1]) if energy_curve else 0,
            'nearest': nearest_short,
            'repaired': success
        }
        repair_results.append(result)
        
        print(f"\n  [{bug_name}]")
        print(f"    Dist: {orig_dist:.4f} -> warp:{warp_dist:.4f} -> land:{final_dist:.4f} ({improvement:.1%} closer)")
        print(f"    Energy: {energy_curve[0]:.2f} -> {energy_curve[-1]:.2f}")
        print(f"    Nearest: {nearest_short} {'(REPAIRED!)' if success else ''}")
    
    n_repaired = sum(1 for r in repair_results if r['repaired'])
    repair_rate = n_repaired / len(repair_results) if repair_results else 0
    mean_imp = np.mean([r['improvement'] for r in repair_results])
    
    print(f"\n{'='*60}")
    print(f"DUAL-SPACE NAVIGATION: {n_repaired}/{len(repair_results)} ({repair_rate:.0%})")
    print(f"P88 (5D GD): 0/6 | P92 (64D GD): 0/7 | P94 (Dual): {n_repaired}/{len(repair_results)}")
    print(f"Mean improvement: {mean_imp:.1%}")
    print(f"{'='*60}")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Phase 94: Dual-Space Navigation (5D Warp + 64D Landing)', fontsize=14, fontweight='bold')
    
    bugs = [r['bug'] for r in repair_results]
    d0 = [r['initial_dist'] for r in repair_results]
    d1 = [r['after_warp_dist'] for r in repair_results]
    d2 = [r['final_dist'] for r in repair_results]
    x_pos = range(len(bugs))
    
    axes[0].bar([i-0.25 for i in x_pos], d0, 0.25, color='#F44336', label='Original', alpha=0.8)
    axes[0].bar([i for i in x_pos], d1, 0.25, color='#FF9800', label='After 5D Warp', alpha=0.8)
    axes[0].bar([i+0.25 for i in x_pos], d2, 0.25, color='#4CAF50', label='After 64D Land', alpha=0.8)
    axes[0].set_xticks(list(x_pos)); axes[0].set_xticklabels(bugs, rotation=45, ha='right', fontsize=7)
    axes[0].set_ylabel('Distance to Correct'); axes[0].set_title('3-Stage Distance Reduction')
    axes[0].legend(fontsize=7)
    
    methods = ['P88\n5D GD', 'P92\n64D GD', 'P94\nDual']
    rates_c = [0, 0, repair_rate*100]
    axes[1].bar(methods, rates_c, color=['#F44336','#FF9800','#4CAF50'], edgecolor='black')
    axes[1].set_ylabel('Repair Rate (%)'); axes[1].set_title('Dual-Space vs Single-Space')
    for i, v in enumerate(rates_c):
        axes[1].text(i, v+2, f'{v:.0f}%', ha='center', fontweight='bold')
    
    imps = [r['improvement']*100 for r in repair_results]
    bc = ['#4CAF50' if r['repaired'] else '#FF9800' for r in repair_results]
    axes[2].bar(x_pos, imps, color=bc, edgecolor='black')
    axes[2].set_xticks(list(x_pos)); axes[2].set_xticklabels(bugs, rotation=45, ha='right', fontsize=7)
    axes[2].set_ylabel('Improvement (%)'); axes[2].set_title(f'Per-Bug: {n_repaired}/{len(repair_results)}')
    axes[2].axhline(100, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase94_dualspace.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    results = {
        'phase': 94, 'title': 'Dual-Space Navigation (5D Warp + 64D Landing)',
        'cpu_r2': float(cpu_r2), 'n_bugs': len(repair_results),
        'n_repaired': n_repaired, 'repair_rate': float(repair_rate),
        'mean_improvement': float(mean_imp),
        'p88_rate': 0.0, 'p92_rate': 0.0,
        'repairs': repair_results,
        'law': f'Dual-space navigation: 5D semantic warp + 64D computational landing repairs {n_repaired}/{len(repair_results)} bugs.'
    }
    with open(os.path.join(RESULTS_DIR, 'phase94_dualspace.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nPhase 94 complete!")
    return results

if __name__ == '__main__':
    main()
