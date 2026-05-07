"""Phase 93: Memetic Singularity Engine - GA + Gradient Descent hybrid.
P91 (GA only): 1/7, P92 (GD only): 0/7.
Memetic = GA for global jumps + GD for local refinement each generation.
"""
import os, json, sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from sklearn.neural_network import MLPRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(EXPERIMENT_DIR)
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')

def main():
    print("=" * 60)
    print("Phase 93: Memetic Singularity Engine (GA + GD)")
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
    
    # Build 64D Neural CPU
    print("Building 64D Neural CPU...")
    import inspect
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
    print(f"  Neural CPU R2: {cpu_r2:.4f}")
    print(f"  Samples: {len(exec_data)}")
    
    func_list = list(func_means.keys())
    func_vecs = np.array([func_means[f] for f in func_list])
    mins = func_vecs.min(axis=0)
    maxs = func_vecs.max(axis=0)
    spread = maxs - mins
    
    def decode(vec):
        dists = np.linalg.norm(func_vecs - vec.reshape(1, -1), axis=1)
        idx = np.argmin(dists)
        return func_list[idx], dists[idx]
    
    def virtual_execute(vec, x, y):
        features = np.concatenate([vec, [x, y]])
        return cpu.predict(features.reshape(1, -1))[0]
    
    def fitness(vec, target_io):
        total = 0.0
        for (x, y), expected in target_io:
            pred = virtual_execute(vec, x, y)
            total += (pred - expected) ** 2
        return -total / len(target_io)
    
    def local_gd(vec, target_io, steps=10, lr=0.0003):
        """Local gradient descent refinement (the 'meme' part)."""
        current = vec.copy()
        for _ in range(steps):
            grad = np.zeros(64)
            for (x_val, y_val), expected in target_io:
                features = np.concatenate([current, [x_val, y_val]])
                pred = cpu.predict(features.reshape(1, -1))[0]
                error = pred - expected
                for d in range(64):
                    p = current.copy()
                    eps = max(1e-4, abs(current[d]) * 1e-4)
                    p[d] += eps
                    fp = np.concatenate([p, [x_val, y_val]])
                    pred_p = cpu.predict(fp.reshape(1, -1))[0]
                    grad[d] += 2 * error * (pred_p - pred) / eps
            
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 5.0:
                grad = grad * 5.0 / grad_norm
            current -= lr * grad / len(target_io)
            current = np.clip(current, mins - spread * 0.2, maxs + spread * 0.2)
        return current
    
    def memetic_evolve(target_io, pop_size=80, generations=100, gd_steps=5):
        """Memetic algorithm: GA + local GD each generation."""
        population = []
        seed_idx = np.random.choice(len(func_list), min(20, len(func_list)), replace=False)
        for idx in seed_idx:
            population.append(func_vecs[idx] + np.random.randn(64) * spread * 0.03)
        while len(population) < pop_size:
            population.append(mins + np.random.rand(64) * spread)
        
        best_history = []
        
        for gen in range(generations):
            # Evaluate
            scores = [(fitness(ind, target_io), ind) for ind in population]
            scores.sort(key=lambda x: x[0], reverse=True)
            
            best_fit = scores[0][0]
            best_ind = scores[0][1]
            best_func, _ = decode(best_ind)
            best_name = best_func.split('return ')[-1].strip() if 'return' in best_func else best_func[-25:]
            best_history.append(best_fit)
            
            if gen % 20 == 0 or gen == generations - 1:
                print(f"    Gen {gen:3d}: fitness={best_fit:.4f}, decoded={best_name}")
            
            if best_fit > -0.01:
                best_history.extend([best_fit] * (generations - gen - 1))
                break
            
            # Selection
            survivors = [s[1] for s in scores[:pop_size // 5]]
            
            # LOCAL GD on top individuals (Memetic step!)
            refined = []
            for s in survivors[:5]:
                refined.append(local_gd(s, target_io, steps=gd_steps))
            survivors[:5] = refined
            
            # Reproduce
            new_pop = survivors.copy()
            while len(new_pop) < pop_size:
                p1 = survivors[np.random.randint(len(survivors))]
                p2 = survivors[np.random.randint(len(survivors))]
                alpha = np.random.rand(64)
                child = alpha * p1 + (1 - alpha) * p2
                if np.random.rand() < 0.3:
                    child += np.random.randn(64) * spread * 0.05
                    child = np.clip(child, mins - spread * 0.1, maxs + spread * 0.1)
                new_pop.append(child)
            population = new_pop
        
        return best_ind, best_func, best_history
    
    # Targets
    targets = [
        ("x+y", [((1,2),3),((3,5),8),((-1,4),3),((0,0),0),((10,-3),7)]),
        ("x-y", [((5,3),2),((10,4),6),((0,1),-1),((7,7),0),((3,8),-5)]),
        ("x*y", [((2,3),6),((4,5),20),((-1,3),-3),((0,7),0),((1,1),1)]),
        ("max(x,y)", [((3,5),5),((7,2),7),((-1,-3),-1),((0,0),0),((4,4),4)]),
        ("abs(x-y)", [((5,3),2),((3,5),2),((-1,4),5),((7,7),0),((0,3),3)]),
        ("x//y", [((10,3),3),((7,2),3),((15,5),3),((0,1),0),((9,3),3)]),
        ("x%y", [((10,3),1),((7,2),1),((15,5),0),((9,4),1),((8,3),2)]),
    ]
    
    print("\n--- MEMETIC SINGULARITY ENGINE ---")
    results_list = []
    all_curves = []
    
    for name, io in targets:
        print(f"\n  Target: {name}")
        best_vec, best_func, curve = memetic_evolve(io)
        best_short = best_func.split('return ')[-1].strip() if 'return' in best_func else best_func[-25:]
        
        try:
            g2 = {}
            exec(compile(best_func, '<string>', 'exec'), g2)
            fn2 = g2['f']
            try: na = len(inspect.signature(fn2).parameters)
            except: na = 2
            correct = 0
            for (x,y), exp in io:
                actual = fn2(x) if na == 1 else fn2(x, y)
                if isinstance(actual, (int, float, bool)) and abs(float(actual) - exp) < 0.5:
                    correct += 1
            acc = correct / len(io)
        except:
            acc = 0.0
        
        print(f"  -> Discovered: {best_short} (acc={acc:.0%})")
        results_list.append({'target': name, 'discovered': best_short, 'accuracy': float(acc),
                             'final_fitness': float(curve[-1])})
        all_curves.append(curve)
    
    n_ok = sum(1 for r in results_list if r['accuracy'] >= 0.8)
    rate = n_ok / len(results_list)
    
    print(f"\n{'='*60}")
    print(f"MEMETIC ENGINE: {n_ok}/{len(results_list)} ({rate:.0%})")
    print(f"P89 (5D GA): 0/5 | P91 (64D GA): 1/7 | P93 (Memetic): {n_ok}/{len(results_list)}")
    print(f"{'='*60}")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Phase 93: Memetic Singularity Engine (GA + GD)', fontsize=14, fontweight='bold')
    
    colors = ['#E91E63','#2196F3','#4CAF50','#FF9800','#9C27B0','#00BCD4','#795548']
    for i, (curve, r) in enumerate(zip(all_curves, results_list)):
        axes[0].plot(curve, color=colors[i%len(colors)], label=r['target'][:10], linewidth=1.5)
    axes[0].set_xlabel('Generation'); axes[0].set_ylabel('Fitness')
    axes[0].set_title('Memetic Evolution Curves'); axes[0].legend(fontsize=7)
    
    methods = ['P89\n5D GA', 'P91\n64D GA', 'P93\nMemetic']
    rates_cmp = [0, 14.3, rate*100]
    axes[1].bar(methods, rates_cmp, color=['#F44336','#FF9800','#4CAF50'], edgecolor='black')
    axes[1].set_ylabel('Discovery Rate (%)'); axes[1].set_title('GA vs GA+GD')
    for i, v in enumerate(rates_cmp):
        axes[1].text(i, v+2, f'{v:.0f}%', ha='center', fontweight='bold')
    
    accs = [r['accuracy']*100 for r in results_list]
    names = [r['target'][:10] for r in results_list]
    bc = ['#4CAF50' if a>=80 else '#F44336' for a in accs]
    axes[2].bar(range(len(accs)), accs, color=bc, edgecolor='black')
    axes[2].set_xticks(range(len(accs))); axes[2].set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    axes[2].set_ylabel('Accuracy (%)'); axes[2].set_title(f'Per-Target: {n_ok}/{len(results_list)}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase93_memetic.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    results = {
        'phase': 93, 'title': 'Memetic Singularity Engine (GA + GD)',
        'cpu_r2': float(cpu_r2), 'n_targets': len(results_list),
        'n_discovered': n_ok, 'discovery_rate': float(rate),
        'p89_rate': 0.0, 'p91_rate': 0.143,
        'targets': results_list,
        'law': f'Memetic (GA+GD) discovers {n_ok}/{len(results_list)} vs GA-only 1/7. Lamarckian evolution in latent space.'
    }
    with open(os.path.join(RESULTS_DIR, 'phase93_memetic.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nPhase 93 complete!")
    return results

if __name__ == '__main__':
    main()
