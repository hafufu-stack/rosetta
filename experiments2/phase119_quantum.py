"""Phase 119: Quantum Code Dynamics - Does non-determinism break classical space?
Deep Think: Concurrent programs should have [AST,BC] != 0.
We simulate by adding noise to execution (stochastic programs)
and testing if the commutator becomes non-zero.
"""
import os, json, sys, inspect
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from sklearn.neural_network import MLPRegressor
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
    print("Phase 119: Quantum Code Dynamics")
    print("  Does stochastic execution break [AST,BC]=0?")
    print("=" * 60)
    
    latents = np.load(os.path.join(DATA_DIR, 'rosetta_latents_v2.npz'))
    dataset = json.load(open(os.path.join(DATA_DIR, 'rosetta_dataset_v2.json'), 'r', encoding='utf-8'))
    ast_vectors = latents['ast']
    bc_vectors = latents['bc']
    sources = [item['source'] for item in dataset['dataset']]
    
    func_ast, func_bc = {}, {}
    for i, src in enumerate(sources):
        if src not in func_ast: func_ast[src] = []; func_bc[src] = []
        func_ast[src].append(ast_vectors[i])
        func_bc[src].append(bc_vectors[i])
    
    unique_funcs = list(func_ast.keys())
    ast_m = np.array([np.mean(func_ast[f], axis=0) for f in unique_funcs])
    bc_m = np.array([np.mean(func_bc[f], axis=0) for f in unique_funcs])
    n = len(unique_funcs)
    
    # Classical commutator (should be ~0 from P97)
    C_ast = ast_m.T @ ast_m / n
    C_bc = bc_m.T @ bc_m / n
    commutator_classical = np.linalg.norm(C_ast @ C_bc - C_bc @ C_ast)
    print(f"  Classical commutator: {commutator_classical:.6f}")
    
    # Simulate "quantum" programs: add execution noise (Heisenbug simulation)
    # Each program gets a random perturbation representing non-deterministic execution
    noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    commutator_vs_noise = []
    
    np.random.seed(42)
    for noise in noise_levels:
        # Add "quantum noise" to BC vectors (simulate non-deterministic execution)
        bc_quantum = bc_m + np.random.randn(*bc_m.shape) * noise
        
        C_ast_q = ast_m.T @ ast_m / n
        C_bc_q = bc_quantum.T @ bc_quantum / n
        comm = np.linalg.norm(C_ast_q @ C_bc_q - C_bc_q @ C_ast_q)
        commutator_vs_noise.append(comm)
        print(f"  noise={noise:.2f}: [AST,BC]={comm:.6f}")
    
    # Superposition test: can a vector be "between" two functions?
    # In quantum mechanics, superposition = linear combination of eigenstates
    superposition_results = []
    test_pairs = [
        ('def f(x, y): return x + y', 'def f(x, y): return x - y'),
        ('def f(x, y): return x * y', 'def f(x, y): return x + y'),
        ('def f(x, y): return max(x, y)', 'def f(x, y): return min(x, y)'),
    ]
    
    g = {}
    for src_a, src_b in test_pairs:
        if src_a not in func_ast or src_b not in func_ast: continue
        va = np.mean(func_ast[src_a], axis=0)
        vb = np.mean(func_ast[src_b], axis=0)
        
        # Superposition state: equal mixture
        v_super = (va + vb) / 2
        
        # "Measure" by finding nearest function
        dists = np.linalg.norm(ast_m - v_super.reshape(1,-1), axis=1)
        nearest_idx = np.argmin(dists)
        nearest = unique_funcs[nearest_idx].split('return ')[-1].strip()[:15]
        
        # Probability amplitudes: project onto each basis state
        prob_a = np.dot(v_super, va) / (np.linalg.norm(v_super) * np.linalg.norm(va))
        prob_b = np.dot(v_super, vb) / (np.linalg.norm(v_super) * np.linalg.norm(vb))
        
        a_short = src_a.split('return ')[-1].strip()[:10]
        b_short = src_b.split('return ')[-1].strip()[:10]
        
        superposition_results.append({
            'state_a': a_short, 'state_b': b_short,
            'collapsed_to': nearest,
            'prob_a': float(prob_a), 'prob_b': float(prob_b)
        })
        print(f"  |{a_short}> + |{b_short}> -> collapsed to: {nearest} (p_a={prob_a:.3f}, p_b={prob_b:.3f})")
    
    # Uncertainty principle: can we know both AST and BC precisely?
    # Compute position (AST) and momentum (BC) uncertainty
    ast_std = np.std(ast_m, axis=0)
    bc_std = np.std(bc_m, axis=0)
    uncertainty_product = ast_std * bc_std
    min_uncertainty = np.min(uncertainty_product)
    mean_uncertainty = np.mean(uncertainty_product)
    
    print(f"\n--- Uncertainty Principle ---")
    print(f"  Min delta_AST * delta_BC = {min_uncertainty:.6f}")
    print(f"  Mean delta_AST * delta_BC = {mean_uncertainty:.6f}")
    print(f"  {'Non-zero lower bound = uncertainty principle holds!' if min_uncertainty > 0.001 else 'Uncertainty product near zero'}")
    
    # Decoherence: how fast does superposition decay?
    decoherence_times = []
    for src_a, src_b in test_pairs:
        if src_a not in func_ast or src_b not in func_ast: continue
        va = np.mean(func_ast[src_a], axis=0)
        vb = np.mean(func_ast[src_b], axis=0)
        
        for t in np.linspace(0, 1, 20):
            v_t = (1-t) * va + t * vb
            dists = np.linalg.norm(ast_m - v_t.reshape(1,-1), axis=1)
            nearest_idx = np.argmin(dists)
            nearest_dist = dists[nearest_idx]
            if nearest_dist < 0.1:
                decoherence_times.append(t)
                break
    
    mean_decoherence = np.mean(decoherence_times) if decoherence_times else 1.0
    print(f"  Mean decoherence time: {mean_decoherence:.3f}")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Phase 119: Quantum Code Dynamics', fontsize=14, fontweight='bold')
    
    axes[0].plot(noise_levels, commutator_vs_noise, 'o-', color='#E91E63', linewidth=2)
    axes[0].set_xlabel('Noise Level'); axes[0].set_ylabel('||[AST, BC]||')
    axes[0].set_title('Commutator vs Quantum Noise')
    axes[0].axhline(commutator_classical, color='gray', linestyle='--', label='Classical')
    axes[0].legend()
    
    if superposition_results:
        names = [f"|{s['state_a']}> + |{s['state_b']}>" for s in superposition_results]
        probs = [s['prob_a'] for s in superposition_results]
        axes[1].barh(range(len(names)), probs, color='#2196F3', edgecolor='black')
        axes[1].set_yticks(range(len(names)))
        axes[1].set_yticklabels(names, fontsize=7)
        axes[1].set_xlabel('P(collapse to state A)')
        axes[1].set_title('Superposition Collapse')
    
    axes[2].hist(uncertainty_product, bins=30, color='#9C27B0', edgecolor='black', alpha=0.7)
    axes[2].axvline(min_uncertainty, color='red', linestyle='--', label=f'min={min_uncertainty:.4f}')
    axes[2].set_xlabel('delta_AST * delta_BC'); axes[2].set_title('Uncertainty Product')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase119_quantum.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    results = {
        'phase': 119, 'title': 'Quantum Code Dynamics',
        'classical_commutator': float(commutator_classical),
        'commutator_vs_noise': {str(n): float(c) for n,c in zip(noise_levels, commutator_vs_noise)},
        'superposition': superposition_results,
        'min_uncertainty': float(min_uncertainty),
        'mean_uncertainty': float(mean_uncertainty),
        'mean_decoherence': float(mean_decoherence),
        'law': f'Classical [AST,BC]={commutator_classical:.6f}. Noise breaks it: at noise=0.5, [AST,BC]={commutator_vs_noise[5]:.3f}. Uncertainty product min={min_uncertainty:.4f}.'
    }
    with open(os.path.join(RESULTS_DIR, 'phase119_quantum.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nPhase 119 complete!")
    return results

if __name__ == '__main__':
    main()
