"""Phase 153: Bell's Inequality Violation
Are the 30.6 hidden bits local or non-local?
Compute CHSH inequality on entangled AST-BC pairs.
S <= 2 (classical) vs S = 2.82 (quantum Tsirelson bound).
"""
import os, json, sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from scipy import stats
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
    print("Phase 153: Bell's Inequality Violation")
    print("  Are the 30.6 hidden bits local or non-local?")
    print("=" * 60)

    latents = np.load(os.path.join(DATA_DIR, 'rosetta_latents_v2.npz'))
    ast_vectors = latents['ast']
    bc_vectors = latents['bc']
    dataset = json.load(open(os.path.join(DATA_DIR, 'rosetta_dataset_v2.json'), 'r', encoding='utf-8'))
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

    # CHSH inequality: S = |E(a,b) - E(a,b') + E(a',b) + E(a',b')|
    # Classical: S <= 2, Quantum: S <= 2*sqrt(2) ~ 2.828

    # Choose measurement axes (random unit vectors in 64D)
    np.random.seed(42)
    n_trials = 100
    S_values = []

    for trial in range(n_trials):
        # 4 measurement directions
        a = np.random.randn(64); a /= np.linalg.norm(a)
        a_prime = np.random.randn(64); a_prime /= np.linalg.norm(a_prime)
        b = np.random.randn(64); b /= np.linalg.norm(b)
        b_prime = np.random.randn(64); b_prime /= np.linalg.norm(b_prime)

        # Correlation function E(x,y) = mean of sign(AST.x) * sign(BC.y)
        def E(axis_a, axis_b):
            proj_a = np.sign(ast_m @ axis_a)
            proj_b = np.sign(bc_m @ axis_b)
            return np.mean(proj_a * proj_b)

        S = abs(E(a, b) - E(a, b_prime) + E(a_prime, b) + E(a_prime, b_prime))
        S_values.append(float(S))

    S_values = np.array(S_values)
    mean_S = float(np.mean(S_values))
    max_S = float(np.max(S_values))
    n_violations = int(np.sum(S_values > 2.0))

    print(f"  Mean CHSH S: {mean_S:.4f}")
    print(f"  Max S: {max_S:.4f}")
    print(f"  Bell violations (S > 2): {n_violations}/{n_trials} ({n_violations/n_trials*100:.1f}%)")
    print(f"  Tsirelson bound (2.828) reached: {'YES' if max_S > 2.828 else 'NO'}")

    # Optimal measurement: find axes that maximize S
    print("\n--- Optimized Bell Test ---")
    from scipy.optimize import minimize as sp_min

    def neg_S(params):
        params = params.reshape(4, 64)
        axes = [p / (np.linalg.norm(p) + 1e-10) for p in params]
        def E(i, j):
            return np.mean(np.sign(ast_m @ axes[i]) * np.sign(bc_m @ axes[j]))
        return -(abs(E(0, 2) - E(0, 3) + E(1, 2) + E(1, 3)))

    best_S_opt = 0
    for attempt in range(5):
        x0 = np.random.randn(4 * 64)
        res = sp_min(neg_S, x0, method='L-BFGS-B', options={'maxiter': 100})
        S_opt = -res.fun
        if S_opt > best_S_opt:
            best_S_opt = S_opt

    print(f"  Optimized S: {best_S_opt:.4f}")
    print(f"  Classical limit: 2.000")
    print(f"  Tsirelson bound: 2.828")

    violation = best_S_opt > 2.0
    print(f"\n  BELL INEQUALITY VIOLATED: {'YES!' if violation else 'NO'}")
    if violation:
        print(f"  -> Hidden variables are NON-LOCAL (no local hidden variable theory)")
        print(f"  -> The 30.6 bits are entangled across the ENTIRE space")
    else:
        print(f"  -> Hidden variables could be local")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Phase 153: Bell's Inequality Violation", fontsize=14, fontweight='bold')

    axes[0].hist(S_values, bins=30, color='#E91E63', edgecolor='black', alpha=0.7)
    axes[0].axvline(2.0, color='black', linewidth=2, linestyle='--', label='Classical limit (S=2)')
    axes[0].axvline(2.828, color='blue', linewidth=2, linestyle=':', label='Tsirelson (S=2.83)')
    axes[0].axvline(best_S_opt, color='red', linewidth=2, label=f'Best S={best_S_opt:.3f}')
    axes[0].set_xlabel('CHSH S value'); axes[0].legend(fontsize=7)
    axes[0].set_title(f'Bell Test: {n_violations}/{n_trials} violations')

    axes[1].bar(['Classical\nlimit', 'Mean S', 'Max S', 'Optimized S', 'Tsirelson\nbound'],
               [2.0, mean_S, max_S, best_S_opt, 2.828],
               color=['gray', '#2196F3', '#FF9800', '#F44336', '#9C27B0'], edgecolor='black')
    axes[1].axhline(2.0, color='black', linestyle='--'); axes[1].set_title('S values')

    axes[2].text(0.5, 0.6, 'Hidden Variables\nare', ha='center', va='center', fontsize=16, transform=axes[2].transAxes)
    axes[2].text(0.5, 0.35, 'NON-LOCAL' if violation else 'LOCAL', ha='center', va='center',
                fontsize=28, fontweight='bold', color='#F44336' if violation else '#4CAF50', transform=axes[2].transAxes)
    axes[2].text(0.5, 0.12, f'S = {best_S_opt:.3f} {">" if violation else "<="} 2', ha='center', va='center',
                fontsize=14, transform=axes[2].transAxes)
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase153_bell.png'), dpi=150, bbox_inches='tight')
    plt.close()

    results = {
        'phase': 153, 'title': "Bell's Inequality Violation",
        'mean_S': mean_S, 'max_S': max_S, 'optimized_S': float(best_S_opt),
        'n_violations': n_violations, 'n_trials': n_trials,
        'bell_violated': bool(violation),
        'law': f'CHSH S={best_S_opt:.3f}. Bell violated: {violation}. {n_violations}/{n_trials} random violations. Hidden variables are {"NON-LOCAL" if violation else "local"}.'
    }
    with open(os.path.join(RESULTS_DIR, 'phase153_bell.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nPhase 153 complete!")
    return results

if __name__ == '__main__':
    main()
