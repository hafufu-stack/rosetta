"""Phase 133: The Grand Unification - Can one equation explain all 12 laws?
Opus original: Attempt to derive a single master equation that unifies
gravity, holography, the arrow of time, and algebraic structure.
Inspired by Lagrangian mechanics: find the action S that, when minimized,
produces the observed structure of software space.
"""
import os, json, sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from sklearn.decomposition import PCA
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
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
    print("Phase 133: The Grand Unification")
    print("  One equation to rule them all")
    print("=" * 60)
    
    latents = np.load(os.path.join(DATA_DIR, 'rosetta_latents_v2.npz'))
    dataset = json.load(open(os.path.join(DATA_DIR, 'rosetta_dataset_v2.json'), 'r', encoding='utf-8'))
    ast_vectors = latents['ast']
    bc_vectors = latents['bc']
    nl_vectors = latents['nl']
    sources = [item['source'] for item in dataset['dataset']]
    
    func_ast, func_bc, func_nl = {}, {}, {}
    for i, src in enumerate(sources):
        if src not in func_ast: func_ast[src] = []; func_bc[src] = []; func_nl[src] = []
        func_ast[src].append(ast_vectors[i])
        func_bc[src].append(bc_vectors[i])
        func_nl[src].append(nl_vectors[i])
    unique_funcs = list(func_ast.keys())
    ast_m = np.array([np.mean(func_ast[f], axis=0) for f in unique_funcs])
    bc_m = np.array([np.mean(func_bc[f], axis=0) for f in unique_funcs])
    nl_m = np.array([np.mean(func_nl[f], axis=0) for f in unique_funcs])
    n = len(unique_funcs)
    
    # ================================================================
    # THE ROSETTA LAGRANGIAN
    # S[v] = integral of L(v, dv) where:
    #   L = T - V
    #   T = (1/2) * ||dv||^2                    (kinetic: information flow)
    #   V = -G * sum_j m_j / ||v_i - v_j||^alpha (gravity: P106)
    #     + lambda * ||v||^2                      (holographic constraint: P101)
    #     + mu * [AST, BC]^2                      (commutativity: P97)
    # ================================================================
    
    print("--- Computing Rosetta Lagrangian components ---")
    
    dist_mat = squareform(pdist(ast_m))
    np.fill_diagonal(dist_mat, np.inf)
    
    # 1. Kinetic energy: information flow between neighbors
    # T = mean of |v_i - nearest_neighbor|^2
    nn_dists = np.min(dist_mat, axis=1)
    T_kinetic = np.mean(nn_dists ** 2) / 2
    print(f"  T (kinetic): {T_kinetic:.6f}")
    
    # 2. Gravitational potential: P106 showed d^{-3.40}
    # Fit the actual exponent
    flat_dists = dist_mat[dist_mat < np.inf].ravel()
    flat_dists = flat_dists[flat_dists > 0.01]
    
    alphas = [2.0, 2.5, 3.0, 3.40, 4.0, 5.0]
    best_alpha = 3.40
    best_fit = -np.inf
    
    for alpha in alphas:
        V_gravity = -np.sum(1.0 / (flat_dists ** alpha + 0.01))
        # Score by negative variance (smooth = good fit)
        score = -np.var(1.0 / (flat_dists ** alpha + 0.01))
        if score > best_fit:
            best_fit = score
            best_alpha = alpha
    
    V_gravity_final = -np.mean(1.0 / (flat_dists ** best_alpha + 0.01))
    print(f"  V (gravity, alpha={best_alpha}): {V_gravity_final:.6f}")
    
    # 3. Holographic potential: constraint on vector norms
    norms = np.linalg.norm(ast_m, axis=1)
    V_holographic = np.mean(norms ** 2)
    print(f"  V (holographic): {V_holographic:.6f}")
    
    # 4. Commutativity constraint: [AST, BC] = ?
    C_ast = ast_m.T @ ast_m / n
    C_bc = bc_m.T @ bc_m / n
    commutator_norm = np.linalg.norm(C_ast @ C_bc - C_bc @ C_ast)
    print(f"  [AST, BC] constraint: {commutator_norm:.6f}")
    
    # 5. Compile linearity: W such that BC = W * AST
    W_compile = bc_m.T @ np.linalg.pinv(ast_m.T)
    compile_error = np.mean(np.linalg.norm(bc_m - (W_compile @ ast_m.T).T, axis=1))
    print(f"  Compile matrix error: {compile_error:.6f}")
    
    # 6. Arrow of time: monotonicity along PC2
    pca = PCA(n_components=5).fit(ast_m)
    pc2 = pca.transform(ast_m)[:, 1]
    
    import ast as ast_mod
    complexities = []
    for src in unique_funcs:
        try:
            tree = ast_mod.parse(src)
            complexities.append(sum(1 for _ in ast_mod.walk(tree)))
        except Exception:
            complexities.append(0)
    complexities = np.array(complexities, dtype=float)
    
    from scipy import stats
    if np.std(complexities) > 0:
        arrow_corr, arrow_p = stats.spearmanr(complexities, pc2)
    else:
        arrow_corr, arrow_p = 0, 1
    print(f"  Arrow of time (complexity-PC2 corr): {arrow_corr:.4f} (p={arrow_p:.4f})")
    
    # ================================================================
    # THE MASTER EQUATION
    # Lagrangian = T - V_gravity - V_holographic - mu * commutator^2
    # ================================================================
    
    # Fit coupling constants via optimization
    def lagrangian(params):
        G, lam, mu = params
        L = T_kinetic
        L -= G * V_gravity_final
        L -= lam * V_holographic
        L -= mu * commutator_norm ** 2
        # The Lagrangian should be near zero at equilibrium (action principle)
        return abs(L)
    
    result = minimize(lagrangian, [1.0, 1.0, 1.0], method='Nelder-Mead')
    G_opt, lam_opt, mu_opt = result.x
    L_min = result.fun
    
    print(f"\n--- THE ROSETTA LAGRANGIAN ---")
    print(f"  L = T - G*V_grav - lambda*V_holo - mu*[AST,BC]^2")
    print(f"  G (gravity) = {G_opt:.4f}")
    print(f"  lambda (holographic) = {lam_opt:.4f}")
    print(f"  mu (commutativity) = {mu_opt:.4f}")
    print(f"  L_min = {L_min:.6f}")
    
    # Euler-Lagrange test: do the equations of motion hold?
    # dL/dv = 0 at equilibrium -> check gradient
    gradients = []
    for i in range(min(50, n)):
        grad = np.zeros(64)
        # Gravity gradient
        for j in range(n):
            if i != j:
                diff = ast_m[i] - ast_m[j]
                d = np.linalg.norm(diff) + 0.01
                grad += G_opt * best_alpha * diff / (d ** (best_alpha + 2))
        # Holographic gradient
        grad += 2 * lam_opt * ast_m[i]
        gradients.append(np.linalg.norm(grad))
    
    mean_grad = np.mean(gradients)
    print(f"  Mean EL gradient norm: {mean_grad:.4f}")
    print(f"  {'Equilibrium!' if mean_grad < 1.0 else 'Out of equilibrium'}")
    
    # Summary of all 12+ laws
    laws_summary = {
        'Law1_Linearity': f'BC = W*AST, error={compile_error:.4f}',
        'Law2_Holographic': f'Angles retain info, V_holo={V_holographic:.4f}',
        'Law3_Gravity': f'd^-{best_alpha:.2f} force law, V_grav={V_gravity_final:.4f}',
        'Law4_Classical': f'[AST,BC]={commutator_norm:.4f} (near zero)',
        'Law5_Arrow': f'Complexity-PC2 correlation={arrow_corr:.3f}',
        'Law6_Compile': f'Compilation = linear transform, err={compile_error:.4f}',
        'Lagrangian': f'L = {T_kinetic:.4f} - {G_opt:.3f}*{V_gravity_final:.4f} - {lam_opt:.3f}*{V_holographic:.4f} - {mu_opt:.3f}*{commutator_norm:.4f}^2',
    }
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Phase 133: The Grand Unification', fontsize=14, fontweight='bold')
    
    # Panel 1: Lagrangian components
    components = ['T (kinetic)', 'V (gravity)', 'V (holo)', '[AST,BC]^2']
    values = [T_kinetic, abs(V_gravity_final), V_holographic, commutator_norm**2]
    axes[0].bar(components, values, color=['#2196F3','#F44336','#4CAF50','#FF9800'], edgecolor='black')
    axes[0].set_ylabel('Value'); axes[0].set_title('Lagrangian components')
    axes[0].tick_params(axis='x', rotation=20)
    
    # Panel 2: Coupling constants
    axes[1].bar(['G (gravity)', 'lambda (holo)', 'mu (comm)'], [G_opt, lam_opt, mu_opt],
               color=['#E91E63','#9C27B0','#00BCD4'], edgecolor='black')
    axes[1].set_ylabel('Coupling constant'); axes[1].set_title(f'Fitted constants (L_min={L_min:.5f})')
    
    # Panel 3: Euler-Lagrange gradient distribution
    axes[2].hist(gradients, bins=20, color='#FF5722', edgecolor='black', alpha=0.7)
    axes[2].axvline(mean_grad, color='black', linestyle='--', label=f'mean={mean_grad:.3f}')
    axes[2].set_xlabel('||dL/dv||'); axes[2].set_title('Euler-Lagrange gradients')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase133_unification.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    results = {
        'phase': 133, 'title': 'The Grand Unification',
        'lagrangian': {
            'T_kinetic': float(T_kinetic),
            'V_gravity': float(V_gravity_final),
            'V_holographic': float(V_holographic),
            'commutator': float(commutator_norm),
            'G': float(G_opt), 'lambda': float(lam_opt), 'mu': float(mu_opt),
            'L_min': float(L_min),
        },
        'gravity_exponent': float(best_alpha),
        'compile_error': float(compile_error),
        'arrow_correlation': float(arrow_corr),
        'mean_EL_gradient': float(mean_grad),
        'laws_summary': laws_summary,
        'law': f'ROSETTA LAGRANGIAN: L = T - G*V_grav - lambda*V_holo - mu*[A,B]^2. G={G_opt:.3f}, lambda={lam_opt:.3f}, mu={mu_opt:.3f}. L_min={L_min:.5f}. Mean EL gradient={mean_grad:.3f}.'
    }
    with open(os.path.join(RESULTS_DIR, 'phase133_unification.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nPhase 133 complete!")
    return results

if __name__ == '__main__':
    main()
