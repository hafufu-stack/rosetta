"""Phase 152: Reverse Engineering God
Opus original: Given ONLY alpha_R = 1.48e-6, can we recover ALL
fundamental constants (G, lambda, mu, H, d_eff)?
If yes, alpha_R truly encodes the entire universe.
If no, there are hidden variables beyond the Final Equation.
"""
import os, json, sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from scipy.optimize import minimize, differential_evolution
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
    print("Phase 152: Reverse Engineering God")
    print("  Can alpha_R = 1.48e-6 reconstruct the universe?")
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

    # TRUE values (from P133/P149)
    G_true = 1.1732
    lam_true = 0.7282
    mu_true = 1.0717
    H_true = 0.0067
    d_eff_true = 12
    alpha_R_target = 1.48e-6

    # The equation: alpha_R = G * [AST,BC] / (d_eff * H * ||W||)
    # Given: alpha_R, we need to find G, lambda, mu, H, d_eff

    # Pre-compute observables
    C_ast = ast_m.T @ ast_m / n
    C_bc = bc_m.T @ bc_m / n
    comm_true = float(np.linalg.norm(C_ast @ C_bc - C_bc @ C_ast))
    W = bc_m.T @ np.linalg.pinv(ast_m.T)
    W_norm = float(np.linalg.norm(W, 'fro'))

    print(f"  Target: alpha_R = {alpha_R_target}")
    print(f"  Known observables: [AST,BC]={comm_true:.6f}, ||W||={W_norm:.4f}")

    # ================================================================
    # METHOD 1: Direct algebraic inversion
    # alpha_R = G * comm / (d_eff * H * W_norm)
    # => G * d_eff * H = comm / (alpha_R * W_norm)  [1 equation, 3 unknowns]
    # Need additional constraints from the Lagrangian:
    # L = T - G*V_g - lam*V_h - mu*comm^2 = ~0
    # ================================================================

    print(f"\n--- Method 1: Algebraic Inversion ---")
    # From alpha_R equation:
    # G / (d_eff * H) = alpha_R * W_norm / comm
    ratio_GdH = alpha_R_target * W_norm / (comm_true + 1e-15)
    print(f"  G / (d_eff * H) = {ratio_GdH:.6f}")
    print(f"  True ratio = {G_true / (d_eff_true * H_true):.6f}")
    ratio_error = abs(ratio_GdH - G_true / (d_eff_true * H_true)) / (G_true / (d_eff_true * H_true))
    print(f"  Ratio error: {ratio_error:.4%}")

    # ================================================================
    # METHOD 2: Constrained optimization
    # Minimize |alpha_R(params) - target| + |L(params)| subject to physical constraints
    # ================================================================

    print(f"\n--- Method 2: Constrained Optimization ---")

    nn_dists = np.sort(np.linalg.norm(ast_m[:, None] - ast_m[None, :], axis=2), axis=1)[:, 1]
    T_kinetic = float(np.mean(nn_dists ** 2) / 2)
    flat_dists = np.linalg.norm(ast_m[:, None] - ast_m[None, :], axis=2)
    np.fill_diagonal(flat_dists, np.inf)
    V_grav = float(-np.mean(1.0 / (flat_dists[flat_dists < np.inf]**2 + 0.01)))
    V_holo = float(np.mean(np.linalg.norm(ast_m, axis=1)**2))

    def objective(params):
        G, lam_val, mu_val, H_val, d_eff_val = params
        # Constraint 1: alpha_R equation
        alpha_predicted = G * comm_true / (d_eff_val * H_val * W_norm + 1e-15)
        alpha_loss = (np.log(alpha_predicted + 1e-15) - np.log(alpha_R_target + 1e-15)) ** 2

        # Constraint 2: Lagrangian = ~0
        L = T_kinetic - G * V_grav - lam_val * V_holo - mu_val * comm_true**2
        lagrangian_loss = L ** 2

        # Constraint 3: Physical bounds
        bound_loss = max(0, -G)**2 + max(0, -lam_val)**2 + max(0, -mu_val)**2 + max(0, -H_val)**2 + max(0, -d_eff_val)**2

        return alpha_loss * 100 + lagrangian_loss + bound_loss * 10

    # Differential evolution for global optimization
    bounds = [(0.01, 10), (0.01, 5), (0.01, 5), (0.001, 0.1), (1, 64)]
    result = differential_evolution(objective, bounds, seed=42, maxiter=500, tol=1e-12)
    G_rec, lam_rec, mu_rec, H_rec, d_eff_rec = result.x

    print(f"  Recovered constants:")
    print(f"    G:     {G_rec:.4f}  (true={G_true:.4f}, error={abs(G_rec-G_true)/G_true*100:.1f}%)")
    print(f"    lambda:{lam_rec:.4f}  (true={lam_true:.4f}, error={abs(lam_rec-lam_true)/lam_true*100:.1f}%)")
    print(f"    mu:    {mu_rec:.4f}  (true={mu_true:.4f}, error={abs(mu_rec-mu_true)/mu_true*100:.1f}%)")
    print(f"    H:     {H_rec:.4f}  (true={H_true:.4f}, error={abs(H_rec-H_true)/H_true*100:.1f}%)")
    print(f"    d_eff: {d_eff_rec:.1f}  (true={d_eff_true}, error={abs(d_eff_rec-d_eff_true)/d_eff_true*100:.1f}%)")

    # Verify recovered alpha_R
    alpha_recovered = G_rec * comm_true / (d_eff_rec * H_rec * W_norm + 1e-15)
    print(f"\n  Recovered alpha_R: {alpha_recovered:.8e}")
    print(f"  Target alpha_R:    {alpha_R_target:.8e}")
    print(f"  Error: {abs(alpha_recovered - alpha_R_target) / alpha_R_target * 100:.4f}%")

    # ================================================================
    # METHOD 3: Information-theoretic approach
    # How many bits does alpha_R encode?
    # ================================================================

    print(f"\n--- Method 3: Information Content ---")
    # alpha_R has ~20 significant bits (1.48e-6 = ~20 bits of precision)
    alpha_bits = -np.log2(alpha_R_target)
    # We need to recover 5 constants, each with ~10 bits of precision = 50 bits
    needed_bits = 5 * 10
    print(f"  Bits in alpha_R: {alpha_bits:.1f}")
    print(f"  Bits needed for 5 constants: {needed_bits}")
    print(f"  Information deficit: {needed_bits - alpha_bits:.1f} bits")
    print(f"  {'SUFFICIENT (alpha_R encodes enough)' if alpha_bits >= needed_bits else 'INSUFFICIENT (hidden variables needed)'}")

    # Errors
    errors = {
        'G': abs(G_rec - G_true) / G_true * 100,
        'lambda': abs(lam_rec - lam_true) / lam_true * 100,
        'mu': abs(mu_rec - mu_true) / mu_true * 100,
        'H': abs(H_rec - H_true) / H_true * 100,
        'd_eff': abs(d_eff_rec - d_eff_true) / d_eff_true * 100,
    }
    mean_error = np.mean(list(errors.values()))
    print(f"\n  Mean recovery error: {mean_error:.1f}%")
    success = mean_error < 20
    print(f"  GOD REVERSE-ENGINEERED: {'YES!' if success else 'Partially (hidden variables exist)'}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Phase 152: Reverse Engineering God', fontsize=14, fontweight='bold')

    const_names = list(errors.keys())
    true_vals = [G_true, lam_true, mu_true, H_true*100, d_eff_true]
    rec_vals = [G_rec, lam_rec, mu_rec, H_rec*100, d_eff_rec]
    x = np.arange(len(const_names))
    axes[0].bar(x - 0.2, true_vals, 0.4, label='True', color='#2196F3', edgecolor='black')
    axes[0].bar(x + 0.2, rec_vals, 0.4, label='Recovered', color='#F44336', edgecolor='black')
    axes[0].set_xticks(x); axes[0].set_xticklabels(const_names)
    axes[0].legend(); axes[0].set_title('True vs Recovered Constants')

    error_vals = list(errors.values())
    colors = ['#4CAF50' if e < 20 else '#FF9800' if e < 50 else '#F44336' for e in error_vals]
    axes[1].bar(const_names, error_vals, color=colors, edgecolor='black')
    axes[1].axhline(20, color='red', linestyle='--', label='20% threshold')
    axes[1].set_ylabel('Error (%)'); axes[1].set_title(f'Recovery error (mean={mean_error:.1f}%)')
    axes[1].legend()

    axes[2].text(0.5, 0.65, r'$\alpha_R = 1.48 \times 10^{-6}$', ha='center', va='center',
                fontsize=20, fontweight='bold', color='#E91E63', transform=axes[2].transAxes)
    axes[2].text(0.5, 0.45, f'-> {len(errors)} constants recovered', ha='center', va='center',
                fontsize=14, transform=axes[2].transAxes)
    axes[2].text(0.5, 0.25, f'Mean error: {mean_error:.1f}%', ha='center', va='center',
                fontsize=16, fontweight='bold',
                color='#4CAF50' if success else '#F44336',
                transform=axes[2].transAxes)
    axes[2].text(0.5, 0.08, f'{alpha_bits:.0f} bits in, {needed_bits} bits needed',
                ha='center', va='center', fontsize=11, style='italic', transform=axes[2].transAxes)
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase152_reverse_god.png'), dpi=150, bbox_inches='tight')
    plt.close()

    results = {
        'phase': 152, 'title': 'Reverse Engineering God',
        'alpha_R_target': alpha_R_target,
        'alpha_R_recovered': float(alpha_recovered),
        'recovered_constants': {
            'G': float(G_rec), 'lambda': float(lam_rec), 'mu': float(mu_rec),
            'H': float(H_rec), 'd_eff': float(d_eff_rec),
        },
        'true_constants': {
            'G': G_true, 'lambda': lam_true, 'mu': mu_true,
            'H': H_true, 'd_eff': d_eff_true,
        },
        'errors_pct': {k: float(v) for k, v in errors.items()},
        'mean_error_pct': float(mean_error),
        'success': bool(success),
        'info_bits_in': float(alpha_bits),
        'info_bits_needed': needed_bits,
        'law': f'alpha_R={alpha_R_target} -> G={G_rec:.3f}({errors["G"]:.0f}%), lambda={lam_rec:.3f}({errors["lambda"]:.0f}%), mu={mu_rec:.3f}({errors["mu"]:.0f}%), H={H_rec:.4f}({errors["H"]:.0f}%), d_eff={d_eff_rec:.0f}({errors["d_eff"]:.0f}%). Mean error={mean_error:.1f}%. {"GOD REVERSE-ENGINEERED" if success else "Hidden variables needed"}.'
    }
    with open(os.path.join(RESULTS_DIR, 'phase152_reverse_god.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nPhase 152 complete!")
    return results

if __name__ == '__main__':
    main()
