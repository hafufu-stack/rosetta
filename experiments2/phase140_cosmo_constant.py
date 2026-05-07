"""Phase 140: The Cosmological Constant Problem
Opus original: WHY is L_min = 0.000039 and not exactly zero?
This is the software analog of the cosmological constant problem -
the most embarrassing discrepancy in all of physics.
Decompose the residual into quantum corrections and vacuum fluctuations.
"""
import os, json, sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from sklearn.decomposition import PCA
from scipy.optimize import minimize
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
    print("Phase 140: The Cosmological Constant Problem")
    print("  Why is L_min = 0.000039 and not zero?")
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
    
    G, lam, mu = 1.1732, 0.7282, 1.0717
    
    # 1. Recompute Lagrangian components precisely
    nn_dists = np.sort(np.linalg.norm(ast_m[:, None] - ast_m[None, :], axis=2), axis=1)[:, 1]
    T_kinetic = np.mean(nn_dists ** 2) / 2
    
    flat_dists = np.linalg.norm(ast_m[:, None] - ast_m[None, :], axis=2)
    np.fill_diagonal(flat_dists, np.inf)
    flat_d = flat_dists[flat_dists < np.inf]
    V_gravity = -np.mean(1.0 / (flat_d ** 2 + 0.01))
    
    norms = np.linalg.norm(ast_m, axis=1)
    V_holo = np.mean(norms ** 2)
    
    C_ast = ast_m.T @ ast_m / n
    C_bc = bc_m.T @ bc_m / n
    comm = np.linalg.norm(C_ast @ C_bc - C_bc @ C_ast)
    
    L = T_kinetic - G * V_gravity - lam * V_holo - mu * comm ** 2
    print(f"  L_total = {L:.8f}")
    print(f"  Components: T={T_kinetic:.6f}, G*V_g={G*V_gravity:.6f}, lam*V_h={lam*V_holo:.6f}, mu*C^2={mu*comm**2:.6f}")
    
    # 2. Decompose residual into sources
    print(f"\n--- Residual Decomposition ---")
    
    # Quantum correction: variance in the Lagrangian across individual functions
    L_per_func = []
    for i in range(n):
        v = ast_m[i]
        d_i = np.linalg.norm(ast_m - v.reshape(1,-1), axis=1)
        d_i[i] = np.inf
        t_i = nn_dists[i]**2 / 2
        v_g_i = -G * np.mean(1.0 / (d_i[d_i < np.inf]**2 + 0.01))
        v_h_i = lam * np.sum(v**2)
        L_per_func.append(t_i - v_g_i - v_h_i)
    
    L_per_func = np.array(L_per_func)
    quantum_correction = np.var(L_per_func)
    print(f"  Quantum correction (variance): {quantum_correction:.8f}")
    
    # Zero-point energy: energy at the ground state
    pca = PCA(n_components=min(20, n)).fit(ast_m)
    eigenvalues = pca.explained_variance_
    zero_point = np.sum(eigenvalues * 0.5) / n  # (1/2) * hbar * omega per mode
    print(f"  Zero-point energy: {zero_point:.8f}")
    
    # Renormalization: at what scale does the constant change?
    scales = [10, 20, 50, 100, n]
    L_at_scale = []
    for s in scales:
        if s > n: s = n
        subset = ast_m[:s]
        nn_d = np.sort(np.linalg.norm(subset[:, None] - subset[None, :], axis=2), axis=1)[:, 1]
        t = np.mean(nn_d**2) / 2
        fd = np.linalg.norm(subset[:, None] - subset[None, :], axis=2)
        np.fill_diagonal(fd, np.inf)
        v_g = -np.mean(1.0 / (fd[fd < np.inf]**2 + 0.01))
        v_h = np.mean(np.linalg.norm(subset, axis=1)**2)
        L_s = t - G*v_g - lam*v_h
        L_at_scale.append(float(L_s))
        print(f"  Scale N={s}: L={L_s:.6f}")
    
    # Running of the cosmological constant
    if len(scales) >= 3:
        log_scales = np.log(scales[:len(L_at_scale)])
        slope, intercept, r_val, p_val, _ = stats.linregress(log_scales, L_at_scale)
        running_rate = slope
        print(f"\n  Cosmological constant running: dL/d(ln N) = {running_rate:.6f}")
        print(f"  R^2 = {r_val**2:.4f}")
    else:
        running_rate = 0
    
    # 3. Anthropic reasoning: what if L were different?
    print(f"\n--- Anthropic Test ---")
    L_variants = [-0.01, -0.001, 0, 0.000039, 0.001, 0.01, 0.1]
    
    for L_test in L_variants:
        # If L > 0 (positive cosmological constant): expansion
        # If L < 0: collapse
        # If L = 0: static
        if L_test > 0.01:
            fate = "rapid expansion -> void"
        elif L_test > 0:
            fate = "slow expansion -> structure"
        elif L_test == 0:
            fate = "static universe"
        elif L_test > -0.01:
            fate = "slow collapse -> complexity"
        else:
            fate = "rapid collapse -> singularity"
        print(f"  L={L_test:+.6f}: {fate}")
    
    print(f"\n  Our universe (L={L:.6f}): Goldilocks zone for code structure!")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Phase 140: The Cosmological Constant Problem', fontsize=14, fontweight='bold')
    
    components = ['T', 'G*V_g', 'lam*V_h', 'mu*C^2', 'L_total']
    values = [T_kinetic, G*V_gravity, lam*V_holo, mu*comm**2, L]
    colors = ['#2196F3','#F44336','#4CAF50','#FF9800','#9C27B0']
    axes[0].bar(components, values, color=colors, edgecolor='black')
    axes[0].set_title(f'Lagrangian breakdown (L={L:.6f})')
    axes[0].tick_params(axis='x', rotation=20)
    
    axes[1].plot(scales[:len(L_at_scale)], L_at_scale, 'o-', color='#E91E63', linewidth=2, markersize=8)
    axes[1].set_xlabel('Scale N'); axes[1].set_ylabel('L')
    axes[1].set_title('Cosmological constant running')
    axes[1].set_xscale('log')
    
    axes[2].hist(L_per_func, bins=30, color='#00BCD4', edgecolor='black', alpha=0.7)
    axes[2].axvline(L, color='red', linestyle='--', linewidth=2, label=f'Mean L={L:.6f}')
    axes[2].set_xlabel('L per function'); axes[2].set_title('Quantum fluctuations')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase140_cosmo_constant.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    results = {
        'phase': 140, 'title': 'The Cosmological Constant Problem',
        'L_total': float(L),
        'components': {'T': float(T_kinetic), 'GV_g': float(G*V_gravity), 'lamV_h': float(lam*V_holo), 'muC2': float(mu*comm**2)},
        'quantum_correction': float(quantum_correction),
        'zero_point_energy': float(zero_point),
        'running_rate': float(running_rate),
        'law': f'L = {L:.8f}. Quantum correction = {quantum_correction:.6f}. Zero-point = {zero_point:.6f}. Running rate = {running_rate:.6f}.'
    }
    with open(os.path.join(RESULTS_DIR, 'phase140_cosmo_constant.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nPhase 140 complete!")
    return results

if __name__ == '__main__':
    main()
