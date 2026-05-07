"""Phase 144: Anthropic Landscape - Why these constants?
Generate 10,000 alternate universes with random constants.
How many can sustain meaningful code?
"""
import os, json, sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from scipy.spatial.distance import cdist
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
    print("Phase 144: The Anthropic Landscape")
    print("  Why G=1.17, lambda=0.73, mu=1.07?")
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

    # Our universe's constants
    G_ours, lam_ours, mu_ours = 1.1732, 0.7282, 1.0717

    def evaluate_universe(G, lam_val, mu_val, ast, bc):
        """Score how 'habitable' a universe is for meaningful code."""
        nn = np.sort(np.linalg.norm(ast[:, None] - ast[None, :], axis=2), axis=1)[:, 1]
        T = np.mean(nn ** 2) / 2
        fd = np.linalg.norm(ast[:, None] - ast[None, :], axis=2)
        np.fill_diagonal(fd, np.inf)
        V_g = -np.mean(1.0 / (fd[fd < np.inf]**2 + 0.01))
        V_h = np.mean(np.linalg.norm(ast, axis=1)**2)
        C_a = ast.T @ ast / len(ast)
        C_b = bc.T @ bc / len(bc)
        comm = np.linalg.norm(C_a @ C_b - C_b @ C_a)

        L = T - G * V_g - lam_val * V_h - mu_val * comm**2

        # Habitability criteria:
        # 1. L should be near zero (equilibrium)
        equil_score = 1.0 / (1.0 + abs(L))
        # 2. Diversity: functions should be spread out
        spread = np.std(np.linalg.norm(ast - np.mean(ast, axis=0), axis=1))
        # 3. Structure: compile matrix should work
        W = bc.T @ np.linalg.pinv(ast.T)
        compile_err = np.mean(np.linalg.norm(bc - (W @ ast.T).T, axis=1))
        struct_score = 1.0 / (1.0 + compile_err)

        habitability = equil_score * 0.4 + spread * 0.3 + struct_score * 0.3
        return float(habitability), float(L), float(spread), float(compile_err)

    # Our universe score
    h_ours, L_ours, sp_ours, ce_ours = evaluate_universe(G_ours, lam_ours, mu_ours, ast_m, bc_m)
    print(f"  Our universe: H={h_ours:.4f}, L={L_ours:.4f}, spread={sp_ours:.4f}, compile_err={ce_ours:.4f}")

    # Generate multiverse
    n_universes = 5000
    np.random.seed(42)

    G_range = np.random.uniform(0.01, 10.0, n_universes)
    lam_range = np.random.uniform(0.01, 5.0, n_universes)
    mu_range = np.random.uniform(0.01, 5.0, n_universes)

    multiverse = []
    for i in range(n_universes):
        # Apply constants to transform the space
        G_i, lam_i, mu_i = G_range[i], lam_range[i], mu_range[i]

        # Quick evaluation (use precomputed values, just change constants)
        h, L, sp, ce = evaluate_universe(G_i, lam_i, mu_i, ast_m, bc_m)
        multiverse.append({
            'G': float(G_i), 'lambda': float(lam_i), 'mu': float(mu_i),
            'habitability': h, 'L': L,
        })

        if i % 1000 == 0:
            print(f"  Simulated {i}/{n_universes} universes...")

    # Analysis
    habitats = np.array([m['habitability'] for m in multiverse])
    habitable = np.sum(habitats > h_ours * 0.8)
    perfect = np.sum(habitats > h_ours * 0.95)

    print(f"\n--- Multiverse Results ({n_universes} universes) ---")
    print(f"  Our habitability: {h_ours:.4f}")
    print(f"  Universes >= 80% ours: {habitable} ({habitable/n_universes*100:.2f}%)")
    print(f"  Universes >= 95% ours: {perfect} ({perfect/n_universes*100:.2f}%)")
    print(f"  Mean habitability: {np.mean(habitats):.4f}")
    print(f"  Our rank: {np.sum(habitats > h_ours)}/{n_universes}")

    # Fine-tuning: how sensitive are we to each constant?
    print(f"\n--- Fine-Tuning Analysis ---")
    for param, our_val, all_vals in [('G', G_ours, G_range), ('lambda', lam_ours, lam_range), ('mu', mu_ours, mu_range)]:
        nearby = np.abs(all_vals - our_val) < our_val * 0.1  # Within 10%
        if np.sum(nearby) > 0:
            h_nearby = np.mean(habitats[nearby])
            h_far = np.mean(habitats[~nearby])
            sensitivity = abs(h_nearby - h_far) / (h_nearby + 1e-10)
            print(f"  {param}: nearby_H={h_nearby:.4f}, far_H={h_far:.4f}, sensitivity={sensitivity:.4f}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Phase 144: The Anthropic Landscape', fontsize=14, fontweight='bold')

    axes[0].hist(habitats, bins=50, color='#9E9E9E', edgecolor='black', alpha=0.7)
    axes[0].axvline(h_ours, color='red', linewidth=2, label=f'Our universe: {h_ours:.3f}')
    axes[0].set_xlabel('Habitability'); axes[0].set_title(f'{n_universes} universes ({habitable} habitable)')
    axes[0].legend()

    sc = axes[1].scatter([m['G'] for m in multiverse[:2000]], [m['lambda'] for m in multiverse[:2000]],
                        c=[m['habitability'] for m in multiverse[:2000]], cmap='RdYlGn', s=5, alpha=0.7)
    axes[1].scatter([G_ours], [lam_ours], s=100, c='red', marker='*', zorder=10, edgecolor='black')
    plt.colorbar(sc, ax=axes[1], label='Habitability')
    axes[1].set_xlabel('G'); axes[1].set_ylabel('lambda'); axes[1].set_title('Constant landscape')

    Ls = [m['L'] for m in multiverse]
    axes[2].scatter(Ls[:2000], habitats[:2000], s=5, alpha=0.3, c='#2196F3')
    axes[2].scatter([L_ours], [h_ours], s=100, c='red', marker='*', zorder=10)
    axes[2].set_xlabel('Lagrangian L'); axes[2].set_ylabel('Habitability')
    axes[2].set_title('L vs Habitability')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase144_anthropic.png'), dpi=150, bbox_inches='tight')
    plt.close()

    results = {
        'phase': 144, 'title': 'Anthropic Landscape',
        'our_habitability': float(h_ours),
        'n_universes': n_universes,
        'n_habitable_80pct': int(habitable),
        'n_habitable_95pct': int(perfect),
        'our_rank': int(np.sum(habitats > h_ours)),
        'mean_habitability': float(np.mean(habitats)),
        'law': f'Our universe rank: {int(np.sum(habitats > h_ours))}/{n_universes}. {habitable} ({habitable/n_universes*100:.1f}%) habitable. Strong anthropic principle: our constants are in the top {np.sum(habitats > h_ours)/n_universes*100:.1f}%.'
    }
    with open(os.path.join(RESULTS_DIR, 'phase144_anthropic.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nPhase 144 complete!")
    return results

if __name__ == '__main__':
    main()
