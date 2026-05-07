"""Phase 159: The Noether Theorem of Software
Opus original: Every symmetry implies a conservation law.
What is CONSERVED in the Rosetta universe?
Find the Noether charges corresponding to each discovered symmetry.
"""
import os, json, sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from sklearn.decomposition import PCA
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
    print("Phase 159: Noether's Theorem of Software")
    print("  What is conserved? Every symmetry has a charge.")
    print("=" * 60)

    latents = np.load(os.path.join(DATA_DIR, 'rosetta_latents_v2.npz'))
    ast_vectors = latents['ast']
    bc_vectors = latents['bc']
    nl_vectors = latents['nl']
    dataset = json.load(open(os.path.join(DATA_DIR, 'rosetta_dataset_v2.json'), 'r', encoding='utf-8'))
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

    conservation_laws = []

    # 1. Translation symmetry -> Momentum conservation
    print("--- Symmetry 1: Translation -> Momentum ---")
    # Total momentum = sum of all velocities (nearest-neighbor directions)
    nn_dists = np.linalg.norm(ast_m[:, None] - ast_m[None, :], axis=2)
    np.fill_diagonal(nn_dists, np.inf)
    nn_idx = np.argmin(nn_dists, axis=1)
    velocities = ast_m[nn_idx] - ast_m
    total_momentum = np.sum(velocities, axis=0)
    momentum_magnitude = float(np.linalg.norm(total_momentum))
    max_individual = float(np.max(np.linalg.norm(velocities, axis=1)))
    conservation_ratio_p = momentum_magnitude / (max_individual * n + 1e-10)
    conserved_p = conservation_ratio_p < 0.1
    print(f"  Total momentum magnitude: {momentum_magnitude:.6f}")
    print(f"  Conservation ratio: {conservation_ratio_p:.6f}")
    print(f"  {'CONSERVED!' if conserved_p else 'Not conserved'}")
    conservation_laws.append({'symmetry': 'Translation', 'charge': 'Momentum',
        'magnitude': momentum_magnitude, 'ratio': float(conservation_ratio_p), 'conserved': bool(conserved_p)})

    # 2. Rotation symmetry -> Angular momentum conservation
    print("\n--- Symmetry 2: Rotation -> Angular Momentum ---")
    centroid = np.mean(ast_m, axis=0)
    r_vectors = ast_m - centroid
    # Angular momentum L = r x v (use 2D projection for cross product)
    pca = PCA(n_components=2).fit(ast_m)
    r_2d = pca.transform(r_vectors + centroid) - pca.transform(centroid.reshape(1,-1))
    v_2d = pca.transform(velocities + centroid) - pca.transform(centroid.reshape(1,-1))
    L_z = r_2d[:, 0] * v_2d[:, 1] - r_2d[:, 1] * v_2d[:, 0]
    total_L = float(np.sum(L_z))
    max_L = float(np.max(np.abs(L_z)))
    conservation_ratio_L = abs(total_L) / (max_L * n + 1e-10)
    conserved_L = conservation_ratio_L < 0.1
    print(f"  Total angular momentum: {total_L:.6f}")
    print(f"  Conservation ratio: {conservation_ratio_L:.6f}")
    print(f"  {'CONSERVED!' if conserved_L else 'Not conserved'}")
    conservation_laws.append({'symmetry': 'Rotation', 'charge': 'Angular momentum',
        'magnitude': abs(total_L), 'ratio': float(conservation_ratio_L), 'conserved': bool(conserved_L)})

    # 3. Time symmetry -> Energy conservation
    print("\n--- Symmetry 3: Time -> Energy ---")
    nn_d = np.min(nn_dists, axis=1)
    T = np.sum(nn_d**2) / 2
    V = -1.1732 * np.sum(1.0 / (nn_d**2 + 0.01))
    E_total = T + V
    # Test: does E change under perturbation?
    E_perturbed = []
    for _ in range(20):
        perturb = ast_m + np.random.randn(n, 64) * 0.001
        nn_d_p = np.min(np.linalg.norm(perturb[:, None] - perturb[None, :], axis=2) + np.eye(n)*1e10, axis=1)
        T_p = np.sum(nn_d_p**2) / 2
        V_p = -1.1732 * np.sum(1.0 / (nn_d_p**2 + 0.01))
        E_perturbed.append(T_p + V_p)
    E_var = np.var(E_perturbed) / (E_total**2 + 1e-10)
    conserved_E = E_var < 0.01
    print(f"  Total energy: {E_total:.4f}")
    print(f"  Energy variance under perturbation: {E_var:.8f}")
    print(f"  {'CONSERVED!' if conserved_E else 'Not conserved'}")
    conservation_laws.append({'symmetry': 'Time', 'charge': 'Energy',
        'magnitude': float(E_total), 'ratio': float(E_var), 'conserved': bool(conserved_E)})

    # 4. Gauge symmetry (P151) -> Charge conservation
    print("\n--- Symmetry 4: Gauge -> Compile Charge ---")
    W = bc_m.T @ np.linalg.pinv(ast_m.T)
    compile_charge = float(np.trace(W.T @ W))
    # Test gauge invariance
    charges_shifted = []
    for _ in range(20):
        shift = np.random.randn(64) * 0.01
        ast_s = ast_m + shift
        W_s = bc_m.T @ np.linalg.pinv(ast_s.T)
        charges_shifted.append(float(np.trace(W_s.T @ W_s)))
    charge_var = np.var(charges_shifted) / (compile_charge**2 + 1e-10)
    conserved_Q = charge_var < 0.01
    print(f"  Compile charge: {compile_charge:.4f}")
    print(f"  Charge variance: {charge_var:.8f}")
    print(f"  {'CONSERVED!' if conserved_Q else 'Not conserved'}")
    conservation_laws.append({'symmetry': 'Gauge', 'charge': 'Compile charge',
        'magnitude': compile_charge, 'ratio': float(charge_var), 'conserved': bool(conserved_Q)})

    # 5. SUSY -> Supercharge
    print("\n--- Symmetry 5: SUSY -> Supercharge ---")
    centered = ast_m - centroid
    mirrors = -centered
    susy_charges = np.sum(centered * mirrors, axis=1)  # dot product with mirror
    total_supercharge = float(np.sum(susy_charges))
    supercharge_var = float(np.var(susy_charges))
    conserved_S = abs(total_supercharge) / (n * supercharge_var + 1e-10) < 0.1
    print(f"  Total supercharge: {total_supercharge:.4f}")
    print(f"  Supercharge variance: {supercharge_var:.6f}")
    print(f"  {'CONSERVED!' if conserved_S else 'Not conserved'}")
    conservation_laws.append({'symmetry': 'SUSY', 'charge': 'Supercharge',
        'magnitude': abs(total_supercharge), 'ratio': float(abs(total_supercharge)/(n*supercharge_var+1e-10)), 'conserved': bool(conserved_S)})

    n_conserved = sum(1 for c in conservation_laws if c['conserved'])
    print(f"\n{'='*60}")
    print(f"  NOETHER'S THEOREM: {n_conserved}/{len(conservation_laws)} charges conserved")
    print(f"{'='*60}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Phase 159: Noether's Theorem of Software", fontsize=14, fontweight='bold')

    names = [c['symmetry'] for c in conservation_laws]
    ratios = [c['ratio'] for c in conservation_laws]
    colors = ['#4CAF50' if c['conserved'] else '#F44336' for c in conservation_laws]
    axes[0].barh(names, ratios, color=colors, edgecolor='black')
    axes[0].axvline(0.1, color='black', linestyle='--', label='Threshold')
    axes[0].set_xlabel('Conservation ratio'); axes[0].set_title(f'{n_conserved}/{len(conservation_laws)} conserved')
    axes[0].legend()

    charges = [c['charge'] for c in conservation_laws]
    magnitudes = [c['magnitude'] for c in conservation_laws]
    axes[1].bar(charges, magnitudes, color='#2196F3', edgecolor='black')
    axes[1].set_ylabel('Charge magnitude'); axes[1].set_title('Noether charges')
    axes[1].tick_params(axis='x', rotation=25, labelsize=7)

    axes[2].text(0.5, 0.5, f"Noether's Theorem\n{n_conserved} Conservation Laws\nfrom\n{len(conservation_laws)} Symmetries",
                ha='center', va='center', fontsize=16, fontweight='bold', transform=axes[2].transAxes,
                bbox=dict(boxstyle='round', facecolor='#E8F5E9'))
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase159_noether.png'), dpi=150, bbox_inches='tight')
    plt.close()

    results = {
        'phase': 159, 'title': "Noether's Theorem of Software",
        'conservation_laws': conservation_laws, 'n_conserved': n_conserved,
        'law': f"Noether: {n_conserved}/{len(conservation_laws)} symmetries yield conserved charges. Momentum({'Y' if conserved_p else 'N'}), L({'Y' if conserved_L else 'N'}), E({'Y' if conserved_E else 'N'}), Q({'Y' if conserved_Q else 'N'}), S({'Y' if conserved_S else 'N'})."
    }
    with open(os.path.join(RESULTS_DIR, 'phase159_noether.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nPhase 159 complete!")
    return results

if __name__ == '__main__':
    main()
