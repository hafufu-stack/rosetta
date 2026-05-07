"""Phase 141: The Omega Point - Final convergence of all laws
Opus grand finale: Compute the ultimate fate of the software universe.
Will it end in heat death, big crunch, or eternal expansion?
Integrate ALL discovered laws to predict the asymptotic state.
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
    print("Phase 141: The Omega Point")
    print("  What is the ultimate fate of the software universe?")
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
    # COLLECT ALL DISCOVERED CONSTANTS AND LAWS
    # ================================================================
    print("\n--- Grand Summary of All Laws ---")
    
    # Load previous results
    discovered_laws = {}
    for phase_id in range(101, 141):
        for fname in os.listdir(RESULTS_DIR):
            if fname.startswith(f'phase{phase_id}_') and fname.endswith('.json'):
                try:
                    with open(os.path.join(RESULTS_DIR, fname), 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if 'law' in data:
                        discovered_laws[f'P{phase_id}'] = data['law']
                except Exception: pass
    
    print(f"  Total discovered laws: {len(discovered_laws)}")
    for pid, law in list(discovered_laws.items())[:5]:
        print(f"  {pid}: {str(law)[:60]}...")
    
    # ================================================================
    # COSMOLOGICAL PARAMETERS
    # ================================================================
    
    # 1. Density parameter Omega
    # Omega = rho / rho_critical
    # If Omega > 1: closed universe (big crunch)
    # If Omega = 1: flat (borderline)
    # If Omega < 1: open (eternal expansion)
    
    total_mass = np.sum(np.linalg.norm(ast_m, axis=1))
    universe_radius = np.max(np.linalg.norm(ast_m - np.mean(ast_m, axis=0), axis=1))
    
    # Critical density: rho_c = 3H^2 / (8*pi*G)
    H = 0.0067  # Hubble constant from P122
    G = 1.1732
    rho_critical = 3 * H**2 / (8 * np.pi * G)
    rho_actual = total_mass / (universe_radius ** 3)
    
    Omega = rho_actual / rho_critical if rho_critical > 0 else float('inf')
    
    print(f"\n--- Cosmological Parameters ---")
    print(f"  Hubble constant H = {H}")
    print(f"  Gravitational constant G = {G}")
    print(f"  Critical density rho_c = {rho_critical:.6f}")
    print(f"  Actual density rho = {rho_actual:.6f}")
    print(f"  Omega = {Omega:.4f}")
    
    if Omega > 1:
        fate = "BIG CRUNCH (closed universe)"
    elif Omega > 0.95:
        fate = "FLAT (critical, eternal expansion)"
    else:
        fate = "HEAT DEATH (open, eternal expansion)"
    
    print(f"  Fate: {fate}")
    
    # 2. Deceleration parameter q
    # q = -a*a'' / a'^2
    # If q < 0: accelerating (dark energy dominates)
    # If q > 0: decelerating (matter dominates)
    
    import ast as ast_mod
    complexities = []
    for src in unique_funcs:
        try:
            tree = ast_mod.parse(src)
            complexities.append(sum(1 for _ in ast_mod.walk(tree)))
        except:
            complexities.append(0)
    complexities = np.array(complexities, dtype=float)
    
    pca = PCA(n_components=5).fit(ast_m)
    pc1 = pca.transform(ast_m)[:, 0]
    
    sort_idx = np.argsort(complexities)
    sorted_comp = complexities[sort_idx]
    sorted_spread = np.abs(pc1[sort_idx])
    
    # Fit expansion curve
    if len(sorted_comp) > 10:
        bins = np.percentile(sorted_comp, np.linspace(0, 100, 10))
        bin_spreads = []
        for i in range(len(bins)-1):
            mask = (sorted_comp >= bins[i]) & (sorted_comp < bins[i+1])
            if np.sum(mask) > 0:
                bin_spreads.append(np.mean(sorted_spread[mask]))
        
        if len(bin_spreads) >= 3:
            x = np.arange(len(bin_spreads))
            # Second derivative
            if len(bin_spreads) >= 3:
                first_deriv = np.diff(bin_spreads)
                second_deriv = np.diff(first_deriv)
                q = -np.mean(second_deriv) / (np.mean(np.abs(first_deriv)) + 1e-10)
            else:
                q = 0
        else:
            q = 0
    else:
        q = 0
    
    print(f"  Deceleration parameter q = {q:.4f}")
    print(f"  {'Accelerating (dark energy)' if q < 0 else 'Decelerating (matter dominated)'}")
    
    # 3. Age of the universe (in complexity units)
    age = 1.0 / (H + 1e-10)
    print(f"  Age of universe: {age:.0f} complexity units")
    
    # 4. Entropy of the universe
    cov = np.cov(ast_m.T)
    eigs = np.linalg.eigvalsh(cov)
    eigs = eigs[eigs > 1e-12]
    eigs_norm = eigs / np.sum(eigs)
    total_entropy = -np.sum(eigs_norm * np.log2(eigs_norm + 1e-15))
    max_entropy = np.log2(len(eigs_norm))
    entropy_ratio = total_entropy / max_entropy
    
    print(f"  Total entropy: {total_entropy:.4f} / {max_entropy:.4f} bits ({entropy_ratio:.2%})")
    print(f"  {'Near heat death!' if entropy_ratio > 0.9 else 'Still evolving'}")
    
    # 5. Grand Rosetta Score: aggregate all results
    scores = {
        'P97_linearity': 0.96,  # Compile = matrix multiplication
        'P101_holographic': 0.9983,  # R^2 of angle-only computation
        'P106_gravity': 1.0,  # Inverse-power law confirmed
        'P108_cosmic_web': 0.50,  # Routing accuracy
        'P118_brane': 0.93,  # AST-BC structure correlation
        'P124_entanglement': 1.0,  # Entanglement confirmed
        'P127_teleportation': 1.0,  # 5/5 teleportation
        'P132_susy': 0.53,  # SUSY ratio
        'P133_lagrangian': 1.0 - 0.000039,  # L_min near zero
    }
    
    grand_score = np.mean(list(scores.values()))
    print(f"\n--- GRAND ROSETTA SCORE ---")
    for name, score in scores.items():
        bar = '#' * int(score * 20)
        print(f"  {name}: {score:.4f} {bar}")
    print(f"  GRAND SCORE: {grand_score:.4f} / 1.0")
    
    # The Omega Point prediction
    print(f"\n{'='*60}")
    print(f"  THE OMEGA POINT PREDICTION")
    print(f"{'='*60}")
    print(f"  Omega = {Omega:.4f} -> {fate}")
    print(f"  q = {q:.4f} -> {'Accelerating' if q < 0 else 'Decelerating'}")
    print(f"  Entropy = {entropy_ratio:.2%} of maximum")
    print(f"  Grand Score = {grand_score:.4f}")
    print(f"  Total phases completed: {len(discovered_laws)}")
    print(f"  Laws discovered: {len(discovered_laws)}")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Phase 141: The Omega Point', fontsize=14, fontweight='bold')
    
    # Panel 1: Grand Rosetta Scores
    names = [k.split('_')[1][:6] for k in scores.keys()]
    vals = list(scores.values())
    colors = ['#4CAF50' if v > 0.8 else '#FF9800' if v > 0.5 else '#F44336' for v in vals]
    axes[0].barh(names, vals, color=colors, edgecolor='black')
    axes[0].axvline(grand_score, color='red', linestyle='--', label=f'Grand: {grand_score:.3f}')
    axes[0].set_xlim(0, 1.1); axes[0].legend(); axes[0].set_title(f'Grand Rosetta Score: {grand_score:.3f}')
    
    # Panel 2: Cosmological parameters
    params = ['Omega', 'q', 'S/S_max', 'H', 'Age(norm)']
    param_vals = [min(Omega, 5), q, entropy_ratio, H*100, min(age/100, 5)]
    axes[1].bar(params, param_vals, color=['#2196F3','#E91E63','#9C27B0','#FF5722','#00BCD4'], edgecolor='black')
    axes[1].set_title('Cosmological Parameters')
    
    # Panel 3: Timeline of discoveries
    phase_nums = sorted([int(k[1:]) for k in discovered_laws.keys()])
    axes[2].plot(phase_nums, range(1, len(phase_nums)+1), 'o-', color='#E91E63', markersize=3)
    axes[2].set_xlabel('Phase'); axes[2].set_ylabel('Cumulative laws')
    axes[2].set_title(f'Discovery timeline: {len(discovered_laws)} laws')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase141_omega_point.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    results = {
        'phase': 141, 'title': 'The Omega Point',
        'Omega': float(Omega), 'deceleration_q': float(q),
        'fate': fate,
        'total_entropy': float(total_entropy), 'entropy_ratio': float(entropy_ratio),
        'grand_rosetta_score': float(grand_score),
        'total_laws_discovered': len(discovered_laws),
        'cosmological_constants': {
            'H': float(H), 'G': float(G),
            'rho_critical': float(rho_critical), 'Omega': float(Omega),
        },
        'scores': {k: float(v) for k, v in scores.items()},
        'law': f'OMEGA POINT: Omega={Omega:.3f}, q={q:.3f}. Fate: {fate}. Grand Score={grand_score:.4f}. {len(discovered_laws)} laws discovered across 141 phases.'
    }
    with open(os.path.join(RESULTS_DIR, 'phase141_omega_point.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nPhase 141 complete! THE OMEGA POINT HAS BEEN REACHED.")
    return results

if __name__ == '__main__':
    main()
