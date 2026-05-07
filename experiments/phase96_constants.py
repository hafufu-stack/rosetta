"""Phase 96: The Rosetta Constant - Universal constants of the software universe.
Search for dimensionless constants that appear across all laws.
Like physics has alpha=1/137, does software physics have universal constants?
"""
import os, json, sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(EXPERIMENT_DIR)
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')

def main():
    print("=" * 60)
    print("Phase 96: The Rosetta Constant")
    print("  Universal constants of the software universe")
    print("=" * 60)
    
    latents = np.load(os.path.join(BASE_DIR, 'data', 'rosetta_latents_v2.npz'))
    dataset = json.load(open(os.path.join(BASE_DIR, 'data', 'rosetta_dataset_v2.json'), 'r', encoding='utf-8'))
    ast_vectors = latents['ast']
    bytecode_vectors = latents['bc']
    sources = [item['source'] for item in dataset['dataset']]
    
    func_to_ast = {}
    func_to_bc = {}
    for i, src in enumerate(sources):
        if src not in func_to_ast:
            func_to_ast[src] = []
            func_to_bc[src] = []
        func_to_ast[src].append(ast_vectors[i])
        func_to_bc[src].append(bytecode_vectors[i])
    
    ast_means = {s: np.mean(v, axis=0) for s, v in func_to_ast.items()}
    bc_means = {s: np.mean(v, axis=0) for s, v in func_to_bc.items()}
    unique_funcs = list(ast_means.keys())
    
    ast_all = np.array([ast_means[f] for f in unique_funcs])
    bc_all = np.array([bc_means[f] for f in unique_funcs])
    
    # === CONSTANT 1: The Mass Ratio ===
    # Ratio of AST "mass" (norm) to Bytecode "mass"
    ast_norms = np.linalg.norm(ast_all, axis=1)
    bc_norms = np.linalg.norm(bc_all, axis=1)
    mass_ratios = ast_norms / (bc_norms + 1e-10)
    mean_ratio = np.mean(mass_ratios)
    std_ratio = np.std(mass_ratios)
    cv_ratio = std_ratio / mean_ratio  # coefficient of variation
    
    print(f"\n--- Constant 1: The Mass Ratio (AST/Bytecode) ---")
    print(f"  Mean: {mean_ratio:.4f}")
    print(f"  Std:  {std_ratio:.4f}")
    print(f"  CV:   {cv_ratio:.4f}")
    print(f"  {'UNIVERSAL' if cv_ratio < 0.3 else 'Variable'} (CV < 0.3 = universal)")
    
    # === CONSTANT 2: The Duality Angle ===
    # Angle between AST and Bytecode representations of same function
    cos_angles = []
    for f in unique_funcs:
        a = ast_means[f]
        b = bc_means[f]
        cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
        cos_angles.append(cos)
    cos_angles = np.array(cos_angles)
    angles_deg = np.arccos(np.clip(cos_angles, -1, 1)) * 180 / np.pi
    mean_angle = np.mean(angles_deg)
    std_angle = np.std(angles_deg)
    
    print(f"\n--- Constant 2: The Duality Angle (AST-Bytecode) ---")
    print(f"  Mean angle: {mean_angle:.1f} deg")
    print(f"  Std:        {std_angle:.1f} deg")
    print(f"  {'UNIVERSAL' if std_angle/mean_angle < 0.3 else 'Variable'}")
    
    # === CONSTANT 3: The Packing Fraction ===
    # How much of the 64D hypersphere is "used" by programs?
    centroid = np.mean(ast_all, axis=0)
    dists_from_center = np.linalg.norm(ast_all - centroid, axis=1)
    max_dist = np.max(dists_from_center)
    mean_dist = np.mean(dists_from_center)
    packing = mean_dist / max_dist  # 0=all at center, 1=all at edge
    
    print(f"\n--- Constant 3: The Packing Fraction ---")
    print(f"  Mean/Max distance: {packing:.4f}")
    print(f"  (0=clustered, 1=shell-like)")
    
    # === CONSTANT 4: The Spectral Ratio ===
    # Ratio of first eigenvalue to total variance
    pca = PCA(n_components=min(20, len(unique_funcs)))
    pca.fit(ast_all)
    spectral_ratio = pca.explained_variance_ratio_[0]
    
    print(f"\n--- Constant 4: The Spectral Dominance ---")
    print(f"  PC1 variance ratio: {spectral_ratio:.4f}")
    
    # === CONSTANT 5: The Golden Ratio Test ===
    # Is there a golden-ratio-like constant in the eigenvalue spectrum?
    eigenvalues = pca.explained_variance_
    consec_ratios = eigenvalues[:-1] / eigenvalues[1:]
    
    golden = (1 + np.sqrt(5)) / 2  # 1.618...
    nearest_to_golden = min(consec_ratios, key=lambda x: abs(x - golden))
    golden_idx = list(consec_ratios).index(nearest_to_golden)
    
    print(f"\n--- Constant 5: Golden Ratio Test ---")
    print(f"  Nearest eigenvalue ratio to phi (1.618): {nearest_to_golden:.4f} at gap {golden_idx+1}")
    print(f"  Deviation from phi: {abs(nearest_to_golden - golden):.4f}")
    
    # === CONSTANT 6: The Universality Index ===
    # Ratio of inter-class to intra-class distances (Fisher criterion)
    import inspect
    op_labels = {}
    g = {}
    for f in unique_funcs:
        try:
            exec(compile(f, '<string>', 'exec'), g)
            fn = g['f']
            try: n = len(inspect.signature(fn).parameters)
            except: n = 2
            # Test signature
            try:
                r1 = fn(3) if n == 1 else fn(3, 5)
                r2 = fn(2) if n == 1 else fn(2, 3)
                op_labels[f] = f"type_{hash(f) % 10}"
            except:
                op_labels[f] = "error"
        except:
            op_labels[f] = "parse_error"
    
    # Use simple operation categorization
    for f in unique_funcs:
        if 'x + y' in f: op_labels[f] = 'add'
        elif 'x - y' in f: op_labels[f] = 'sub'
        elif 'x * y' in f: op_labels[f] = 'mul'
        elif 'x > y' in f or 'x < y' in f or 'x ==' in f: op_labels[f] = 'cmp'
        elif 'max(' in f or 'min(' in f: op_labels[f] = 'minmax'
    
    # Compute Fisher criterion for labeled functions
    classes = {}
    for f in unique_funcs:
        label = op_labels.get(f, 'other')
        if label in ['add', 'sub', 'mul', 'cmp', 'minmax']:
            if label not in classes:
                classes[label] = []
            classes[label].append(ast_means[f])
    
    if len(classes) >= 2:
        global_mean = np.mean(ast_all, axis=0)
        inter_class = 0
        intra_class = 0
        for label, vecs in classes.items():
            vecs = np.array(vecs)
            class_mean = np.mean(vecs, axis=0)
            inter_class += len(vecs) * np.linalg.norm(class_mean - global_mean)**2
            intra_class += sum(np.linalg.norm(v - class_mean)**2 for v in vecs)
        
        fisher = inter_class / (intra_class + 1e-10)
        print(f"\n--- Constant 6: The Fisher Criterion ---")
        print(f"  Inter/Intra class ratio: {fisher:.4f}")
        print(f"  (Higher = better separated operation types)")
    else:
        fisher = 0.0
    
    # === Compile the Rosetta Constants ===
    constants = {
        'alpha_mass': {'value': float(mean_ratio), 'cv': float(cv_ratio),
                       'description': 'AST/Bytecode mass ratio'},
        'theta_duality': {'value': float(mean_angle), 'std': float(std_angle),
                          'description': 'AST-Bytecode duality angle (degrees)'},
        'rho_packing': {'value': float(packing),
                        'description': 'Packing fraction of program hypersphere'},
        'lambda_spectral': {'value': float(spectral_ratio),
                            'description': 'PC1 dominance ratio'},
        'phi_golden': {'value': float(nearest_to_golden), 'gap_position': int(golden_idx+1),
                       'deviation': float(abs(nearest_to_golden - golden)),
                       'description': 'Nearest eigenvalue ratio to golden ratio'},
        'F_fisher': {'value': float(fisher),
                     'description': 'Fisher separability of operation types'}
    }
    
    print(f"\n{'='*60}")
    print(f"THE ROSETTA CONSTANTS")
    print(f"{'='*60}")
    for name, info in constants.items():
        print(f"  {name} = {info['value']:.4f}  ({info['description']})")
    
    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Phase 96: The Rosetta Constants', fontsize=14, fontweight='bold')
    
    axes[0,0].hist(mass_ratios, bins=30, color='#2196F3', edgecolor='black', alpha=0.7)
    axes[0,0].axvline(mean_ratio, color='red', linestyle='--', label=f'mean={mean_ratio:.3f}')
    axes[0,0].set_xlabel('AST/Bytecode Mass Ratio'); axes[0,0].set_title('alpha_mass')
    axes[0,0].legend()
    
    axes[0,1].hist(angles_deg, bins=30, color='#4CAF50', edgecolor='black', alpha=0.7)
    axes[0,1].axvline(mean_angle, color='red', linestyle='--', label=f'mean={mean_angle:.1f} deg')
    axes[0,1].set_xlabel('Duality Angle (deg)'); axes[0,1].set_title('theta_duality')
    axes[0,1].legend()
    
    axes[0,2].hist(dists_from_center, bins=30, color='#FF9800', edgecolor='black', alpha=0.7)
    axes[0,2].axvline(mean_dist, color='red', linestyle='--', label=f'packing={packing:.3f}')
    axes[0,2].set_xlabel('Distance from Centroid'); axes[0,2].set_title('rho_packing')
    axes[0,2].legend()
    
    axes[1,0].bar(range(1, len(eigenvalues[:15])+1), eigenvalues[:15]/eigenvalues[0],
                  color='#9C27B0', edgecolor='black')
    axes[1,0].set_xlabel('PC'); axes[1,0].set_ylabel('Normalized Eigenvalue')
    axes[1,0].set_title(f'lambda_spectral = {spectral_ratio:.3f}')
    
    axes[1,1].plot(range(1, len(consec_ratios[:15])+1), consec_ratios[:15], 'o-', color='#E91E63')
    axes[1,1].axhline(golden, color='gold', linestyle='--', linewidth=2, label=f'phi = {golden:.3f}')
    axes[1,1].set_xlabel('Gap Position'); axes[1,1].set_ylabel('Eigenvalue Ratio')
    axes[1,1].set_title(f'phi_golden: nearest = {nearest_to_golden:.3f}')
    axes[1,1].legend()
    
    const_names = list(constants.keys())
    const_vals = [constants[k]['value'] for k in const_names]
    axes[1,2].barh(const_names, const_vals, color=['#2196F3','#4CAF50','#FF9800','#9C27B0','#E91E63','#795548'])
    axes[1,2].set_xlabel('Value'); axes[1,2].set_title('All Rosetta Constants')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase96_constants.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    results = {
        'phase': 96, 'title': 'The Rosetta Constants',
        'constants': constants,
        'law': f"Six universal constants of the software universe: alpha_mass={mean_ratio:.3f}, theta_duality={mean_angle:.1f} deg, rho_packing={packing:.3f}, lambda_spectral={spectral_ratio:.3f}, phi_golden={nearest_to_golden:.3f}, F_fisher={fisher:.3f}"
    }
    with open(os.path.join(RESULTS_DIR, 'phase96_constants.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nPhase 96 complete!")
    return results

if __name__ == '__main__':
    main()
