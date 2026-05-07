"""Phase 122: Semantic Dark Energy - Is the software universe expanding?
Deep Think: Measure if software evolves toward greater complexity (expansion).
We analyze AST complexity progression and compute a Hubble constant.
"""
import os, json, sys, ast, types
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

def ast_complexity(source):
    """Compute AST complexity: number of nodes in the tree."""
    try:
        tree = ast.parse(source)
        return sum(1 for _ in ast.walk(tree))
    except Exception:
        return 0

def main():
    print("=" * 60)
    print("Phase 122: Semantic Dark Energy")
    print("  Is the software universe accelerating?")
    print("=" * 60)
    
    latents = np.load(os.path.join(DATA_DIR, 'rosetta_latents_v2.npz'))
    dataset = json.load(open(os.path.join(DATA_DIR, 'rosetta_dataset_v2.json'), 'r', encoding='utf-8'))
    ast_vectors = latents['ast']
    sources = [item['source'] for item in dataset['dataset']]
    
    func_ast = {}
    for i, src in enumerate(sources):
        if src not in func_ast: func_ast[src] = []
        func_ast[src].append(ast_vectors[i])
    
    unique_funcs = list(func_ast.keys())
    ast_m = np.array([np.mean(func_ast[f], axis=0) for f in unique_funcs])
    n = len(unique_funcs)
    
    pca = PCA(n_components=10).fit(ast_m)
    ast_pca = pca.transform(ast_m)
    
    # 1. Assign "cosmic time" based on AST complexity
    complexities = [ast_complexity(src) for src in unique_funcs]
    complexities = np.array(complexities, dtype=float)
    
    # Sort by complexity = sort by "cosmic time"
    sort_idx = np.argsort(complexities)
    sorted_complexities = complexities[sort_idx]
    sorted_pca = ast_pca[sort_idx]
    
    # 2. Compute "expansion" - mean pairwise distance at each complexity level
    # Group by complexity bins
    bins = np.linspace(0, max(complexities), 10)
    expansion_data = []
    
    for i in range(len(bins)-1):
        mask = (complexities >= bins[i]) & (complexities < bins[i+1])
        if np.sum(mask) >= 3:
            cluster = ast_m[mask]
            pdists = []
            for a in range(len(cluster)):
                for b in range(a+1, len(cluster)):
                    pdists.append(np.linalg.norm(cluster[a] - cluster[b]))
            mean_dist = np.mean(pdists) if pdists else 0
            expansion_data.append({
                'complexity_bin': float(bins[i]),
                'n_funcs': int(np.sum(mask)),
                'mean_pairwise_dist': float(mean_dist),
            })
            print(f"  Complexity {bins[i]:.0f}-{bins[i+1]:.0f}: {np.sum(mask)} funcs, mean dist = {mean_dist:.4f}")
    
    # 3. Hubble constant: rate of expansion
    if len(expansion_data) >= 3:
        x = [d['complexity_bin'] for d in expansion_data]
        y = [d['mean_pairwise_dist'] for d in expansion_data]
        slope, intercept, r_val, p_val, std_err = stats.linregress(x, y)
        hubble = slope
        print(f"\n--- Hubble Constant of Software ---")
        print(f"  H_0 = {hubble:.4f} (dist/complexity-unit)")
        print(f"  R-squared: {r_val**2:.4f}")
        print(f"  {'Expanding!' if hubble > 0 else 'Contracting!'}")
    else:
        hubble = 0; r_val = 0
    
    # 4. Dark energy detection: is expansion accelerating?
    # Compute second derivative (acceleration)
    if len(expansion_data) >= 4:
        dists = [d['mean_pairwise_dist'] for d in expansion_data]
        accelerations = []
        for i in range(1, len(dists)-1):
            accel = dists[i+1] - 2*dists[i] + dists[i-1]
            accelerations.append(accel)
        mean_accel = np.mean(accelerations)
        print(f"  Mean acceleration: {mean_accel:.4f}")
        print(f"  {'Dark energy detected!' if mean_accel > 0 else 'No dark energy (deceleration)'}")
    else:
        mean_accel = 0
    
    # 5. Red-shift analogy: do complex programs have "shifted" spectra?
    # Compare eigenvalue ratios of simple vs complex programs
    simple_mask = complexities <= np.median(complexities)
    complex_mask = complexities > np.median(complexities)
    
    simple_cov = np.cov(ast_m[simple_mask].T)
    complex_cov = np.cov(ast_m[complex_mask].T)
    
    simple_eigs = np.sort(np.linalg.eigvalsh(simple_cov))[::-1]
    complex_eigs = np.sort(np.linalg.eigvalsh(complex_cov))[::-1]
    
    # Red-shift = ratio of eigenvalues
    red_shift = complex_eigs[:5] / (simple_eigs[:5] + 1e-10)
    mean_red_shift = float(np.mean(red_shift))
    print(f"\n--- Red-shift ---")
    print(f"  Eigenvalue ratio (complex/simple): {', '.join(f'{r:.3f}' for r in red_shift)}")
    print(f"  Mean red-shift z = {mean_red_shift:.3f}")
    print(f"  {'Red-shifted (universe expanding)' if mean_red_shift > 1 else 'Blue-shifted (contracting)'}")
    
    # 6. Cosmic microwave background: residual "temperature" after PCA
    residuals = ast_m - pca.inverse_transform(ast_pca)
    temperature = np.std(residuals)
    print(f"\n--- Cosmic Microwave Background ---")
    print(f"  Residual temperature (noise after 10 PCs): {temperature:.6f}")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Phase 122: Semantic Dark Energy', fontsize=14, fontweight='bold')
    
    if expansion_data:
        x = [d['complexity_bin'] for d in expansion_data]
        y = [d['mean_pairwise_dist'] for d in expansion_data]
        axes[0].scatter(x, y, s=60, c='#E91E63', edgecolor='black', zorder=5)
        if hubble != 0:
            x_fit = np.linspace(min(x), max(x), 100)
            axes[0].plot(x_fit, slope*x_fit + intercept, 'k--',
                        label=f'H_0={hubble:.3f} (R2={r_val**2:.3f})')
            axes[0].legend()
        axes[0].set_xlabel('AST Complexity (cosmic time)')
        axes[0].set_ylabel('Mean pairwise distance')
        axes[0].set_title('Software Universe Expansion')
    
    axes[1].bar(range(5), red_shift, color='#FF5722', edgecolor='black')
    axes[1].axhline(1.0, color='gray', linestyle='--', label='No shift')
    axes[1].set_xlabel('Eigenvalue index'); axes[1].set_ylabel('Red-shift z')
    axes[1].set_title(f'Spectral Red-shift (mean z={mean_red_shift:.3f})')
    axes[1].legend()
    
    axes[2].imshow(np.std(residuals.reshape(-1, 8, 8)[:20], axis=0), cmap='coolwarm')
    axes[2].set_title(f'CMB: residual temp = {temperature:.5f}')
    axes[2].set_xlabel('Dim group'); axes[2].set_ylabel('Dim group')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase122_dark_energy.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    results = {
        'phase': 122, 'title': 'Semantic Dark Energy',
        'hubble_constant': float(hubble),
        'hubble_r_squared': float(r_val**2),
        'mean_acceleration': float(mean_accel),
        'mean_red_shift': mean_red_shift,
        'cmb_temperature': float(temperature),
        'expansion_data': expansion_data,
        'law': f'H_0={hubble:.4f} (R2={r_val**2:.3f}). Accel={mean_accel:.4f}. Red-shift z={mean_red_shift:.3f}. CMB temp={temperature:.5f}.'
    }
    with open(os.path.join(RESULTS_DIR, 'phase122_dark_energy.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nPhase 122 complete!")
    return results

if __name__ == '__main__':
    main()
