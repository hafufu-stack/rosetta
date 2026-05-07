"""Phase 106: The Gravity Equation - Quantifying gravitational attraction in program space.
P105 showed hubs with high degree (x+y=9, abs(x/y)=12). Why?
Test if attraction follows an inverse-square law like real gravity:
  F ~ m1 * m2 / d^2
where mass = number of related functions (or AST complexity).
"""
import os, json, sys, inspect, ast
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from scipy.optimize import curve_fit
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
    print("Phase 106: The Gravity Equation")
    print("  Does attraction follow an inverse-square law?")
    print("=" * 60)
    
    latents = np.load(os.path.join(DATA_DIR, 'rosetta_latents_v2.npz'))
    dataset = json.load(open(os.path.join(DATA_DIR, 'rosetta_dataset_v2.json'), 'r', encoding='utf-8'))
    ast_vectors = latents['ast']
    sources = [item['source'] for item in dataset['dataset']]
    
    func_to_vecs = {}
    for i, src in enumerate(sources):
        if src not in func_to_vecs: func_to_vecs[src] = []
        func_to_vecs[src].append(ast_vectors[i])
    func_means = {s: np.mean(v, axis=0) for s, v in func_to_vecs.items()}
    unique_funcs = list(func_means.keys())
    all_vecs = np.array([func_means[f] for f in unique_funcs])
    n = len(unique_funcs)
    
    # Define "mass" of a function
    func_mass = {}
    for f in unique_funcs:
        try:
            tree = ast.parse(f)
            nodes = sum(1 for _ in ast.walk(tree))
            func_mass[f] = nodes
        except:
            func_mass[f] = 1
    
    # Compute pairwise distances and "attraction" (similarity)
    print("  Computing pairwise gravity...")
    distances = []
    attractions = []
    mass_products = []
    
    # Sample pairs (full n^2 would be too large)
    np.random.seed(42)
    n_pairs = min(5000, n * (n - 1) // 2)
    sampled_pairs = set()
    while len(sampled_pairs) < n_pairs:
        i, j = np.random.randint(0, n, 2)
        if i != j and (i, j) not in sampled_pairs and (j, i) not in sampled_pairs:
            sampled_pairs.add((i, j))
    
    for i, j in sampled_pairs:
        d = np.linalg.norm(all_vecs[i] - all_vecs[j])
        if d < 1e-6: continue
        m1 = func_mass[unique_funcs[i]]
        m2 = func_mass[unique_funcs[j]]
        cos_sim = np.dot(all_vecs[i], all_vecs[j]) / (np.linalg.norm(all_vecs[i]) * np.linalg.norm(all_vecs[j]) + 1e-10)
        attraction = max(0, cos_sim)  # Only attractive forces
        
        distances.append(d)
        attractions.append(attraction)
        mass_products.append(m1 * m2)
    
    distances = np.array(distances)
    attractions = np.array(attractions)
    mass_products = np.array(mass_products)
    
    # Fit power law: attraction ~ G * mass_product / d^alpha
    valid = (distances > 0.01) & (attractions > 0)
    d_valid = distances[valid]
    a_valid = attractions[valid]
    m_valid = mass_products[valid]
    
    # Fit: log(attraction) = log(G) + beta*log(mass) - alpha*log(d)
    log_d = np.log(d_valid + 1e-10)
    log_a = np.log(a_valid + 1e-10)
    log_m = np.log(m_valid + 1e-10)
    
    # Linear regression: log_a = c0 + c1*log_m + c2*log_d
    A_mat = np.column_stack([np.ones(len(log_d)), log_m, log_d])
    try:
        coeffs, residuals, _, _ = np.linalg.lstsq(A_mat, log_a, rcond=None)
        c0, beta, neg_alpha = coeffs
        alpha = -neg_alpha  # Should be positive (attraction decreases with distance)
        G = np.exp(c0)
        
        # R2
        predicted = A_mat @ coeffs
        ss_res = np.sum((log_a - predicted)**2)
        ss_tot = np.sum((log_a - np.mean(log_a))**2)
        r2 = 1 - ss_res / ss_tot
    except:
        alpha = 0; beta = 0; G = 0; r2 = 0
    
    print(f"\n--- The Gravity Equation ---")
    print(f"  F = G * M^beta / d^alpha")
    print(f"  G (gravitational constant): {G:.6f}")
    print(f"  alpha (distance exponent):  {alpha:.3f}  (2.0 = inverse-square)")
    print(f"  beta (mass exponent):       {beta:.3f}  (1.0 = linear)")
    print(f"  R2 of fit:                  {r2:.4f}")
    print(f"  {'INVERSE-SQUARE LAW!' if 1.5 < alpha < 2.5 else f'Power law with alpha={alpha:.2f}'}")
    
    # Distance distribution analysis
    print(f"\n--- Distance Statistics ---")
    print(f"  Mean distance:   {np.mean(distances):.4f}")
    print(f"  Median distance: {np.median(distances):.4f}")
    
    # Hub mass correlation
    from scipy.spatial.distance import pdist, squareform
    dist_mat = squareform(pdist(all_vecs))
    nn_counts = []
    masses = []
    threshold = np.percentile(dist_mat[dist_mat > 0], 15)
    for i, f in enumerate(unique_funcs):
        neighbors = np.sum(dist_mat[i] < threshold) - 1
        nn_counts.append(neighbors)
        masses.append(func_mass[f])
    
    mass_degree_corr = np.corrcoef(masses, nn_counts)[0, 1]
    print(f"\n  Mass-Degree correlation: {mass_degree_corr:.3f}")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Phase 106: The Gravity Equation (alpha={alpha:.2f})', fontsize=14, fontweight='bold')
    
    axes[0].scatter(d_valid, a_valid, alpha=0.1, s=5, c=np.log(m_valid+1), cmap='viridis')
    axes[0].set_xlabel('Distance d'); axes[0].set_ylabel('Attraction (cosine sim)')
    axes[0].set_title(f'Gravity: F ~ M^{beta:.2f} / d^{alpha:.2f} (R2={r2:.3f})')
    
    axes[1].scatter(log_d, log_a, alpha=0.1, s=3, color='#2196F3')
    x_fit = np.linspace(log_d.min(), log_d.max(), 100)
    y_fit = c0 + beta * np.mean(log_m) + neg_alpha * x_fit
    axes[1].plot(x_fit, y_fit, 'r-', linewidth=2, label=f'slope={neg_alpha:.2f}')
    axes[1].set_xlabel('log(distance)'); axes[1].set_ylabel('log(attraction)')
    axes[1].set_title('Log-Log Gravity Plot')
    axes[1].legend()
    
    axes[2].scatter(masses, nn_counts, alpha=0.3, s=20, color='#4CAF50')
    axes[2].set_xlabel('Function Mass (AST nodes)')
    axes[2].set_ylabel('Number of Neighbors')
    axes[2].set_title(f'Mass-Degree Corr={mass_degree_corr:.3f}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase106_gravity.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    results = {
        'phase': 106, 'title': 'The Gravity Equation',
        'G': float(G), 'alpha': float(alpha), 'beta': float(beta),
        'r2': float(r2), 'mass_degree_corr': float(mass_degree_corr),
        'mean_distance': float(np.mean(distances)),
        'law': f'Software gravity: F = {G:.4f} * M^{beta:.2f} / d^{alpha:.2f} (R2={r2:.3f}). {"Inverse-square law confirmed!" if 1.5<alpha<2.5 else f"Power law alpha={alpha:.2f}"}'
    }
    with open(os.path.join(RESULTS_DIR, 'phase106_gravity.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nPhase 106 complete!")
    return results

if __name__ == '__main__':
    main()
