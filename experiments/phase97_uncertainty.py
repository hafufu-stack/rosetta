"""Phase 97: The Uncertainty Principle - Can you know both structure and behavior?
In quantum mechanics, position*momentum >= hbar/2.
Here: can you know both AST (structure) and Bytecode (behavior) simultaneously?
Measure the product of uncertainties in the dual representation.
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
    print("Phase 97: The Uncertainty Principle")
    print("  Can you know both structure AND behavior precisely?")
    print("=" * 60)
    
    latents = np.load(os.path.join(BASE_DIR, 'data', 'rosetta_latents_v2.npz'))
    dataset = json.load(open(os.path.join(BASE_DIR, 'data', 'rosetta_dataset_v2.json'), 'r', encoding='utf-8'))
    ast_vectors = latents['ast']
    bc_vectors = latents['bc']
    sources = [item['source'] for item in dataset['dataset']]
    
    # Group by function
    func_ast = {}; func_bc = {}
    for i, src in enumerate(sources):
        if src not in func_ast:
            func_ast[src] = []; func_bc[src] = []
        func_ast[src].append(ast_vectors[i])
        func_bc[src].append(bc_vectors[i])
    
    unique = list(func_ast.keys())
    
    # For each function with multiple samples: measure spread in AST vs BC
    ast_spreads = []
    bc_spreads = []
    products = []
    func_names = []
    
    for f in unique:
        if len(func_ast[f]) < 3:
            continue
        a_vecs = np.array(func_ast[f])
        b_vecs = np.array(func_bc[f])
        
        # Spread = mean distance from centroid (uncertainty)
        a_spread = np.mean(np.linalg.norm(a_vecs - np.mean(a_vecs, axis=0), axis=1))
        b_spread = np.mean(np.linalg.norm(b_vecs - np.mean(b_vecs, axis=0), axis=1))
        
        ast_spreads.append(a_spread)
        bc_spreads.append(b_spread)
        products.append(a_spread * b_spread)
        func_names.append(f[:40])
    
    ast_spreads = np.array(ast_spreads)
    bc_spreads = np.array(bc_spreads)
    products = np.array(products)
    
    print(f"  Functions with 3+ samples: {len(products)}")
    print(f"  AST spread: mean={np.mean(ast_spreads):.4f}, std={np.std(ast_spreads):.4f}")
    print(f"  BC spread:  mean={np.mean(bc_spreads):.4f}, std={np.std(bc_spreads):.4f}")
    print(f"  Product:    mean={np.mean(products):.6f}, min={np.min(products):.6f}")
    
    # Is there a lower bound? (Uncertainty principle = product has minimum)
    min_product = np.min(products)
    percentile_5 = np.percentile(products, 5)
    
    print(f"\n--- Uncertainty Analysis ---")
    print(f"  Min product (hbar/2 analog): {min_product:.6f}")
    print(f"  5th percentile:              {percentile_5:.6f}")
    print(f"  Median product:              {np.median(products):.6f}")
    
    # Correlation between AST and BC spreads
    correlation = np.corrcoef(ast_spreads, bc_spreads)[0, 1]
    print(f"  Correlation(AST_spread, BC_spread): {correlation:.4f}")
    print(f"  {'ANTI-correlated (uncertainty-like!)' if correlation < -0.3 else 'Positive or weak'}")
    
    # Per-dimension uncertainty
    print(f"\n--- Per-Dimension Uncertainty ---")
    dim_products = []
    for d in range(min(64, ast_vectors.shape[1])):
        a_std = np.std(ast_vectors[:, d])
        b_std = np.std(bc_vectors[:, d])
        dim_products.append(a_std * b_std)
    dim_products = np.array(dim_products)
    print(f"  Mean dim product: {np.mean(dim_products):.6f}")
    print(f"  Min dim product:  {np.min(dim_products):.6f}")
    
    # Commutator test: does [AST, BC] = 0?
    # Use correlation matrix as proxy for commutator
    ast_means = np.array([np.mean(func_ast[f], axis=0) for f in unique])
    bc_means = np.array([np.mean(func_bc[f], axis=0) for f in unique])
    
    # Cross-correlation matrices
    C_ab = np.corrcoef(ast_means.T, bc_means.T)[:64, 64:]
    C_ba = np.corrcoef(bc_means.T, ast_means.T)[:64, 64:]
    commutator_norm = np.linalg.norm(C_ab - C_ba.T)
    
    print(f"\n--- Commutator Test ---")
    print(f"  ||[AST, BC]|| = {commutator_norm:.6f}")
    print(f"  {'NON-COMMUTING (quantum-like!)' if commutator_norm > 0.1 else 'Nearly commuting (classical)'}")
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Phase 97: The Uncertainty Principle of Code', fontsize=14, fontweight='bold')
    
    axes[0, 0].scatter(ast_spreads, bc_spreads, alpha=0.5, c=products, cmap='viridis', s=20)
    axes[0, 0].set_xlabel('AST Spread (structure uncertainty)')
    axes[0, 0].set_ylabel('BC Spread (behavior uncertainty)')
    axes[0, 0].set_title(f'Uncertainty Scatter (corr={correlation:.3f})')
    cb = plt.colorbar(axes[0, 0].collections[0], ax=axes[0, 0])
    cb.set_label('Product')
    
    axes[0, 1].hist(products, bins=30, color='#E91E63', edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(min_product, color='blue', linestyle='--', label=f'min={min_product:.4f}')
    axes[0, 1].axvline(percentile_5, color='green', linestyle='--', label=f'5th pct={percentile_5:.4f}')
    axes[0, 1].set_xlabel('AST_spread * BC_spread')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Uncertainty Product Distribution')
    axes[0, 1].legend()
    
    axes[1, 0].bar(range(len(dim_products)), dim_products, color='#2196F3', alpha=0.7)
    axes[1, 0].set_xlabel('Dimension')
    axes[1, 0].set_ylabel('sigma_AST * sigma_BC')
    axes[1, 0].set_title('Per-Dimension Uncertainty Product')
    
    im = axes[1, 1].imshow(C_ab[:20, :20], cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    axes[1, 1].set_xlabel('BC dimension')
    axes[1, 1].set_ylabel('AST dimension')
    axes[1, 1].set_title(f'Cross-Correlation (commutator={commutator_norm:.3f})')
    plt.colorbar(im, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase97_uncertainty.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    results = {
        'phase': 97, 'title': 'The Uncertainty Principle of Code',
        'n_functions': len(products),
        'mean_ast_spread': float(np.mean(ast_spreads)),
        'mean_bc_spread': float(np.mean(bc_spreads)),
        'min_product': float(min_product),
        'percentile_5_product': float(percentile_5),
        'median_product': float(np.median(products)),
        'spread_correlation': float(correlation),
        'commutator_norm': float(commutator_norm),
        'per_dim_mean_product': float(np.mean(dim_products)),
        'law': f'Code uncertainty: AST-BC spread correlation={correlation:.3f}, commutator_norm={commutator_norm:.3f}. {"Non-commuting (quantum-like)" if commutator_norm > 0.1 else "Nearly commuting (classical)"}'
    }
    with open(os.path.join(RESULTS_DIR, 'phase97_uncertainty.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nPhase 97 complete!")
    return results

if __name__ == '__main__':
    main()
