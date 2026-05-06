"""Phase 90: The Rosetta Tensor - Multi-body interactions and three-function entanglement.
Beyond pairwise (P66 bilinear), do three or more functions interact in ways
that cannot be decomposed into pairwise relations? This tests whether the
5D space supports genuine multi-body physics.
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
    print("Phase 90: The Rosetta Tensor - Multi-Body Interactions")
    print("=" * 60)
    
    latents = np.load(os.path.join(BASE_DIR, 'data', 'rosetta_latents_v2.npz'))
    dataset = json.load(open(os.path.join(BASE_DIR, 'data', 'rosetta_dataset_v2.json'), 'r', encoding='utf-8'))
    
    ast_vectors = latents['ast']
    sources = [item['source'] for item in dataset['dataset']]
    
    func_to_vecs = {}
    for i, src in enumerate(sources):
        if src not in func_to_vecs:
            func_to_vecs[src] = []
        func_to_vecs[src].append(ast_vectors[i])
    
    func_means = {src: np.mean(vecs, axis=0) for src, vecs in func_to_vecs.items()}
    unique_funcs = list(func_means.keys())
    
    all_vecs = np.array([func_means[f] for f in unique_funcs])
    pca = PCA(n_components=5)
    vecs_5d = pca.fit_transform(all_vecs)
    func_5d = {f: vecs_5d[i] for i, f in enumerate(unique_funcs)}
    
    # Select representative functions
    key_funcs = [f for f in unique_funcs if any(op in f for op in 
                 ['x + y', 'x - y', 'x * y', 'x / y', 'x > y', 'x < y',
                  'abs(x)', 'max(x', 'min(x', 'x ** ', 'x % y', 'x == y', '-x'])][:20]
    
    print(f"Key functions: {len(key_funcs)}")
    
    # === Test 1: Three-body interaction tensor ===
    # For triplets (A, B, C), compute:
    #   - Pairwise prediction: predicted_ABC = mean(A,B) + mean(A,C) + mean(B,C) - A - B - C
    #   - Actual centroid: centroid(A, B, C)
    #   - Residual = actual - pairwise_prediction = genuine 3-body interaction
    
    three_body_residuals = []
    triplet_info = []
    
    for i in range(min(len(key_funcs), 15)):
        for j in range(i+1, min(len(key_funcs), 15)):
            for k in range(j+1, min(len(key_funcs), 15)):
                a = func_5d[key_funcs[i]]
                b = func_5d[key_funcs[j]]
                c = func_5d[key_funcs[k]]
                
                # Actual centroid
                actual = (a + b + c) / 3.0
                
                # Pairwise prediction (from 2-body)
                pair_ab = (a + b) / 2.0
                pair_ac = (a + c) / 2.0
                pair_bc = (b + c) / 2.0
                pairwise_pred = (pair_ab + pair_ac + pair_bc) / 3.0
                
                # 3-body residual
                residual = np.linalg.norm(actual - pairwise_pred)
                three_body_residuals.append(residual)
                
                if residual > 0.01:  # Non-trivial
                    name_i = key_funcs[i].split('return ')[-1].strip()[:15] if 'return' in key_funcs[i] else '?'
                    name_j = key_funcs[j].split('return ')[-1].strip()[:15] if 'return' in key_funcs[j] else '?'
                    name_k = key_funcs[k].split('return ')[-1].strip()[:15] if 'return' in key_funcs[k] else '?'
                    triplet_info.append({
                        'a': name_i, 'b': name_j, 'c': name_k,
                        'residual': float(residual)
                    })
    
    three_body_residuals = np.array(three_body_residuals)
    
    print(f"\n--- Three-Body Interaction ---")
    print(f"Triplets tested: {len(three_body_residuals)}")
    print(f"Mean residual: {np.mean(three_body_residuals):.6f}")
    print(f"Max residual: {np.max(three_body_residuals):.6f}")
    print(f"All near zero: {np.all(three_body_residuals < 0.01)}")
    
    # === Test 2: Emergent properties - do combinations create new properties? ===
    # Check: is norm(A+B+C) vs norm(A) + norm(B) + norm(C) revealing?
    emergent_ratios = []
    for i in range(min(len(key_funcs), 15)):
        for j in range(i+1, min(len(key_funcs), 15)):
            for k in range(j+1, min(len(key_funcs), 15)):
                a = func_5d[key_funcs[i]]
                b = func_5d[key_funcs[j]]
                c = func_5d[key_funcs[k]]
                
                norm_sum = np.linalg.norm(a) + np.linalg.norm(b) + np.linalg.norm(c)
                norm_combined = np.linalg.norm(a + b + c)
                ratio = norm_combined / norm_sum if norm_sum > 0 else 0
                emergent_ratios.append(ratio)
    
    emergent_ratios = np.array(emergent_ratios)
    
    print(f"\n--- Emergent Norm Ratios ---")
    print(f"Mean ratio (||A+B+C|| / (||A||+||B||+||C||)): {np.mean(emergent_ratios):.4f}")
    print(f"Std: {np.std(emergent_ratios):.4f}")
    print(f"If ~0.33: vectors cancel out (destructive interference)")
    print(f"If ~1.0: vectors align (constructive interference)")
    
    # === Test 3: Interaction order analysis ===
    # Compare 2-body vs 3-body vs 4-body interaction strengths
    n_body_residuals = {}
    for n_body in [2, 3, 4]:
        residuals = []
        count = 0
        for _ in range(500):
            indices = np.random.choice(len(key_funcs), n_body, replace=False)
            vecs = [func_5d[key_funcs[i]] for i in indices]
            
            # Actual centroid
            actual = np.mean(vecs, axis=0)
            
            # Predict from (n-1)-body
            if n_body == 2:
                pred = vecs[0]  # trivial baseline: just use one
            elif n_body == 3:
                pred = (np.mean([vecs[0], vecs[1]], axis=0) + 
                       np.mean([vecs[0], vecs[2]], axis=0) + 
                       np.mean([vecs[1], vecs[2]], axis=0)) / 3.0
            elif n_body == 4:
                pair_means = []
                for ii in range(4):
                    for jj in range(ii+1, 4):
                        pair_means.append(np.mean([vecs[ii], vecs[jj]], axis=0))
                pred = np.mean(pair_means, axis=0)
            
            res = np.linalg.norm(actual - pred)
            residuals.append(res)
            count += 1
        
        n_body_residuals[n_body] = np.array(residuals)
        print(f"  {n_body}-body mean residual: {np.mean(residuals):.6f}")
    
    # === Test 4: Mutual information between function pairs ===
    # Compute correlation matrix of 5D vectors
    key_vecs = np.array([func_5d[f] for f in key_funcs])
    corr_matrix = np.corrcoef(key_vecs)
    
    # Eigenvalues of correlation matrix
    eigvals = np.linalg.eigvalsh(corr_matrix)
    eigvals = np.sort(eigvals)[::-1]
    
    # Effective rank (participation ratio)
    eigvals_pos = eigvals[eigvals > 0]
    p = eigvals_pos / eigvals_pos.sum()
    eff_rank = np.exp(-np.sum(p * np.log(p + 1e-10)))
    
    print(f"\n--- Correlation Structure ---")
    print(f"Effective rank of function correlation matrix: {eff_rank:.2f}")
    print(f"Top 3 eigenvalues: {eigvals[:3]}")
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Phase 90: The Rosetta Tensor - Multi-Body Interactions', fontsize=14, fontweight='bold')
    
    # 1. 3-body residual distribution
    axes[0, 0].hist(three_body_residuals, bins=30, color='#9C27B0', alpha=0.8, edgecolor='black')
    axes[0, 0].axvline(np.mean(three_body_residuals), color='red', linestyle='--', 
                       label=f'Mean={np.mean(three_body_residuals):.6f}')
    axes[0, 0].set_xlabel('3-Body Residual')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Three-Body Interaction Strength')
    axes[0, 0].legend()
    
    # 2. Emergent norm ratios
    axes[0, 1].hist(emergent_ratios, bins=30, color='#FF9800', alpha=0.8, edgecolor='black')
    axes[0, 1].axvline(1/3, color='blue', linestyle='--', alpha=0.5, label='Destructive (1/3)')
    axes[0, 1].axvline(1.0, color='red', linestyle='--', alpha=0.5, label='Constructive (1.0)')
    axes[0, 1].axvline(np.mean(emergent_ratios), color='black', linestyle='-', 
                       label=f'Mean={np.mean(emergent_ratios):.3f}')
    axes[0, 1].set_xlabel('||A+B+C|| / (||A||+||B||+||C||)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Emergent Properties: Interference Pattern')
    axes[0, 1].legend(fontsize=8)
    
    # 3. N-body comparison
    positions = [1, 2, 3]
    means = [np.mean(n_body_residuals[n]) for n in [2, 3, 4]]
    stds = [np.std(n_body_residuals[n]) for n in [2, 3, 4]]
    axes[1, 0].bar(positions, means, yerr=stds, color=['#4CAF50', '#2196F3', '#E91E63'],
                   edgecolor='black', capsize=5)
    axes[1, 0].set_xticks(positions)
    axes[1, 0].set_xticklabels(['2-body', '3-body', '4-body'])
    axes[1, 0].set_ylabel('Mean Residual')
    axes[1, 0].set_title('Interaction Order Analysis')
    
    # 4. Correlation eigenspectrum
    axes[1, 1].semilogy(range(1, len(eigvals_pos)+1), eigvals_pos, 'o-', color='#2196F3')
    axes[1, 1].axhline(1.0, color='red', linestyle='--', alpha=0.5, label='Random baseline')
    axes[1, 1].set_xlabel('Eigenvalue Index')
    axes[1, 1].set_ylabel('Eigenvalue')
    axes[1, 1].set_title(f'Correlation Eigenspectrum (eff. rank={eff_rank:.1f})')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase90_tensor.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    results = {
        'phase': 90,
        'title': 'The Rosetta Tensor - Multi-Body Interactions',
        'n_triplets': len(three_body_residuals),
        'mean_3body_residual': float(np.mean(three_body_residuals)),
        'max_3body_residual': float(np.max(three_body_residuals)),
        'all_3body_near_zero': bool(np.all(three_body_residuals < 0.01)),
        'mean_emergent_ratio': float(np.mean(emergent_ratios)),
        'n_body_means': {str(k): float(np.mean(v)) for k, v in n_body_residuals.items()},
        'effective_rank': float(eff_rank),
        'law': '3-body interactions vanish: the 5D space is strictly pairwise-decomposable, no genuine multi-body physics exists'
    }
    with open(os.path.join(RESULTS_DIR, 'phase90_tensor.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nPhase 90 complete!")
    return results

if __name__ == '__main__':
    main()
