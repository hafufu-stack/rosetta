"""Phase 127: ER=EPR - Semantic Teleportation via Entanglement
Use Schmidt decomposition to find maximally entangled pairs,
then test if repairing one side 'teleports' the fix to the other.
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
    print("Phase 127: ER=EPR - Semantic Teleportation")
    print("  Can bug fixes teleport via entanglement?")
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
    
    # 1. Find maximally entangled pairs via cross-correlation
    cross_corr = ast_m @ bc_m.T  # (n, n) cross-space correlation
    np.fill_diagonal(cross_corr, -np.inf)
    
    # Top entangled pairs (highest cross-space correlation to DIFFERENT functions)
    entangled_pairs = []
    for i in range(n):
        j = np.argmax(cross_corr[i])
        corr = cross_corr[i, j]
        if i != j:
            entangled_pairs.append((i, j, float(corr)))
    entangled_pairs.sort(key=lambda x: x[2], reverse=True)
    
    print("--- Top entangled pairs (AST_i <-> BC_j) ---")
    for i, j, c in entangled_pairs[:5]:
        fi = unique_funcs[i].split('return ')[-1].strip()[:15]
        fj = unique_funcs[j].split('return ')[-1].strip()[:15]
        print(f"  AST({fi}) <-> BC({fj}): corr={c:.4f}")
    
    # 2. Teleportation test: modify AST of function A,
    # check if the entangled partner B's BC representation responds
    bug_pairs = [
        ('def f(x, y): return x + y', 'def f(x, y): return x - y'),
        ('def f(x, y): return x * y', 'def f(x, y): return x / y'),
        ('def f(x, y): return x > y', 'def f(x, y): return x < y'),
        ('def f(x, y): return max(x, y)', 'def f(x, y): return min(x, y)'),
        ('def f(x): return abs(x)', 'def f(x): return -x'),
        ('def f(x, y): return x == y', 'def f(x, y): return x != y'),
    ]
    
    teleport_results = []
    
    for buggy_src, target_src in bug_pairs:
        if buggy_src not in func_ast or target_src not in func_ast: continue
        buggy_idx = unique_funcs.index(buggy_src)
        target_idx = unique_funcs.index(target_src)
        
        buggy_ast = ast_m[buggy_idx]
        target_ast = ast_m[target_idx]
        
        # Repair vector in AST space
        repair_vec = target_ast - buggy_ast
        
        # Find the most entangled BC partner for buggy function
        bc_partner_idx = np.argmax(cross_corr[buggy_idx])
        
        # Apply repair in BC space via entanglement channel
        # Teleportation: map AST repair to BC space via cross-correlation structure
        W_teleport = bc_m.T @ np.linalg.pinv(ast_m.T)  # AST->BC mapping
        repair_bc = W_teleport @ repair_vec
        
        # Apply teleported repair
        repaired_bc = bc_m[buggy_idx] + repair_bc
        
        # Find nearest function to repaired BC vector
        bc_dists = np.linalg.norm(bc_m - repaired_bc.reshape(1,-1), axis=1)
        nearest_idx = np.argmin(bc_dists)
        nearest_func = unique_funcs[nearest_idx].split('return ')[-1].strip()[:15]
        
        success = nearest_idx == target_idx
        
        buggy_label = buggy_src.split('return ')[-1].strip()[:10]
        target_label = target_src.split('return ')[-1].strip()[:10]
        
        teleport_results.append({
            'buggy': buggy_label, 'target': target_label,
            'teleported_to': nearest_func, 'success': bool(success),
            'distance_to_target': float(bc_dists[target_idx])
        })
        
        status = "TELEPORTED!" if success else f"landed on: {nearest_func}"
        print(f"  {buggy_label} -> {target_label}: {status}")
    
    # 3. Entanglement strength vs teleportation success
    # Compute Bell inequality violation analog
    bell_violations = []
    np.random.seed(42)
    for _ in range(100):
        i, j = np.random.randint(0, n, 2)
        if i == j: continue
        # CHSH-like correlation
        # Measure AST_i, BC_i, AST_j, BC_j correlations
        a_b_same = np.dot(ast_m[i], bc_m[i]) / (np.linalg.norm(ast_m[i]) * np.linalg.norm(bc_m[i]) + 1e-10)
        a_b_diff = np.dot(ast_m[i], bc_m[j]) / (np.linalg.norm(ast_m[i]) * np.linalg.norm(bc_m[j]) + 1e-10)
        bell = abs(a_b_same) - abs(a_b_diff)
        bell_violations.append(float(bell))
    
    mean_bell = np.mean(bell_violations)
    print(f"\n--- Bell inequality analog ---")
    print(f"  Mean Bell violation: {mean_bell:.4f}")
    print(f"  {'Entanglement confirmed (>0)' if mean_bell > 0 else 'No entanglement'}")
    
    # Summary
    n_success = sum(1 for r in teleport_results if r['success'])
    total = len(teleport_results)
    rate = n_success / total * 100 if total > 0 else 0
    print(f"\n--- Teleportation success rate: {n_success}/{total} ({rate:.0f}%) ---")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Phase 127: ER=EPR Semantic Teleportation', fontsize=14, fontweight='bold')
    
    labels = [r['buggy'] for r in teleport_results]
    colors = ['#4CAF50' if r['success'] else '#F44336' for r in teleport_results]
    axes[0].barh(labels, [1 if r['success'] else 0 for r in teleport_results], color=colors, edgecolor='black')
    axes[0].set_xlabel('Success'); axes[0].set_title(f'Teleportation: {n_success}/{total} ({rate:.0f}%)')
    
    axes[1].hist(bell_violations, bins=30, color='#2196F3', edgecolor='black', alpha=0.7)
    axes[1].axvline(0, color='red', linestyle='--', label='Classical bound')
    axes[1].set_xlabel('Bell violation'); axes[1].set_title(f'Bell inequality (mean={mean_bell:.3f})')
    axes[1].legend()
    
    corr_matrix = cross_corr[:20, :20]
    im = axes[2].imshow(corr_matrix, cmap='RdBu_r', aspect='auto')
    plt.colorbar(im, ax=axes[2]); axes[2].set_title('AST-BC cross-correlation (top 20)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase127_teleportation.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    results = {
        'phase': 127, 'title': 'ER=EPR Semantic Teleportation',
        'teleportation_results': teleport_results,
        'success_rate_pct': float(rate),
        'mean_bell_violation': float(mean_bell),
        'law': f'Teleportation success: {n_success}/{total} ({rate:.0f}%). Bell violation={mean_bell:.4f}.'
    }
    with open(os.path.join(RESULTS_DIR, 'phase127_teleportation.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nPhase 127 complete!")
    return results

if __name__ == '__main__':
    main()
