"""Phase 130: The Turing Horizon
Map halting/non-halting programs and find the geometric boundary
(event horizon) separating decidable from undecidable computation.
"""
import os, json, sys, ast, dis, io
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXP2_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(EXP2_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')

def classify_halting(source):
    """Classify a function as 'halts', 'maybe', or 'diverges'."""
    try:
        tree = ast.parse(source)
    except Exception:
        return 'unknown'
    
    code_str = source.lower()
    
    # Check for explicit loops or recursion indicators
    has_while = 'while' in code_str
    has_for = 'for' in code_str
    has_recursion = False
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            fname = node.name
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    if isinstance(child.func, ast.Name) and child.func.id == fname:
                        has_recursion = True
    
    if has_recursion or has_while:
        return 'maybe'
    elif has_for:
        return 'halts_bounded'
    else:
        return 'halts'

def main():
    print("=" * 60)
    print("Phase 130: The Turing Horizon")
    print("  Where is the geometric boundary of decidability?")
    print("=" * 60)
    
    latents = np.load(os.path.join(DATA_DIR, 'rosetta_latents_v2.npz'))
    dataset = json.load(open(os.path.join(DATA_DIR, 'rosetta_dataset_v2.json'), 'r', encoding='utf-8'))
    ast_vectors = latents['ast']
    bc_vectors = latents['bc']
    sources = [item['source'] for item in dataset['dataset']]
    
    func_ast = {}
    for i, src in enumerate(sources):
        if src not in func_ast: func_ast[src] = []
        func_ast[src].append(ast_vectors[i])
    unique_funcs = list(func_ast.keys())
    ast_m = np.array([np.mean(func_ast[f], axis=0) for f in unique_funcs])
    n = len(unique_funcs)
    
    # 1. Classify all functions
    classifications = [classify_halting(src) for src in unique_funcs]
    
    class_counts = {}
    for c in classifications:
        class_counts[c] = class_counts.get(c, 0) + 1
    print(f"  Classifications: {class_counts}")
    
    # 2. Compute "halting complexity" for each function
    # More complex = closer to the Turing horizon
    halting_scores = []
    for idx, src in enumerate(unique_funcs):
        try:
            tree = ast.parse(src)
            n_nodes = sum(1 for _ in ast.walk(tree))
            depth = max((sum(1 for _ in ast.walk(child)) for child in ast.walk(tree)), default=0)
            
            # Bytecode instruction count
            code = compile(src, '<string>', 'exec')
            bc_count = 0
            for const in code.co_consts:
                if hasattr(const, 'co_code'):
                    bc_count = len(const.co_code)
            
            score = n_nodes + depth + bc_count / 10
        except Exception:
            score = 0
        halting_scores.append(score)
    
    halting_scores = np.array(halting_scores)
    
    # 3. Gravitational field around the "Turing horizon"
    # Compute mass (neighbor density) at each point
    pca = PCA(n_components=10).fit(ast_m)
    ast_pca = pca.transform(ast_m)
    
    dist_mat = cdist(ast_m, ast_m)
    np.fill_diagonal(dist_mat, np.inf)
    
    # Local density = number of neighbors within radius
    radius = np.median(dist_mat[dist_mat < np.inf])
    densities = np.sum(dist_mat < radius, axis=1)
    
    # Gravitational potential
    G = 1.0
    potentials = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                d = dist_mat[i, j]
                potentials[i] -= G * densities[j] / (d + 0.01)
    
    # 4. Find the Turing horizon: boundary between simple and complex functions
    median_complexity = np.median(halting_scores)
    simple_mask = halting_scores <= median_complexity
    complex_mask = halting_scores > median_complexity
    
    # Distance between simple and complex centroids
    simple_centroid = np.mean(ast_m[simple_mask], axis=0)
    complex_centroid = np.mean(ast_m[complex_mask], axis=0)
    horizon_distance = np.linalg.norm(simple_centroid - complex_centroid)
    
    # Midpoint = the Turing horizon
    horizon_point = (simple_centroid + complex_centroid) / 2
    
    # How many functions are near the horizon?
    dist_to_horizon = np.linalg.norm(ast_m - horizon_point.reshape(1,-1), axis=1)
    near_horizon = np.sum(dist_to_horizon < horizon_distance * 0.3)
    
    print(f"\n--- Turing Horizon ---")
    print(f"  Horizon distance: {horizon_distance:.4f}")
    print(f"  Functions near horizon: {near_horizon}/{n}")
    print(f"  Mean gravity (simple): {np.mean(potentials[simple_mask]):.2f}")
    print(f"  Mean gravity (complex): {np.mean(potentials[complex_mask]):.2f}")
    
    # 5. Escape velocity: can a function in the complex zone reach simplicity?
    escape_velocities = []
    for idx in range(n):
        if complex_mask[idx]:
            v_escape = np.sqrt(2 * abs(potentials[idx]))
            escape_velocities.append(float(v_escape))
    
    mean_escape = np.mean(escape_velocities) if escape_velocities else 0
    print(f"  Mean escape velocity (complex -> simple): {mean_escape:.4f}")
    
    # 6. Event horizon radius: below this, no information escapes
    # Schwarzschild analog: r_s = 2GM/c^2
    total_mass = np.sum(densities)
    r_schwarzschild = 2 * G * total_mass / (n * np.mean(dist_mat[dist_mat < np.inf]) ** 2)
    print(f"  Schwarzschild radius analog: {r_schwarzschild:.4f}")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Phase 130: The Turing Horizon', fontsize=14, fontweight='bold')
    
    pca_2d = PCA(n_components=2).fit_transform(ast_m)
    sc = axes[0].scatter(pca_2d[:,0], pca_2d[:,1], c=halting_scores, cmap='RdYlGn_r', s=20, alpha=0.7)
    plt.colorbar(sc, ax=axes[0], label='Halting complexity')
    horizon_2d = PCA(n_components=2).fit(ast_m).transform(horizon_point.reshape(1,-1))
    axes[0].scatter(horizon_2d[0,0], horizon_2d[0,1], s=200, c='black', marker='D', zorder=10, label='Horizon')
    axes[0].legend(); axes[0].set_title('Turing Horizon in PC1-PC2')
    
    sc2 = axes[1].scatter(pca_2d[:,0], pca_2d[:,1], c=potentials, cmap='inferno', s=20, alpha=0.7)
    plt.colorbar(sc2, ax=axes[1], label='Grav. potential')
    axes[1].set_title('Gravitational field')
    
    axes[2].scatter(halting_scores, potentials, s=15, alpha=0.5, c='#2196F3')
    axes[2].axvline(median_complexity, color='red', linestyle='--', label=f'Horizon (complexity={median_complexity:.0f})')
    axes[2].set_xlabel('Halting complexity'); axes[2].set_ylabel('Gravitational potential')
    axes[2].set_title('Complexity vs Gravity'); axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase130_turing_horizon.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    results = {
        'phase': 130, 'title': 'The Turing Horizon',
        'classifications': class_counts,
        'horizon_distance': float(horizon_distance),
        'near_horizon_count': int(near_horizon),
        'mean_escape_velocity': float(mean_escape),
        'schwarzschild_radius': float(r_schwarzschild),
        'law': f'Turing horizon at distance {horizon_distance:.3f}. {near_horizon} functions near boundary. Escape velocity={mean_escape:.3f}. Schwarzschild radius={r_schwarzschild:.3f}.'
    }
    with open(os.path.join(RESULTS_DIR, 'phase130_turing_horizon.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nPhase 130 complete!")
    return results

if __name__ == '__main__':
    main()
