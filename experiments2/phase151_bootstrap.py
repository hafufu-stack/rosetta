"""Phase 151: The Bootstrap Paradox
Opus original: What happens when the observer forces itself to the center?
P147 showed Rosetta is NEAR the center. Now we MOVE it to the exact center.
Does the universe adapt, break, or reveal something new?
"""
import os, json, sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXP2_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(EXP2_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')

def ast_fingerprint(source, dim=64):
    import ast
    try:
        tree = ast.parse(source)
        feat = np.zeros(dim)
        for i, node in enumerate(ast.walk(tree)):
            h = hash(type(node).__name__) % dim
            feat[h] += 1; feat[(h+1)%dim] += 0.1*min(i,20); feat[(h+7)%dim] += 0.05
        norm = np.linalg.norm(feat)
        return feat / norm if norm > 0 else feat
    except: return np.zeros(dim)

def main():
    print("=" * 60)
    print("Phase 151: The Bootstrap Paradox")
    print("  Force the observer to the center. What breaks?")
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
    centroid = np.mean(ast_m, axis=0)

    # 1. Measure ORIGINAL universe properties
    W_orig = bc_m.T @ np.linalg.pinv(ast_m.T)
    orig_compile_err = float(np.mean(np.linalg.norm(bc_m - (W_orig @ ast_m.T).T, axis=1)))
    orig_spread = float(np.std(np.linalg.norm(ast_m - centroid, axis=1)))

    C_ast = ast_m.T @ ast_m / n; C_bc = bc_m.T @ bc_m / n
    orig_comm = float(np.linalg.norm(C_ast @ C_bc - C_bc @ C_ast))

    # Original gravity
    dists_orig = np.sort(np.linalg.norm(ast_m[:, None] - ast_m[None, :], axis=2), axis=1)
    orig_nn_dist = float(np.mean(dists_orig[:, 1]))

    print(f"  ORIGINAL universe:")
    print(f"    Compile error: {orig_compile_err:.6f}")
    print(f"    Spread: {orig_spread:.6f}")
    print(f"    Commutator: {orig_comm:.6f}")
    print(f"    Mean NN distance: {orig_nn_dist:.6f}")

    # 2. Extract Rosetta's own code and create observer vector
    import ast
    rosetta_files = []
    for d in [os.path.join(BASE_DIR, 'experiments'), EXP2_DIR]:
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.endswith('.py') and f.startswith('phase'):
                    rosetta_files.append(os.path.join(d, f))

    rosetta_funcs = []
    for fp in rosetta_files:
        try:
            with open(fp, 'r', encoding='utf-8') as f:
                src = f.read()
            tree = ast.parse(src)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_src = ast.get_source_segment(src, node)
                    if func_src and 50 < len(func_src) < 5000:
                        rosetta_funcs.append(func_src)
        except: pass

    # Create mean observer vector
    rosetta_fps = np.array([ast_fingerprint(s) for s in rosetta_funcs])
    observer_vector = np.mean(rosetta_fps, axis=0)
    # Project into universe via weighted average
    cos_sims = ast_m @ observer_vector / (np.linalg.norm(ast_m, axis=1) * np.linalg.norm(observer_vector) + 1e-10)
    weights = np.maximum(cos_sims, 0) ** 2
    observer_pos = (weights[:, None] * ast_m).sum(axis=0) / (np.sum(weights) + 1e-10)

    observer_dist = float(np.linalg.norm(observer_pos - centroid))
    print(f"\n  Observer position: {observer_dist:.4f} from center")

    # 3. BOOTSTRAP: Force observer to exact center
    # Strategy: translate entire space so observer IS the centroid
    shift = centroid - observer_pos
    ast_bootstrapped = ast_m + shift  # Move everything
    # Now recompute centroid (should be near observer)
    new_centroid = np.mean(ast_bootstrapped, axis=0)

    # But the REAL bootstrap: warp space around the observer
    # Each point gets pulled toward the observer proportionally
    alpha_warp = 0.3  # Warp strength
    ast_warped = ast_m.copy()
    for i in range(n):
        direction = centroid - ast_m[i]
        dist_to_obs = np.linalg.norm(ast_m[i] - observer_pos)
        # Closer to observer = less warping, far = more warping
        warp_factor = alpha_warp * (1 - np.exp(-dist_to_obs))
        ast_warped[i] += direction * warp_factor

    # 4. Measure WARPED universe properties
    W_warped = bc_m.T @ np.linalg.pinv(ast_warped.T)
    warped_compile_err = float(np.mean(np.linalg.norm(bc_m - (W_warped @ ast_warped.T).T, axis=1)))
    warped_centroid = np.mean(ast_warped, axis=0)
    warped_spread = float(np.std(np.linalg.norm(ast_warped - warped_centroid, axis=1)))

    C_warp = ast_warped.T @ ast_warped / n
    warped_comm = float(np.linalg.norm(C_warp @ C_bc - C_bc @ C_warp))

    dists_warped = np.sort(np.linalg.norm(ast_warped[:, None] - ast_warped[None, :], axis=2), axis=1)
    warped_nn_dist = float(np.mean(dists_warped[:, 1]))

    print(f"\n  WARPED universe (observer at center):")
    print(f"    Compile error: {warped_compile_err:.6f} (delta={warped_compile_err-orig_compile_err:+.6f})")
    print(f"    Spread: {warped_spread:.6f} (delta={warped_spread-orig_spread:+.6f})")
    print(f"    Commutator: {warped_comm:.6f} (delta={warped_comm-orig_comm:+.6f})")
    print(f"    Mean NN distance: {warped_nn_dist:.6f} (delta={warped_nn_dist-orig_nn_dist:+.6f})")

    # 5. Does the universe resist? Or does it accommodate?
    compile_change = (warped_compile_err - orig_compile_err) / orig_compile_err * 100
    spread_change = (warped_spread - orig_spread) / orig_spread * 100
    comm_change = (warped_comm - orig_comm) / (orig_comm + 1e-10) * 100

    print(f"\n--- Bootstrap Result ---")
    print(f"  Compile error change: {compile_change:+.2f}%")
    print(f"  Spread change: {spread_change:+.2f}%")
    print(f"  Commutator change: {comm_change:+.2f}%")

    if abs(compile_change) < 5 and abs(spread_change) < 5:
        verdict = "UNIVERSE ACCOMMODATES (observer-invariant!)"
    elif compile_change > 20:
        verdict = "UNIVERSE BREAKS (observer destroys structure)"
    else:
        verdict = "UNIVERSE RESISTS (partial deformation)"
    print(f"  Verdict: {verdict}")

    # 6. Gauge invariance test: is the physics independent of where we put the origin?
    print(f"\n--- Gauge Invariance Test ---")
    random_shifts = np.random.randn(10, 64) * orig_spread
    gauge_errors = []
    for shift in random_shifts:
        ast_shifted = ast_m + shift
        W_shift = bc_m.T @ np.linalg.pinv(ast_shifted.T)
        err = float(np.mean(np.linalg.norm(bc_m - (W_shift @ ast_shifted.T).T, axis=1)))
        gauge_errors.append(err)

    gauge_variance = np.var(gauge_errors) / orig_compile_err ** 2
    print(f"  Compile error variance under shifts: {gauge_variance:.6f}")
    print(f"  Gauge invariant: {'YES' if gauge_variance < 0.01 else 'NO'}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Phase 151: The Bootstrap Paradox', fontsize=14, fontweight='bold')

    metrics = ['Compile err', 'Spread', 'Commutator', 'NN dist']
    orig_vals = [orig_compile_err, orig_spread, orig_comm, orig_nn_dist]
    warp_vals = [warped_compile_err, warped_spread, warped_comm, warped_nn_dist]
    x = np.arange(len(metrics))
    axes[0].bar(x - 0.2, orig_vals, 0.4, label='Original', color='#2196F3', edgecolor='black')
    axes[0].bar(x + 0.2, warp_vals, 0.4, label='Bootstrapped', color='#F44336', edgecolor='black')
    axes[0].set_xticks(x); axes[0].set_xticklabels(metrics, fontsize=8)
    axes[0].legend(); axes[0].set_title('Original vs Bootstrapped')

    changes = [compile_change, spread_change, comm_change]
    colors = ['#4CAF50' if abs(c) < 5 else '#F44336' for c in changes]
    axes[1].bar(['Compile', 'Spread', 'Commutator'], changes, color=colors, edgecolor='black')
    axes[1].set_ylabel('% Change'); axes[1].set_title(f'Deformation ({verdict[:20]})')
    axes[1].axhline(0, color='black', linewidth=0.5)

    axes[2].bar(range(len(gauge_errors)), gauge_errors, color='#9C27B0', edgecolor='black', alpha=0.7)
    axes[2].axhline(orig_compile_err, color='red', linestyle='--', label=f'Original={orig_compile_err:.4f}')
    axes[2].set_xlabel('Random shift'); axes[2].set_ylabel('Compile error')
    axes[2].set_title(f'Gauge invariance (var={gauge_variance:.5f})'); axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase151_bootstrap.png'), dpi=150, bbox_inches='tight')
    plt.close()

    results = {
        'phase': 151, 'title': 'The Bootstrap Paradox',
        'original': {'compile_err': orig_compile_err, 'spread': orig_spread, 'comm': orig_comm},
        'warped': {'compile_err': warped_compile_err, 'spread': warped_spread, 'comm': warped_comm},
        'changes_pct': {'compile': float(compile_change), 'spread': float(spread_change), 'comm': float(comm_change)},
        'verdict': verdict,
        'gauge_variance': float(gauge_variance),
        'law': f'Bootstrap: compile {compile_change:+.1f}%, spread {spread_change:+.1f}%, comm {comm_change:+.1f}%. {verdict}. Gauge variance={gauge_variance:.5f}.'
    }
    with open(os.path.join(RESULTS_DIR, 'phase151_bootstrap.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nPhase 151 complete!")
    return results

if __name__ == '__main__':
    main()
