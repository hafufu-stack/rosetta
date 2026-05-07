"""Phase 145: Imaginary Time & Tachyon Compilation
Wick rotation: multiply time axis by i to enter imaginary time.
Can we break the causal speed limit?
"""
import os, json, sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from sklearn.decomposition import PCA
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
    print("Phase 145: Imaginary Time & Tachyon Compilation")
    print("  Wick rotation into imaginary time")
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

    # PC2 = arrow of time (from P112)
    pca = PCA(n_components=5).fit(ast_m)
    ast_pca = pca.transform(ast_m)
    pc2 = ast_pca[:, 1]  # Time axis

    # 1. Wick rotation: t -> i*t (multiply time component by i)
    # In real space, this means: rotate the time axis by 90 degrees
    print("--- Wick Rotation ---")

    # Create complexified space: real part = spatial dims, imaginary = time
    ast_complex = ast_pca.copy().astype(complex)
    ast_complex[:, 1] *= 1j  # Wick rotation on PC2

    # Lorentzian -> Euclidean metric
    # In Lorentz: ds^2 = -dt^2 + dx^2
    # After Wick: ds^2 = dt_E^2 + dx^2 (all positive = Euclidean)

    # Compute distances in both metrics
    real_dists = np.linalg.norm(ast_pca[:20, None] - ast_pca[None, :20], axis=2)

    # Lorentzian distance (pseudo-Riemannian)
    lorentz_dists = np.zeros((20, 20))
    for i in range(20):
        for j in range(20):
            dt = ast_pca[i, 1] - ast_pca[j, 1]
            dx = np.linalg.norm(ast_pca[i, [0,2,3,4]] - ast_pca[j, [0,2,3,4]])
            lorentz_dists[i, j] = np.sqrt(abs(-dt**2 + dx**2))

    # Euclidean distance (after Wick rotation)
    eucl_dists = np.zeros((20, 20))
    for i in range(20):
        for j in range(20):
            diff = ast_complex[i, :5] - ast_complex[j, :5]
            eucl_dists[i, j] = np.sqrt(abs(np.sum(np.abs(diff)**2)))

    print(f"  Real distance range: [{np.min(real_dists[real_dists>0]):.4f}, {np.max(real_dists):.4f}]")
    print(f"  Lorentz distance range: [{np.min(lorentz_dists[lorentz_dists>0]):.4f}, {np.max(lorentz_dists):.4f}]")
    print(f"  Euclidean (Wick) range: [{np.min(eucl_dists[eucl_dists>0]):.4f}, {np.max(eucl_dists):.4f}]")

    # 2. Tachyon detection: pairs with spacelike separation in Lorentz metric
    # Spacelike = dx > dt (faster than light)
    n_spacelike = 0
    n_timelike = 0
    n_lightlike = 0
    tachyon_pairs = []

    for i in range(min(50, n)):
        for j in range(i+1, min(50, n)):
            dt = abs(ast_pca[i, 1] - ast_pca[j, 1])
            dx = np.linalg.norm(ast_pca[i, [0,2,3,4]] - ast_pca[j, [0,2,3,4]])
            if abs(dx - dt) < 0.01:
                n_lightlike += 1
            elif dx > dt:
                n_spacelike += 1
                if len(tachyon_pairs) < 5:
                    fi = unique_funcs[i].split('return ')[-1].strip()[:12]
                    fj = unique_funcs[j].split('return ')[-1].strip()[:12]
                    tachyon_pairs.append({'a': fi, 'b': fj, 'dx': float(dx), 'dt': float(dt), 'speed': float(dx/max(dt,1e-10))})
            else:
                n_timelike += 1

    total_pairs = n_spacelike + n_timelike + n_lightlike
    print(f"\n--- Causal Structure ---")
    print(f"  Timelike (causal): {n_timelike} ({n_timelike/total_pairs*100:.1f}%)")
    print(f"  Spacelike (tachyon): {n_spacelike} ({n_spacelike/total_pairs*100:.1f}%)")
    print(f"  Lightlike: {n_lightlike} ({n_lightlike/total_pairs*100:.1f}%)")

    print(f"\n--- Top Tachyon Pairs ---")
    for tp in tachyon_pairs:
        print(f"  {tp['a']} <-> {tp['b']}: speed={tp['speed']:.2f}c")

    # 3. Time travel: can Wick-rotated compile matrix predict "future" functions?
    print("\n--- Retrocausal Compilation ---")
    # Sort by complexity (time axis)
    import ast as ast_mod
    complexities = []
    for src in unique_funcs:
        try:
            tree = ast_mod.parse(src)
            complexities.append(sum(1 for _ in ast_mod.walk(tree)))
        except: complexities.append(0)
    sort_idx = np.argsort(complexities)

    # Train on "past" (simple), predict "future" (complex)
    n_past = n // 2
    past_idx = sort_idx[:n_past]
    future_idx = sort_idx[n_past:]

    W_past = bc_m[past_idx].T @ np.linalg.pinv(ast_m[past_idx].T)
    future_pred = (W_past @ ast_m[future_idx].T).T
    future_err = np.mean(np.linalg.norm(bc_m[future_idx] - future_pred, axis=1))

    # Full compile error for comparison
    W_full = bc_m.T @ np.linalg.pinv(ast_m.T)
    full_err = np.mean(np.linalg.norm(bc_m - (W_full @ ast_m.T).T, axis=1))

    retrocausal_ratio = 1 - (future_err / (full_err + 1e-10))
    print(f"  Past->Future compile error: {future_err:.4f}")
    print(f"  Full compile error: {full_err:.4f}")
    print(f"  Retrocausal prediction: {retrocausal_ratio:.2%} of full accuracy")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Phase 145: Imaginary Time & Tachyon Compilation', fontsize=14, fontweight='bold')

    axes[0].pie([n_timelike, n_spacelike, n_lightlike],
               labels=['Timelike', 'Spacelike (tachyon)', 'Lightlike'],
               colors=['#2196F3', '#F44336', '#4CAF50'], autopct='%1.0f%%')
    axes[0].set_title('Causal Structure')

    if tachyon_pairs:
        speeds = [tp['speed'] for tp in tachyon_pairs]
        labels = [f"{tp['a'][:6]}-{tp['b'][:6]}" for tp in tachyon_pairs]
        axes[1].barh(labels, speeds, color='#E91E63', edgecolor='black')
        axes[1].axvline(1.0, color='black', linestyle='--', label='Light speed')
        axes[1].set_xlabel('Speed (c)'); axes[1].set_title('Top Tachyons'); axes[1].legend()

    axes[2].bar(['Past->Future', 'Full'], [future_err, full_err],
               color=['#FF9800', '#2196F3'], edgecolor='black')
    axes[2].set_ylabel('Compile Error'); axes[2].set_title(f'Retrocausal prediction ({retrocausal_ratio:.0%})')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase145_wick.png'), dpi=150, bbox_inches='tight')
    plt.close()

    results = {
        'phase': 145, 'title': 'Imaginary Time & Tachyon Compilation',
        'n_timelike': n_timelike, 'n_spacelike': n_spacelike, 'n_lightlike': n_lightlike,
        'tachyon_fraction': float(n_spacelike / total_pairs),
        'top_tachyons': tachyon_pairs,
        'retrocausal_ratio': float(retrocausal_ratio),
        'future_compile_err': float(future_err), 'full_compile_err': float(full_err),
        'law': f'Causal structure: {n_spacelike} spacelike ({n_spacelike/total_pairs*100:.0f}% tachyonic). Retrocausal prediction: {retrocausal_ratio:.0%}.'
    }
    with open(os.path.join(RESULTS_DIR, 'phase145_wick.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nPhase 145 complete!")
    return results

if __name__ == '__main__':
    main()
