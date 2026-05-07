"""Phase 126: Hyperbolic Holography & AdS/CFT
Remap vectors to Poincare ball (hyperbolic space) and retry P125's stress test.
If program space is naturally hyperbolic, boundary info should survive erasure.
"""
import os, json, sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
import inspect
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXP2_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(EXP2_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')

def euclidean_to_poincare(x, c=1.0):
    """Map Euclidean vectors to the Poincare ball model of hyperbolic space."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    # Exponential map: tanh(c * ||x||) * x/||x||
    norms_safe = np.maximum(norms, 1e-10)
    scale = np.tanh(c * norms) / norms_safe
    return x * scale

def poincare_distance(u, v, c=1.0):
    """Compute hyperbolic distance in the Poincare ball."""
    diff_sq = np.sum((u - v) ** 2, axis=-1)
    u_sq = np.sum(u ** 2, axis=-1)
    v_sq = np.sum(v ** 2, axis=-1)
    denom = (1 - u_sq) * (1 - v_sq)
    denom = np.maximum(denom, 1e-10)
    arg = 1 + 2 * diff_sq / denom
    return np.arccosh(np.maximum(arg, 1.0)) / c

def main():
    print("=" * 60)
    print("Phase 126: Hyperbolic Holography & AdS/CFT")
    print("  Does hyperbolic geometry save holographic decoding?")
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
    
    # Reduce to 16D for tractable computation
    pca = PCA(n_components=16).fit(ast_m)
    ast_16d = pca.transform(ast_m)
    
    # Map to Poincare ball
    curvatures = [0.5, 1.0, 2.0, 5.0]
    
    # Build execution dataset
    exec_data = []
    for idx, src in enumerate(unique_funcs):
        try:
            env = {}
            exec(src, env)
            func = [v for v in env.values() if callable(v)][0]
            n_params = len(inspect.signature(func).parameters)
            tests = [(2,3),(5,7),(1,4),(3,2)] if n_params == 2 else [(3,),(7,),(2,),(5,)]
            for args in tests:
                try:
                    result = func(*args[:n_params])
                    if isinstance(result, (int, float)) and abs(result) < 1e6:
                        exec_data.append({'idx': idx, 'args': list(args[:n_params]), 'result': float(result)})
                except Exception: pass
        except Exception: pass
    
    print(f"  Execution samples: {len(exec_data)}")
    
    def build_dataset(vectors):
        X, y = [], []
        for d in exec_data:
            v = vectors[d['idx']]
            args = d['args'] + [0] * (2 - len(d['args']))
            X.append(np.concatenate([v, args]))
            y.append(d['result'])
        return np.array(X), np.array(y)
    
    def test_accuracy(vectors, label):
        X, y = build_dataset(vectors)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
        model = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=1000, random_state=42, early_stopping=True, learning_rate_init=0.001)
        model.fit(X_tr, y_tr)
        r2_full = r2_score(y_te, model.predict(X_te))
        
        # Erasure test: zero out 50% of dims
        n_dims = vectors.shape[1]
        np.random.seed(42)
        dims_to_zero = np.random.choice(n_dims, n_dims // 2, replace=False)
        X_tr_dam = X_tr.copy(); X_te_dam = X_te.copy()
        X_tr_dam[:, dims_to_zero] = 0; X_te_dam[:, dims_to_zero] = 0
        model2 = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=1000, random_state=42, early_stopping=True, learning_rate_init=0.001)
        model2.fit(X_tr_dam, y_tr)
        r2_erased = r2_score(y_te, model2.predict(X_te_dam))
        
        print(f"  {label}: full R2={r2_full:.4f}, 50% erased R2={r2_erased:.4f}")
        return r2_full, r2_erased
    
    results_data = {}
    
    # Euclidean baseline (16D)
    r2_euc_full, r2_euc_erased = test_accuracy(ast_16d, "Euclidean 16D")
    results_data['euclidean'] = {'full': float(r2_euc_full), 'erased_50pct': float(r2_euc_erased)}
    
    # Hyperbolic at various curvatures
    for c in curvatures:
        hyp_vectors = euclidean_to_poincare(ast_16d, c=c)
        r2_f, r2_e = test_accuracy(hyp_vectors, f"Poincare c={c}")
        results_data[f'poincare_c{c}'] = {'full': float(r2_f), 'erased_50pct': float(r2_e)}
    
    # Hyperbolic distance properties
    best_c = curvatures[np.argmax([results_data[f'poincare_c{c}']['erased_50pct'] for c in curvatures])]
    hyp_best = euclidean_to_poincare(ast_16d, c=best_c)
    
    euc_dists = np.linalg.norm(ast_16d[:20, None] - ast_16d[None, :20], axis=2)
    hyp_dists = np.array([[poincare_distance(hyp_best[i:i+1], hyp_best[j:j+1], c=best_c) 
                          for j in range(20)] for i in range(20)])
    
    dist_corr = np.corrcoef(euc_dists.ravel(), hyp_dists.ravel())[0,1]
    print(f"\n  Euclidean-Hyperbolic distance correlation: {dist_corr:.4f}")
    print(f"  Best curvature: c={best_c}")
    
    # Boundary concentration: how much info is near the boundary of the Poincare ball?
    norms_hyp = np.linalg.norm(hyp_best, axis=1)
    near_boundary = np.mean(norms_hyp > 0.9)
    print(f"  Fraction near boundary (|x|>0.9): {near_boundary:.2%}")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Phase 126: Hyperbolic Holography (AdS/CFT)', fontsize=14, fontweight='bold')
    
    labels = ['Euclidean'] + [f'Poincare c={c}' for c in curvatures]
    full_r2s = [results_data['euclidean']['full']] + [results_data[f'poincare_c{c}']['full'] for c in curvatures]
    erased_r2s = [results_data['euclidean']['erased_50pct']] + [results_data[f'poincare_c{c}']['erased_50pct'] for c in curvatures]
    
    x_pos = np.arange(len(labels))
    axes[0].bar(x_pos - 0.15, full_r2s, 0.3, label='Full', color='#2196F3', edgecolor='black')
    axes[0].bar(x_pos + 0.15, erased_r2s, 0.3, label='50% erased', color='#F44336', edgecolor='black')
    axes[0].set_xticks(x_pos); axes[0].set_xticklabels(labels, rotation=30, fontsize=7)
    axes[0].set_ylabel('R2 Score'); axes[0].legend(); axes[0].set_title('Euclidean vs Hyperbolic')
    
    axes[1].hist(norms_hyp, bins=30, color='#9C27B0', edgecolor='black', alpha=0.7)
    axes[1].axvline(0.9, color='red', linestyle='--', label='Boundary zone')
    axes[1].set_xlabel(f'Poincare ball radius (c={best_c})'); axes[1].set_title('Boundary concentration')
    axes[1].legend()
    
    pca_2d = PCA(n_components=2).fit_transform(hyp_best)
    sc = axes[2].scatter(pca_2d[:,0], pca_2d[:,1], c=norms_hyp, cmap='magma', s=20, alpha=0.7)
    plt.colorbar(sc, ax=axes[2], label='|x| (radius)')
    axes[2].set_title('Hyperbolic embedding (2D projection)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase126_hyperbolic.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    results = {
        'phase': 126, 'title': 'Hyperbolic Holography & AdS/CFT',
        'results': results_data, 'best_curvature': float(best_c),
        'dist_correlation': float(dist_corr), 'boundary_fraction': float(near_boundary),
        'law': f'Best curvature c={best_c}. Euclidean erased={r2_euc_erased:.4f}. Poincare erased={results_data[f"poincare_c{best_c}"]["erased_50pct"]:.4f}. Boundary fraction={near_boundary:.2%}.'
    }
    with open(os.path.join(RESULTS_DIR, 'phase126_hyperbolic.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nPhase 126 complete!")
    return results

if __name__ == '__main__':
    main()
