"""Phase 125: The Holographic Stress Test - Can we decode programs from partial boundaries?
Opus original: P101 showed angles retain 99.9% accuracy. But how ROBUST is this?
We systematically destroy boundary information and measure the critical threshold.
This determines the "error correction capacity" of the holographic code.
"""
import os, json, sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
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
    print("Phase 125: Holographic Stress Test")
    print("  How much boundary info can we destroy before decoding fails?")
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
    
    # Build execution dataset for Neural CPU
    exec_data = []
    for idx, src in enumerate(unique_funcs):
        try:
            env = {}
            exec(src, env)
            func = [v for v in env.values() if callable(v)][0]
            import inspect
            n_params = len(inspect.signature(func).parameters)
            
            test_inputs = [(2, 3), (5, 7), (1, 4), (3, 2)] if n_params == 2 else [(3,), (7,), (2,), (5,)]
            for args in test_inputs:
                try:
                    result = func(*args[:n_params])
                    if isinstance(result, (int, float)) and abs(result) < 1e6:
                        exec_data.append({
                            'func_idx': idx,
                            'args': list(args[:n_params]),
                            'result': float(result),
                        })
                except Exception:
                    pass
        except Exception:
            pass
    
    print(f"  Execution dataset: {len(exec_data)} samples")
    
    if len(exec_data) < 50:
        print("  ERROR: not enough exec data")
        return
    
    # Prepare training data: (holographic_vector, args) -> result
    X_full = []
    y_full = []
    for d in exec_data:
        v = ast_m[d['func_idx']]
        args = d['args']
        if len(args) == 1: args = args + [0]
        # Full vector feature
        X_full.append(np.concatenate([v, args]))
        y_full.append(d['result'])
    
    X_full = np.array(X_full)
    y_full = np.array(y_full)
    
    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.3, random_state=42)
    
    # Baseline: full 64D vectors
    from sklearn.metrics import r2_score
    
    # Train a simple MLP
    def train_cpu(X_tr, X_te, y_tr, y_te):
        model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42, early_stopping=True)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        return r2_score(y_te, y_pred)
    
    print("\n--- Stress Test: Dimension Zeroing ---")
    
    # Test: zero out increasing fractions of dimensions
    erasure_fracs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    r2_vs_erasure = []
    
    np.random.seed(42)
    for frac in erasure_fracs:
        n_zero = int(64 * frac)
        dims_to_zero = np.random.choice(64, n_zero, replace=False)
        
        X_tr_dam = X_train.copy()
        X_te_dam = X_test.copy()
        X_tr_dam[:, dims_to_zero] = 0
        X_te_dam[:, dims_to_zero] = 0
        
        r2 = train_cpu(X_tr_dam, X_te_dam, y_train, y_test)
        r2_vs_erasure.append(r2)
        print(f"  Erasure {frac*100:.0f}% ({n_zero} dims): R2 = {r2:.4f}")
    
    # Find critical threshold: where R2 drops below 0.5
    critical_frac = 1.0
    for i, r2 in enumerate(r2_vs_erasure):
        if r2 < 0.5:
            critical_frac = erasure_fracs[i]
            break
    
    print(f"\n  Critical threshold: {critical_frac*100:.0f}% erasure")
    
    # Noise injection test
    print("\n--- Stress Test: Gaussian Noise Injection ---")
    noise_levels = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0]
    r2_vs_noise = []
    
    for noise in noise_levels:
        X_tr_noisy = X_train.copy()
        X_te_noisy = X_test.copy()
        X_tr_noisy[:, :64] += np.random.randn(len(X_train), 64) * noise
        X_te_noisy[:, :64] += np.random.randn(len(X_test), 64) * noise
        
        # Re-normalize to unit sphere (holographic)
        norms_tr = np.linalg.norm(X_tr_noisy[:, :64], axis=1, keepdims=True)
        norms_tr[norms_tr < 1e-10] = 1
        X_tr_noisy[:, :64] /= norms_tr
        
        norms_te = np.linalg.norm(X_te_noisy[:, :64], axis=1, keepdims=True)
        norms_te[norms_te < 1e-10] = 1
        X_te_noisy[:, :64] /= norms_te
        
        r2 = train_cpu(X_tr_noisy, X_te_noisy, y_train, y_test)
        r2_vs_noise.append(r2)
        print(f"  Noise {noise:.1f}: R2 = {r2:.4f}")
    
    # Holographic error correction capacity
    # How many bits can be corrupted before information is lost?
    error_capacity = critical_frac * 64  # dimensions that can be erased
    print(f"\n--- Holographic Error Correction ---")
    print(f"  Can erase {error_capacity:.0f}/64 dimensions without losing >50% accuracy")
    print(f"  Error correction rate: {critical_frac*100:.0f}%")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Phase 125: Holographic Stress Test', fontsize=14, fontweight='bold')
    
    axes[0].plot([f*100 for f in erasure_fracs], r2_vs_erasure, 'o-', 
                color='#E91E63', linewidth=2, markersize=8)
    axes[0].axhline(0.5, color='gray', linestyle='--', label='R2=0.5 threshold')
    axes[0].axvline(critical_frac*100, color='red', linestyle=':', label=f'Critical: {critical_frac*100:.0f}%')
    axes[0].set_xlabel('% Dimensions Erased'); axes[0].set_ylabel('R2 Score')
    axes[0].set_title('Dimension Erasure Stress Test'); axes[0].legend()
    
    axes[1].plot(noise_levels, r2_vs_noise, 's-', color='#2196F3', linewidth=2, markersize=8)
    axes[1].set_xlabel('Noise Level (sigma)'); axes[1].set_ylabel('R2 Score')
    axes[1].set_title('Noise Injection (holographic re-normalized)')
    
    # Error correction capacity visualization
    axes[2].bar(['Erasable', 'Essential'], [error_capacity, 64-error_capacity],
               color=['#4CAF50', '#F44336'], edgecolor='black')
    axes[2].set_ylabel('Dimensions'); 
    axes[2].set_title(f'Error Correction: {error_capacity:.0f}/64 dims dispensable')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase125_stress_test.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    results = {
        'phase': 125, 'title': 'Holographic Stress Test',
        'r2_vs_erasure': {f'{f*100:.0f}%': float(r) for f, r in zip(erasure_fracs, r2_vs_erasure)},
        'r2_vs_noise': {str(n): float(r) for n, r in zip(noise_levels, r2_vs_noise)},
        'critical_erasure_threshold': float(critical_frac),
        'error_correction_capacity': float(error_capacity),
        'law': f'Holographic error correction: {error_capacity:.0f}/64 dims can be erased (R2>0.5). Critical threshold: {critical_frac*100:.0f}%. Noise tolerance: sigma={noise_levels[next((i for i,r in enumerate(r2_vs_noise) if r<0.5), len(noise_levels)-1)]:.1f}.'
    }
    with open(os.path.join(RESULTS_DIR, 'phase125_stress_test.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nPhase 125 complete!")
    return results

if __name__ == '__main__':
    main()
