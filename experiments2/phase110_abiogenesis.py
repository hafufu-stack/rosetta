"""Phase 110: Dark Matter Abiogenesis - Creating alien code from the void.
P104 showed 100% of space is dark matter. What programs exist in the voids?
"""
import os, json, sys, inspect
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor
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
    print("Phase 110: Dark Matter Abiogenesis")
    print("  Creating alien code from the void")
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
    
    nn = NearestNeighbors(n_neighbors=3)
    nn.fit(all_vecs)
    
    # Build Neural CPU
    print("  Building Neural CPU...")
    exec_data = []
    g = {}
    for func_src in unique_funcs:
        vec = func_means[func_src]
        try:
            exec(compile(func_src, '<string>', 'exec'), g)
            fn = g['f']
            try: n_args = len(inspect.signature(fn).parameters)
            except: n_args = 2
            for x_val in [-3, -1, 0, 1, 3, 5, 7]:
                for y_val in [-3, -1, 0, 1, 3, 5, 7]:
                    try:
                        result = fn(x_val) if n_args == 1 else fn(x_val, y_val)
                        if isinstance(result, (int, float, bool)) and abs(float(result)) < 1e6:
                            features = np.concatenate([vec, [x_val, y_val]])
                            exec_data.append((features, float(result)))
                    except: pass
        except: pass
    
    X_cpu = np.array([d[0] for d in exec_data])
    y_cpu = np.array([d[1] for d in exec_data])
    cpu = MLPRegressor(hidden_layer_sizes=(256, 128, 64), max_iter=2000, random_state=42,
                       early_stopping=True, validation_fraction=0.1)
    cpu.fit(X_cpu, y_cpu)
    
    # Find the deepest voids
    print("\n  Searching for deepest voids...")
    centroid = np.mean(all_vecs, axis=0)
    mins = all_vecs.min(axis=0)
    maxs = all_vecs.max(axis=0)
    spread = maxs - mins
    
    np.random.seed(42)
    void_probes = []
    n_candidates = 10000
    
    for _ in range(n_candidates):
        probe = mins + np.random.rand(64) * spread
        dists, _ = nn.kneighbors(probe.reshape(1, -1))
        min_dist = dists[0, 0]
        void_probes.append((probe, min_dist))
    
    # Sort by isolation (most distant from all known programs)
    void_probes.sort(key=lambda x: -x[1])
    deepest_voids = void_probes[:50]
    
    print(f"  Deepest void distance: {deepest_voids[0][1]:.4f}")
    print(f"  Top 10 void distances: {[f'{v[1]:.3f}' for v in deepest_voids[:10]]}")
    
    # Decode alien code from voids using Neural CPU
    print("\n  Decoding alien code from voids...")
    test_inputs = [(0,0),(1,1),(2,3),(5,5),(-1,2),(3,-1)]
    
    alien_signatures = []
    for i, (void_vec, void_dist) in enumerate(deepest_voids[:20]):
        predictions = []
        for x, y in test_inputs:
            features = np.concatenate([void_vec, [x, y]])
            pred = cpu.predict(features.reshape(1, -1))[0]
            predictions.append(round(pred, 2))
        
        # Find nearest known function for comparison
        _, idx = nn.kneighbors(void_vec.reshape(1, -1))
        nearest = unique_funcs[idx[0, 0]]
        nearest_short = nearest.split('return ')[-1].strip() if 'return' in nearest else '?'
        
        alien_signatures.append({
            'void_id': i,
            'isolation': float(void_dist),
            'predictions': predictions,
            'nearest_known': nearest_short
        })
        
        if i < 5:
            print(f"  Alien #{i}: isolation={void_dist:.3f}, outputs={predictions[:4]}, nearest={nearest_short}")
    
    # Analyze alien code patterns
    all_preds = np.array([a['predictions'] for a in alien_signatures])
    pred_variance = np.mean(np.var(all_preds, axis=0))
    
    # Compare to known functions
    known_preds = []
    for func_src in unique_funcs[:20]:
        g2 = {}
        try:
            exec(compile(func_src, '<string>', 'exec'), g2)
            fn = g2['f']
            try: n_args = len(inspect.signature(fn).parameters)
            except: n_args = 2
            preds = []
            for x, y in test_inputs:
                try:
                    r = fn(x) if n_args == 1 else fn(x, y)
                    preds.append(float(r) if isinstance(r, (int, float, bool)) else 0)
                except: preds.append(0)
            known_preds.append(preds)
        except: pass
    
    known_variance = np.mean(np.var(np.array(known_preds), axis=0)) if known_preds else 0
    
    print(f"\n--- Alien Code Analysis ---")
    print(f"  Alien prediction variance: {pred_variance:.4f}")
    print(f"  Known prediction variance: {known_variance:.4f}")
    print(f"  Alien/Known ratio: {pred_variance/known_variance:.2f}x" if known_variance > 0 else "  N/A")
    
    # Check if alien outputs are "interesting" (not just noise)
    alien_patterns = []
    for sig in alien_signatures:
        preds = sig['predictions']
        # Check for patterns: constant, linear, periodic
        is_constant = np.std(preds) < 0.5
        diffs = np.diff(preds)
        is_linear = np.std(diffs) < 0.5 * np.mean(np.abs(diffs) + 1e-10)
        alien_patterns.append({'constant': is_constant, 'linear': is_linear})
    
    n_constant = sum(1 for p in alien_patterns if p['constant'])
    n_linear = sum(1 for p in alien_patterns if p['linear'])
    n_complex = len(alien_patterns) - n_constant - n_linear
    
    print(f"  Constant aliens: {n_constant}")
    print(f"  Linear aliens: {n_linear}")
    print(f"  Complex aliens: {n_complex}")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Phase 110: Dark Matter Abiogenesis', fontsize=14, fontweight='bold')
    
    isolations = [a['isolation'] for a in alien_signatures]
    axes[0].hist(isolations, bins=20, color='#9C27B0', edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Isolation Distance')
    axes[0].set_title(f'Void Depth Distribution (max={max(isolations):.3f})')
    
    for i, sig in enumerate(alien_signatures[:8]):
        axes[1].plot(range(len(sig['predictions'])), sig['predictions'],
                    'o-', alpha=0.6, label=f"Alien #{i}" if i < 4 else None, linewidth=1.5)
    axes[1].set_xlabel('Test Input Index')
    axes[1].set_ylabel('Predicted Output')
    axes[1].set_title('Alien Code Signatures')
    axes[1].legend(fontsize=7)
    
    axes[2].pie([n_constant, n_linear, n_complex],
               labels=[f'Constant\n{n_constant}', f'Linear\n{n_linear}', f'Complex\n{n_complex}'],
               colors=['#CCCCCC', '#4CAF50', '#E91E63'], autopct='%1.0f%%',
               startangle=90, textprops={'fontsize': 10})
    axes[2].set_title('Alien Code Taxonomy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase110_abiogenesis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    results = {
        'phase': 110, 'title': 'Dark Matter Abiogenesis',
        'n_voids_probed': len(alien_signatures),
        'max_isolation': float(max(isolations)),
        'alien_variance': float(pred_variance),
        'known_variance': float(known_variance),
        'n_constant': n_constant, 'n_linear': n_linear, 'n_complex': n_complex,
        'alien_samples': alien_signatures[:5],
        'law': f'Dark matter abiogenesis: {n_complex} complex aliens found in voids (isolation up to {max(isolations):.3f}). Alien variance={pred_variance:.2f} vs known={known_variance:.2f}.'
    }
    with open(os.path.join(RESULTS_DIR, 'phase110_abiogenesis.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nPhase 110 complete!")
    return results

if __name__ == '__main__':
    main()
