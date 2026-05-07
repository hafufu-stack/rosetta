"""Phase 104: The Dark Matter Census - What lives in the voids of program space?
Most of the 64D hypersphere is empty. What programs would exist there?
Generate random vectors in voids and decode them.
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
RESULTS_DIR = os.path.join(EXP2_DIR, 'results')
FIGURES_DIR = os.path.join(EXP2_DIR, 'figures')

def main():
    print("=" * 60)
    print("Phase 104: The Dark Matter Census")
    print("  What lives in the voids of program space?")
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
    
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(all_vecs)
    
    centroid = np.mean(all_vecs, axis=0)
    centered = all_vecs - centroid
    
    # Build Neural CPU
    print("Building Neural CPU...")
    exec_data = []
    g = {}
    for func_src in unique_funcs:
        vec = func_means[func_src]
        try:
            exec(compile(func_src, '<string>', 'exec'), g)
            fn = g['f']
            try: n_args = len(inspect.signature(fn).parameters)
            except: n_args = 2
            for x_val in [-2, -1, 0, 1, 2, 3]:
                for y_val in [-2, -1, 0, 1, 2, 3]:
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
    
    # Generate void probes
    print("\nProbing the voids...")
    mins = all_vecs.min(axis=0)
    maxs = all_vecs.max(axis=0)
    spread = maxs - mins
    
    n_probes = 1000
    void_probes = []
    void_dists = []
    inhabited_probes = []
    inhabited_dists = []
    
    for _ in range(n_probes * 3):
        probe = mins + np.random.rand(64) * spread
        dist, _ = nn.kneighbors(probe.reshape(1, -1))
        d = dist[0, 0]
        
        if d > np.percentile([np.linalg.norm(all_vecs[i] - all_vecs[j])
                              for i in range(min(50, len(all_vecs)))
                              for j in range(i+1, min(50, len(all_vecs)))], 75):
            if len(void_probes) < n_probes // 2:
                void_probes.append(probe)
                void_dists.append(d)
        else:
            if len(inhabited_probes) < n_probes // 2:
                inhabited_probes.append(probe)
                inhabited_dists.append(d)
        
        if len(void_probes) >= n_probes // 2 and len(inhabited_probes) >= n_probes // 2:
            break
    
    void_probes = np.array(void_probes) if void_probes else np.zeros((1, 64))
    inhabited_probes = np.array(inhabited_probes) if inhabited_probes else np.zeros((1, 64))
    
    print(f"  Void probes: {len(void_probes)}")
    print(f"  Inhabited probes: {len(inhabited_probes)}")
    
    # Test Neural CPU predictions in voids vs inhabited regions
    test_inputs = [(1,2),(3,5),(0,0),(-1,1)]
    
    void_predictions = []
    inhabited_predictions = []
    
    for probe in void_probes[:100]:
        preds = []
        for x, y in test_inputs:
            features = np.concatenate([probe, [x, y]])
            preds.append(cpu.predict(features.reshape(1, -1))[0])
        void_predictions.append(preds)
    
    for probe in inhabited_probes[:100]:
        preds = []
        for x, y in test_inputs:
            features = np.concatenate([probe, [x, y]])
            preds.append(cpu.predict(features.reshape(1, -1))[0])
        inhabited_predictions.append(preds)
    
    void_preds = np.array(void_predictions)
    inhab_preds = np.array(inhabited_predictions)
    
    void_var = np.mean(np.var(void_preds, axis=0))
    inhab_var = np.mean(np.var(inhab_preds, axis=0))
    void_range = np.mean(np.max(void_preds, axis=0) - np.min(void_preds, axis=0))
    inhab_range = np.mean(np.max(inhab_preds, axis=0) - np.min(inhab_preds, axis=0))
    
    print(f"\n--- Dark Matter Properties ---")
    print(f"  Void prediction variance:      {void_var:.4f}")
    print(f"  Inhabited prediction variance:  {inhab_var:.4f}")
    print(f"  Void / Inhabited ratio:         {void_var/inhab_var:.2f}x")
    print(f"  Void prediction range:          {void_range:.4f}")
    print(f"  Inhabited prediction range:     {inhab_range:.4f}")
    
    # Volume fraction
    total_volume_probes = 5000
    void_count = 0
    median_nn_dist = np.median([np.linalg.norm(all_vecs[i] - all_vecs[(i+1)%len(all_vecs)])
                                for i in range(min(200, len(all_vecs)))])
    
    for _ in range(total_volume_probes):
        probe = mins + np.random.rand(64) * spread
        dist, _ = nn.kneighbors(probe.reshape(1, -1))
        if dist[0, 0] > median_nn_dist:
            void_count += 1
    
    void_fraction = void_count / total_volume_probes
    dark_matter_fraction = void_fraction
    
    print(f"\n--- Volume Census ---")
    print(f"  Dark matter fraction: {dark_matter_fraction:.1%}")
    print(f"  Visible matter:       {1-dark_matter_fraction:.1%}")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Phase 104: The Dark Matter Census', fontsize=14, fontweight='bold')
    
    axes[0].hist(void_dists, bins=30, alpha=0.6, color='#9C27B0', label='Void probes', edgecolor='black')
    axes[0].hist(inhabited_dists, bins=30, alpha=0.6, color='#4CAF50', label='Inhabited', edgecolor='black')
    axes[0].set_xlabel('Distance to Nearest Program')
    axes[0].set_title('Void vs Inhabited Regions')
    axes[0].legend()
    
    categories = ['Void\n(Dark Matter)', 'Inhabited\n(Visible)']
    variances = [void_var, inhab_var]
    axes[1].bar(categories, variances, color=['#9C27B0', '#4CAF50'], edgecolor='black')
    axes[1].set_ylabel('Prediction Variance')
    axes[1].set_title(f'CPU Prediction Stability ({void_var/inhab_var:.1f}x)')
    
    axes[2].pie([dark_matter_fraction, 1-dark_matter_fraction],
                labels=[f'Dark Matter\n{dark_matter_fraction:.0%}',
                        f'Visible\n{1-dark_matter_fraction:.0%}'],
                colors=['#9C27B0', '#4CAF50'], autopct='%1.0f%%',
                startangle=90, textprops={'fontsize': 10})
    axes[2].set_title('Volume Census of Program Space')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase104_dark_matter.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    results = {
        'phase': 104, 'title': 'The Dark Matter Census',
        'dark_matter_fraction': float(dark_matter_fraction),
        'visible_matter_fraction': float(1-dark_matter_fraction),
        'void_prediction_var': float(void_var),
        'inhabited_prediction_var': float(inhab_var),
        'void_inhab_ratio': float(void_var/inhab_var) if inhab_var > 0 else 0,
        'law': f'Program space is {dark_matter_fraction:.0%} dark matter (voids). CPU predictions in voids have {void_var/inhab_var:.1f}x higher variance - these are "impossible programs".'
    }
    with open(os.path.join(RESULTS_DIR, 'phase104_dark_matter.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nPhase 104 complete!")
    return results

if __name__ == '__main__':
    main()
