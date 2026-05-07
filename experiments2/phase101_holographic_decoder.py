"""Phase 101: The Holographic Decoder - Programs decoded from angular info only.
P98 proved sphere projection preserves 108% info. Can we decode code from angles alone?
"""
import os, json, sys, inspect
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
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
    print("Phase 101: The Holographic Decoder")
    print("  Decoding programs from angular info only")
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
    
    # Normalize to unit sphere (holographic projection)
    centroid = np.mean(all_vecs, axis=0)
    centered = all_vecs - centroid
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    unit_vecs = centered / (norms + 1e-10)
    
    print(f"  Functions: {len(unique_funcs)}")
    print(f"  Projecting 64D -> unit sphere (angles only)")
    
    # Build TWO Neural CPUs: one with full 64D, one with unit sphere (angles only)
    print("\nBuilding Neural CPUs...")
    exec_data_full = []
    exec_data_holo = []
    g = {}
    for i, func_src in enumerate(unique_funcs):
        vec_full = func_means[func_src]
        vec_holo = unit_vecs[i]
        try:
            exec(compile(func_src, '<string>', 'exec'), g)
            fn = g['f']
            try: n_args = len(inspect.signature(fn).parameters)
            except: n_args = 2
            for x_val in [-2, -1, 0, 1, 2, 3, 5]:
                for y_val in [-2, -1, 0, 1, 2, 3, 5]:
                    try:
                        result = fn(x_val) if n_args == 1 else fn(x_val, y_val)
                        if isinstance(result, (int, float, bool)) and abs(float(result)) < 1e6:
                            feat_full = np.concatenate([vec_full, [x_val, y_val]])
                            feat_holo = np.concatenate([vec_holo, [x_val, y_val]])
                            exec_data_full.append((feat_full, float(result)))
                            exec_data_holo.append((feat_holo, float(result)))
                    except: pass
        except: pass
    
    X_full = np.array([d[0] for d in exec_data_full])
    y_full = np.array([d[1] for d in exec_data_full])
    X_holo = np.array([d[0] for d in exec_data_holo])
    y_holo = np.array([d[1] for d in exec_data_holo])
    
    cpu_full = MLPRegressor(hidden_layer_sizes=(256, 128, 64), max_iter=2000,
                            random_state=42, early_stopping=True, validation_fraction=0.1)
    cpu_full.fit(X_full, y_full)
    r2_full = cpu_full.score(X_full, y_full)
    
    cpu_holo = MLPRegressor(hidden_layer_sizes=(256, 128, 64), max_iter=2000,
                            random_state=42, early_stopping=True, validation_fraction=0.1)
    cpu_holo.fit(X_holo, y_holo)
    r2_holo = cpu_holo.score(X_holo, y_holo)
    
    print(f"  Full 64D CPU R2:        {r2_full:.4f}")
    print(f"  Holographic CPU R2:     {r2_holo:.4f}")
    print(f"  Info retention:         {r2_holo/r2_full*100:.1f}%")
    
    # NN classification on sphere vs full
    op_labels = {}
    for f in unique_funcs:
        if 'x + y' in f: op_labels[f] = 'add'
        elif 'x - y' in f: op_labels[f] = 'sub'
        elif 'x * y' in f: op_labels[f] = 'mul'
        elif 'max(' in f or 'min(' in f: op_labels[f] = 'minmax'
        elif 'x > y' in f or 'x < y' in f: op_labels[f] = 'cmp'
        else: op_labels[f] = 'other'
    
    labeled = [(i, f) for i, f in enumerate(unique_funcs) if op_labels[f] != 'other']
    
    def nn_acc(vecs, labeled):
        correct = 0
        for i, (idx, _) in enumerate(labeled):
            dists = np.array([np.linalg.norm(vecs[idx] - vecs[j]) for j, _ in labeled])
            dists[i] = float('inf')
            nn = np.argmin(dists)
            if op_labels[labeled[nn][1]] == op_labels[labeled[i][1]]:
                correct += 1
        return correct / len(labeled)
    
    acc_full = nn_acc(all_vecs, labeled)
    acc_holo = nn_acc(unit_vecs, labeled)
    
    print(f"\n  NN classify (full):     {acc_full:.3f}")
    print(f"  NN classify (holo):     {acc_holo:.3f}")
    print(f"  Classification retain:  {acc_holo/acc_full*100:.1f}%")
    
    holographic = r2_holo/r2_full >= 0.95
    print(f"\n  HOLOGRAPHIC DECODING: {'SUCCESS' if holographic else 'PARTIAL'}")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Phase 101: The Holographic Decoder', fontsize=14, fontweight='bold')
    
    cats = ['Full 64D', 'Holographic\n(angles only)']
    axes[0].bar(cats, [r2_full, r2_holo], color=['#2196F3', '#4CAF50'], edgecolor='black')
    axes[0].set_ylabel('Neural CPU R2')
    axes[0].set_title(f'CPU Accuracy: {r2_holo/r2_full*100:.1f}% retained')
    for i, v in enumerate([r2_full, r2_holo]):
        axes[0].text(i, v+0.01, f'{v:.4f}', ha='center', fontweight='bold')
    
    axes[1].bar(cats, [acc_full*100, acc_holo*100], color=['#FF9800', '#E91E63'], edgecolor='black')
    axes[1].set_ylabel('NN Classification (%)')
    axes[1].set_title(f'Classification: {acc_holo/acc_full*100:.1f}% retained')
    
    axes[2].hist(norms.flatten(), bins=30, color='#9C27B0', edgecolor='black', alpha=0.7)
    axes[2].set_xlabel('Vector Norm (discarded info)')
    axes[2].set_ylabel('Count')
    axes[2].set_title(f'Norms discarded by Holographic projection')
    axes[2].axvline(np.mean(norms), color='red', linestyle='--', label=f'mean={np.mean(norms):.3f}')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase101_holographic_decoder.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    results = {
        'phase': 101, 'title': 'The Holographic Decoder',
        'r2_full': float(r2_full), 'r2_holographic': float(r2_holo),
        'cpu_retention_pct': float(r2_holo/r2_full*100),
        'nn_acc_full': float(acc_full), 'nn_acc_holo': float(acc_holo),
        'classify_retention_pct': float(acc_holo/acc_full*100),
        'holographic_success': holographic,
        'law': f'Holographic Decoder: angles-only CPU retains {r2_holo/r2_full*100:.1f}% of R2 ({r2_holo:.4f} vs {r2_full:.4f}). {"The boundary encodes the volume!" if holographic else "Partial info loss."}'
    }
    with open(os.path.join(RESULTS_DIR, 'phase101_holographic_decoder.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nPhase 101 complete!")
    return results

if __name__ == '__main__':
    main()
