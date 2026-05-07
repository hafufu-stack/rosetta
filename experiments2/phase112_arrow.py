"""Phase 112: The Arrow of Time - Does code evolution have a direction?
As programs evolve (add -> mul -> pow), is there an entropy increase?
Test if there's a thermodynamic arrow in program space.
"""
import os, json, sys, ast as ast_mod
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
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
    print("Phase 112: The Arrow of Time")
    print("  Does code evolution have a preferred direction?")
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
    
    # Define "complexity" of each function
    complexity = {}
    for f in unique_funcs:
        try:
            tree = ast_mod.parse(f)
            nodes = sum(1 for _ in ast_mod.walk(tree))
            depth = 0
            for node in ast_mod.walk(tree):
                for child in ast_mod.iter_child_nodes(node):
                    pass
            complexity[f] = nodes
        except:
            complexity[f] = 1
    
    # Define evolutionary chains (simple -> complex)
    chains = [
        ['def f(x, y): return x', 'def f(x, y): return x + y', 'def f(x, y): return x * y',
         'def f(x, y): return x ** y'],
        ['def f(x, y): return 0', 'def f(x, y): return x', 'def f(x, y): return abs(x)',
         'def f(x, y): return abs(x + y)'],
        ['def f(x, y): return x + y', 'def f(x, y): return x + y + 1',
         'def f(x, y): return (x + y) * 2'],
        ['def f(x, y): return x > y', 'def f(x, y): return max(x, y)',
         'def f(x, y): return abs(max(x, y))'],
    ]
    
    # For each chain, compute the "direction" in latent space
    print("\n--- Evolutionary Chains ---")
    chain_results = []
    chain_vectors = []
    
    for chain in chains:
        valid = [f for f in chain if f in func_means]
        if len(valid) < 2:
            continue
        
        vecs = np.array([func_means[f] for f in valid])
        comps = [complexity.get(f, 0) for f in valid]
        
        # Direction = vector from simplest to most complex
        direction = vecs[-1] - vecs[0]
        direction_norm = direction / (np.linalg.norm(direction) + 1e-10)
        
        # Project all steps onto this direction
        projections = [(vecs[i] - vecs[0]) @ direction_norm for i in range(len(vecs))]
        
        # Is the projection monotonically increasing?
        is_monotonic = all(projections[i] <= projections[i+1] for i in range(len(projections)-1))
        
        chain_name = ' -> '.join([f.split('return ')[-1].strip()[:15] for f in valid])
        
        chain_results.append({
            'chain': chain_name,
            'length': len(valid),
            'projections': [float(p) for p in projections],
            'monotonic': is_monotonic,
            'complexities': comps
        })
        chain_vectors.append(direction_norm)
        
        mono_str = 'MONOTONIC' if is_monotonic else 'non-monotonic'
        print(f"  {chain_name}: {mono_str}")
        print(f"    Projections: {[f'{p:.3f}' for p in projections]}")
    
    # Global arrow: is there a universal direction of increasing complexity?
    if chain_vectors:
        chain_vecs = np.array(chain_vectors)
        # Compute mean direction
        mean_direction = np.mean(chain_vecs, axis=0)
        mean_direction = mean_direction / (np.linalg.norm(mean_direction) + 1e-10)
        
        # How aligned are the individual chains?
        alignments = [np.dot(cv, mean_direction) for cv in chain_vecs]
        mean_alignment = np.mean(alignments)
        
        print(f"\n--- Universal Arrow ---")
        print(f"  Mean alignment: {mean_alignment:.4f}")
        print(f"  {'UNIVERSAL ARROW EXISTS!' if mean_alignment > 0.5 else 'No universal arrow'}")
    else:
        mean_alignment = 0
        alignments = []
    
    # Entropy along complexity gradient
    comp_arr = np.array([complexity[f] for f in unique_funcs])
    centroid = np.mean(all_vecs, axis=0)
    dists_from_center = np.linalg.norm(all_vecs - centroid, axis=1)
    
    comp_dist_corr = np.corrcoef(comp_arr, dists_from_center)[0, 1]
    print(f"\n--- Complexity-Distance Correlation ---")
    print(f"  Complexity vs distance from center: {comp_dist_corr:.4f}")
    print(f"  {'Complex = peripheral' if comp_dist_corr > 0.2 else 'Complex = central' if comp_dist_corr < -0.2 else 'No pattern'}")
    
    # PC1 correlation with complexity
    from sklearn.decomposition import PCA
    pca = PCA(n_components=5)
    vecs_5d = pca.fit_transform(all_vecs)
    pc_corrs = [np.corrcoef(comp_arr, vecs_5d[:, i])[0, 1] for i in range(5)]
    
    print(f"\n--- PC Correlations with Complexity ---")
    for i, c in enumerate(pc_corrs):
        print(f"  PC{i+1}: r={c:.4f}")
    
    max_pc_corr = max(pc_corrs, key=abs)
    max_pc_idx = pc_corrs.index(max_pc_corr)
    
    print(f"\n  Strongest: PC{max_pc_idx+1} (r={max_pc_corr:.4f})")
    print(f"  {'ARROW OF TIME = PC' + str(max_pc_idx+1) + '!' if abs(max_pc_corr) > 0.3 else 'No strong arrow'}")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Phase 112: The Arrow of Time', fontsize=14, fontweight='bold')
    
    for i, cr in enumerate(chain_results):
        c = '#4CAF50' if cr['monotonic'] else '#F44336'
        axes[0].plot(range(len(cr['projections'])), cr['projections'],
                    'o-', color=c, linewidth=2, label=cr['chain'][:20])
    axes[0].set_xlabel('Evolution Step')
    axes[0].set_ylabel('Projection onto Arrow')
    axes[0].set_title('Evolutionary Chains')
    axes[0].legend(fontsize=6)
    
    axes[1].scatter(comp_arr, dists_from_center, alpha=0.3, s=20, c='#2196F3')
    axes[1].set_xlabel('Complexity (AST nodes)')
    axes[1].set_ylabel('Distance from Center')
    axes[1].set_title(f'Complexity vs Position (r={comp_dist_corr:.3f})')
    
    axes[2].bar([f'PC{i+1}' for i in range(5)], pc_corrs,
               color=['#E91E63' if abs(c) == abs(max_pc_corr) else '#CCCCCC' for c in pc_corrs],
               edgecolor='black')
    axes[2].set_ylabel('Correlation with Complexity')
    axes[2].set_title(f'Arrow Direction = PC{max_pc_idx+1} (r={max_pc_corr:.3f})')
    axes[2].axhline(0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase112_arrow.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    n_mono = sum(1 for c in chain_results if c['monotonic'])
    
    results = {
        'phase': 112, 'title': 'The Arrow of Time',
        'n_chains': len(chain_results),
        'n_monotonic': n_mono,
        'mean_alignment': float(mean_alignment),
        'comp_dist_corr': float(comp_dist_corr),
        'pc_complexity_corrs': [float(c) for c in pc_corrs],
        'strongest_pc': max_pc_idx + 1,
        'strongest_pc_corr': float(max_pc_corr),
        'chains': chain_results,
        'law': f'Arrow of time: {n_mono}/{len(chain_results)} chains monotonic. Mean alignment={mean_alignment:.3f}. Complexity axis = PC{max_pc_idx+1} (r={max_pc_corr:.3f}).'
    }
    with open(os.path.join(RESULTS_DIR, 'phase112_arrow.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nPhase 112 complete!")
    return results

if __name__ == '__main__':
    main()
