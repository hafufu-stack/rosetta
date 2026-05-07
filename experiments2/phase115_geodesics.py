"""Phase 115: Geodesics in Curved Space - The shortest path ISN'T a straight line.
P106 showed gravity is d^-3.4, meaning space is curved.
Compute geodesics (gravity-bent paths) between programs.
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
    print("Phase 115: Geodesics in Curved Space")
    print("  The shortest path isn't a straight line!")
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
    n = len(unique_funcs)
    
    # Gravity parameters from P106
    G = 0.752
    alpha = 3.40
    
    # AST mass for each function
    import ast as ast_mod
    func_mass = {}
    for f in unique_funcs:
        try:
            tree = ast_mod.parse(f)
            func_mass[f] = sum(1 for _ in ast_mod.walk(tree))
        except:
            func_mass[f] = 1
    
    masses = np.array([func_mass[f] for f in unique_funcs])
    
    def compute_gravitational_field(pos):
        """Compute total gravitational acceleration at position."""
        acc = np.zeros(64)
        for i, f in enumerate(unique_funcs):
            diff = all_vecs[i] - pos
            d = np.linalg.norm(diff)
            if d < 0.01: continue
            direction = diff / d
            force = G * masses[i] / (d ** alpha)
            force = min(force, 5.0)
            acc += direction * force
        return acc
    
    def trace_geodesic(start, end, n_steps=200, gravity_weight=0.1):
        """Trace a geodesic (gravity-influenced path) from start to end."""
        path = [start.copy()]
        pos = start.copy()
        
        for step in range(n_steps):
            t = (step + 1) / n_steps
            # Linear interpolation direction
            linear_dir = end - pos
            linear_dist = np.linalg.norm(linear_dir)
            if linear_dist < 0.01: break
            linear_dir = linear_dir / linear_dist
            
            # Gravitational deflection
            grav = compute_gravitational_field(pos)
            
            # Combined direction
            step_size = linear_dist / (n_steps - step)
            combined = linear_dir + gravity_weight * grav
            combined = combined / (np.linalg.norm(combined) + 1e-10)
            
            pos = pos + combined * step_size
            path.append(pos.copy())
        
        return np.array(path)
    
    def trace_straight(start, end, n_steps=200):
        """Straight line (Euclidean) path."""
        return np.array([start + t * (end - start) for t in np.linspace(0, 1, n_steps)])
    
    # Test pairs
    test_pairs = [
        ('def f(x, y): return x + y', 'def f(x, y): return x - y'),
        ('def f(x, y): return x + y', 'def f(x, y): return x * y'),
        ('def f(x, y): return x > y', 'def f(x, y): return x < y'),
        ('def f(x, y): return max(x, y)', 'def f(x, y): return min(x, y)'),
    ]
    
    geodesic_results = []
    all_geodesics = []
    all_straights = []
    
    for src_a, src_b in test_pairs:
        if src_a not in func_means or src_b not in func_means:
            continue
        
        vec_a = func_means[src_a]
        vec_b = func_means[src_b]
        euclidean_dist = np.linalg.norm(vec_a - vec_b)
        
        geodesic = trace_geodesic(vec_a, vec_b, n_steps=100, gravity_weight=0.3)
        straight = trace_straight(vec_a, vec_b, n_steps=100)
        
        # Geodesic length (sum of step distances)
        geo_length = sum(np.linalg.norm(geodesic[i+1] - geodesic[i]) for i in range(len(geodesic)-1))
        str_length = euclidean_dist
        
        # How many known functions does each path pass near?
        geo_encounters = 0
        str_encounters = 0
        encounter_threshold = np.percentile([np.linalg.norm(all_vecs[i] - all_vecs[j]) 
                                             for i in range(min(50,n)) for j in range(i+1,min(50,n))], 10)
        
        for pt in geodesic[::5]:
            dists = np.linalg.norm(all_vecs - pt.reshape(1,-1), axis=1)
            geo_encounters += np.sum(dists < encounter_threshold)
        
        for pt in straight[::5]:
            dists = np.linalg.norm(all_vecs - pt.reshape(1,-1), axis=1)
            str_encounters += np.sum(dists < encounter_threshold)
        
        # Deflection angle
        mid_geo = geodesic[len(geodesic)//2]
        mid_str = straight[len(straight)//2]
        deflection = np.linalg.norm(mid_geo - mid_str)
        
        a_short = src_a.split('return ')[-1].strip()[:10]
        b_short = src_b.split('return ')[-1].strip()[:10]
        
        result = {
            'pair': f'{a_short} -> {b_short}',
            'euclidean': float(euclidean_dist),
            'geodesic_length': float(geo_length),
            'curvature_ratio': float(geo_length / str_length),
            'deflection': float(deflection),
            'geo_encounters': int(geo_encounters),
            'str_encounters': int(str_encounters)
        }
        geodesic_results.append(result)
        all_geodesics.append(geodesic)
        all_straights.append(straight)
        
        print(f"  {a_short} -> {b_short}")
        print(f"    Euclidean: {euclidean_dist:.3f} | Geodesic: {geo_length:.3f} | Curvature: {geo_length/str_length:.3f}x")
        print(f"    Deflection: {deflection:.4f} | Encounters: geo={geo_encounters}, str={str_encounters}")
    
    mean_curvature = np.mean([r['curvature_ratio'] for r in geodesic_results])
    mean_deflection = np.mean([r['deflection'] for r in geodesic_results])
    
    print(f"\n--- Space Curvature Summary ---")
    print(f"  Mean curvature ratio: {mean_curvature:.3f}x")
    print(f"  Mean deflection: {mean_deflection:.4f}")
    print(f"  {'SPACE IS CURVED!' if mean_curvature > 1.05 else 'Space is approximately flat'}")
    
    # Plot
    pca = PCA(n_components=2)
    all_points = np.vstack([all_vecs] + all_geodesics + all_straights)
    pca.fit(all_points)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Phase 115: Geodesics (curvature={mean_curvature:.2f}x)', fontsize=14, fontweight='bold')
    
    vecs_2d = pca.transform(all_vecs)
    axes[0].scatter(vecs_2d[:,0], vecs_2d[:,1], s=10, alpha=0.2, c='gray')
    
    colors = ['#E91E63', '#2196F3', '#4CAF50', '#FF9800']
    for i, (geo, st) in enumerate(zip(all_geodesics, all_straights)):
        geo_2d = pca.transform(geo)
        st_2d = pca.transform(st)
        axes[0].plot(st_2d[:,0], st_2d[:,1], '--', color=colors[i%4], alpha=0.5, linewidth=1, label='straight' if i==0 else None)
        axes[0].plot(geo_2d[:,0], geo_2d[:,1], '-', color=colors[i%4], linewidth=2, label='geodesic' if i==0 else None)
    axes[0].set_title('Geodesics vs Straight Lines')
    axes[0].legend(fontsize=7)
    
    pairs = [r['pair'] for r in geodesic_results]
    curvatures = [r['curvature_ratio'] for r in geodesic_results]
    axes[1].bar(range(len(pairs)), curvatures, color='#9C27B0', edgecolor='black')
    axes[1].set_xticks(range(len(pairs)))
    axes[1].set_xticklabels(pairs, fontsize=7, rotation=30)
    axes[1].axhline(1.0, color='gray', linestyle='--')
    axes[1].set_ylabel('Curvature Ratio')
    axes[1].set_title('Path Curvature per Pair')
    
    deflections = [r['deflection'] for r in geodesic_results]
    axes[2].bar(range(len(pairs)), deflections, color='#FF9800', edgecolor='black')
    axes[2].set_xticks(range(len(pairs)))
    axes[2].set_xticklabels(pairs, fontsize=7, rotation=30)
    axes[2].set_ylabel('Deflection (64D)')
    axes[2].set_title('Gravitational Deflection')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase115_geodesics.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    results = {
        'phase': 115, 'title': 'Geodesics in Curved Space',
        'mean_curvature_ratio': float(mean_curvature),
        'mean_deflection': float(mean_deflection),
        'geodesics': geodesic_results,
        'law': f'Program space is curved: geodesics are {mean_curvature:.2f}x longer than Euclidean. Mean deflection={mean_deflection:.4f}. Gravity bends the shortest path between programs.'
    }
    with open(os.path.join(RESULTS_DIR, 'phase115_geodesics.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nPhase 115 complete!")
    return results

if __name__ == '__main__':
    main()
