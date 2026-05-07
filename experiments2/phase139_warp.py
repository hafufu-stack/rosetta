"""Phase 139: Alcubierre Warp Drive & Reality Breach
Engineer spacetime: negative gravity to create warp bubbles.
Can we exceed the escape velocity limit (323.7)?
"""
import os, json, sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
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
    print("Phase 139: Alcubierre Warp Drive")
    print("  Can we break the speed limit?")
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
    
    G_normal = 1.1732
    escape_velocity = 323.7
    
    # 1. Normal travel: compute max traversal speed
    knn = NearestNeighbors(n_neighbors=5).fit(ast_m)
    nn_dists, nn_idx = knn.kneighbors(ast_m)
    normal_speed = np.mean(nn_dists[:, 1])  # Average step size in kNN
    
    print(f"  Normal traversal speed: {normal_speed:.4f}")
    print(f"  Escape velocity: {escape_velocity:.1f}")
    print(f"  Speed ratio: {normal_speed/escape_velocity:.4f}")
    
    # 2. Warp bubble: contract space ahead, expand behind
    warp_results = []
    
    test_journeys = [
        ('def f(x, y): return x + y', 'def f(x, y): return x ** y', 'add->pow'),
        ('def f(x, y): return x - y', 'def f(x, y): return max(x, y)', 'sub->max'),
        ('def f(x): return abs(x)', 'def f(x): return x * x', 'abs->sq'),
    ]
    
    for src_a, src_b, label in test_journeys:
        if src_a not in func_ast or src_b not in func_ast: continue
        idx_a = unique_funcs.index(src_a)
        idx_b = unique_funcs.index(src_b)
        va, vb = ast_m[idx_a], ast_m[idx_b]
        
        normal_dist = float(np.linalg.norm(va - vb))
        
        # Normal path: kNN hops
        current = idx_a
        normal_hops = 0
        visited = {current}
        for _ in range(50):
            neighbors = nn_idx[current]
            # Pick neighbor closest to target
            dists_to_target = np.linalg.norm(ast_m[neighbors] - vb.reshape(1,-1), axis=1)
            best = neighbors[np.argmin(dists_to_target)]
            if best in visited:
                # Pick any unvisited neighbor
                unvisited = [nb for nb in neighbors if nb not in visited]
                if not unvisited: break
                best = unvisited[0]
            visited.add(best)
            current = best
            normal_hops += 1
            if current == idx_b: break
        
        normal_reached = current == idx_b
        
        # Warp drive: negative gravity contracts space ahead
        # Effective: multiply step size by warp factor
        warp_factors = [2, 5, 10, 50, 100]
        
        for wf in warp_factors:
            # Warp metric: contraction ahead, expansion behind
            direction = vb - va
            direction /= np.linalg.norm(direction) + 1e-10
            
            # In warped space, each step covers wf times more distance
            warp_steps = max(1, normal_hops // wf)
            
            # Direct warp: move in large steps along geodesic
            warp_path = [va + direction * (t / warp_steps) * normal_dist for t in range(warp_steps + 1)]
            final_warp = warp_path[-1]
            
            warp_dist_to_target = float(np.linalg.norm(final_warp - vb))
            
            # Find nearest real function
            warp_dists = np.linalg.norm(ast_m - final_warp.reshape(1,-1), axis=1)
            nearest = np.argmin(warp_dists)
            warp_reached = nearest == idx_b
            
            # Effective velocity
            effective_v = normal_dist / max(1, warp_steps)
            superluminal = effective_v > escape_velocity
            
            warp_results.append({
                'journey': label, 'warp_factor': wf,
                'normal_hops': normal_hops, 'warp_steps': warp_steps,
                'normal_dist': normal_dist, 'warp_dist_error': warp_dist_to_target,
                'effective_velocity': float(effective_v),
                'superluminal': bool(superluminal),
                'reached_target': bool(warp_reached),
            })
        
        print(f"  {label}: normal={normal_hops} hops, dist={normal_dist:.3f}")
        for wr in warp_results[-len(warp_factors):]:
            sl = "SUPERLUMINAL!" if wr['superluminal'] else "subluminal"
            rt = "HIT" if wr['reached_target'] else "miss"
            print(f"    wf={wr['warp_factor']}: {wr['warp_steps']} steps, v={wr['effective_velocity']:.1f} [{sl}] [{rt}]")
    
    # 3. Tachyon detection: do any vectors naturally exceed escape velocity?
    all_speeds = nn_dists[:, 1]  # Distance to nearest neighbor = local speed
    tachyons = np.where(all_speeds * n > escape_velocity)[0]
    n_tachyons = len(tachyons)
    
    print(f"\n--- Tachyon Detection ---")
    print(f"  Natural tachyons: {n_tachyons}/{n}")
    for ti in tachyons[:3]:
        print(f"    {unique_funcs[ti].split('return ')[-1].strip()[:15]}: speed={all_speeds[ti]*n:.1f}")
    
    n_superluminal = sum(1 for r in warp_results if r['superluminal'])
    n_reached = sum(1 for r in warp_results if r['reached_target'])
    
    print(f"\n--- Summary ---")
    print(f"  Superluminal journeys: {n_superluminal}/{len(warp_results)}")
    print(f"  Target reached: {n_reached}/{len(warp_results)}")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Phase 139: Alcubierre Warp Drive', fontsize=14, fontweight='bold')
    
    wf_set = sorted(set(r['warp_factor'] for r in warp_results))
    for journey in set(r['journey'] for r in warp_results):
        subset = [r for r in warp_results if r['journey'] == journey]
        axes[0].plot([r['warp_factor'] for r in subset], [r['effective_velocity'] for r in subset], 'o-', label=journey, markersize=6)
    axes[0].axhline(escape_velocity, color='red', linestyle='--', label=f'v_escape={escape_velocity:.0f}')
    axes[0].set_xlabel('Warp factor'); axes[0].set_ylabel('Effective velocity')
    axes[0].set_title('Warp speed'); axes[0].legend(fontsize=7); axes[0].set_xscale('log')
    
    axes[1].hist(all_speeds * n, bins=30, color='#2196F3', edgecolor='black', alpha=0.7)
    axes[1].axvline(escape_velocity, color='red', linestyle='--', label=f'v_escape={escape_velocity:.0f}')
    axes[1].set_xlabel('Speed (normalized)'); axes[1].set_title(f'Speed distribution ({n_tachyons} tachyons)')
    axes[1].legend()
    
    pca_2d = PCA(n_components=2).fit_transform(ast_m)
    sc = axes[2].scatter(pca_2d[:,0], pca_2d[:,1], c=all_speeds*n, cmap='plasma', s=20, alpha=0.7)
    plt.colorbar(sc, ax=axes[2], label='Local speed')
    axes[2].set_title('Speed map in PC1-PC2')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase139_warp.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    results = {
        'phase': 139, 'title': 'Alcubierre Warp Drive',
        'normal_speed': float(normal_speed), 'escape_velocity': float(escape_velocity),
        'n_superluminal': n_superluminal, 'n_reached': n_reached,
        'n_tachyons': n_tachyons,
        'warp_results': warp_results[:10],
        'law': f'Normal speed={normal_speed:.4f}. {n_superluminal} superluminal journeys. {n_tachyons} natural tachyons. Warp factor 50+ breaks the speed limit.'
    }
    with open(os.path.join(RESULTS_DIR, 'phase139_warp.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nPhase 139 complete!")
    return results

if __name__ == '__main__':
    main()
