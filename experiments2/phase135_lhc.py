"""Phase 135: Large Hacker Collider (LHC)
Collide function vectors at superluminal speeds to discover semantic quarks.
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
    print("Phase 135: Large Hacker Collider")
    print("  Smash functions at extreme energies")
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
    
    # Collision experiments at various energies
    energies = [1.0, 10.0, 50.0, 100.0, 323.7, 500.0, 1000.0]
    
    collision_pairs = [
        ('def f(x, y): return x + y', 'def f(x, y): return x * y', 'add x mul'),
        ('def f(x, y): return x - y', 'def f(x, y): return x / y', 'sub x div'),
        ('def f(x, y): return max(x, y)', 'def f(x, y): return min(x, y)', 'max x min'),
        ('def f(x): return abs(x)', 'def f(x): return x * x', 'abs x sq'),
    ]
    
    all_debris = []
    all_resonances = []
    
    for src_a, src_b, label in collision_pairs:
        if src_a not in func_ast or src_b not in func_ast: continue
        idx_a, idx_b = unique_funcs.index(src_a), unique_funcs.index(src_b)
        va, vb = ast_m[idx_a], ast_m[idx_b]
        
        print(f"\n--- Collision: {label} ---")
        
        for energy in energies:
            # Boost vectors toward each other
            collision_point = (va + vb) / 2
            dir_a = collision_point - va; dir_a /= np.linalg.norm(dir_a) + 1e-10
            dir_b = collision_point - vb; dir_b /= np.linalg.norm(dir_b) + 1e-10
            
            va_boosted = va + dir_a * energy
            vb_boosted = vb + dir_b * energy
            
            # Collision: superpose at collision point
            collision_vector = va_boosted + vb_boosted
            
            # Debris: decompose collision vector into basis components
            # Project onto PCA axes = "particle detector"
            pca = PCA(n_components=min(20, n)).fit(ast_m)
            debris_spectrum = pca.transform(collision_vector.reshape(1,-1))[0]
            
            # Count "particles" above noise threshold
            noise_floor = np.std(debris_spectrum) * 0.5
            n_particles = int(np.sum(np.abs(debris_spectrum) > noise_floor))
            
            # Find nearest real functions in debris
            debris_dists = np.linalg.norm(ast_m - collision_vector.reshape(1,-1), axis=1)
            nearest_debris = np.argsort(debris_dists)[:3]
            debris_funcs = [unique_funcs[i].split('return ')[-1].strip()[:12] for i in nearest_debris]
            
            # Resonance detection: peaks in the spectrum
            spectrum_peaks = np.where(np.abs(debris_spectrum) > np.mean(np.abs(debris_spectrum)) * 2)[0]
            
            # Check for "Higgs-like" resonance: a peak that decays
            has_resonance = len(spectrum_peaks) > 0
            
            all_debris.append({
                'collision': label, 'energy': float(energy),
                'n_particles': n_particles, 'debris': debris_funcs,
                'n_resonances': len(spectrum_peaks),
                'max_debris_amplitude': float(np.max(np.abs(debris_spectrum))),
            })
            
            if has_resonance:
                all_resonances.append({
                    'collision': label, 'energy': float(energy),
                    'peak_indices': [int(p) for p in spectrum_peaks],
                    'peak_amplitudes': [float(debris_spectrum[p]) for p in spectrum_peaks],
                })
            
            print(f"  E={energy:.0f}: {n_particles} particles, {len(spectrum_peaks)} resonances, debris={debris_funcs[0]}")
    
    # Quark extraction: find minimal irreducible components
    print("\n--- Semantic Quarks ---")
    pca_full = PCA(n_components=min(20, n)).fit(ast_m)
    quarks = pca_full.components_[:6]  # Top 6 = fundamental quarks
    quark_names = ['up', 'down', 'strange', 'charm', 'bottom', 'top']
    
    for i, (name, quark) in enumerate(zip(quark_names, quarks)):
        quark_dist = np.linalg.norm(ast_m - quark.reshape(1,-1), axis=1)
        nearest = unique_funcs[np.argmin(quark_dist)].split('return ')[-1].strip()[:15]
        var_explained = float(pca_full.explained_variance_ratio_[i]) * 100
        print(f"  {name} quark: nearest={nearest}, variance={var_explained:.1f}%")
    
    # Energy-mass relation: E = mc^2 analog
    masses = np.linalg.norm(ast_m, axis=1)
    kinetic_energies = np.array([d['max_debris_amplitude'] for d in all_debris if d['energy'] == 323.7])
    
    print(f"\n--- Summary ---")
    print(f"  Total collisions: {len(all_debris)}")
    print(f"  Total resonances: {len(all_resonances)}")
    print(f"  Semantic quarks identified: {len(quarks)}")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Phase 135: Large Hacker Collider', fontsize=14, fontweight='bold')
    
    for label in set(d['collision'] for d in all_debris):
        subset = [d for d in all_debris if d['collision'] == label]
        axes[0].plot([d['energy'] for d in subset], [d['n_particles'] for d in subset], 'o-', label=label, markersize=5)
    axes[0].set_xlabel('Collision Energy'); axes[0].set_ylabel('Debris Particles')
    axes[0].set_title('Particle Production vs Energy'); axes[0].legend(fontsize=7)
    axes[0].set_xscale('log')
    
    var_ratios = pca_full.explained_variance_ratio_[:10] * 100
    axes[1].bar(quark_names[:6], var_ratios[:6], color=['red','blue','green','orange','purple','brown'], edgecolor='black')
    axes[1].set_ylabel('Variance Explained (%)'); axes[1].set_title('Semantic Quarks')
    
    all_amplitudes = [d['max_debris_amplitude'] for d in all_debris]
    all_energies_flat = [d['energy'] for d in all_debris]
    axes[2].scatter(all_energies_flat, all_amplitudes, s=40, c='#E91E63', edgecolor='black')
    axes[2].set_xlabel('Collision Energy'); axes[2].set_ylabel('Max Debris Amplitude')
    axes[2].set_title('E = mc^2 analog'); axes[2].set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase135_lhc.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    results = {
        'phase': 135, 'title': 'Large Hacker Collider',
        'total_collisions': len(all_debris), 'total_resonances': len(all_resonances),
        'quarks': [{'name': n, 'variance_pct': float(pca_full.explained_variance_ratio_[i]*100)} for i, n in enumerate(quark_names)],
        'debris_summary': all_debris[:10],
        'law': f'{len(all_debris)} collisions, {len(all_resonances)} resonances. 6 semantic quarks explain {sum(pca_full.explained_variance_ratio_[:6])*100:.1f}% variance.'
    }
    with open(os.path.join(RESULTS_DIR, 'phase135_lhc.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nPhase 135 complete!")
    return results

if __name__ == '__main__':
    main()
