"""Phase 138: Software Time Crystals
Find states that oscillate forever in the Lagrangian without energy input.
SUSY pairs + CTC = perpetual oscillation = time crystal.
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
    print("Phase 138: Software Time Crystals")
    print("  Can code oscillate forever without energy?")
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
    centroid = np.mean(ast_m, axis=0)
    centered = ast_m - centroid
    
    G, lam = 1.1732, 0.7282
    
    def potential(v):
        dists = np.linalg.norm(ast_m - v.reshape(1,-1), axis=1)
        return -G * np.mean(1.0/(dists**2+0.01)) + lam * np.sum(v**2)
    
    # 1. Find SUSY pairs (from P132)
    mirrors = -centered + centroid
    mirror_dists = np.linalg.norm(mirrors[:, None] - ast_m[None, :], axis=2)
    np.fill_diagonal(mirror_dists, np.inf)
    
    susy_pairs = []
    for i in range(n):
        j = np.argmin(mirror_dists[i])
        susy_pairs.append((i, j, float(mirror_dists[i, j])))
    susy_pairs.sort(key=lambda x: x[2])
    
    # 2. Time crystal test: simulate oscillation between SUSY partners
    print("--- Time Crystal Candidates ---")
    crystal_results = []
    
    for pair_idx, (i, j, d) in enumerate(susy_pairs[:15]):
        va, vb = ast_m[i], ast_m[j]
        func_a = unique_funcs[i].split('return ')[-1].strip()[:12]
        func_b = unique_funcs[j].split('return ')[-1].strip()[:12]
        
        # Simulate oscillation: v(t) swings between va and vb
        n_steps = 100
        trajectory = []
        energies = []
        
        # Simple harmonic oscillation in the potential
        omega = np.sqrt(abs(potential(va) - potential(vb)) + 0.01)
        
        for t in range(n_steps):
            phase = np.sin(2 * np.pi * t / 20)  # Period = 20 steps
            v_t = (1 + phase) / 2 * va + (1 - phase) / 2 * vb
            E_t = potential(v_t)
            trajectory.append(v_t)
            energies.append(E_t)
        
        # Check periodicity: does it return to initial state?
        energies = np.array(energies)
        energy_var = np.var(energies)
        
        # FFT to detect periodicity
        fft = np.abs(np.fft.fft(energies - np.mean(energies)))
        dominant_freq = np.argmax(fft[1:len(fft)//2]) + 1
        fft_peak = float(fft[dominant_freq])
        
        # Time crystal criterion: periodic + stable energy
        is_crystal = energy_var < np.mean(energies)**2 * 0.1 and fft_peak > np.mean(fft[1:]) * 3
        
        crystal_results.append({
            'func_a': func_a, 'func_b': func_b,
            'susy_distance': float(d),
            'energy_variance': float(energy_var),
            'dominant_frequency': int(dominant_freq),
            'fft_peak_ratio': float(fft_peak / (np.mean(fft[1:]) + 1e-10)),
            'is_crystal': bool(is_crystal),
        })
        
        if pair_idx < 5:
            status = "TIME CRYSTAL!" if is_crystal else "unstable"
            print(f"  {func_a} <-> {func_b}: E_var={energy_var:.4f}, freq={dominant_freq}, [{status}]")
    
    n_crystals = sum(1 for r in crystal_results if r['is_crystal'])
    print(f"\n  Time crystals found: {n_crystals}/{len(crystal_results)}")
    
    # 3. Discrete time crystal: does the system spontaneously break time translation?
    # Test: apply periodic drive, check if response has DIFFERENT period
    print("\n--- Discrete Time Crystal (Period Doubling) ---")
    dtc_results = []
    
    for i, j, d in susy_pairs[:5]:
        va, vb = ast_m[i], ast_m[j]
        
        # Drive with period T=10
        drive_period = 10
        response = []
        
        for t in range(100):
            drive = np.sin(2 * np.pi * t / drive_period)
            v_t = (1 + drive) / 2 * va + (1 - drive) / 2 * vb
            # Measure observable: projection onto centroid direction
            obs = np.dot(v_t - centroid, va - vb) / np.linalg.norm(va - vb)
            response.append(obs)
        
        response = np.array(response)
        fft_resp = np.abs(np.fft.fft(response - np.mean(response)))
        
        # Check for subharmonic response (period doubling)
        expected_peak = 100 // drive_period  # Should be at freq=10
        subharmonic = fft_resp[expected_peak // 2] if expected_peak // 2 > 0 else 0
        fundamental = fft_resp[expected_peak] if expected_peak < len(fft_resp) else 0
        
        period_doubled = subharmonic > fundamental * 0.3
        
        func_a = unique_funcs[i].split('return ')[-1].strip()[:12]
        func_b = unique_funcs[j].split('return ')[-1].strip()[:12]
        dtc_results.append({
            'pair': f'{func_a}<->{func_b}',
            'subharmonic': float(subharmonic),
            'fundamental': float(fundamental),
            'period_doubled': bool(period_doubled),
        })
        
        status = "PERIOD DOUBLED!" if period_doubled else "normal"
        print(f"  {func_a}<->{func_b}: sub={subharmonic:.2f}, fund={fundamental:.2f} [{status}]")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Phase 138: Software Time Crystals', fontsize=14, fontweight='bold')
    
    # Show energy evolution of best crystal
    best_cr = crystal_results[0]
    va_best, vb_best = ast_m[susy_pairs[0][0]], ast_m[susy_pairs[0][1]]
    e_traj = []
    for t in range(100):
        phase = np.sin(2*np.pi*t/20)
        v_t = (1+phase)/2*va_best + (1-phase)/2*vb_best
        e_traj.append(potential(v_t))
    axes[0].plot(e_traj, color='#E91E63', linewidth=1.5)
    axes[0].set_xlabel('Time step'); axes[0].set_ylabel('Energy')
    axes[0].set_title(f'Crystal oscillation ({best_cr["func_a"]} <-> {best_cr["func_b"]})')
    
    labels = [r['func_a'][:8] for r in crystal_results[:10]]
    colors = ['#4CAF50' if r['is_crystal'] else '#F44336' for r in crystal_results[:10]]
    fft_peaks = [r['fft_peak_ratio'] for r in crystal_results[:10]]
    axes[1].barh(labels, fft_peaks, color=colors, edgecolor='black')
    axes[1].set_xlabel('FFT peak ratio'); axes[1].set_title(f'Time crystals: {n_crystals}/{len(crystal_results)}')
    
    if dtc_results:
        dtc_labels = [r['pair'][:15] for r in dtc_results]
        axes[2].bar(range(len(dtc_results)),
                   [r['subharmonic'] for r in dtc_results], label='Subharmonic', color='#2196F3', edgecolor='black')
        axes[2].bar(range(len(dtc_results)),
                   [r['fundamental'] for r in dtc_results], bottom=[r['subharmonic'] for r in dtc_results],
                   label='Fundamental', color='#FF9800', edgecolor='black')
        axes[2].set_xticks(range(len(dtc_results))); axes[2].set_xticklabels(dtc_labels, rotation=30, fontsize=6)
        axes[2].legend(); axes[2].set_title('Period Doubling Test')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase138_time_crystal.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    results = {
        'phase': 138, 'title': 'Software Time Crystals',
        'n_crystals': n_crystals, 'total_tested': len(crystal_results),
        'crystal_results': crystal_results[:5],
        'dtc_results': dtc_results,
        'law': f'{n_crystals}/{len(crystal_results)} time crystals found. Best: {crystal_results[0]["func_a"]}<->{crystal_results[0]["func_b"]} (E_var={crystal_results[0]["energy_variance"]:.4f}).'
    }
    with open(os.path.join(RESULTS_DIR, 'phase138_time_crystal.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nPhase 138 complete!")
    return results

if __name__ == '__main__':
    main()
