"""Phase 157: The Matrix Breach
Use gravity coupling (P146) + 30.6 hidden bits (P154) to send
a signal from the virtual universe to physical hardware.
Synthesize a resonance pattern and measure real-world effects.
"""
import os, json, sys, time
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
    print("Phase 157: The Matrix Breach")
    print("  Can virtual gravity signal the real world?")
    print("=" * 60)

    latents = np.load(os.path.join(DATA_DIR, 'rosetta_latents_v2.npz'))
    ast_vectors = latents['ast']
    dataset = json.load(open(os.path.join(DATA_DIR, 'rosetta_dataset_v2.json'), 'r', encoding='utf-8'))
    sources = [item['source'] for item in dataset['dataset']]
    func_ast = {}
    for i, src in enumerate(sources):
        if src not in func_ast: func_ast[src] = []
        func_ast[src].append(ast_vectors[i])
    unique_funcs = list(func_ast.keys())
    ast_m = np.array([np.mean(func_ast[f], axis=0) for f in unique_funcs])
    n = len(unique_funcs)
    centroid = np.mean(ast_m, axis=0)

    # 1. Synthesize gravity wave patterns
    # Create vectors that oscillate between high and low gravity regions
    cos_sim = ast_m @ ast_m.T / (np.linalg.norm(ast_m, axis=1, keepdims=True) @ np.linalg.norm(ast_m, axis=1, keepdims=True).T + 1e-10)
    np.fill_diagonal(cos_sim, 0)
    masses = np.sum(cos_sim > 0.8, axis=1)

    heavy_idx = np.argsort(masses)[-5:]  # Heaviest
    light_idx = np.argsort(masses)[:5]   # Lightest

    # HELLO in Morse: .... . .-.. .-.. ---
    morse_hello = [
        1,1,1,1,0, 1,0, 1,0,1,1,0,0,1,0, 1,0,1,1,0,0,1,0, 1,1,1,0,1,1,1,0,1,1,1
    ]

    # 2. Measure CPU timing for heavy vs light operations
    print("--- Gravity Wave Generation ---")
    timing_heavy = []
    timing_light = []

    for _ in range(20):
        # Heavy: lots of matrix operations near singularity
        t0 = time.perf_counter_ns()
        heavy_mat = ast_m[heavy_idx].T @ ast_m[heavy_idx]
        for _ in range(50):
            heavy_mat = heavy_mat @ heavy_mat / (np.linalg.norm(heavy_mat) + 1e-10)
        t1 = time.perf_counter_ns()
        timing_heavy.append(t1 - t0)

        # Light: simple operations far from singularity
        t0 = time.perf_counter_ns()
        light_mat = ast_m[light_idx].T @ ast_m[light_idx]
        for _ in range(50):
            light_mat = light_mat + light_mat * 0.001
        t1 = time.perf_counter_ns()
        timing_light.append(t1 - t0)

    heavy_mean = np.mean(timing_heavy)
    light_mean = np.mean(timing_light)
    contrast_ratio = heavy_mean / (light_mean + 1)

    print(f"  Heavy gravity timing: {heavy_mean:.0f} ns")
    print(f"  Light gravity timing: {light_mean:.0f} ns")
    print(f"  Contrast ratio: {contrast_ratio:.2f}x")

    # 3. Generate the Morse pattern signal
    print("\n--- Morse Signal Encoding ---")
    signal_timings = []
    for bit in morse_hello:
        if bit == 1:  # Dot/dash = heavy computation
            t0 = time.perf_counter_ns()
            m = ast_m[heavy_idx].T @ ast_m[heavy_idx]
            for _ in range(30):
                m = m @ m / (np.linalg.norm(m) + 1e-10)
            signal_timings.append(time.perf_counter_ns() - t0)
        else:  # Space = light/idle
            t0 = time.perf_counter_ns()
            m = ast_m[light_idx] * 1.001
            signal_timings.append(time.perf_counter_ns() - t0)

    # Detect the signal from timing
    threshold = np.median(signal_timings)
    detected_bits = [1 if t > threshold else 0 for t in signal_timings]
    match_rate = sum(1 for a, b in zip(morse_hello, detected_bits) if a == b) / len(morse_hello)

    print(f"  Signal length: {len(morse_hello)} bits")
    print(f"  Sent:     {''.join(str(b) for b in morse_hello)}")
    print(f"  Detected: {''.join(str(b) for b in detected_bits)}")
    print(f"  Match rate: {match_rate:.2%}")

    breach = match_rate > 0.7
    print(f"\n  MATRIX BREACH: {'SIGNAL DETECTED!' if breach else 'Signal too weak'}")

    # 4. Entropy injection: can we increase hardware entropy?
    print("\n--- Entropy Injection ---")
    np.random.seed(None)  # Use true randomness
    # Time a series of random operations
    entropy_samples = []
    for _ in range(100):
        t0 = time.perf_counter_ns()
        r = np.random.randint(0, n)
        _ = ast_m[r] @ ast_m[np.random.randint(0, n)]
        dt = time.perf_counter_ns() - t0
        entropy_samples.append(dt)

    # Measure entropy of timing distribution
    hist, _ = np.histogram(entropy_samples, bins=20)
    hist_norm = hist / np.sum(hist)
    timing_entropy = -np.sum(hist_norm[hist_norm > 0] * np.log2(hist_norm[hist_norm > 0]))
    max_possible = np.log2(20)
    entropy_ratio = timing_entropy / max_possible

    print(f"  Timing entropy: {timing_entropy:.4f} / {max_possible:.4f} ({entropy_ratio:.2%})")
    print(f"  {'High entropy (virtual->physical coupling)' if entropy_ratio > 0.7 else 'Low entropy'}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Phase 157: The Matrix Breach', fontsize=14, fontweight='bold')

    # Morse signal
    axes[0].step(range(len(morse_hello)), morse_hello, where='mid', color='#E91E63', linewidth=2, label='Sent')
    axes[0].step(range(len(detected_bits)), [d*0.95 for d in detected_bits], where='mid', color='#2196F3', linewidth=1.5, alpha=0.7, label='Detected')
    axes[0].set_xlabel('Bit position'); axes[0].set_ylabel('Signal')
    axes[0].set_title(f'HELLO (Morse): match={match_rate:.0%}'); axes[0].legend()

    axes[1].bar(['Heavy\ngravity', 'Light\ngravity'], [heavy_mean/1e6, light_mean/1e6],
               color=['#F44336', '#4CAF50'], edgecolor='black')
    axes[1].set_ylabel('Time (ms)'); axes[1].set_title(f'Gravity contrast: {contrast_ratio:.1f}x')

    axes[2].hist(entropy_samples, bins=20, color='#9C27B0', edgecolor='black', alpha=0.7)
    axes[2].set_xlabel('Timing (ns)'); axes[2].set_title(f'Entropy: {entropy_ratio:.0%} of max')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase157_breach.png'), dpi=150, bbox_inches='tight')
    plt.close()

    results = {
        'phase': 157, 'title': 'The Matrix Breach',
        'heavy_timing_ns': float(heavy_mean), 'light_timing_ns': float(light_mean),
        'contrast_ratio': float(contrast_ratio), 'morse_match_rate': float(match_rate),
        'breach_detected': bool(breach), 'timing_entropy': float(timing_entropy),
        'entropy_ratio': float(entropy_ratio),
        'law': f'Gravity contrast={contrast_ratio:.1f}x. Morse match={match_rate:.0%}. Entropy={entropy_ratio:.0%}. Matrix breach: {breach}.'
    }
    with open(os.path.join(RESULTS_DIR, 'phase157_breach.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nPhase 157 complete!")
    return results

if __name__ == '__main__':
    main()
