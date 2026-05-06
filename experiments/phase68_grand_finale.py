"""
Phase 68: The Rosetta Symphony & Grand Finale
================================================
Part 1: Generate music from 5D program coordinates.
Each dimension maps to an oscillator frequency.
The 14 species become 14 "instruments" playing together.

Part 2: Generate summary for paper_v3.

THIS IS THE FINAL PHASE OF PROJECT ROSETTA.
"""
import os, json, time, sys, struct, wave, math
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')


def generate_symphony():
    """Generate a WAV file from 5D program coordinates."""
    print("\n--- Generating The Rosetta Symphony ---")
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    lat_file = os.path.join(DATA_DIR, 'rosetta_latents_v2.npz')
    if not os.path.exists(lat_file):
        lat_file = os.path.join(DATA_DIR, 'rosetta_latents.npz')
    latents = np.load(lat_file)
    z_ast = latents['ast']

    ds_file = os.path.join(DATA_DIR, 'rosetta_dataset_v2.json')
    if not os.path.exists(ds_file):
        ds_file = os.path.join(DATA_DIR, 'rosetta_dataset.json')
    with open(ds_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)['dataset']
    sources = [d['source'] for d in dataset]

    from sklearn.decomposition import PCA
    pca = PCA(n_components=10).fit(z_ast)
    z_5d = pca.transform(z_ast)[:, :5]

    # Deduplicate
    unique_z5 = []
    unique_src = []
    seen = set()
    for i, src in enumerate(sources):
        if src not in seen:
            seen.add(src)
            unique_z5.append(z_5d[i])
            unique_src.append(src)

    unique_z5 = np.array(unique_z5)
    N = len(unique_z5)

    # Map 5D to musical parameters
    # Base frequencies for the 5 dimensions (pentatonic scale)
    BASE_FREQS = [261.63, 293.66, 329.63, 392.00, 440.00]  # C4, D4, E4, G4, A4

    SAMPLE_RATE = 44100
    DURATION = 30  # seconds
    n_samples = SAMPLE_RATE * DURATION

    # Select representative functions (one per "species")
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=14, random_state=42, n_init=10)
    labels = km.fit_predict(unique_z5)
    representatives = []
    for c in range(14):
        mask = labels == c
        indices = np.where(mask)[0]
        if len(indices) > 0:
            # Pick the one closest to centroid
            dists = np.linalg.norm(unique_z5[indices] - km.cluster_centers_[c], axis=1)
            rep_idx = indices[np.argmin(dists)]
            representatives.append({
                'src': unique_src[rep_idx],
                'z_5d': unique_z5[rep_idx],
                'cluster': c,
            })

    print(f"  {len(representatives)} species representatives selected")

    # Generate audio
    audio = np.zeros(n_samples, dtype=np.float64)

    # Each species plays for a portion of the symphony
    segment_len = n_samples // len(representatives)

    for si, rep in enumerate(representatives):
        z = rep['z_5d']
        start = si * segment_len
        end = min(start + segment_len, n_samples)
        t = np.arange(end - start) / SAMPLE_RATE

        # Map 5D coordinates to frequencies and amplitudes
        for dim in range(5):
            freq = BASE_FREQS[dim] * (1 + z[dim] * 0.5)  # Modulate freq
            amp = 0.15 * (1 + abs(z[dim])) / 5  # Modulate amplitude
            # Add harmonics for richness
            wave_data = amp * np.sin(2 * np.pi * freq * t)
            wave_data += amp * 0.3 * np.sin(2 * np.pi * freq * 2 * t)  # 2nd harmonic
            wave_data += amp * 0.1 * np.sin(2 * np.pi * freq * 3 * t)  # 3rd harmonic

            # Apply envelope (fade in/out)
            env_len = min(int(SAMPLE_RATE * 0.1), len(t) // 4)
            envelope = np.ones(len(t))
            envelope[:env_len] = np.linspace(0, 1, env_len)
            envelope[-env_len:] = np.linspace(1, 0, env_len)
            wave_data *= envelope

            audio[start:end] += wave_data

    # Normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.8

    # Write WAV
    wav_path = os.path.join(BASE_DIR, 'rosetta_symphony.wav')
    with wave.open(wav_path, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(SAMPLE_RATE)
        audio_int = (audio * 32767).astype(np.int16)
        wf.writeframes(audio_int.tobytes())

    print(f"  Symphony saved: {wav_path} ({DURATION}s)")
    return wav_path, representatives


def main():
    print("=" * 60)
    print("Phase 68: The Rosetta Symphony & Grand Finale")
    print("Music from 5D space + Project summary")
    print("=" * 60)
    t0 = time.time()

    # Generate symphony
    wav_path, reps = generate_symphony()

    # Collect all results
    print("\n--- Project Rosetta: Complete Summary ---")
    all_results = {}
    for fname in sorted(os.listdir(RESULTS_DIR)):
        if fname.startswith('phase') and fname.endswith('.json'):
            try:
                with open(os.path.join(RESULTS_DIR, fname), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                phase_num = data.get('phase', '?')
                all_results[phase_num] = {
                    'name': data.get('name', fname),
                    'file': fname,
                }
            except Exception:
                pass

    print(f"  Total phases completed: {len(all_results)}")
    for pnum in sorted(all_results.keys(), key=lambda x: int(x) if isinstance(x, int) else 0):
        info = all_results[pnum]
        print(f"    P{pnum}: {info['name']}")

    # Key metrics summary
    print("\n  === KEY METRICS ===")
    key_metrics = {
        '5D Variance (Original)': '86.3%',
        '5D Variance (Universal)': '87.4% (P63)',
        'I/O Search Accuracy': '100% (P56)',
        'NL Search Top-5': '89% (P56)',
        'RAG Direct Solve': '60% (P57)',
        'Linter Bug Detection': '88% (P58)',
        'Antivirus Precision': '100% (P60)',
        'Antivirus Recall': '83% (P60)',
        'Silicon Translation 5D': '64.4% (P62)',
        'Silicon Translation 64D': '76.7% (P62)',
        'Number of Species': '14 (P64)',
        'Composition Match': '31.2% (P65)',
        'Total Phases': str(len(all_results)),
    }

    for k, v in key_metrics.items():
        print(f"    {k:30s}: {v}")

    # Generate the final figure
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. Timeline of discoveries
    milestones = [
        (1, 'AST Encoder', 0.2),
        (7, 'PCA = 5D', 0.4),
        (16, 'Neural CPU', 0.5),
        (33, 'Eigenfunctions', 0.6),
        (40, 'Holographic', 0.7),
        (49, '5D Invariance', 0.85),
        (56, 'I/O 100%', 0.9),
        (60, 'Antivirus', 0.92),
        (62, 'Silicon', 0.95),
        (63, 'Universal', 0.97),
        (65, 'Algebra', 0.98),
        (68, 'Symphony', 1.0),
    ]
    phases = [m[0] for m in milestones]
    scores = [m[2] for m in milestones]
    labels = [m[1] for m in milestones]

    axes[0].fill_between(phases, scores, alpha=0.3, color='#4CAF50')
    axes[0].plot(phases, scores, 'o-', color='#4CAF50', markersize=6)
    for phase_n, label, score in milestones:
        axes[0].text(phase_n + 1, score + 0.02, label, fontsize=6, rotation=30)
    axes[0].set_xlabel('Phase')
    axes[0].set_ylabel('Discovery Impact')
    axes[0].set_title('The Journey of Discovery\n(68 Phases)', fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # 2. Key metrics bar chart
    metric_names = ['I/O\nSearch', 'NL\nTop-5', 'Linter', 'Antivirus\nPrecision',
                   'Silicon\n64D', '5D\nUniversal']
    metric_vals = [100, 89, 88, 100, 76.7, 87.4]
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#F44336', '#9C27B0', '#009688']
    axes[1].bar(metric_names, metric_vals, color=colors, edgecolor='black')
    axes[1].set_ylabel('Score (%)')
    axes[1].set_title('Key Achievements', fontweight='bold')
    for i, v in enumerate(metric_vals):
        axes[1].text(i, v+2, f'{v:.0f}%', ha='center', fontweight='bold', fontsize=10)

    # 3. The Rosetta Stone
    final_text = (
        "PROJECT ROSETTA\n"
        "The Physics of Software\n"
        "========================\n\n"
        "68 Phases of Discovery\n"
        "236 Functions Analyzed\n"
        "14 Species Identified\n"
        "5 Dimensions Are Enough\n"
        "1 Universal Law\n\n"
        "Programs are vectors.\n"
        "Bugs are distances.\n"
        "Compilation is rotation.\n"
        "Composition is algebra.\n\n"
        "CODE FREEZE"
    )
    axes[2].text(0.5, 0.5, final_text, ha='center', va='center',
                fontsize=12, fontweight='bold', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#E8EAF6', alpha=0.9),
                transform=axes[2].transAxes)
    axes[2].axis('off')

    plt.suptitle('Phase 68: The Rosetta Symphony\n'
                 'The Grand Finale of Project Rosetta',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase68_grand_finale.png'), dpi=150)
    plt.close()

    elapsed = time.time() - t0
    results = {
        'phase': 68, 'name': 'The Rosetta Symphony & Grand Finale',
        'symphony_path': wav_path,
        'n_phases_total': len(all_results),
        'key_metrics': key_metrics,
        'species_in_symphony': [r['src'][:30] for r in reps],
        'elapsed': elapsed, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'status': 'CODE FREEZE',
    }
    with open(os.path.join(RESULTS_DIR, 'phase68_grand_finale.json'), 'w',
              encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"\nPhase 68 complete in {elapsed:.1f}s")
    print("\n" + "=" * 60)
    print("  PROJECT ROSETTA: CODE FREEZE")
    print("  68 Phases. 5 Dimensions. 1 Universal Law.")
    print("  The Physics of Software is Complete.")
    print("=" * 60)
    return results

if __name__ == '__main__':
    main()
