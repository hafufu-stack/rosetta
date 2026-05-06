"""
Phase 78: The Duality Theorem
================================
Programs have TWO faces: STRUCTURE (AST) and BEHAVIOR (I/O).

Is the relationship between structure-space and behavior-space
a DUALITY? Like wave-particle duality in physics?

Test: compute the correlation between AST-distance and
behavioral-distance for all pairs. If they're correlated
but not identical, there's a duality.
"""
import os, json, time, inspect
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')


def main():
    print("=" * 60)
    print("Phase 78: The Duality Theorem")
    print("Structure vs Behavior: Wave-Particle Duality?")
    print("=" * 60)
    t0 = time.time()
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

    # Build behavioral fingerprints for each function
    print("\n--- Building Behavioral Fingerprints ---")
    test_inputs_1 = [1, 2, -1, 3, 5, 0, -5, 10]
    test_inputs_2 = [(1,2), (2,3), (-1,1), (3,5), (5,7), (0,0), (-2,3), (10,1)]

    behaviors = {}
    src_to_z = {}
    seen = set()
    for i, src in enumerate(sources):
        if src in seen:
            continue
        seen.add(src)
        src_to_z[src] = z_5d[i]

        try:
            ns = {}
            exec(compile(src, '<string>', 'exec'), ns)
            fn = [v for k, v in ns.items()
                  if callable(v) and not k.startswith('_')][0]
            sig = inspect.signature(fn)
            n_params = len(sig.parameters)

            fp = []
            if n_params == 1:
                for v in test_inputs_1:
                    try:
                        r = fn(v)
                        if isinstance(r, (int, float)):
                            r = float(r)
                            if abs(r) < 1e6 and not np.isnan(r):
                                fp.append(r)
                                continue
                    except Exception:
                        pass
                    fp.append(0.0)
            elif n_params == 2:
                for a, b in test_inputs_2:
                    try:
                        r = fn(a, b)
                        if isinstance(r, (int, float)):
                            r = float(r)
                            if abs(r) < 1e6 and not np.isnan(r):
                                fp.append(r)
                                continue
                    except Exception:
                        pass
                    fp.append(0.0)
            else:
                continue

            if len(fp) >= 5:
                behaviors[src] = np.array(fp[:8])
        except Exception:
            pass

    print(f"  Functions with behavior: {len(behaviors)}")

    # Compute pairwise distances in BOTH spaces
    print("\n--- Computing Pairwise Distances ---")
    func_list = list(behaviors.keys())
    N = len(func_list)
    n_pairs = N * (N - 1) // 2

    structural_dists = []
    behavioral_dists = []

    for i in range(N):
        for j in range(i+1, N):
            s_dist = float(np.linalg.norm(src_to_z[func_list[i]] - src_to_z[func_list[j]]))
            b_dist = float(np.linalg.norm(behaviors[func_list[i]] - behaviors[func_list[j]]))
            structural_dists.append(s_dist)
            behavioral_dists.append(b_dist)

    structural_dists = np.array(structural_dists)
    behavioral_dists = np.array(behavioral_dists)

    # Correlation
    correlation = float(np.corrcoef(structural_dists, behavioral_dists)[0, 1])
    print(f"\n  Pairs computed: {n_pairs}")
    print(f"  Structure-Behavior correlation: {correlation:.4f}")

    # Is it a duality? (correlated but not identical)
    is_duality = 0.2 < abs(correlation) < 0.95
    print(f"  Duality detected: {is_duality}")

    # Find interesting cases
    # High structural distance but low behavioral distance (look different, act same)
    s_norm = (structural_dists - structural_dists.mean()) / (structural_dists.std() + 1e-8)
    b_norm = (behavioral_dists - behavioral_dists.mean()) / (behavioral_dists.std() + 1e-8)

    diff = s_norm - b_norm  # Positive = structurally far but behaviorally close

    # Top "looks different, acts same" pairs
    top_same_idx = np.argsort(diff)[-5:]
    print("\n  === Looks Different, Acts Same ===")
    same_examples = []
    for idx in top_same_idx:
        i = 0
        count = 0
        for ii in range(N):
            for jj in range(ii+1, N):
                if count == idx:
                    i, j = ii, jj
                    break
                count += 1
            else:
                continue
            break
        f1 = func_list[i]
        f2 = func_list[j]
        s1 = f1.split('return ')[1][:20] if 'return' in f1 else '?'
        s2 = f2.split('return ')[1][:20] if 'return' in f2 else '?'
        print(f"    {s1:20s} vs {s2:20s}: "
              f"struct={structural_dists[idx]:.3f}, behav={behavioral_dists[idx]:.3f}")
        same_examples.append({'f1': f1, 'f2': f2,
                            'struct_dist': float(structural_dists[idx]),
                            'behav_dist': float(behavioral_dists[idx])})

    # Top "looks same, acts different" pairs
    top_diff_idx = np.argsort(-diff)[-5:]
    print("\n  === Looks Same, Acts Different ===")
    diff_examples = []
    for idx in top_diff_idx:
        count = 0
        for ii in range(N):
            for jj in range(ii+1, N):
                if count == idx:
                    i, j = ii, jj
                    break
                count += 1
            else:
                continue
            break
        f1 = func_list[i]
        f2 = func_list[j]
        s1 = f1.split('return ')[1][:20] if 'return' in f1 else '?'
        s2 = f2.split('return ')[1][:20] if 'return' in f2 else '?'
        print(f"    {s1:20s} vs {s2:20s}: "
              f"struct={structural_dists[idx]:.3f}, behav={behavioral_dists[idx]:.3f}")
        diff_examples.append({'f1': f1, 'f2': f2,
                            'struct_dist': float(structural_dists[idx]),
                            'behav_dist': float(behavioral_dists[idx])})

    print(f"\n  =======================================")
    print(f"  THE DUALITY THEOREM")
    print(f"  Structure-Behavior correlation: {correlation:.4f}")
    print(f"  Verdict: {'DUALITY EXISTS' if is_duality else 'NO DUALITY'}")
    print(f"  =======================================")

    elapsed = time.time() - t0
    results = {
        'phase': 78, 'name': 'The Duality Theorem',
        'n_functions': N, 'n_pairs': n_pairs,
        'correlation': correlation,
        'is_duality': bool(is_duality),
        'same_examples': same_examples,
        'diff_examples': diff_examples,
        'elapsed': elapsed, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase78_duality.json'), 'w',
              encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    # Figure
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Scatter plot: structure vs behavior distances
    sample_idx = np.random.choice(len(structural_dists),
                                  min(2000, len(structural_dists)), replace=False)
    axes[0].scatter(structural_dists[sample_idx], behavioral_dists[sample_idx],
                   alpha=0.1, s=5, c='#2196F3')
    axes[0].set_xlabel('Structural Distance (5D)')
    axes[0].set_ylabel('Behavioral Distance (I/O)')
    axes[0].set_title(f'Structure vs Behavior\ncorr={correlation:.3f}', fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # 2. Distribution of distances
    axes[1].hist(structural_dists, bins=50, alpha=0.5, color='#4CAF50',
                label='Structural', density=True, edgecolor='black')
    axes[1].hist(behavioral_dists / (behavioral_dists.max() + 1e-8) *
                structural_dists.max(), bins=50, alpha=0.5,
                color='#FF9800', label='Behavioral (scaled)', density=True,
                edgecolor='black')
    axes[1].set_xlabel('Distance')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Distance Distributions', fontweight='bold')
    axes[1].legend()

    # 3. The duality theorem
    verdict = ("THE DUALITY THEOREM\n\n"
              f"Structure-Behavior\n"
              f"Correlation: {correlation:.3f}\n\n")
    if is_duality:
        verdict += "Programs have DUAL nature:\n"
        verdict += "Structure != Behavior\n"
        verdict += "but they're CORRELATED!\n\n"
        verdict += "Like wave-particle\nduality in physics."
        bg = '#E8F5E9'
    else:
        verdict += "No duality detected."
        bg = '#FFF3E0'
    axes[2].text(0.5, 0.5, verdict, ha='center', va='center',
                fontsize=12, fontweight='bold', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor=bg, alpha=0.8),
                transform=axes[2].transAxes)
    axes[2].axis('off')

    plt.suptitle('Phase 78: The Duality Theorem\n'
                 'Wave-Particle Duality of Software',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase78_duality.png'), dpi=150)
    plt.close()
    print(f"\nPhase 78 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
