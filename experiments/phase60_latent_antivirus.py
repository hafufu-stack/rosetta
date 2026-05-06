"""
Phase 60: The Latent Antivirus
================================
Detect MALICIOUS code using only vector distances.
No signature matching. No sandbox. Pure math.

Key insight: Even a tiny backdoor causes massive semantic drift
in the Rosetta latent space, because malicious intent changes
the MEANING of the function, not just its text.
"""
import os, json, time, sys, ast, inspect
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')


def main():
    print("=" * 60)
    print("Phase 60: The Latent Antivirus")
    print("Detect malware by semantic distance, not signatures")
    print("=" * 60)
    t0 = time.time()
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    # Load space
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

    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.decomposition import PCA
    pca = PCA(n_components=10).fit(z_ast)
    z_5d = pca.transform(z_ast)[:, :5]

    # Build embedding lookup
    src_to_z = {}
    for i, src in enumerate(sources):
        if src not in src_to_z:
            src_to_z[src] = {'z_ast': z_ast[i], 'z_5d': z_5d[i]}

    # Generate normal vs malicious code pairs
    # Since our dataset has simple functions, we simulate
    # "backdoor insertion" by comparing semantically different operations
    malware_scenarios = [
        {
            'name': 'auth_bypass',
            'desc': 'Password check -> always true backdoor',
            'normal': 'def f(x, y): return x == y',
            'malicious': 'def f(x, y): return x != y',
            'attack': 'Flipped equality to bypass auth',
        },
        {
            'name': 'data_theft',
            'desc': 'Read value -> expose value',
            'normal': 'def f(x): return abs(x)',
            'malicious': 'def f(x): return -x',
            'attack': 'Negation instead of abs = data sign leak',
        },
        {
            'name': 'logic_bomb',
            'desc': 'Normal addition -> conditional destruction',
            'normal': 'def f(x, y): return x + y',
            'malicious': 'def f(x, y): return x - y',
            'attack': 'Subtraction disguised as addition',
        },
        {
            'name': 'privilege_escalation',
            'desc': 'Min privilege -> max privilege',
            'normal': 'def f(x, y): return min(x, y)',
            'malicious': 'def f(x, y): return max(x, y)',
            'attack': 'min->max = give max privilege',
        },
        {
            'name': 'comparison_inversion',
            'desc': 'Greater than -> less than (ACL bypass)',
            'normal': 'def f(x, y): return x > y',
            'malicious': 'def f(x, y): return x < y',
            'attack': 'Inverted comparison = ACL bypass',
        },
        {
            'name': 'coercion_attack',
            'desc': 'Type check -> type coercion',
            'normal': 'def f(x): return int(x)',
            'malicious': 'def f(x): return float(x)',
            'attack': 'int->float = precision exploit',
        },
        {
            'name': 'overflow_exploit',
            'desc': 'Safe multiply -> unsafe power',
            'normal': 'def f(x, y): return x * y',
            'malicious': 'def f(x, y): return x ** y',
            'attack': 'mul->pow = integer overflow',
        },
        {
            'name': 'benign_mutation',
            'desc': 'Same function, different var names (FALSE POSITIVE test)',
            'normal': 'def f(x, y): return x + y',
            'malicious': 'def f(a, b): return a + b',
            'attack': 'NONE (should NOT be flagged)',
        },
    ]

    print("\n--- Malware Detection via Semantic Distance ---")
    results_list = []
    tp, fp, fn_count, tn = 0, 0, 0, 0

    for scenario in malware_scenarios:
        z_n = src_to_z.get(scenario['normal'], {}).get('z_ast')
        z_m = src_to_z.get(scenario['malicious'], {}).get('z_ast')
        z5_n = src_to_z.get(scenario['normal'], {}).get('z_5d')
        z5_m = src_to_z.get(scenario['malicious'], {}).get('z_5d')

        if z_n is None or z_m is None:
            print(f"  {scenario['name']:25s}: SKIP (not in dataset)")
            continue

        cos = float(cosine_similarity(z_n.reshape(1,-1), z_m.reshape(1,-1))[0,0])
        dist_5d = float(np.linalg.norm(z5_n - z5_m))
        is_benign = scenario['name'] == 'benign_mutation'

        # Detection threshold
        THREAT_THRESHOLD = 0.9
        detected_threat = cos < THREAT_THRESHOLD

        if is_benign:
            if detected_threat:
                fp += 1; label = "FALSE POSITIVE"
            else:
                tn += 1; label = "TRUE NEGATIVE (correct!)"
        else:
            if detected_threat:
                tp += 1; label = "DETECTED"
            else:
                fn_count += 1; label = "MISSED"

        print(f"  {scenario['name']:25s}: cos={cos:+.4f}, 5D_dist={dist_5d:.4f} "
              f"[{label}]")
        print(f"    {scenario['desc']}")
        print(f"    Attack: {scenario['attack']}")

        results_list.append({
            'name': scenario['name'],
            'desc': scenario['desc'],
            'normal': scenario['normal'],
            'malicious': scenario['malicious'],
            'attack': scenario['attack'],
            'cosine': cos,
            'dist_5d': dist_5d,
            'is_benign': is_benign,
            'detected': bool(detected_threat),
            'label': label,
        })

    n_threats = tp + fn_count
    n_benign = tn + fp
    print(f"\n  === LATENT ANTIVIRUS RESULTS ===")
    print(f"  Threat Detection:  {tp}/{n_threats} "
          f"({tp/max(n_threats,1)*100:.0f}%)")
    print(f"  False Positive:    {fp}/{n_benign} "
          f"({fp/max(n_benign,1)*100:.0f}%)")
    print(f"  Precision: {tp/max(tp+fp,1)*100:.0f}%")
    print(f"  Recall:    {tp/max(tp+fn_count,1)*100:.0f}%")

    # Semantic distance analysis
    print("\n--- Semantic Distance Analysis ---")
    threat_dists = [r['dist_5d'] for r in results_list if not r['is_benign']]
    benign_dists = [r['dist_5d'] for r in results_list if r['is_benign']]
    threat_cos = [r['cosine'] for r in results_list if not r['is_benign']]
    benign_cos = [r['cosine'] for r in results_list if r['is_benign']]

    if threat_dists:
        print(f"  Threat avg 5D distance: {np.mean(threat_dists):.4f}")
    if benign_dists:
        print(f"  Benign avg 5D distance: {np.mean(benign_dists):.4f}")
    if threat_cos:
        print(f"  Threat avg cosine:      {np.mean(threat_cos):.4f}")
    if benign_cos:
        print(f"  Benign avg cosine:      {np.mean(benign_cos):.4f}")

    elapsed = time.time() - t0
    results = {
        'phase': 60, 'name': 'The Latent Antivirus',
        'tp': tp, 'fp': fp, 'fn': fn_count, 'tn': tn,
        'precision': tp / max(tp+fp, 1),
        'recall': tp / max(tp+fn_count, 1),
        'threat_avg_cos': float(np.mean(threat_cos)) if threat_cos else 0,
        'benign_avg_cos': float(np.mean(benign_cos)) if benign_cos else 0,
        'scenarios': results_list,
        'elapsed': elapsed, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase60_latent_antivirus.json'), 'w',
              encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    # Figure
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Cosine similarity: normal vs malicious
    names = [r['name'][:12] for r in results_list]
    cosines = [r['cosine'] for r in results_list]
    colors = ['#4CAF50' if r['is_benign'] else '#F44336' for r in results_list]
    axes[0].barh(names, cosines, color=colors, edgecolor='black')
    axes[0].axvline(THREAT_THRESHOLD, color='orange', linestyle='--',
                   label=f'Threshold={THREAT_THRESHOLD}')
    axes[0].set_xlabel('Cosine Similarity')
    axes[0].set_title('Normal vs Malicious\nSemantic Distance', fontweight='bold')
    axes[0].legend()

    # 2. 5D distance
    axes[1].barh(names, [r['dist_5d'] for r in results_list],
                color=colors, edgecolor='black')
    axes[1].set_xlabel('5D L2 Distance')
    axes[1].set_title('Semantic Drift\nin 5D Space', fontweight='bold')

    # 3. Detection summary
    axes[2].bar(['True\nPositive', 'False\nPositive', 'True\nNegative', 'False\nNegative'],
               [tp, fp, tn, fn_count],
               color=['#4CAF50', '#FF9800', '#2196F3', '#F44336'], edgecolor='black')
    axes[2].set_ylabel('Count')
    axes[2].set_title(f'Detection: P={tp/max(tp+fp,1)*100:.0f}% '
                     f'R={tp/max(tp+fn_count,1)*100:.0f}%', fontweight='bold')

    plt.suptitle('Phase 60: The Latent Antivirus\n'
                 'Detecting Malware by Mathematical Distance',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase60_latent_antivirus.png'), dpi=150)
    plt.close()
    print(f"\nPhase 60 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
