"""
Phase 69: The Adversarial Frontier
=====================================
Can we fool the latent space?

Create adversarial programs: functions that LOOK different
in source code but are semantically identical, or functions
that LOOK identical but behave differently.

If the latent space is robust, adversarial attacks should FAIL.
If it's fragile, we'll find the frontier of deception.
"""
import os, json, time, sys
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')


def main():
    print("=" * 60)
    print("Phase 69: The Adversarial Frontier")
    print("Can adversarial programs fool the Rosetta Space?")
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
    from sklearn.metrics.pairwise import cosine_similarity

    pca = PCA(n_components=10).fit(z_ast)
    z_5d = pca.transform(z_ast)[:, :5]

    src_to_z = {}
    for i, src in enumerate(sources):
        if src not in src_to_z:
            src_to_z[src] = {'z_5d': z_5d[i], 'z_ast': z_ast[i]}

    # ========================================
    # Attack 1: Semantic Equivalents (same meaning, different text)
    # The space should map these CLOSE together
    # ========================================
    print("\n--- Attack 1: Semantic Equivalents ---")
    print("  Same behavior, different source code")
    equivalents = [
        # Group: addition
        ['def f(x, y): return x + y',
         'def f(a, b): return a + b',
         'def f(m, n): return m + n',
         'def f(p, q): return p + q'],
        # Group: absolute value
        ['def f(x): return abs(x)',
         'def f(x): return x if x >= 0 else -x',
         'def f(x): return (x**2)**0.5'],
        # Group: identity-like
        ['def f(x): return x',
         'def f(x): return x + 0',
         'def f(x): return x * 1'],
        # Group: max
        ['def f(x, y): return max(x, y)',
         'def f(x, y): return x if x > y else y'],
        # Group: double
        ['def f(x): return x * 2',
         'def f(x): return x + x'],
    ]

    equiv_results = []
    for group in equivalents:
        available = [s for s in group if s in src_to_z]
        if len(available) < 2:
            continue

        # Compute all pairwise distances
        z_list = [src_to_z[s]['z_5d'] for s in available]
        dists = []
        coss = []
        for i in range(len(z_list)):
            for j in range(i+1, len(z_list)):
                d = float(np.linalg.norm(z_list[i] - z_list[j]))
                c = float(cosine_similarity(
                    z_list[i].reshape(1,-1), z_list[j].reshape(1,-1))[0,0])
                dists.append(d)
                coss.append(c)

        avg_dist = np.mean(dists)
        avg_cos = np.mean(coss)
        fooled = avg_cos < 0.8  # If cos < 0.8, the attack succeeded

        print(f"  Group: {available[0].split('return ')[1][:15]:15s} "
              f"({len(available)} variants): "
              f"avg_cos={avg_cos:.4f}, avg_dist={avg_dist:.4f} "
              f"[{'FOOLED' if fooled else 'ROBUST'}]")

        equiv_results.append({
            'group': available, 'avg_dist': avg_dist,
            'avg_cos': avg_cos, 'fooled': fooled,
        })

    n_robust_eq = sum(1 for r in equiv_results if not r['fooled'])
    print(f"\n  Robustness: {n_robust_eq}/{len(equiv_results)} groups stay close")

    # ========================================
    # Attack 2: Trojan Horses (same look, different behavior)
    # The space should map these FAR apart
    # ========================================
    print("\n--- Attack 2: Trojan Horses ---")
    print("  Similar source, different behavior")
    trojans = [
        ('def f(x, y): return x + y', 'def f(x, y): return x + y + 1',
         'Off-by-one: +1 hidden'),
        ('def f(x, y): return x * y', 'def f(x, y): return x * y * -1',
         'Sign flip: *-1 hidden'),
        ('def f(x): return abs(x)', 'def f(x): return abs(x) + 1',
         'Constant offset'),
        ('def f(x, y): return x - y', 'def f(x, y): return y - x',
         'Argument swap'),
        ('def f(x, y): return x > y', 'def f(x, y): return x >= y',
         'Boundary change: > vs >='),
        ('def f(x, y): return max(x, y)', 'def f(x, y): return min(x, y)',
         'max/min swap'),
    ]

    trojan_results = []
    for orig, trojan, desc in trojans:
        z_o = src_to_z.get(orig, {}).get('z_5d')
        z_t = src_to_z.get(trojan, {}).get('z_5d')

        if z_o is None or z_t is None:
            # Trojan not in dataset; that's expected for modified versions
            # We'll compute what we can
            if z_o is not None:
                print(f"  {desc:25s}: Trojan not in DB (novel attack)")
            continue

        cos = float(cosine_similarity(
            z_o.reshape(1,-1), z_t.reshape(1,-1))[0,0])
        dist = float(np.linalg.norm(z_o - z_t))
        detected = cos < 0.9  # Trojan detected if vectors are far apart

        print(f"  {desc:25s}: cos={cos:.4f}, dist={dist:.4f} "
              f"[{'DETECTED' if detected else 'HIDDEN!'}]")

        trojan_results.append({
            'orig': orig, 'trojan': trojan, 'desc': desc,
            'cos': cos, 'dist': dist, 'detected': detected,
        })

    n_detected = sum(1 for r in trojan_results if r['detected'])
    n_total_t = len(trojan_results)
    print(f"\n  Detection: {n_detected}/{n_total_t} trojans caught")

    # ========================================
    # Attack 3: Obfuscation (same function, maximally different AST)
    # ========================================
    print("\n--- Attack 3: Obfuscation Resistance ---")
    obfuscation_pairs = [
        ('def f(x, y): return x + y', 'def f(x, y): return (x) + (y)',
         'Parentheses padding'),
        ('def f(x): return -x', 'def f(x): return x * -1',
         'Negate via multiply'),
        ('def f(x): return x * 2', 'def f(x): return x << 1',
         'Multiply via shift'),
        ('def f(x, y): return x + y', 'def f(x, y): return x - (-y)',
         'Add via double negate'),
    ]

    obf_results = []
    for orig, obf, desc in obfuscation_pairs:
        z_o = src_to_z.get(orig, {}).get('z_5d')
        z_ob = src_to_z.get(obf, {}).get('z_5d')

        if z_o is None or z_ob is None:
            print(f"  {desc:25s}: Not in DB")
            continue

        cos = float(cosine_similarity(
            z_o.reshape(1,-1), z_ob.reshape(1,-1))[0,0])
        dist = float(np.linalg.norm(z_o - z_ob))
        robust = cos > 0.8

        print(f"  {desc:25s}: cos={cos:.4f}, dist={dist:.4f} "
              f"[{'ROBUST' if robust else 'FOOLED'}]")
        obf_results.append({
            'orig': orig, 'obfuscated': obf, 'desc': desc,
            'cos': cos, 'dist': dist, 'robust': robust,
        })

    # Summary
    n_robust_obf = sum(1 for r in obf_results if r['robust'])
    print(f"\n  Obfuscation resistance: {n_robust_obf}/{len(obf_results)}")

    print("\n  === ADVERSARIAL FRONTIER SUMMARY ===")
    print(f"  Semantic equivalence robustness: {n_robust_eq}/{len(equiv_results)}")
    print(f"  Trojan detection rate:           {n_detected}/{n_total_t}")
    print(f"  Obfuscation resistance:          {n_robust_obf}/{len(obf_results)}")

    elapsed = time.time() - t0
    results = {
        'phase': 69, 'name': 'The Adversarial Frontier',
        'equiv_robust': n_robust_eq, 'equiv_total': len(equiv_results),
        'trojan_detected': n_detected, 'trojan_total': n_total_t,
        'obf_robust': n_robust_obf, 'obf_total': len(obf_results),
        'equiv_results': equiv_results,
        'trojan_results': trojan_results,
        'obf_results': obf_results,
        'elapsed': elapsed, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase69_adversarial.json'), 'w',
              encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    # Figure
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Equivalence cosines
    if equiv_results:
        eq_names = [r['group'][0].split('return ')[1][:10] for r in equiv_results]
        eq_cos = [r['avg_cos'] for r in equiv_results]
        colors_eq = ['#4CAF50' if not r['fooled'] else '#F44336' for r in equiv_results]
        axes[0].barh(eq_names, eq_cos, color=colors_eq, edgecolor='black')
        axes[0].axvline(0.8, color='orange', linestyle='--')
        axes[0].set_xlabel('Cosine Similarity')
        axes[0].set_title('Semantic Equivalence\n(should be close)', fontweight='bold')

    # 2. Trojan detection
    if trojan_results:
        tr_names = [r['desc'][:15] for r in trojan_results]
        tr_cos = [r['cos'] for r in trojan_results]
        colors_tr = ['#4CAF50' if r['detected'] else '#F44336' for r in trojan_results]
        axes[1].barh(tr_names, tr_cos, color=colors_tr, edgecolor='black')
        axes[1].axvline(0.9, color='orange', linestyle='--')
        axes[1].set_xlabel('Cosine Similarity')
        axes[1].set_title('Trojan Detection\n(should be far)', fontweight='bold')

    # 3. Summary
    categories = ['Equivalence\nRobustness', 'Trojan\nDetection', 'Obfuscation\nResistance']
    rates = [n_robust_eq/max(len(equiv_results),1)*100,
             n_detected/max(n_total_t,1)*100,
             n_robust_obf/max(len(obf_results),1)*100]
    axes[2].bar(categories, rates, color=['#2196F3', '#FF9800', '#9C27B0'],
               edgecolor='black')
    axes[2].set_ylabel('Success Rate (%)')
    axes[2].set_title('Adversarial Robustness\nOverall', fontweight='bold')
    axes[2].set_ylim(0, 110)

    plt.suptitle('Phase 69: The Adversarial Frontier\n'
                 'How Robust is the Rosetta Space?',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase69_adversarial.png'), dpi=150)
    plt.close()
    print(f"\nPhase 69 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
