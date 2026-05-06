"""
Phase 65: The Composition Algebra
===================================
BONUS PHASE (Opus's idea)

THE ULTIMATE ALGEBRAIC TEST:
Is f composed with g (f o g) equal to f + g in latent space?

If composition is linear, then 5D space has ALGEBRAIC STRUCTURE.
This would mean programs form a vector space where:
  - Addition = composition
  - Scalar multiplication = parameter scaling
  - The origin = identity function

This is the mathematical equivalent of discovering that
"programming" is secretly "linear algebra".
"""
import os, json, time, sys, inspect
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')


def main():
    print("=" * 60)
    print("Phase 65: The Composition Algebra")
    print("Is f(g(x)) = f + g in latent space?")
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

    # Build lookup
    src_to_z = {}
    for i, src in enumerate(sources):
        if src not in src_to_z:
            src_to_z[src] = {'z_5d': z_5d[i], 'z_ast': z_ast[i]}

    # Find unary functions (composable: f(x) -> y)
    unary_funcs = {}
    for src, info in src_to_z.items():
        try:
            ns = {}
            exec(compile(src, '<string>', 'exec'), ns)
            fn = [v for k, v in ns.items()
                  if callable(v) and not k.startswith('_')][0]
            sig = inspect.signature(fn)
            if len(sig.parameters) == 1:
                # Test it works with numeric input
                test_vals = [1, 2, -1, 3, 5]
                works = True
                for v in test_vals:
                    try:
                        r = fn(v)
                        if not isinstance(r, (int, float)):
                            works = False; break
                        if abs(float(r)) > 1e6 or np.isnan(float(r)):
                            works = False; break
                    except Exception:
                        works = False; break
                if works:
                    unary_funcs[src] = {'fn': fn, 'z_5d': info['z_5d'],
                                       'z_ast': info['z_ast']}
        except Exception:
            pass

    print(f"  Composable unary functions: {len(unary_funcs)}")

    # Test composition algebra
    # For each pair (f, g), compute:
    #   1. z(f) + z(g)  (vector sum)
    #   2. z(f o g)     (embedding of composed function)
    # If they're similar, composition IS addition!

    print("\n--- Testing: f(g(x)) == f + g ? ---")
    func_items = list(unary_funcs.items())[:30]  # Limit for speed

    composition_results = []
    for i, (src_f, info_f) in enumerate(func_items):
        for j, (src_g, info_g) in enumerate(func_items):
            if i == j:
                continue

            fn_f = info_f['fn']
            fn_g = info_g['fn']

            # Compose: h(x) = f(g(x))
            try:
                test_vals = [1, 2, -1, 3, 5]
                composed_results = []
                for v in test_vals:
                    r = fn_f(fn_g(v))
                    if not isinstance(r, (int, float)):
                        break
                    if abs(float(r)) > 1e6 or np.isnan(float(r)):
                        break
                    composed_results.append(float(r))
                else:
                    if len(composed_results) == len(test_vals):
                        # Compute z(f) + z(g)
                        z_sum_5d = info_f['z_5d'] + info_g['z_5d']
                        z_sum_ast = info_f['z_ast'] + info_g['z_ast']

                        # Find nearest function to z_sum in the database
                        all_z5 = np.array([v['z_5d'] for v in src_to_z.values()])
                        all_srcs = list(src_to_z.keys())
                        dists = np.linalg.norm(all_z5 - z_sum_5d, axis=1)
                        nn_idx = np.argmin(dists)
                        nn_src = all_srcs[nn_idx]
                        nn_dist = dists[nn_idx]

                        # Check if nearest neighbor behaves like f(g(x))
                        try:
                            ns2 = {}
                            exec(compile(nn_src, '<string>', 'exec'), ns2)
                            fn_nn = [v for k, v in ns2.items()
                                     if callable(v) and not k.startswith('_')][0]
                            sig_nn = inspect.signature(fn_nn)
                            n_match = 0
                            for k, v in enumerate(test_vals):
                                try:
                                    if len(sig_nn.parameters) == 1:
                                        r_nn = float(fn_nn(v))
                                    else:
                                        continue
                                    if abs(r_nn - composed_results[k]) < 0.01:
                                        n_match += 1
                                except Exception:
                                    pass
                            match_rate = n_match / len(test_vals)
                        except Exception:
                            match_rate = 0.0

                        # Also compute cosine between z_sum and z(nearest)
                        cos_5d = float(cosine_similarity(
                            z_sum_5d.reshape(1,-1),
                            src_to_z[nn_src]['z_5d'].reshape(1,-1))[0,0])

                        composition_results.append({
                            'f': src_f, 'g': src_g,
                            'nn_src': nn_src,
                            'nn_dist': float(nn_dist),
                            'match_rate': float(match_rate),
                            'cos_5d': cos_5d,
                        })
            except Exception:
                pass

    if not composition_results:
        print("  No valid compositions found")
        elapsed = time.time() - t0
        results = {'phase': 65, 'name': 'The Composition Algebra',
                   'n_compositions': 0, 'elapsed': elapsed}
        with open(os.path.join(RESULTS_DIR, 'phase65_composition.json'), 'w',
                  encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        return results

    # Analyze results
    avg_match = np.mean([r['match_rate'] for r in composition_results])
    avg_dist = np.mean([r['nn_dist'] for r in composition_results])
    avg_cos = np.mean([r['cos_5d'] for r in composition_results])
    n_perfect = sum(1 for r in composition_results if r['match_rate'] == 1.0)

    print(f"\n  Total compositions tested: {len(composition_results)}")
    print(f"  Average match rate: {avg_match*100:.1f}%")
    print(f"  Perfect matches: {n_perfect}/{len(composition_results)} "
          f"({n_perfect/max(len(composition_results),1)*100:.1f}%)")
    print(f"  Average NN distance: {avg_dist:.4f}")
    print(f"  Average cosine(sum, nn): {avg_cos:.4f}")

    # Show best and worst compositions
    sorted_results = sorted(composition_results,
                          key=lambda x: x['match_rate'], reverse=True)

    print(f"\n--- Best Compositions (f+g approx= fog) ---")
    for r in sorted_results[:8]:
        f_short = r['f'].split('return ')[1][:12] if 'return' in r['f'] else '?'
        g_short = r['g'].split('return ')[1][:12] if 'return' in r['g'] else '?'
        nn_short = r['nn_src'].split('return ')[1][:15] if 'return' in r['nn_src'] else '?'
        status = "OK" if r['match_rate'] == 1.0 else f"{r['match_rate']*100:.0f}%"
        print(f"  [{status:4s}] {f_short:12s} o {g_short:12s} -> {nn_short}")

    print(f"\n--- Worst Compositions ---")
    for r in sorted_results[-5:]:
        f_short = r['f'].split('return ')[1][:12] if 'return' in r['f'] else '?'
        g_short = r['g'].split('return ')[1][:12] if 'return' in r['g'] else '?'
        nn_short = r['nn_src'].split('return ')[1][:15] if 'return' in r['nn_src'] else '?'
        print(f"  [{r['match_rate']*100:.0f}%] {f_short:12s} o {g_short:12s} -> {nn_short}")

    # The verdict
    is_algebraic = avg_match > 0.3
    print(f"\n  =======================================")
    print(f"  Composition Algebra:")
    print(f"    Match rate:  {avg_match*100:.1f}%")
    print(f"    Perfect:     {n_perfect}/{len(composition_results)}")
    print(f"    Verdict:     {'ALGEBRAIC STRUCTURE EXISTS' if is_algebraic else 'NON-LINEAR'}")
    print(f"  =======================================")

    elapsed = time.time() - t0
    results = {
        'phase': 65, 'name': 'The Composition Algebra',
        'n_compositions': len(composition_results),
        'avg_match_rate': float(avg_match),
        'n_perfect': n_perfect,
        'avg_nn_dist': float(avg_dist),
        'avg_cos': float(avg_cos),
        'is_algebraic': bool(is_algebraic),
        'best_compositions': sorted_results[:10],
        'worst_compositions': sorted_results[-5:],
        'elapsed': elapsed, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase65_composition.json'), 'w',
              encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    # Figure
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Match rate distribution
    match_rates = [r['match_rate'] for r in composition_results]
    axes[0].hist(match_rates, bins=20, color='#2196F3', edgecolor='black')
    axes[0].axvline(avg_match, color='red', linestyle='--',
                   label=f'Mean={avg_match:.2f}')
    axes[0].set_xlabel('Match Rate')
    axes[0].set_ylabel('Count')
    axes[0].set_title('f(g(x)) vs f+g Match\nDistribution', fontweight='bold')
    axes[0].legend()

    # 2. Distance vs match rate
    axes[1].scatter([r['nn_dist'] for r in composition_results],
                   [r['match_rate'] for r in composition_results],
                   c='#4CAF50', alpha=0.3, s=20)
    axes[1].set_xlabel('NN Distance')
    axes[1].set_ylabel('Match Rate')
    axes[1].set_title('Distance vs Correctness\n(closer=better?)', fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    # 3. The algebra verdict
    verdict = ("THE COMPOSITION ALGEBRA\n\n"
              f"Tested: {len(composition_results)} compositions\n"
              f"Match rate: {avg_match*100:.1f}%\n"
              f"Perfect: {n_perfect}\n\n")
    if is_algebraic:
        verdict += "f(g(x)) ~ f + g\nALGEBRAIC STRUCTURE EXISTS!"
        bg = '#E8F5E9'
    else:
        verdict += "Composition is\nnon-linear in this space"
        bg = '#FFF3E0'
    axes[2].text(0.5, 0.5, verdict, ha='center', va='center',
                fontsize=13, fontweight='bold', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor=bg, alpha=0.8),
                transform=axes[2].transAxes)
    axes[2].axis('off')

    plt.suptitle('Phase 65: The Composition Algebra\n'
                 'Is Programming Secretly Linear Algebra?',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase65_composition.png'), dpi=150)
    plt.close()
    print(f"\nPhase 65 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
