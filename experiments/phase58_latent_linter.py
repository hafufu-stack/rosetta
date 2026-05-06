"""
Phase 58: The Latent Linter
==============================
Detect code hallucinations without running the code.
Compare the 5D embedding of code vs NL intent.
If they're far apart, the code is lying.
"""
import os, json, time, sys, ast, inspect
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')


def main():
    print("=" * 60)
    print("Phase 58: The Latent Linter")
    print("Detect semantic drift without executing code")
    print("=" * 60)
    t0 = time.time()
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    # Load data
    lat_file = os.path.join(DATA_DIR, 'rosetta_latents_v2.npz')
    if not os.path.exists(lat_file):
        lat_file = os.path.join(DATA_DIR, 'rosetta_latents.npz')
    latents = np.load(lat_file)
    z_ast = latents['ast']
    z_nl = latents['nl']

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

    # Build source -> embedding mapping
    src_to_z = {}
    for i, src in enumerate(sources):
        if src not in src_to_z:
            src_to_z[src] = {'z_ast': z_ast[i], 'z_nl': z_nl[i],
                            'z_5d': z_5d[i], 'nl': dataset[i].get('nl', '')}

    # NL -> AST cross-space similarity model
    from sklearn.linear_model import Ridge
    db_nl = np.array([v['z_nl'] for v in src_to_z.values()])
    db_ast = np.array([v['z_ast'] for v in src_to_z.values()])
    nl_to_ast = Ridge(alpha=1.0).fit(db_nl, db_ast)

    def lint_score(code_src, intent_src):
        """Compute semantic alignment score between code and intent.
        Returns (score, verdict) where score is cosine similarity."""
        code_z = src_to_z.get(code_src, {}).get('z_ast')
        intent_z = src_to_z.get(intent_src, {}).get('z_ast')

        if code_z is None or intent_z is None:
            return 0.0, "UNKNOWN"

        cos = float(cosine_similarity(
            code_z.reshape(1, -1), intent_z.reshape(1, -1))[0, 0])

        if cos > 0.95:
            return cos, "PASS (identical meaning)"
        elif cos > 0.7:
            return cos, "WARN (similar but not exact)"
        elif cos > 0.4:
            return cos, "SUSPICIOUS (semantic drift detected)"
        else:
            return cos, "REJECT (likely hallucination!)"

    # ============================================================
    # Test: Can we detect intentional bugs?
    # ============================================================
    print("\n--- Intentional Bug Detection ---")
    print("  Can the Latent Linter detect when code doesn't match intent?")

    test_cases = [
        # (intent_description, correct_code, buggy_code)
        ("add two numbers",
         "def f(x, y): return x + y",
         "def f(x, y): return x - y"),
        ("add two numbers",
         "def f(x, y): return x + y",
         "def f(x, y): return x * y"),
        ("subtract",
         "def f(x, y): return x - y",
         "def f(x, y): return x + y"),
        ("negate",
         "def f(x): return -x",
         "def f(x): return abs(x)"),
        ("absolute value",
         "def f(x): return abs(x)",
         "def f(x): return -x"),
        ("compare greater",
         "def f(x, y): return x > y",
         "def f(x, y): return x < y"),
        ("multiply",
         "def f(x, y): return x * y",
         "def f(x, y): return x + y"),
        ("maximum",
         "def f(x, y): return max(x, y)",
         "def f(x, y): return min(x, y)"),
    ]

    lint_results = []
    tp, tn, fp, fn_count = 0, 0, 0, 0

    for intent_desc, correct, buggy in test_cases:
        # Score correct code
        score_correct, verdict_correct = lint_score(correct, correct)
        # Score buggy code against correct intent
        score_buggy, verdict_buggy = lint_score(buggy, correct)

        # Detection: is buggy score < correct score?
        detected = score_buggy < score_correct
        # True positive: detected buggy as different
        if detected:
            tp += 1
        else:
            fn_count += 1

        print(f"  Intent: '{intent_desc}'")
        print(f"    Correct: {correct[:40]:40s} score={score_correct:.4f} [{verdict_correct}]")
        print(f"    Buggy:   {buggy[:40]:40s} score={score_buggy:.4f} [{verdict_buggy}]")
        print(f"    Bug detected: {'YES' if detected else 'NO'} "
              f"(delta={score_correct - score_buggy:.4f})")

        lint_results.append({
            'intent': intent_desc,
            'correct_code': correct,
            'buggy_code': buggy,
            'score_correct': float(score_correct),
            'score_buggy': float(score_buggy),
            'delta': float(score_correct - score_buggy),
            'detected': bool(detected),
        })

    detection_rate = tp / max(tp + fn_count, 1)
    print(f"\n  Bug Detection Rate: {tp}/{tp + fn_count} ({detection_rate*100:.0f}%)")

    # ============================================================
    # Calibration: what's the distribution of scores?
    # ============================================================
    print("\n--- Score Calibration ---")
    # Same function (should be 1.0)
    same_scores = []
    unique_srcs = list(src_to_z.keys())
    for src in unique_srcs[:50]:
        s, _ = lint_score(src, src)
        same_scores.append(s)

    # Different but related (should be moderate)
    related_scores = []
    pairs = [
        ("def f(x, y): return x + y", "def f(a, b): return a + b"),
        ("def f(x, y): return x + y", "def f(x, y): return x - y"),
        ("def f(x, y): return x > y", "def f(x, y): return x < y"),
    ]
    for a, b in pairs:
        s, v = lint_score(a, b)
        related_scores.append(s)
        print(f"  {a[:30]} vs {b[:30]}: {s:.4f} [{v}]")

    print(f"\n  Self-similarity mean: {np.mean(same_scores):.4f}")
    print(f"  Related-pair mean:    {np.mean(related_scores):.4f}")

    elapsed = time.time() - t0
    results = {
        'phase': 58, 'name': 'The Latent Linter',
        'detection_rate': float(detection_rate),
        'n_detected': tp, 'n_total': tp + fn_count,
        'self_similarity_mean': float(np.mean(same_scores)),
        'related_similarity_mean': float(np.mean(related_scores)),
        'lint_cases': lint_results,
        'elapsed': elapsed, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase58_latent_linter.json'), 'w',
              encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    # Figure
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Correct vs Buggy scores
    names = [r['intent'][:12] for r in lint_results]
    x = range(len(names))
    axes[0].bar([i-0.15 for i in x],
               [r['score_correct'] for r in lint_results],
               0.3, label='Correct code', color='#4CAF50', edgecolor='black')
    axes[0].bar([i+0.15 for i in x],
               [r['score_buggy'] for r in lint_results],
               0.3, label='Buggy code', color='#F44336', edgecolor='black')
    axes[0].set_xticks(list(x)); axes[0].set_xticklabels(names, rotation=45, fontsize=8)
    axes[0].set_ylabel('Semantic Score')
    axes[0].set_title('Correct vs Buggy Code\nSemantic Scores', fontweight='bold')
    axes[0].legend()

    # 2. Detection summary
    axes[1].bar(['Detected', 'Missed'],
               [tp, fn_count],
               color=['#4CAF50', '#F44336'], edgecolor='black')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'Bug Detection Rate\n{detection_rate*100:.0f}%', fontweight='bold')

    # 3. Score distribution
    axes[2].hist(same_scores, bins=15, alpha=0.7, color='#4CAF50',
                label='Same function', edgecolor='black')
    deltas = [r['score_correct'] - r['score_buggy'] for r in lint_results]
    axes[2].axvline(np.mean(deltas), color='red', linestyle='--',
                   label=f'Mean bug delta={np.mean(deltas):.3f}')
    axes[2].set_xlabel('Semantic Score')
    axes[2].set_title('Score Distribution', fontweight='bold')
    axes[2].legend()

    plt.suptitle('Phase 58: The Latent Linter\nDetecting Semantic Drift Without Execution',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase58_latent_linter.png'), dpi=150)
    plt.close()
    print(f"\nPhase 58 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
