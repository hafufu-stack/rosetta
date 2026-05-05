"""
Phase 13: Semantic Auto-Patching
==================================
Fix bugs using vector arithmetic. No source code editing needed.
V_fixed = V_buggy - V_nl("wrong intent") + V_nl("correct intent")
"""
import os, json, time
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    print("=" * 60)
    print("Phase 13: Semantic Auto-Patching")
    print("=" * 60)
    t0 = time.time()

    ds_file = os.path.join(DATA_DIR, 'rosetta_dataset_v2.json')
    if not os.path.exists(ds_file):
        ds_file = os.path.join(DATA_DIR, 'rosetta_dataset.json')
    with open(ds_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)['dataset']

    lat_file = os.path.join(DATA_DIR, 'rosetta_latents_v2.npz')
    if not os.path.exists(lat_file):
        lat_file = os.path.join(DATA_DIR, 'rosetta_latents.npz')
    latents = np.load(lat_file)
    z_nl, z_ast = latents['nl'], latents['ast']

    with open(os.path.join(DATA_DIR, 'char_vocab.json'), 'r') as f:
        vd = json.load(f)
    idx2char = {int(i): c for c, i in vd['char2idx'].items()}
    V = len(vd['char2idx'])

    from experiments.phase9_generative_decompiler import CodeDecoder, decode_tokens
    decoder = CodeDecoder(64, V, hidden=128, max_len=80).to(DEVICE)
    decoder.load_state_dict(torch.load(os.path.join(DATA_DIR, 'decoder.pt'),
                                        map_location=DEVICE, weights_only=True))
    decoder.eval()

    src_to_idx = {}
    nl_to_idx = {}
    for i, d in enumerate(dataset):
        if d['source'] not in src_to_idx:
            src_to_idx[d['source']] = i
        nl_key = d['nl'].lower().strip()
        if nl_key not in nl_to_idx:
            nl_to_idx[nl_key] = i

    def gen(z_vec):
        with torch.no_grad():
            z_t = torch.tensor(z_vec, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            tokens = decoder(z_t)
            return decode_tokens(tokens[0].cpu().numpy(), idx2char)

    def get_nl_vec(text):
        """Find NL vector by partial match."""
        text_l = text.lower()
        for nl_key, idx in nl_to_idx.items():
            if text_l in nl_key or nl_key in text_l:
                return z_nl[idx]
        return None

    # Bug scenarios: (buggy_src, correct_src, wrong_intent_nl, correct_intent_nl)
    bugs = [
        ("def f(x, y): return x + y", "def f(x, y): return x - y",
         "add", "subtract",
         "Bug: adds instead of subtracting"),
        ("def f(x, y): return x - y", "def f(x, y): return x + y",
         "subtract", "add",
         "Bug: subtracts instead of adding"),
        ("def f(x, y): return x * y", "def f(x, y): return x / y",
         "multiply", "divide",
         "Bug: multiplies instead of dividing"),
        ("def f(x, y): return x > y", "def f(x, y): return x < y",
         "greater than", "less than",
         "Bug: wrong comparison direction"),
        ("def f(x, y): return x == y", "def f(x, y): return x != y",
         "equals", "not equal",
         "Bug: equality instead of inequality"),
        ("def f(s): return s.upper()", "def f(s): return s.lower()",
         "uppercase", "lowercase",
         "Bug: uppercases instead of lowercasing"),
        ("def f(x): return x + 1", "def f(x): return x - 1",
         "increment", "decrement",
         "Bug: increments instead of decrementing"),
        ("def f(x, y): return max(x, y)", "def f(x, y): return min(x, y)",
         "maximum", "minimum",
         "Bug: max instead of min"),
    ]

    patch_results = []
    n_fixed_exact, n_fixed_semantic, total = 0, 0, 0

    print("\n--- Auto-Patching Results ---")
    for buggy_src, correct_src, wrong_nl, correct_nl, desc in bugs:
        if buggy_src not in src_to_idx:
            continue

        idx_buggy = src_to_idx[buggy_src]
        z_buggy = z_ast[idx_buggy]

        # Get NL vectors for wrong and correct intents
        v_wrong = get_nl_vec(wrong_nl)
        v_correct = get_nl_vec(correct_nl)

        if v_wrong is None or v_correct is None:
            print(f"  SKIP: {desc} (NL not found)")
            continue

        total += 1

        # Apply semantic patch: V_fixed = V_buggy - V_wrong + V_correct
        z_patched = z_buggy - v_wrong + v_correct
        z_patched = z_patched / (np.linalg.norm(z_patched) + 1e-8)

        # Also try with scaling
        z_patched_scaled = z_buggy - 0.5 * v_wrong + 0.5 * v_correct
        z_patched_scaled = z_patched_scaled / (np.linalg.norm(z_patched_scaled) + 1e-8)

        # Decode
        buggy_code = gen(z_buggy)
        patched_code = gen(z_patched)
        patched_scaled = gen(z_patched_scaled)

        # Check fix
        exact_fix = patched_code.strip() == correct_src.strip()
        # Semantic: check if key operation matches
        correct_op = correct_src.split('return ')[-1].strip() if 'return ' in correct_src else ''
        patched_op = patched_code.split('return ')[-1].strip() if 'return ' in patched_code else ''
        # Normalize variable names for comparison
        for old, new in [('x','_'), ('y','_'), ('a','_'), ('b','_'),
                         ('p','_'), ('q','_'), ('m','_'), ('n','_')]:
            correct_op = correct_op.replace(old, '_')
            patched_op = patched_op.replace(old, '_')
        semantic_fix = correct_op == patched_op

        if exact_fix: n_fixed_exact += 1
        if semantic_fix: n_fixed_semantic += 1

        status = "EXACT" if exact_fix else ("SEM" if semantic_fix else "X")
        print(f"\n  {desc}")
        print(f"    Buggy:   {buggy_src} -> decoded: {buggy_code[:50]}")
        print(f"    Patched: {patched_code[:50]} [{status}]")
        print(f"    Scaled:  {patched_scaled[:50]}")
        print(f"    Target:  {correct_src}")

        patch_results.append({
            'desc': desc, 'buggy': buggy_src, 'correct': correct_src,
            'buggy_decoded': buggy_code, 'patched': patched_code,
            'patched_scaled': patched_scaled,
            'exact_fix': exact_fix, 'semantic_fix': semantic_fix,
        })

    print(f"\n--- Summary ---")
    print(f"  Total bugs: {total}")
    print(f"  Exact fixes: {n_fixed_exact}/{total} ({n_fixed_exact/max(total,1):.0%})")
    print(f"  Semantic fixes: {n_fixed_semantic}/{total} ({n_fixed_semantic/max(total,1):.0%})")

    elapsed = time.time() - t0
    results = {
        'phase': 13, 'name': 'Semantic Auto-Patching',
        'total_bugs': total,
        'exact_fixes': n_fixed_exact, 'semantic_fixes': n_fixed_semantic,
        'exact_rate': n_fixed_exact/max(total,1),
        'semantic_rate': n_fixed_semantic/max(total,1),
        'details': patch_results,
        'elapsed': elapsed, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase13_auto_patching.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Figure
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    bars = axes[0].bar(['Exact Fix', 'Semantic Fix'],
                       [n_fixed_exact/max(total,1), n_fixed_semantic/max(total,1)],
                       color=['#E91E63','#4CAF50'], edgecolor='black')
    for b, v in zip(bars, [n_fixed_exact/max(total,1), n_fixed_semantic/max(total,1)]):
        axes[0].text(b.get_x()+b.get_width()/2, v+0.03, f'{v:.0%}',
                     ha='center', fontweight='bold', fontsize=14)
    axes[0].set_ylabel('Rate'); axes[0].set_ylim(0, 1.1)
    axes[0].set_title('Bug Fix Success Rate', fontweight='bold')

    # Per-bug breakdown
    bug_names = [r['desc'][:25] for r in patch_results]
    sem_vals = [1 if r['semantic_fix'] else 0 for r in patch_results]
    axes[1].barh(bug_names, sem_vals, color=['#4CAF50' if v else '#F44336' for v in sem_vals],
                edgecolor='black')
    axes[1].set_xlabel('Fixed (1=Yes)')
    axes[1].set_title('Per-Bug Results', fontweight='bold')

    plt.suptitle('Phase 13: Semantic Auto-Patching\n'
                 'V_fixed = V_buggy - V_nl(wrong) + V_nl(correct)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase13_auto_patching.png'), dpi=150)
    plt.close()
    print(f"\nPhase 13 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
