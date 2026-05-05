"""
Phase 4: Semantic Binary Arithmetic
====================================
Test: V(add_bytecode) - V(NL_"add") + V(NL_"subtract") ~ V(subtract_bytecode)

Can we manipulate binary code by doing vector arithmetic in meaning space?
"""
import os, json, time
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def main():
    print("=" * 60)
    print("Phase 4: Semantic Binary Arithmetic")
    print("=" * 60)
    t0 = time.time()

    # Load dataset + latents
    with open(os.path.join(DATA_DIR, 'rosetta_dataset.json'), 'r', encoding='utf-8') as f:
        data = json.load(f)
    dataset = data['dataset']

    latents = np.load(os.path.join(DATA_DIR, 'rosetta_latents.npz'))
    z_nl = latents['nl']
    z_ast = latents['ast']
    z_bc = latents['bc']

    # Build source -> index mapping (use first occurrence for each source)
    src_to_idx = {}
    for i, d in enumerate(dataset):
        src = d['source']
        if src not in src_to_idx:
            src_to_idx[src] = i

    # Define analogy triplets: (A, B, C) -> A - B + C should ~ D
    # "add" is to "subtract" as X_bytecode is to Y_bytecode
    analogies = [
        # (source_A, source_B, source_C, expected_source_D, description)
        ("def f(x, y): return x + y", "def f(x, y): return x - y",
         "def f(x, y): return x * y", "def f(x, y): return x / y",
         "add:sub :: mul:?div"),

        ("def f(x, y): return x + y", "def f(x, y): return x * y",
         "def f(x, y): return x - y", "def f(x, y): return x / y",
         "add:mul :: sub:?div"),

        ("def f(x, y): return max(x, y)", "def f(x, y): return min(x, y)",
         "def f(x, y): return x > y", "def f(x, y): return x == y",
         "max:min :: gt:?eq"),

        ("def f(x): return -x", "def f(x): return abs(x)",
         "def f(x): return x * x", "def f(x): return x * x * x",
         "neg:abs :: square:?cube"),

        ("def f(s): return s.upper()", "def f(s): return s.lower()",
         "def f(s): return len(s)", "def f(s): return s[::-1]",
         "upper:lower :: len:?reverse"),
    ]

    # Run analogies on all three modalities
    analogy_results = []

    for src_a, src_b, src_c, src_d, desc in analogies:
        if not all(s in src_to_idx for s in [src_a, src_b, src_c, src_d]):
            print(f"  SKIP {desc}: missing source")
            continue

        ia, ib, ic, id_ = (src_to_idx[s] for s in [src_a, src_b, src_c, src_d])

        result = {'analogy': desc}

        for name, z in [('NL', z_nl), ('AST', z_ast), ('Bytecode', z_bc)]:
            # Analogy: A - B + C should be close to D
            v_pred = z[ia] - z[ib] + z[ic]
            v_pred = v_pred / (np.linalg.norm(v_pred) + 1e-8)  # Normalize

            # Find nearest neighbor among all unique sources
            best_sim = -1
            best_src = None
            all_sims = {}
            for src, idx in src_to_idx.items():
                sim = cosine_sim(v_pred, z[idx])
                all_sims[src[:30]] = sim
                if sim > best_sim:
                    best_sim = sim
                    best_src = src

            target_sim = cosine_sim(v_pred, z[id_])
            is_correct = best_src == src_d

            result[f'{name}_target_sim'] = float(target_sim)
            result[f'{name}_best_sim'] = float(best_sim)
            result[f'{name}_correct'] = is_correct
            result[f'{name}_best'] = best_src[:40] if best_src else 'N/A'

            # Rank of target
            sims_list = sorted(all_sims.values(), reverse=True)
            rank = sims_list.index(target_sim) + 1 if target_sim in sims_list else -1
            result[f'{name}_rank'] = rank

        analogy_results.append(result)
        print(f"\n  {desc}")
        for name in ['NL', 'AST', 'Bytecode']:
            status = 'OK' if result[f'{name}_correct'] else 'X'
            print(f"    {name}: sim={result[f'{name}_target_sim']:.3f} "
                  f"rank={result[f'{name}_rank']} [{status}]")

    # Summary statistics
    total = len(analogy_results)
    for name in ['NL', 'AST', 'Bytecode']:
        correct = sum(1 for r in analogy_results if r.get(f'{name}_correct', False))
        avg_sim = np.mean([r[f'{name}_target_sim'] for r in analogy_results])
        avg_rank = np.mean([r[f'{name}_rank'] for r in analogy_results
                           if r[f'{name}_rank'] > 0])
        print(f"\n{name} Analogy Summary: {correct}/{total} correct, "
              f"avg sim={avg_sim:.3f}, avg rank={avg_rank:.1f}")

    # === Bonus: Direct semantic patching ===
    print("\n--- Bonus: Semantic Patching ---")
    print("Can we 'rewrite' bytecode using NL vector arithmetic?")

    patch_results = []
    for src_a, src_b, src_c, src_d, desc in analogies:
        if not all(s in src_to_idx for s in [src_a, src_b, src_c, src_d]):
            continue
        ia, ib, ic, id_ = (src_to_idx[s] for s in [src_a, src_b, src_c, src_d])

        # Cross-modal: NL_A - NL_B + BC_C -> should approximate BC_D
        v_patched = z_nl[ia] - z_nl[ib] + z_bc[ic]
        v_patched = v_patched / (np.linalg.norm(v_patched) + 1e-8)

        sim_to_target = cosine_sim(v_patched, z_bc[id_])
        patch_results.append({
            'desc': desc,
            'sim': float(sim_to_target),
        })
        print(f"  {desc}: NL_diff + BC = sim {sim_to_target:.3f} to target BC")

    elapsed = time.time() - t0
    results = {
        'phase': 4,
        'name': 'Semantic Binary Arithmetic',
        'analogies': analogy_results,
        'patches': patch_results,
        'summary': {},
        'elapsed': elapsed,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    for name in ['NL', 'AST', 'Bytecode']:
        correct = sum(1 for r in analogy_results if r.get(f'{name}_correct', False))
        results['summary'][f'{name}_accuracy'] = correct / max(total, 1)
        results['summary'][f'{name}_avg_sim'] = float(
            np.mean([r[f'{name}_target_sim'] for r in analogy_results]))

    with open(os.path.join(RESULTS_DIR, 'phase4_semantic_arithmetic.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Visualization
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Analogy accuracy by modality
    modalities = ['NL', 'AST', 'Bytecode']
    accs = [results['summary'][f'{m}_accuracy'] for m in modalities]
    sims = [results['summary'][f'{m}_avg_sim'] for m in modalities]

    x = np.arange(len(modalities))
    bars1 = axes[0].bar(x - 0.15, accs, 0.3, label='Accuracy', color='#2196F3',
                        edgecolor='black')
    bars2 = axes[0].bar(x + 0.15, sims, 0.3, label='Avg Similarity', color='#FF9800',
                        edgecolor='black')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(modalities, fontsize=12)
    axes[0].set_ylabel('Score', fontsize=12)
    axes[0].set_title('Analogy Test Results', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].set_ylim(0, 1.1)
    for b, v in zip(bars1, accs):
        axes[0].text(b.get_x()+b.get_width()/2, v+0.02, f'{v:.0%}',
                     ha='center', fontweight='bold', fontsize=11)

    # Cross-modal patching results
    if patch_results:
        descs = [p['desc'].split('::')[0] for p in patch_results]
        sims_p = [p['sim'] for p in patch_results]
        bars3 = axes[1].barh(descs, sims_p, color='#9C27B0', edgecolor='black')
        for b, v in zip(bars3, sims_p):
            axes[1].text(v + 0.01, b.get_y() + b.get_height()/2,
                         f'{v:.3f}', va='center', fontweight='bold', fontsize=11)
        axes[1].set_xlabel('Cosine Similarity to Target', fontsize=12)
        axes[1].set_title('Semantic Patching\n(NL diff + Bytecode)', fontsize=13,
                          fontweight='bold')
        axes[1].set_xlim(0, 1)

    plt.suptitle('Phase 4: Semantic Binary Arithmetic',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase4_semantic_arithmetic.png'), dpi=150)
    plt.close()

    print(f"\nPhase 4 complete in {elapsed:.1f}s")
    return results


if __name__ == '__main__':
    main()
