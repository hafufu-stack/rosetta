"""
Phase 6: The Linear Decompiler
================================
Use pseudo-inverse of W_compile to decompile: Bin -> PL -> NL.
Can a single matrix inversion replace a complex rule-based decompiler?
"""
import os, json, time
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def main():
    print("=" * 60)
    print("Phase 6: The Linear Decompiler")
    print("=" * 60)
    t0 = time.time()

    # Load v2 latents (from Phase 5's scaled training)
    lat_file = os.path.join(DATA_DIR, 'rosetta_latents_v2.npz')
    if not os.path.exists(lat_file):
        lat_file = os.path.join(DATA_DIR, 'rosetta_latents.npz')
    latents = np.load(lat_file)
    z_nl, z_ast, z_bc = latents['nl'], latents['ast'], latents['bc']
    labels = latents['labels']
    N, D = z_nl.shape
    print(f"Loaded {N} vectors of dim {D}")

    # Load dataset for source lookup
    ds_file = os.path.join(DATA_DIR, 'rosetta_dataset_v2.json')
    if not os.path.exists(ds_file):
        ds_file = os.path.join(DATA_DIR, 'rosetta_dataset.json')
    with open(ds_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)['dataset']

    # Build source index
    src_to_idx = {}
    for i, d in enumerate(dataset):
        if d['source'] not in src_to_idx:
            src_to_idx[d['source']] = i

    # Train/test split
    n_train = int(N * 0.8)
    perm = np.random.RandomState(42).permutation(N)
    train_idx, test_idx = perm[:n_train], perm[n_train:]

    # === 1. Learn forward maps ===
    print("\n--- Learning forward maps ---")
    # PL -> Bin
    reg_fwd = Ridge(alpha=1.0).fit(z_ast[train_idx], z_bc[train_idx])
    r2_fwd = r2_score(z_bc[test_idx], reg_fwd.predict(z_ast[test_idx]))
    print(f"  PL->Bin R2: {r2_fwd:.4f}")

    # NL -> PL
    reg_nl_pl = Ridge(alpha=1.0).fit(z_nl[train_idx], z_ast[train_idx])

    # === 2. Learn INVERSE maps (The Decompiler!) ===
    print("\n--- Learning inverse maps (DECOMPILER) ---")
    # Bin -> PL (inverse compiler)
    reg_inv = Ridge(alpha=1.0).fit(z_bc[train_idx], z_ast[train_idx])
    z_ast_decompiled = reg_inv.predict(z_bc[test_idx])
    r2_inv = r2_score(z_ast[test_idx], z_ast_decompiled)

    # Bin -> NL (direct meaning recovery)
    reg_bin_nl = Ridge(alpha=1.0).fit(z_bc[train_idx], z_nl[train_idx])
    z_nl_recovered = reg_bin_nl.predict(z_bc[test_idx])
    r2_bin_nl = r2_score(z_nl[test_idx], z_nl_recovered)

    print(f"  Bin->PL (decompile) R2: {r2_inv:.4f}")
    print(f"  Bin->NL (meaning)   R2: {r2_bin_nl:.4f}")

    # === 3. Pseudo-inverse of W_compile ===
    print("\n--- Pseudo-inverse analysis ---")
    W_compile = reg_fwd.coef_  # (D, D)
    W_pinv = np.linalg.pinv(W_compile)  # Moore-Penrose pseudo-inverse
    print(f"  W_compile condition number: {np.linalg.cond(W_compile):.2f}")

    # Apply pseudo-inverse: Bin * pinv(W) -> reconstructed PL
    z_ast_pinv = z_bc[test_idx] @ W_pinv
    cos_pinv = [cosine_sim(z_ast_pinv[i], z_ast[test_idx][i])
                for i in range(len(test_idx))]
    r2_pinv = r2_score(z_ast[test_idx], z_ast_pinv)
    print(f"  Pseudo-inverse R2: {r2_pinv:.4f}")
    print(f"  Pseudo-inverse mean cos: {np.mean(cos_pinv):.4f}")

    # === 4. Retrieval accuracy (decompile then find nearest source) ===
    print("\n--- Retrieval from decompiled vectors ---")
    # For each test sample, decompile Bin->PL, then find nearest AST
    unique_srcs = list(src_to_idx.keys())
    unique_ast = np.array([z_ast[src_to_idx[s]] for s in unique_srcs])

    correct_ridge, correct_pinv, total = 0, 0, 0
    for i, ti in enumerate(test_idx):
        true_label = labels[ti]
        # Ridge decompiler
        sims_r = z_ast_decompiled[i] @ unique_ast.T
        pred_r = np.argmax(sims_r)
        if labels[src_to_idx[unique_srcs[pred_r]]] == true_label:
            correct_ridge += 1
        # Pseudo-inverse decompiler
        sims_p = z_ast_pinv[i] @ unique_ast.T
        pred_p = np.argmax(sims_p)
        if labels[src_to_idx[unique_srcs[pred_p]]] == true_label:
            correct_pinv += 1
        total += 1

    acc_ridge = correct_ridge / max(total, 1)
    acc_pinv = correct_pinv / max(total, 1)
    print(f"  Ridge decompiler retrieval: {acc_ridge:.1%}")
    print(f"  Pseudo-inverse retrieval:   {acc_pinv:.1%}")

    # === 5. Round-trip: PL -> compile -> decompile -> PL ===
    print("\n--- Round-trip: compile then decompile ---")
    z_bc_compiled = reg_fwd.predict(z_ast[test_idx])
    z_ast_roundtrip = reg_inv.predict(z_bc_compiled)
    cos_roundtrip = [cosine_sim(z_ast_roundtrip[i], z_ast[test_idx][i])
                     for i in range(len(test_idx))]
    r2_roundtrip = r2_score(z_ast[test_idx], z_ast_roundtrip)
    print(f"  Round-trip R2: {r2_roundtrip:.4f}")
    print(f"  Round-trip mean cos: {np.mean(cos_roundtrip):.4f}")

    elapsed = time.time() - t0
    results = {
        'phase': 6, 'name': 'The Linear Decompiler',
        'forward_r2': float(r2_fwd),
        'inverse_r2_ridge': float(r2_inv),
        'inverse_r2_pinv': float(r2_pinv),
        'bin_to_nl_r2': float(r2_bin_nl),
        'retrieval_ridge': float(acc_ridge),
        'retrieval_pinv': float(acc_pinv),
        'roundtrip_r2': float(r2_roundtrip),
        'roundtrip_cos': float(np.mean(cos_roundtrip)),
        'condition_number': float(np.linalg.cond(W_compile)),
        'elapsed': elapsed,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase6_linear_decompiler.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Figure
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # R2 scores
    names = ['PL->Bin\n(Compile)', 'Bin->PL\n(Ridge)', 'Bin->PL\n(Pinv)',
             'Bin->NL\n(Meaning)', 'Roundtrip']
    vals = [r2_fwd, r2_inv, r2_pinv, r2_bin_nl, r2_roundtrip]
    colors = ['#E91E63','#2196F3','#9C27B0','#4CAF50','#FF9800']
    bars = axes[0].bar(names, vals, color=colors, edgecolor='black')
    for b, v in zip(bars, vals):
        y = max(0, v) + 0.02
        axes[0].text(b.get_x()+b.get_width()/2, y, f'{v:.3f}',
                     ha='center', fontweight='bold', fontsize=10)
    axes[0].set_ylabel('R2 Score'); axes[0].axhline(0, color='gray', ls='--')
    axes[0].set_title('Linear Transform Quality', fontweight='bold')

    # Retrieval accuracy
    bars2 = axes[1].bar(['Ridge\nDecompiler', 'Pseudo-inverse\nDecompiler'],
                        [acc_ridge, acc_pinv],
                        color=['#2196F3','#9C27B0'], edgecolor='black')
    for b, v in zip(bars2, [acc_ridge, acc_pinv]):
        axes[1].text(b.get_x()+b.get_width()/2, v+0.02, f'{v:.1%}',
                     ha='center', fontweight='bold', fontsize=14)
    axes[1].set_ylabel('Accuracy'); axes[1].set_ylim(0, 1.1)
    axes[1].set_title('Decompilation Retrieval', fontweight='bold')

    # Round-trip cosine distribution
    axes[2].hist(cos_roundtrip, bins=30, color='#FF9800', edgecolor='black')
    axes[2].axvline(np.mean(cos_roundtrip), color='red', ls='--',
                    label=f'Mean={np.mean(cos_roundtrip):.3f}')
    axes[2].set_xlabel('Cosine Similarity')
    axes[2].set_title('Round-trip: Compile -> Decompile', fontweight='bold')
    axes[2].legend()

    plt.suptitle('Phase 6: The Linear Decompiler', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase6_linear_decompiler.png'), dpi=150)
    plt.close()

    print(f"\nPhase 6 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
