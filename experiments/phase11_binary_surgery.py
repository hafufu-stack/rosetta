"""
Phase 11: Mechanistic Binary Surgery
=======================================
Hack the compiler by intervening on SVD axes during compilation.
Flip the "arithmetic" axis -> does add become subtract?
"""
import os, json, time
import numpy as np
import torch
from sklearn.linear_model import Ridge

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    print("=" * 60)
    print("Phase 11: Mechanistic Binary Surgery")
    print("=" * 60)
    t0 = time.time()

    # Load data
    ds_file = os.path.join(DATA_DIR, 'rosetta_dataset_v2.json')
    if not os.path.exists(ds_file):
        ds_file = os.path.join(DATA_DIR, 'rosetta_dataset.json')
    with open(ds_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)['dataset']

    lat_file = os.path.join(DATA_DIR, 'rosetta_latents_v2.npz')
    if not os.path.exists(lat_file):
        lat_file = os.path.join(DATA_DIR, 'rosetta_latents.npz')
    latents = np.load(lat_file)
    z_ast, z_bc = latents['ast'], latents['bc']
    N, D = z_ast.shape

    src_to_idx = {}
    for i, d in enumerate(dataset):
        if d['source'] not in src_to_idx:
            src_to_idx[d['source']] = i

    # Load decoder
    with open(os.path.join(DATA_DIR, 'char_vocab.json'), 'r') as f:
        vocab_data = json.load(f)
    char2idx = vocab_data['char2idx']
    idx2char = {int(i): c for c, i in char2idx.items()}
    V = len(char2idx)

    from experiments.phase9_generative_decompiler import CodeDecoder, decode_tokens
    decoder = CodeDecoder(64, V, hidden=128, max_len=80).to(DEVICE)
    decoder.load_state_dict(torch.load(os.path.join(DATA_DIR, 'decoder.pt'),
                                        map_location=DEVICE, weights_only=True))
    decoder.eval()

    # Learn W_compile and its inverse
    n_train = int(N * 0.8)
    perm = np.random.RandomState(42).permutation(N)
    train_i = perm[:n_train]
    reg_fwd = Ridge(alpha=1.0).fit(z_ast[train_i], z_bc[train_i])
    reg_inv = Ridge(alpha=1.0).fit(z_bc[train_i], z_ast[train_i])
    W = reg_fwd.coef_
    U, S, Vt = np.linalg.svd(W)

    print(f"W_compile top 4 singular values: {S[:4].round(3)}")

    def gen_code(z_vec):
        with torch.no_grad():
            z_t = torch.tensor(z_vec, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            tokens = decoder(z_t)
            return decode_tokens(tokens[0].cpu().numpy(), idx2char)

    def intervene_compile(z_ast_vec, axis, scale):
        """Compile AST->Bin with intervention on SVD axis."""
        # Normal compilation: z_bc = z_ast @ W.T + bias
        z_bc_normal = reg_fwd.predict(z_ast_vec.reshape(1, -1))[0]

        # Project onto SVD space, intervene, project back
        # W = U @ diag(S) @ Vt
        # z_bc = z_ast @ (U @ diag(S) @ Vt).T = z_ast @ Vt.T @ diag(S) @ U.T
        proj = z_ast_vec @ Vt.T  # Project AST into SVD space (D,)
        proj_modified = proj.copy()
        proj_modified[axis] *= scale  # Intervene on axis
        z_bc_modified = proj_modified @ np.diag(S) @ U.T + reg_fwd.intercept_
        return z_bc_normal, z_bc_modified

    # === Surgery experiments ===
    test_sources = [
        "def f(x, y): return x + y",
        "def f(x, y): return x * y",
        "def f(x, y): return x > y",
        "def f(x): return abs(x)",
        "def f(s): return s.upper()",
        "def f(x, y): return x - y",
    ]

    interventions = [
        (0, -1.0, "Flip axis 0 (arithmetic)"),
        (0, 0.0, "Zero axis 0"),
        (0, 2.0, "Amplify axis 0 x2"),
        (1, -1.0, "Flip axis 1"),
        (2, -1.0, "Flip axis 2 (comparison)"),
        (2, 0.0, "Zero axis 2"),
    ]

    surgery_results = []
    print("\n--- Surgery Results ---")

    for src in test_sources:
        if src not in src_to_idx:
            continue
        idx = src_to_idx[src]
        z_in = z_ast[idx]

        # Original (no intervention)
        z_bc_orig = reg_fwd.predict(z_in.reshape(1, -1))[0]
        z_ast_orig = reg_inv.predict(z_bc_orig.reshape(1, -1))[0]
        orig_code = gen_code(z_ast_orig)

        print(f"\n  Input: {src}")
        print(f"  Original decode: {orig_code[:60]}")

        for axis, scale, desc in interventions:
            _, z_bc_mod = intervene_compile(z_in, axis, scale)
            z_ast_mod = reg_inv.predict(z_bc_mod.reshape(1, -1))[0]
            mod_code = gen_code(z_ast_mod)

            # Check if semantics changed
            changed = mod_code.strip() != orig_code.strip()
            cos_to_orig = float(np.dot(z_bc_mod, z_bc_orig) /
                               (np.linalg.norm(z_bc_mod) * np.linalg.norm(z_bc_orig) + 1e-8))

            surgery_results.append({
                'input': src, 'intervention': desc,
                'axis': axis, 'scale': scale,
                'original': orig_code, 'modified': mod_code,
                'changed': changed, 'cos_to_orig': cos_to_orig,
            })
            if changed:
                print(f"    [{desc}] -> {mod_code[:60]} (cos={cos_to_orig:.3f})")

    # Summary
    n_changed = sum(1 for r in surgery_results if r['changed'])
    n_total = len(surgery_results)
    print(f"\n  Total interventions: {n_total}")
    print(f"  Semantics changed: {n_changed} ({n_changed/max(n_total,1):.0%})")

    # Categorize changes
    meaningful = 0
    for r in surgery_results:
        if r['changed'] and 'def f' in r['modified']:
            meaningful += 1
    print(f"  Meaningful changes (valid code): {meaningful}")

    elapsed = time.time() - t0
    results = {
        'phase': 11, 'name': 'Mechanistic Binary Surgery',
        'total_interventions': n_total,
        'changed': n_changed, 'change_rate': n_changed/max(n_total,1),
        'meaningful': meaningful,
        'details': surgery_results[:30],
        'elapsed': elapsed, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase11_binary_surgery.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Figure
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Change rate by axis
    for ax_id in range(4):
        axis_results = [r for r in surgery_results if r['axis'] == ax_id]
        if axis_results:
            rate = sum(1 for r in axis_results if r['changed']) / len(axis_results)
            axes[0].bar(f'Axis {ax_id}\n(s={S[ax_id]:.1f})', rate,
                       color=['#E91E63','#2196F3','#4CAF50','#FF9800'][ax_id],
                       edgecolor='black')
            axes[0].text(ax_id, rate+0.02, f'{rate:.0%}', ha='center',
                        fontweight='bold', fontsize=12)
    axes[0].set_ylabel('Change Rate')
    axes[0].set_title('Semantics Change by SVD Axis', fontweight='bold')
    axes[0].set_ylim(0, 1.1)

    # Cosine similarity to original after intervention
    cos_vals = [r['cos_to_orig'] for r in surgery_results]
    changed_mask = [r['changed'] for r in surgery_results]
    axes[1].hist([c for c, m in zip(cos_vals, changed_mask) if m],
                bins=20, alpha=0.7, color='#F44336', label='Changed')
    axes[1].hist([c for c, m in zip(cos_vals, changed_mask) if not m],
                bins=20, alpha=0.7, color='#4CAF50', label='Unchanged')
    axes[1].set_xlabel('Cosine to Original')
    axes[1].set_title('Intervention Impact', fontweight='bold')
    axes[1].legend()

    plt.suptitle('Phase 11: Mechanistic Binary Surgery',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase11_binary_surgery.png'), dpi=150)
    plt.close()
    print(f"\nPhase 11 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
