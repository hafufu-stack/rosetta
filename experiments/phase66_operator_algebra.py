"""
Phase 66: The Operator Algebra
================================
Deep Think's crucial insight: vector addition is COMMUTATIVE
but function composition is NOT. f(g(x)) != g(f(x)).

Solution: Learn a NON-COMMUTATIVE composition operator.
z_comp = Compose(z_f, z_g) where Compose(a,b) != Compose(b,a).

If this works, programming IS operator algebra.
"""
import os, json, time, sys, inspect
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BilinearComposer(nn.Module):
    """Non-commutative composition operator.
    z_comp = W(z_f, z_g) where W(a,b) != W(b,a)."""
    def __init__(self, dim=5):
        super().__init__()
        # Bilinear layer: inherently non-commutative
        self.bilinear = nn.Bilinear(dim, dim, dim)
        # Extra MLP for expressiveness
        self.mlp = nn.Sequential(
            nn.Linear(dim * 3, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, dim),
        )

    def forward(self, z_f, z_g):
        bi = self.bilinear(z_f, z_g)
        # Concatenate [z_f, z_g, bilinear_output]
        cat = torch.cat([z_f, z_g, bi], dim=-1)
        return self.mlp(cat)


def main():
    print("=" * 60)
    print("Phase 66: The Operator Algebra")
    print("Non-commutative composition: f o g != g o f")
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

    # Find composable unary functions
    unary_funcs = {}
    for src, info in src_to_z.items():
        try:
            ns = {}
            exec(compile(src, '<string>', 'exec'), ns)
            fn = [v for k, v in ns.items()
                  if callable(v) and not k.startswith('_')][0]
            sig = inspect.signature(fn)
            if len(sig.parameters) == 1:
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
                    unary_funcs[src] = {'fn': fn, 'z_5d': info['z_5d']}
        except Exception:
            pass

    print(f"  Composable unary functions: {len(unary_funcs)}")

    # Build composition dataset: (z_f, z_g) -> z_{f o g}
    # where z_{f o g} is the nearest neighbor to the composed behavior
    print("\n--- Building Composition Dataset ---")
    all_z5 = np.array([v['z_5d'] for v in src_to_z.values()])
    all_srcs = list(src_to_z.keys())
    func_items = list(unary_funcs.items())

    train_data = []
    test_vals = [1, 2, -1, 3, 5]

    for i, (src_f, info_f) in enumerate(func_items):
        for j, (src_g, info_g) in enumerate(func_items):
            if i == j:
                continue
            fn_f, fn_g = info_f['fn'], info_g['fn']
            try:
                # f o g
                results_fg = []
                for v in test_vals:
                    r = fn_f(fn_g(v))
                    if not isinstance(r, (int, float)):
                        break
                    r = float(r)
                    if abs(r) > 1e6 or np.isnan(r):
                        break
                    results_fg.append(r)
                if len(results_fg) != len(test_vals):
                    continue

                # Find nearest function to f o g by behavior
                best_src, best_match = None, 0
                for src_k, info_k in src_to_z.items():
                    try:
                        ns2 = {}
                        exec(compile(src_k, '<string>', 'exec'), ns2)
                        fn_k = [v for k, v in ns2.items()
                                if callable(v) and not k.startswith('_')][0]
                        sig_k = inspect.signature(fn_k)
                        if len(sig_k.parameters) != 1:
                            continue
                        n_match = 0
                        for idx, v in enumerate(test_vals):
                            try:
                                rk = float(fn_k(v))
                                if abs(rk - results_fg[idx]) < 0.01:
                                    n_match += 1
                            except Exception:
                                pass
                        if n_match > best_match:
                            best_match = n_match
                            best_src = src_k
                    except Exception:
                        pass

                if best_src and best_match >= 3:
                    z_target = src_to_z[best_src]['z_5d']
                    train_data.append({
                        'z_f': info_f['z_5d'], 'z_g': info_g['z_5d'],
                        'z_target': z_target,
                        'f': src_f, 'g': src_g, 'nn': best_src,
                        'match': best_match,
                    })
            except Exception:
                pass

    print(f"  Composition training pairs: {len(train_data)}")

    if len(train_data) < 10:
        print("  Not enough data for training. Exiting.")
        results = {'phase': 66, 'name': 'The Operator Algebra',
                   'n_pairs': len(train_data), 'error': 'insufficient_data',
                   'elapsed': time.time() - t0}
        with open(os.path.join(RESULTS_DIR, 'phase66_operator_algebra.json'), 'w',
                  encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        return results

    # Prepare tensors
    Z_f = torch.tensor(np.array([d['z_f'] for d in train_data], dtype=np.float32)).to(DEVICE)
    Z_g = torch.tensor(np.array([d['z_g'] for d in train_data], dtype=np.float32)).to(DEVICE)
    Z_target = torch.tensor(np.array([d['z_target'] for d in train_data], dtype=np.float32)).to(DEVICE)

    n = len(train_data)
    split = int(n * 0.8)
    perm = torch.randperm(n)
    train_idx, test_idx = perm[:split], perm[split:]

    # Train composition operator
    composer = BilinearComposer(dim=5).to(DEVICE)
    opt = torch.optim.Adam(composer.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 1000)

    print("\n--- Training Non-Commutative Composer ---")
    for epoch in range(1000):
        composer.train()
        pred = composer(Z_f[train_idx], Z_g[train_idx])
        loss = F.mse_loss(pred, Z_target[train_idx])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()

        if (epoch+1) % 200 == 0:
            composer.eval()
            with torch.no_grad():
                pred_test = composer(Z_f[test_idx], Z_g[test_idx])
                test_loss = F.mse_loss(pred_test, Z_target[test_idx])
                r2 = 1 - test_loss.item() / (Z_target[test_idx].var().item() + 1e-8)
            print(f"  Epoch {epoch+1}/1000: train={loss.item():.6f}, "
                  f"test={test_loss.item():.6f}, R2={r2:.4f}")

    composer.eval()

    # Compare: addition vs operator
    print("\n--- Addition vs Operator Comparison ---")
    with torch.no_grad():
        pred_operator = composer(Z_f[test_idx], Z_g[test_idx]).cpu().numpy()

    z_sum = (Z_f[test_idx] + Z_g[test_idx]).cpu().numpy()
    z_true = Z_target[test_idx].cpu().numpy()

    # For each test sample, find NN and check match
    n_correct_add = 0
    n_correct_op = 0

    for k in range(len(test_idx)):
        idx = test_idx[k].item()
        d = train_data[idx]
        true_z = z_true[k]

        # Addition method
        dists_add = np.linalg.norm(all_z5 - z_sum[k], axis=1)
        nn_add_src = all_srcs[np.argmin(dists_add)]

        # Operator method
        dists_op = np.linalg.norm(all_z5 - pred_operator[k], axis=1)
        nn_op_src = all_srcs[np.argmin(dists_op)]

        # Check if the found function matches behavior
        if nn_add_src == d['nn']:
            n_correct_add += 1
        if nn_op_src == d['nn']:
            n_correct_op += 1

    n_test = len(test_idx)
    acc_add = n_correct_add / max(n_test, 1) * 100
    acc_op = n_correct_op / max(n_test, 1) * 100

    print(f"  Test samples: {n_test}")
    print(f"  Addition (f+g):     {n_correct_add}/{n_test} ({acc_add:.1f}%)")
    print(f"  Operator (W(f,g)):  {n_correct_op}/{n_test} ({acc_op:.1f}%)")
    improvement = acc_op - acc_add
    print(f"  Improvement:        {improvement:+.1f}%")

    # Non-commutativity test: does Compose(f,g) != Compose(g,f)?
    print("\n--- Non-Commutativity Test ---")
    n_noncommute = 0
    n_tested = 0
    noncommute_examples = []

    with torch.no_grad():
        for k in range(min(100, len(train_data))):
            d = train_data[k]
            zf = torch.tensor(d['z_f'].astype(np.float32)).unsqueeze(0).to(DEVICE)
            zg = torch.tensor(d['z_g'].astype(np.float32)).unsqueeze(0).to(DEVICE)

            fg = composer(zf, zg).cpu().numpy()[0]
            gf = composer(zg, zf).cpu().numpy()[0]

            dist_fg_gf = np.linalg.norm(fg - gf)
            n_tested += 1
            if dist_fg_gf > 0.1:
                n_noncommute += 1
                if len(noncommute_examples) < 5:
                    noncommute_examples.append({
                        'f': d['f'], 'g': d['g'],
                        'dist': float(dist_fg_gf),
                    })

    nc_rate = n_noncommute / max(n_tested, 1) * 100
    print(f"  Non-commutative pairs: {n_noncommute}/{n_tested} ({nc_rate:.1f}%)")
    for ex in noncommute_examples:
        f_short = ex['f'].split('return ')[1][:15] if 'return' in ex['f'] else '?'
        g_short = ex['g'].split('return ')[1][:15] if 'return' in ex['g'] else '?'
        print(f"    {f_short} o {g_short}: dist(fg,gf)={ex['dist']:.4f}")

    print(f"\n  =======================================")
    print(f"  THE OPERATOR ALGEBRA RESULTS")
    print(f"  Addition accuracy:  {acc_add:.1f}%")
    print(f"  Operator accuracy:  {acc_op:.1f}%")
    print(f"  Non-commutativity:  {nc_rate:.1f}%")
    print(f"  Verdict: {'OPERATOR ALGEBRA PROVEN' if acc_op > acc_add else 'MORE DATA NEEDED'}")
    print(f"  =======================================")

    elapsed = time.time() - t0
    results = {
        'phase': 66, 'name': 'The Operator Algebra',
        'n_pairs': len(train_data),
        'n_test': n_test,
        'acc_addition': float(acc_add),
        'acc_operator': float(acc_op),
        'improvement': float(improvement),
        'noncommutative_rate': float(nc_rate),
        'noncommute_examples': noncommute_examples,
        'elapsed': elapsed, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase66_operator_algebra.json'), 'w',
              encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    # Figure
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Addition vs Operator accuracy
    axes[0].bar(['Addition\n(f + g)', 'Bilinear\nOperator'],
               [acc_add, acc_op],
               color=['#FF9800', '#4CAF50'], edgecolor='black')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Composition Method\nComparison', fontweight='bold')
    for i, v in enumerate([acc_add, acc_op]):
        axes[0].text(i, v+2, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=14)

    # 2. Non-commutativity distribution
    nc_dists = []
    with torch.no_grad():
        for k in range(min(200, len(train_data))):
            d = train_data[k]
            zf = torch.tensor(d['z_f'].astype(np.float32)).unsqueeze(0).to(DEVICE)
            zg = torch.tensor(d['z_g'].astype(np.float32)).unsqueeze(0).to(DEVICE)
            fg = composer(zf, zg).cpu().numpy()[0]
            gf = composer(zg, zf).cpu().numpy()[0]
            nc_dists.append(float(np.linalg.norm(fg - gf)))

    axes[1].hist(nc_dists, bins=30, color='#9C27B0', edgecolor='black', alpha=0.8)
    axes[1].axvline(0.1, color='red', linestyle='--', label='Threshold')
    axes[1].set_xlabel('||Compose(f,g) - Compose(g,f)||')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'Non-Commutativity\n({nc_rate:.0f}% pairs are non-commutative)',
                     fontweight='bold')
    axes[1].legend()

    # 3. The algebra verdict
    verdict = ("THE OPERATOR ALGEBRA\n\n"
              f"Addition:  {acc_add:.1f}% (commutative)\n"
              f"Operator:  {acc_op:.1f}% (non-commutative)\n"
              f"NC rate:   {nc_rate:.0f}%\n\n")
    if acc_op > acc_add:
        verdict += "Programming IS\nOperator Algebra!"
        bg = '#E8F5E9'
    else:
        verdict += "Need more data\nfor convergence"
        bg = '#FFF3E0'
    axes[2].text(0.5, 0.5, verdict, ha='center', va='center',
                fontsize=13, fontweight='bold', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor=bg, alpha=0.8),
                transform=axes[2].transAxes)
    axes[2].axis('off')

    plt.suptitle('Phase 66: The Operator Algebra\n'
                 'Non-Commutative Composition in 5D Space',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase66_operator_algebra.png'), dpi=150)
    plt.close()
    print(f"\nPhase 66 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
