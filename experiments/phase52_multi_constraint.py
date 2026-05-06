"""
Phase 52: Multi-Constraint Inverse Synthesis
===============================================
P50 proved: 5D optimization reaches loss=10^-15 but hits wrong function.
Root cause: single I/O pair has infinite solutions (non-injective).
Solution: Use 5+ diverse test cases + L2 regularization (Occam's razor).
This is "Differentiable TDD" — synthesize code from test cases.
"""
import os, json, time, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    print("=" * 60)
    print("Phase 52: Multi-Constraint Inverse Synthesis")
    print("Differentiable TDD: code from test cases")
    print("=" * 60)
    t0 = time.time()
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    # Load data
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
    N = len(z_ast)

    # PCA for 5D manifold
    from sklearn.decomposition import PCA
    pca = PCA(n_components=10)
    z_pca = pca.fit_transform(z_ast)
    print(f"  PCA: 5D = {sum(pca.explained_variance_ratio_[:5])*100:.1f}% variance")

    # Build execution dataset
    import inspect
    exec_data = []
    for i, d in enumerate(dataset):
        src = d['source']
        try:
            ns = {}
            exec(compile(src, '<string>', 'exec'), ns)
            fn = [v for k, v in ns.items() if callable(v) and not k.startswith('_')][0]
            sig = inspect.signature(fn)
            n_p = len(sig.parameters)

            tests = []
            if n_p == 1:
                tests = [(-5,), (-1,), (0,), (1,), (3,), (7,), (10,)]
            elif n_p == 2:
                tests = [(-3,-1), (-1,2), (0,0), (1,1), (2,3), (3,5), (5,-2), (7,4)]

            for args in tests:
                try:
                    r = fn(*args)
                    if isinstance(r, (int, float)) and not np.isnan(float(r)):
                        rv = float(r)
                        if abs(rv) < 1e6:
                            pa = list(args) + [0]*(3-len(args))
                            exec_data.append({
                                'func_idx': i, 'z_5d': z_pca[i, :5].tolist(),
                                'args': pa[:3], 'result': rv,
                            })
                except Exception:
                    pass
        except Exception:
            pass
    print(f"  Execution samples: {len(exec_data)}")

    # Train 5D Neural CPU
    X = np.array([[*d['z_5d'], *d['args']] for d in exec_data], dtype=np.float32)
    Y = np.array([d['result'] for d in exec_data], dtype=np.float32)
    X_mean, X_std = X.mean(0), X.std(0) + 1e-8
    Y_mean, Y_std = Y.mean(), Y.std() + 1e-8
    X_t = torch.tensor((X - X_mean) / X_std, dtype=torch.float32).to(DEVICE)
    Y_t = torch.tensor((Y - Y_mean) / Y_std, dtype=torch.float32).to(DEVICE)

    cpu5d = nn.Sequential(
        nn.Linear(8, 256), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, 1)
    ).to(DEVICE)

    opt = torch.optim.Adam(cpu5d.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 600)
    for epoch in range(600):
        pred = cpu5d(X_t).squeeze()
        loss = F.mse_loss(pred, Y_t)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if (epoch+1) % 200 == 0:
            r2 = 1 - loss.item() / (Y_t.var().item() + 1e-8)
            print(f"  Epoch {epoch+1}/600: R2={r2:.4f}")
    cpu5d.eval()

    # Load decoder
    sys.path.insert(0, BASE_DIR)
    with open(os.path.join(DATA_DIR, 'char_vocab.json'), 'r') as f:
        vd = json.load(f)
    idx2char = {int(i): c for c, i in vd['char2idx'].items()}
    V = len(vd['char2idx'])
    from experiments.phase9_generative_decompiler import CodeDecoder, decode_tokens
    decoder = CodeDecoder(64, V, hidden=128, max_len=80).to(DEVICE)
    dec_path = os.path.join(DATA_DIR, 'decoder_robust.pt')
    if not os.path.exists(dec_path):
        dec_path = os.path.join(DATA_DIR, 'decoder.pt')
    decoder.load_state_dict(torch.load(dec_path, map_location=DEVICE, weights_only=True))
    decoder.eval()

    def decode_5d(z5):
        full = np.zeros(10); full[:5] = z5
        z64 = pca.inverse_transform(full.reshape(1, -1))[0]
        with torch.no_grad():
            zt = torch.tensor(z64.astype(np.float32)).unsqueeze(0).to(DEVICE)
            tok = decoder(zt)
            return decode_tokens(tok[0].cpu().numpy(), idx2char)

    # Build ground truth 5D coords for known functions
    src_to_5d = {}
    for i, s in enumerate(sources):
        if s not in src_to_5d:
            src_to_5d[s] = z_pca[i, :5]

    # Multi-constraint targets with DIVERSE test cases
    targets = [
        {'name': 'add', 'true_src': 'def f(x, y): return x + y',
         'io': [(-3, 2, 0, -1), (0, 0, 0, 0), (1, 1, 0, 2),
                (5, -2, 0, 3), (7, 3, 0, 10), (10, 10, 0, 20),
                (-5, -5, 0, -10)]},
        {'name': 'sub', 'true_src': 'def f(x, y): return x - y',
         'io': [(5, 3, 0, 2), (0, 0, 0, 0), (1, 5, 0, -4),
                (10, 3, 0, 7), (-2, -3, 0, 1), (7, 7, 0, 0),
                (3, -1, 0, 4)]},
        {'name': 'mul', 'true_src': 'def f(x, y): return x * y',
         'io': [(2, 3, 0, 6), (0, 5, 0, 0), (-1, 3, 0, -3),
                (4, 4, 0, 16), (1, 7, 0, 7), (-2, -3, 0, 6),
                (5, 1, 0, 5)]},
        {'name': 'neg', 'true_src': 'def f(x): return -x',
         'io': [(5, 0, 0, -5), (-3, 0, 0, 3), (0, 0, 0, 0),
                (1, 0, 0, -1), (10, 0, 0, -10), (-7, 0, 0, 7),
                (100, 0, 0, -100)]},
        {'name': 'abs', 'true_src': 'def f(x): return abs(x)',
         'io': [(-5, 0, 0, 5), (3, 0, 0, 3), (0, 0, 0, 0),
                (-1, 0, 0, 1), (7, 0, 0, 7), (-10, 0, 0, 10),
                (-100, 0, 0, 100)]},
        {'name': 'max', 'true_src': 'def f(x, y): return max(x, y)',
         'io': [(3, 5, 0, 5), (5, 3, 0, 5), (0, 0, 0, 0),
                (-1, 1, 0, 1), (7, 7, 0, 7), (-3, -5, 0, -3),
                (10, 1, 0, 10)]},
    ]

    print("\n--- Multi-Constraint Inverse Synthesis ---")
    print(f"  Each target: 7 diverse I/O constraints + L2 regularization")

    results_list = []
    for target in targets:
        # Compute ground truth 5D coordinates
        true_5d = src_to_5d.get(target['true_src'])
        gt_dist = "N/A"

        # Multi-restart optimization
        best_overall_loss = float('inf')
        best_overall_z = None
        best_overall_code = None

        for restart in range(10):  # 10 random restarts
            z5 = torch.randn(5, device=DEVICE, dtype=torch.float32) * 0.5
            z5.requires_grad_(True)
            opt_inv = torch.optim.Adam([z5], lr=0.05)
            sched_inv = torch.optim.lr_scheduler.CosineAnnealingLR(opt_inv, 500)

            best_loss = float('inf')
            best_z = None

            for step in range(500):
                total_loss = torch.tensor(0.0, device=DEVICE)
                for a1, a2, a3, expected in target['io']:
                    args_n = torch.tensor(
                        [(a1-X_mean[5])/X_std[5], (a2-X_mean[6])/X_std[6],
                         (a3-X_mean[7])/X_std[7]],
                        dtype=torch.float32, device=DEVICE)
                    z5_n = (z5 - torch.tensor(X_mean[:5], device=DEVICE)) / \
                           torch.tensor(X_std[:5], device=DEVICE)
                    inp = torch.cat([z5_n, args_n]).unsqueeze(0)
                    pred = cpu5d(inp).squeeze()
                    exp_n = (expected - Y_mean) / Y_std
                    total_loss = total_loss + (pred - exp_n) ** 2

                # L2 regularization (Occam's razor: prefer simple = near origin)
                l2_reg = 0.001 * torch.sum(z5 ** 2)
                total_loss = total_loss + l2_reg

                opt_inv.zero_grad()
                total_loss.backward()
                opt_inv.step()
                sched_inv.step()

                if total_loss.item() < best_loss:
                    best_loss = total_loss.item()
                    best_z = z5.detach().cpu().numpy().copy()

            if best_loss < best_overall_loss:
                best_overall_loss = best_loss
                best_overall_z = best_z.copy()
                best_overall_code = decode_5d(best_z)

        # Verify
        code = best_overall_code
        verified = False
        n_tests_passed = 0
        try:
            ns = {}
            exec(compile(code, '<string>', 'exec'), ns)
            fn = [v for k, v in ns.items() if callable(v) and not k.startswith('_')][0]
            sig = inspect.signature(fn)
            n_params = len(sig.parameters)

            for a1, a2, a3, expected in target['io']:
                try:
                    if n_params == 1:
                        r = float(fn(a1))
                    else:
                        r = float(fn(a1, a2))
                    if abs(r - expected) < 0.01:
                        n_tests_passed += 1
                except Exception:
                    pass
            verified = (n_tests_passed == len(target['io']))
        except Exception:
            pass

        # Distance to ground truth
        if true_5d is not None:
            gt_dist = f"{np.linalg.norm(best_overall_z - true_5d):.4f}"

        status = "SOLVED!" if verified else f"{n_tests_passed}/{len(target['io'])}"
        print(f"  {target['name']:6s}: {code[:50]:50s} [{status}] "
              f"(loss={best_overall_loss:.2e}, dist={gt_dist})")

        results_list.append({
            'target': target['name'],
            'true_src': target['true_src'],
            'generated_code': code,
            'verified': verified,
            'tests_passed': n_tests_passed,
            'total_tests': len(target['io']),
            'loss': float(best_overall_loss),
            'gt_distance': gt_dist,
            'z_5d': best_overall_z.tolist(),
        })

    n_solved = sum(1 for r in results_list if r['verified'])
    n_total = len(results_list)
    print(f"\n  === MULTI-CONSTRAINT: {n_solved}/{n_total} solved ===")
    print(f"  === P50 (single I/O): 0/6 -> P52 (7 I/O + L2): {n_solved}/6 ===")

    elapsed = time.time() - t0
    results = {
        'phase': 52, 'name': 'Multi-Constraint Inverse Synthesis',
        'method': '7 diverse I/O + L2 regularization + 10 restarts',
        'n_solved': n_solved, 'n_total': n_total,
        'solve_rate': n_solved / max(n_total, 1),
        'p50_solve_rate': 0.0,
        'improvement': f'{n_solved}/6 vs 0/6',
        'synthesis': results_list,
        'elapsed': elapsed, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase52_multi_constraint.json'), 'w',
              encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    # Figure
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Solve rate comparison
    axes[0].bar(['P18\n64D, 1 I/O', 'P50\n5D, 1 I/O', 'P52\n5D, 7 I/O + L2'],
               [0, 0, n_solved/max(n_total,1)*100],
               color=['#9E9E9E', '#FF9800', '#4CAF50'], edgecolor='black')
    axes[0].set_ylabel('% Solved')
    axes[0].set_title('Inverse Synthesis Evolution', fontweight='bold')
    for i, v in enumerate([0, 0, n_solved/max(n_total,1)*100]):
        axes[0].text(i, v+3, f'{v:.0f}%', ha='center', fontweight='bold', fontsize=14)

    # 2. Tests passed per target
    names = [r['target'] for r in results_list]
    passed = [r['tests_passed'] for r in results_list]
    total = [r['total_tests'] for r in results_list]
    colors = ['#4CAF50' if r['verified'] else '#FF9800' for r in results_list]
    axes[1].bar(names, passed, color=colors, edgecolor='black')
    axes[1].axhline(y=7, color='red', linestyle='--', label='All tests (7)')
    axes[1].set_ylabel('Tests Passed')
    axes[1].set_title('Per-Target Test Pass Rate', fontweight='bold')
    axes[1].legend()

    # 3. 5D distance to ground truth
    dists = []
    dist_names = []
    for r in results_list:
        if r['gt_distance'] != 'N/A':
            dists.append(float(r['gt_distance']))
            dist_names.append(r['target'])
    if dists:
        colors_d = ['#4CAF50' if d < 1.0 else '#FF9800' for d in dists]
        axes[2].barh(dist_names, dists, color=colors_d, edgecolor='black')
        axes[2].set_xlabel('L2 Distance to True 5D Coords')
        axes[2].set_title('Proximity to Ground Truth', fontweight='bold')

    plt.suptitle('Phase 52: Multi-Constraint Inverse Synthesis\n'
                 'Differentiable TDD: Code from Test Cases',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase52_multi_constraint.png'), dpi=150)
    plt.close()
    print(f"\nPhase 52 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
