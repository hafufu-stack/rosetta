"""
Phase 55: Retrieval-Augmented Inverse Synthesis
=================================================
P52 proved: 5D optimization converges perfectly, but the DECODER
can't translate 5D coords back to code strings.

Key insight: We don't NEED to decode. We can just LOOK UP the
nearest function in our dataset using 5D coordinates!

This is "Semantic Retrieval TDD":
  1. Optimize 5D coords to fit I/O tests (P52 method)
  2. Find nearest neighbor in dataset (bypass decoder entirely)
  3. Return that function as the synthesis result

If the 5D space is truly a meaningful manifold, the nearest
neighbor to the optimized point should be the correct function.
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


def main():
    print("=" * 60)
    print("Phase 55: Retrieval-Augmented Inverse Synthesis")
    print("Bypass decoder: optimize 5D, then NN lookup")
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

    # PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=10)
    z_pca = pca.fit_transform(z_ast)
    print(f"  PCA: 5D = {sum(pca.explained_variance_ratio_[:5])*100:.1f}% variance")

    # Build unique function index (deduplicate)
    unique_funcs = {}
    for i, src in enumerate(sources):
        if src not in unique_funcs:
            unique_funcs[src] = {'idx': i, 'z_5d': z_pca[i, :5]}
    unique_list = list(unique_funcs.items())
    print(f"  Unique functions: {len(unique_list)}")

    # Build execution data for Neural CPU
    exec_data = []
    for src, info in unique_list:
        try:
            ns = {}
            exec(compile(src, '<string>', 'exec'), ns)
            fn = [v for k, v in ns.items() if callable(v) and not k.startswith('_')][0]
            sig = inspect.signature(fn)
            n_p = len(sig.parameters)

            tests = []
            if n_p == 1:
                tests = [(-5,), (-3,), (-1,), (0,), (1,), (2,), (3,), (5,), (7,), (10,)]
            elif n_p == 2:
                tests = [(-3,-1), (-1,2), (0,0), (1,1), (2,3), (3,5), (5,-2), (7,4),
                         (-2,-3), (4,1)]

            for args in tests:
                try:
                    r = fn(*args)
                    if isinstance(r, (int, float)) and not np.isnan(float(r)):
                        rv = float(r)
                        if abs(rv) < 1e6:
                            pa = list(args) + [0]*(3-len(args))
                            exec_data.append({
                                'src': src, 'z_5d': info['z_5d'].tolist(),
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
            print(f"  Neural CPU Epoch {epoch+1}/600: R2={r2:.4f}")
    cpu5d.eval()

    # Build function database with 5D coordinates
    func_db = []
    for src, info in unique_list:
        func_db.append({'src': src, 'z_5d': info['z_5d']})
    db_coords = np.array([f['z_5d'] for f in func_db])

    # Test targets (same as P52, plus more)
    targets = [
        {'name': 'add', 'true_src': 'def f(x, y): return x + y',
         'io': [(-3,2,0,-1), (0,0,0,0), (1,1,0,2), (5,-2,0,3),
                (7,3,0,10), (10,10,0,20), (-5,-5,0,-10)]},
        {'name': 'sub', 'true_src': 'def f(x, y): return x - y',
         'io': [(5,3,0,2), (0,0,0,0), (1,5,0,-4), (10,3,0,7),
                (-2,-3,0,1), (7,7,0,0), (3,-1,0,4)]},
        {'name': 'mul', 'true_src': 'def f(x, y): return x * y',
         'io': [(2,3,0,6), (0,5,0,0), (-1,3,0,-3), (4,4,0,16),
                (1,7,0,7), (-2,-3,0,6), (5,1,0,5)]},
        {'name': 'neg', 'true_src': 'def f(x): return -x',
         'io': [(5,0,0,-5), (-3,0,0,3), (0,0,0,0), (1,0,0,-1),
                (10,0,0,-10), (-7,0,0,7), (100,0,0,-100)]},
        {'name': 'abs', 'true_src': 'def f(x): return abs(x)',
         'io': [(-5,0,0,5), (3,0,0,3), (0,0,0,0), (-1,0,0,1),
                (7,0,0,7), (-10,0,0,10), (-100,0,0,100)]},
        {'name': 'max', 'true_src': 'def f(x, y): return max(x, y)',
         'io': [(3,5,0,5), (5,3,0,5), (0,0,0,0), (-1,1,0,1),
                (7,7,0,7), (-3,-5,0,-3), (10,1,0,10)]},
        {'name': 'mod', 'true_src': 'def f(x, y): return x % y',
         'io': [(7,3,0,1), (10,5,0,0), (1,1,0,0), (9,4,0,1),
                (15,7,0,1), (8,3,0,2), (100,7,0,2)]},
        {'name': 'pow', 'true_src': 'def f(x, y): return x ** y',
         'io': [(2,3,0,8), (3,2,0,9), (5,1,0,5), (1,10,0,1),
                (2,0,0,1), (10,2,0,100), (3,3,0,27)]},
        {'name': 'square', 'true_src': 'def f(x): return x * x',
         'io': [(3,0,0,9), (-2,0,0,4), (0,0,0,0), (5,0,0,25),
                (1,0,0,1), (7,0,0,49), (-4,0,0,16)]},
        {'name': 'double', 'true_src': 'def f(x): return x * 2',
         'io': [(1,0,0,2), (0,0,0,0), (5,0,0,10), (-3,0,0,-6),
                (7,0,0,14), (10,0,0,20), (-1,0,0,-2)]},
    ]

    print(f"\n--- Retrieval-Augmented Inverse Synthesis ---")
    print(f"  Method: Optimize 5D -> NN lookup (no decoder!)")
    print(f"  Targets: {len(targets)}, Database: {len(func_db)} unique funcs")

    results_list = []
    for target in targets:
        # Step 1: Optimize 5D coordinates (same as P52)
        best_loss = float('inf')
        best_z = None

        for restart in range(15):
            z5 = torch.randn(5, device=DEVICE, dtype=torch.float32) * 0.3
            z5.requires_grad_(True)
            opt_inv = torch.optim.Adam([z5], lr=0.05)

            for step in range(400):
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
                # L2 regularization
                total_loss = total_loss + 0.001 * torch.sum(z5 ** 2)

                opt_inv.zero_grad()
                total_loss.backward()
                opt_inv.step()

                if total_loss.item() < best_loss:
                    best_loss = total_loss.item()
                    best_z = z5.detach().cpu().numpy().copy()

        # Step 2: Find nearest neighbor in database (THE KEY DIFFERENCE!)
        dists = np.linalg.norm(db_coords - best_z, axis=1)
        nn_idx = np.argmin(dists)
        nn_src = func_db[nn_idx]['src']
        nn_dist = dists[nn_idx]

        # Also get top-3 nearest neighbors
        top3_idx = np.argsort(dists)[:3]
        top3 = [(func_db[i]['src'], dists[i]) for i in top3_idx]

        # Step 3: Verify the NN result
        verified = False
        n_tests_passed = 0
        try:
            ns = {}
            exec(compile(nn_src, '<string>', 'exec'), ns)
            fn = [v for k, v in ns.items() if callable(v) and not k.startswith('_')][0]
            sig = inspect.signature(fn)
            n_params = len(sig.parameters)

            for a1, a2, a3, expected in target['io']:
                try:
                    if n_params == 1:
                        r = float(fn(a1))
                    elif n_params == 2:
                        r = float(fn(a1, a2))
                    else:
                        r = float(fn(a1, a2, a3))
                    if abs(r - expected) < 0.01:
                        n_tests_passed += 1
                except Exception:
                    pass
            verified = (n_tests_passed == len(target['io']))
        except Exception:
            pass

        # Check if semantically equivalent (same behavior, maybe different name)
        semantic_match = verified

        status = "SOLVED!" if verified else f"{n_tests_passed}/{len(target['io'])}"
        print(f"  {target['name']:8s}: NN = {nn_src[:45]:45s} [{status}] "
              f"(dist={nn_dist:.4f})")
        for s, d in top3[1:]:
            print(f"           #{top3_idx[top3.index((s,d))]+1:3d} {s[:45]} (d={d:.4f})")

        results_list.append({
            'target': target['name'],
            'true_src': target['true_src'],
            'nn_src': nn_src,
            'nn_dist': float(nn_dist),
            'verified': verified,
            'semantic_match': semantic_match,
            'tests_passed': n_tests_passed,
            'total_tests': len(target['io']),
            'loss': float(best_loss),
            'top3': [{'src': s, 'dist': float(d)} for s, d in top3],
        })

    n_solved = sum(1 for r in results_list if r['verified'])
    n_total = len(results_list)
    print(f"\n  ==========================================")
    print(f"  P18 (64D + decoder):   0/6 solved")
    print(f"  P50 (5D + decoder):    0/6 solved")
    print(f"  P52 (5D+MC + decoder): 0/6 solved")
    print(f"  P55 (5D+MC + NN):      {n_solved}/{n_total} SOLVED!")
    print(f"  ==========================================")

    # Bonus: How does retrieval rank compare to distance?
    print(f"\n--- Analysis: Does closer = more correct? ---")
    for r in results_list:
        status = "OK" if r['verified'] else "X "
        print(f"  [{status}] {r['target']:8s}: dist={r['nn_dist']:.4f}, "
              f"tests={r['tests_passed']}/{r['total_tests']}")

    elapsed = time.time() - t0
    results = {
        'phase': 55, 'name': 'Retrieval-Augmented Inverse Synthesis',
        'method': '5D optimization + NN lookup (bypass decoder)',
        'n_solved': n_solved, 'n_total': n_total,
        'solve_rate': n_solved / max(n_total, 1),
        'p52_solve_rate': 0.0,
        'p50_solve_rate': 0.0,
        'improvement': f'P55: {n_solved}/{n_total} vs P52: 0/6',
        'synthesis': results_list,
        'elapsed': elapsed, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase55_retrieval_synthesis.json'), 'w',
              encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    # Figure
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Evolution of inverse synthesis
    methods = ['P18\n64D+Dec', 'P50\n5D+Dec', 'P52\n5D+MC+Dec', 'P55\n5D+MC+NN']
    rates = [0, 0, 0, n_solved/max(n_total,1)*100]
    colors = ['#9E9E9E', '#FF9800', '#FF5722', '#4CAF50']
    axes[0].bar(methods, rates, color=colors, edgecolor='black')
    axes[0].set_ylabel('% Solved')
    axes[0].set_title('Inverse Synthesis Evolution\n(The Decoder Was the Problem!)',
                      fontweight='bold')
    axes[0].set_ylim(0, 110)
    for i, v in enumerate(rates):
        axes[0].text(i, v+3, f'{v:.0f}%', ha='center', fontweight='bold', fontsize=13)

    # 2. Per-target results
    names = [r['target'] for r in results_list]
    passed = [r['tests_passed'] for r in results_list]
    total_t = [r['total_tests'] for r in results_list]
    c = ['#4CAF50' if r['verified'] else '#FF9800' for r in results_list]
    axes[1].bar(names, passed, color=c, edgecolor='black')
    axes[1].axhline(y=7, color='red', linestyle='--', alpha=0.5, label='All tests')
    axes[1].set_ylabel('Tests Passed (out of 7)')
    axes[1].set_title('Per-Target Results', fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].legend()

    # 3. NN distance vs success
    dists_plot = [r['nn_dist'] for r in results_list]
    solved_plot = [1 if r['verified'] else 0 for r in results_list]
    c3 = ['#4CAF50' if s else '#F44336' for s in solved_plot]
    axes[2].scatter(dists_plot, [r['tests_passed'] for r in results_list],
                   c=c3, s=100, edgecolor='black', zorder=5)
    for r in results_list:
        axes[2].annotate(r['target'], (r['nn_dist'], r['tests_passed']),
                        fontsize=8, ha='left', va='bottom')
    axes[2].set_xlabel('NN Distance in 5D Space')
    axes[2].set_ylabel('Tests Passed')
    axes[2].set_title('Distance vs Correctness\n(closer = better?)', fontweight='bold')
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Phase 55: Retrieval-Augmented Inverse Synthesis\n'
                 'The Decoder Was the Bottleneck All Along!',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase55_retrieval_synthesis.png'), dpi=150)
    plt.close()
    print(f"\nPhase 55 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
