"""
Phase 84: The Latent Calculator
==================================
The ultimate test: can we COMPUTE f(3,5) for addition
PURELY in 5D space, WITHOUT executing any code?

If this works, the 5D space isn't just a representation —
it's a COMPUTATIONAL ENGINE.
"""
import os, json, time, inspect
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')


def main():
    print("=" * 60)
    print("Phase 84: The Latent Calculator")
    print("Computing WITHOUT executing code")
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
    pca = PCA(n_components=10).fit(z_ast)
    z_5d = pca.transform(z_ast)[:, :5]

    src_to_z = {}
    for i, src in enumerate(sources):
        if src not in src_to_z:
            src_to_z[src] = z_5d[i]

    # Build training data: (z_5d, args) -> result
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n--- Building Latent Calculator Training Data ---")
    calc_data = []
    for src, z5 in src_to_z.items():
        try:
            ns = {}
            exec(compile(src, '<string>', 'exec'), ns)
            fn = [v for k, v in ns.items()
                  if callable(v) and not k.startswith('_')][0]
            sig = inspect.signature(fn)
            n_p = len(sig.parameters)

            if n_p == 1:
                test_args = [(-5,), (-2,), (0,), (1,), (2,), (3,), (5,), (7,), (10,)]
            elif n_p == 2:
                test_args = [(-2,3), (0,0), (1,1), (1,2), (2,3), (3,4), (3,5),
                            (5,7), (7,2), (10,3)]
            else:
                continue

            for args in test_args:
                try:
                    result = fn(*args)
                    if not isinstance(result, (int, float)):
                        continue
                    result = float(result)
                    if abs(result) > 1e4 or np.isnan(result):
                        continue
                    padded_args = list(args) + [0] * (2 - len(args))
                    calc_data.append({
                        'z5': z5, 'args': padded_args, 'result': result,
                        'src': src,
                    })
                except Exception:
                    pass
        except Exception:
            pass

    print(f"  Training samples: {len(calc_data)}")

    # Build tensors
    X_z = np.array([d['z5'] for d in calc_data], dtype=np.float32)
    X_args = np.array([d['args'] for d in calc_data], dtype=np.float32)
    Y = np.array([d['result'] for d in calc_data], dtype=np.float32)

    # Normalize
    y_mean, y_std = Y.mean(), Y.std() + 1e-8
    Y_norm = (Y - y_mean) / y_std

    X = np.concatenate([X_z, X_args], axis=1)  # 5 + 2 = 7 features
    X_t = torch.tensor(X).to(DEVICE)
    Y_t = torch.tensor(Y_norm).unsqueeze(1).to(DEVICE)

    # Split
    n = len(X_t)
    perm = torch.randperm(n)
    split = int(n * 0.8)
    train_idx, test_idx = perm[:split], perm[split:]

    # Neural calculator: z5 + args -> result
    class LatentCalculator(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(7, 256), nn.ReLU(), nn.Dropout(0.05),
                nn.Linear(256, 128), nn.ReLU(),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, 1),
            )
        def forward(self, x):
            return self.net(x)

    calc = LatentCalculator().to(DEVICE)
    opt = torch.optim.Adam(calc.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 500)

    print("\n--- Training Latent Calculator ---")
    for epoch in range(500):
        calc.train()
        pred = calc(X_t[train_idx])
        loss = F.mse_loss(pred, Y_t[train_idx])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()

        if (epoch+1) % 100 == 0:
            calc.eval()
            with torch.no_grad():
                pred_test = calc(X_t[test_idx])
                test_loss = F.mse_loss(pred_test, Y_t[test_idx])
                r2 = 1 - test_loss.item() / (Y_t[test_idx].var().item() + 1e-8)
            print(f"  Epoch {epoch+1}/500: train={loss.item():.6f}, "
                  f"test={test_loss.item():.6f}, R2={r2:.4f}")

    # Test the calculator on specific functions
    print("\n--- Latent Calculator Results ---")
    calc.eval()
    test_cases = [
        ('def f(x, y): return x + y', (3, 5), 8),
        ('def f(x, y): return x + y', (10, 3), 13),
        ('def f(x, y): return x * y', (3, 5), 15),
        ('def f(x, y): return x * y', (7, 2), 14),
        ('def f(x, y): return x - y', (7, 3), 4),
        ('def f(x): return abs(x)', (-7, 0), 7),
        ('def f(x): return -x', (5, 0), -5),
        ('def f(x, y): return x ** y', (2, 3), 8),
        ('def f(x, y): return max(x, y)', (3, 5), 5),
        ('def f(x, y): return min(x, y)', (3, 5), 3),
    ]

    calc_results = []
    for src, args, expected in test_cases:
        z5 = src_to_z.get(src)
        if z5 is None:
            continue
        padded = list(args) + [0] * (2 - len(args))
        inp = torch.tensor(np.concatenate([z5, padded]).astype(np.float32)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred_norm = calc(inp).item()
        pred = pred_norm * y_std + y_mean
        error = abs(pred - expected)

        short = src.split('return ')[1][:12] if 'return' in src else '?'
        print(f"  {short:12s}({args[0]},{args[1]}): "
              f"expected={expected:6.1f}, predicted={pred:6.1f}, err={error:.2f}")

        calc_results.append({
            'src': src, 'args': list(args),
            'expected': expected, 'predicted': float(pred),
            'error': float(error),
        })

    avg_error = np.mean([r['error'] for r in calc_results])
    n_close = sum(1 for r in calc_results if r['error'] < 2)
    print(f"\n  Average error: {avg_error:.2f}")
    print(f"  Close predictions (<2): {n_close}/{len(calc_results)}")
    print(f"  Calculator {'WORKS' if n_close > len(calc_results)//2 else 'needs more data'}!")

    elapsed = time.time() - t0
    results = {
        'phase': 84, 'name': 'The Latent Calculator',
        'n_train': len(calc_data), 'r2': float(r2),
        'avg_error': float(avg_error),
        'n_close': n_close, 'n_total': len(calc_results),
        'test_results': calc_results,
        'elapsed': elapsed, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase84_calculator.json'), 'w',
              encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    labels = [f"{r['src'].split('return ')[1][:8]}({r['args'][0]},{r['args'][1]})"
             if 'return' in r['src'] else '?' for r in calc_results]
    expected = [r['expected'] for r in calc_results]
    predicted = [r['predicted'] for r in calc_results]
    x = range(len(labels))
    axes[0].bar([i-0.15 for i in x], expected, 0.3, label='Expected', color='#4CAF50', edgecolor='black')
    axes[0].bar([i+0.15 for i in x], predicted, 0.3, label='Predicted', color='#2196F3', edgecolor='black')
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(labels, rotation=45, fontsize=6)
    axes[0].legend()
    axes[0].set_title('Latent Calculator\n(5D + args -> result)', fontweight='bold')

    axes[1].scatter(expected, predicted, c='#9C27B0', s=50, edgecolors='black')
    lims = [min(min(expected), min(predicted))-1, max(max(expected), max(predicted))+1]
    axes[1].plot(lims, lims, 'r--')
    axes[1].set_xlabel('Expected')
    axes[1].set_ylabel('Predicted')
    axes[1].set_title(f'Accuracy: R2={r2:.3f}\nAvg Error={avg_error:.1f}', fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Phase 84: The Latent Calculator\nComputing Without Code Execution',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase84_calculator.png'), dpi=150)
    plt.close()
    print(f"\nPhase 84 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
