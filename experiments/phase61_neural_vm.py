"""
Phase 61: The Neural Virtual Machine
=======================================
Simulate program EXECUTION as trajectories in latent space.
Instead of running code, watch state vectors move through
the 5D manifold, pulled by the function's gravity.

The Turing Machine, reimagined as continuous dynamics.
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
    print("Phase 61: The Neural Virtual Machine")
    print("Execution = trajectory in latent space")
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
            src_to_z[src] = {'z_ast': z_ast[i], 'z_5d': z_5d[i]}

    # Build step-by-step execution data
    # For each function: (func_5d, state_before) -> state_after
    print("\n--- Building Execution Trajectories ---")
    trajectories = []
    unique_srcs = list(src_to_z.keys())

    for src in unique_srcs:
        try:
            ns = {}
            exec(compile(src, '<string>', 'exec'), ns)
            fn = [v for k, v in ns.items()
                  if callable(v) and not k.startswith('_')][0]
            sig = inspect.signature(fn)
            n_p = len(sig.parameters)

            z5 = src_to_z[src]['z_5d']

            # Generate trajectories: sequence of state transitions
            if n_p == 1:
                test_inputs = [(-5,), (-2,), (0,), (1,), (3,), (5,), (7,), (10,)]
            elif n_p == 2:
                test_inputs = [(-2,3), (0,0), (1,1), (2,3), (3,5), (5,7), (7,-1)]
            else:
                continue

            for args in test_inputs:
                try:
                    result = fn(*args)
                    if not isinstance(result, (int, float)):
                        continue
                    result = float(result)
                    if abs(result) > 1e6 or np.isnan(result):
                        continue

                    # State vector: [arg1, arg2, partial_result, step]
                    # We simulate a 3-step trajectory:
                    # Step 0: Initial state (inputs loaded)
                    # Step 1: Intermediate (partial computation)
                    # Step 2: Final state (result computed)
                    state_0 = list(args) + [0]*(3-len(args)) + [0.0]  # 4D state
                    state_2 = list(args) + [0]*(3-len(args)-1) + [result, 1.0]

                    # Intermediate: linear interpolation + noise
                    mid_result = sum(args) / len(args)  # Rough midpoint
                    state_1 = list(args) + [0]*(3-len(args)-1) + [mid_result, 0.5]

                    trajectories.append({
                        'src': src, 'z_5d': z5.tolist(),
                        'states': [state_0, state_1, state_2],
                        'args': list(args), 'result': result,
                    })
                except Exception:
                    pass
        except Exception:
            pass

    print(f"  Generated {len(trajectories)} trajectories")

    # Train Neural VM: (func_5d, state_t) -> state_t+1
    train_data = []
    for traj in trajectories:
        z5 = traj['z_5d']
        for t in range(len(traj['states']) - 1):
            s_t = traj['states'][t]
            s_next = traj['states'][t + 1]
            train_data.append({
                'input': z5 + s_t,  # 5D func + 4D state = 9D
                'target': s_next,    # 4D next state
            })

    X = np.array([d['input'] for d in train_data], dtype=np.float32)
    Y = np.array([d['target'] for d in train_data], dtype=np.float32)
    X_mean, X_std = X.mean(0), X.std(0) + 1e-8
    Y_mean, Y_std = Y.mean(0), Y.std(0) + 1e-8

    X_t = torch.tensor((X - X_mean) / X_std).to(DEVICE)
    Y_t = torch.tensor((Y - Y_mean) / Y_std).to(DEVICE)

    print(f"  Training samples: {len(train_data)}")

    # Neural VM model
    nvm = nn.Sequential(
        nn.Linear(9, 256), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, 4),  # 4D state output
    ).to(DEVICE)

    opt = torch.optim.Adam(nvm.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 500)

    for epoch in range(500):
        pred = nvm(X_t)
        loss = F.mse_loss(pred, Y_t)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if (epoch+1) % 100 == 0:
            r2 = 1 - loss.item() / (Y_t.var().item() + 1e-8)
            print(f"  NVM Epoch {epoch+1}/500: loss={loss.item():.6f}, R2={r2:.4f}")

    nvm.eval()

    # Simulate execution trajectories
    print("\n--- Simulating Execution Trajectories ---")
    test_funcs = [
        ('def f(x, y): return x + y', (3, 5)),
        ('def f(x, y): return x * y', (3, 5)),
        ('def f(x, y): return x - y', (7, 3)),
        ('def f(x): return -x', (5,)),
        ('def f(x): return abs(x)', (-7,)),
        ('def f(x, y): return x ** y', (2, 3)),
    ]

    sim_results = []
    for src, args in test_funcs:
        z5 = src_to_z.get(src, {}).get('z_5d')
        if z5 is None:
            continue

        # Compute true result
        try:
            ns = {}
            exec(compile(src, '<string>', 'exec'), ns)
            fn = [v for k, v in ns.items()
                  if callable(v) and not k.startswith('_')][0]
            true_result = float(fn(*args))
        except Exception:
            continue

        # Simulate trajectory
        state = list(args) + [0]*(3-len(args)) + [0.0]  # 4D initial
        trajectory = [state.copy()]

        for step in range(5):  # 5 steps of simulation
            inp = np.array(list(z5) + state, dtype=np.float32)
            inp_n = torch.tensor((inp - X_mean) / X_std).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                pred = nvm(inp_n).cpu().numpy()[0]
            state = list(pred * Y_std + Y_mean)
            trajectory.append(state.copy())

        predicted_result = float(trajectory[-1][2])  # 3rd element = result
        error = float(abs(predicted_result - true_result))

        print(f"  {src[:35]:35s} args={args}")
        print(f"    True: {true_result:.2f}, NVM pred: {predicted_result:.2f}, "
              f"err: {error:.2f}")
        print(f"    Trajectory (result dim):", end='')
        for s in trajectory:
            print(f" {s[2]:.2f}", end=' ->')
        print()

        sim_results.append({
            'src': src, 'args': list(args),
            'true_result': float(true_result),
            'predicted': float(predicted_result),
            'error': float(error),
            'trajectory': [[float(x) for x in s] for s in trajectory],
        })

    avg_error = np.mean([r['error'] for r in sim_results])
    print(f"\n  Average prediction error: {avg_error:.4f}")

    elapsed = time.time() - t0
    results = {
        'phase': 61, 'name': 'The Neural Virtual Machine',
        'n_trajectories': len(trajectories),
        'n_train': len(train_data),
        'avg_error': float(avg_error),
        'simulations': sim_results,
        'elapsed': elapsed, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase61_neural_vm.json'), 'w',
              encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    # Figure
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Prediction accuracy
    names = [r['src'].split('return ')[1][:15] if 'return' in r['src']
             else r['src'][:15] for r in sim_results]
    true_vals = [r['true_result'] for r in sim_results]
    pred_vals = [r['predicted'] for r in sim_results]
    x = range(len(names))
    axes[0].bar([i-0.15 for i in x], true_vals, 0.3,
               label='True', color='#4CAF50', edgecolor='black')
    axes[0].bar([i+0.15 for i in x], pred_vals, 0.3,
               label='NVM', color='#2196F3', edgecolor='black')
    axes[0].set_xticks(list(x)); axes[0].set_xticklabels(names, rotation=45, fontsize=8)
    axes[0].set_ylabel('Result')
    axes[0].set_title('True vs NVM Prediction', fontweight='bold')
    axes[0].legend()

    # 2. Trajectories in result dimension
    for r in sim_results[:4]:
        traj_vals = [s[2] for s in r['trajectory']]
        label = r['src'].split('return ')[1][:10] if 'return' in r['src'] else '?'
        axes[1].plot(range(len(traj_vals)), traj_vals, '-o', label=label, markersize=4)
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Result Dimension')
    axes[1].set_title('Execution Trajectories\n(State flowing toward result)',
                     fontweight='bold')
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)

    # 3. Error distribution
    errors = [r['error'] for r in sim_results]
    axes[2].bar(names, errors, color='#FF9800', edgecolor='black')
    axes[2].set_ylabel('Absolute Error')
    axes[2].set_title(f'NVM Prediction Error\n(avg={avg_error:.2f})', fontweight='bold')
    axes[2].tick_params(axis='x', rotation=45)

    plt.suptitle('Phase 61: The Neural Virtual Machine\n'
                 'Program Execution as Trajectories in Latent Space',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase61_neural_vm.png'), dpi=150)
    plt.close()
    print(f"\nPhase 61 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
