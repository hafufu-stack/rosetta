"""
Phase 16: The Neural Execution Engine (Neural CPU)
====================================================
Predict function output from (function_vector, input_args) without
running Python. Pure neural execution.
"""
import os, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NeuralCPU(nn.Module):
    """MLP: (function_vec, input_args) -> output value."""
    def __init__(self, func_dim, arg_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(func_dim + arg_dim, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, func_vec, args_vec):
        x = torch.cat([func_vec, args_vec], dim=-1)
        return self.net(x).squeeze(-1)

def main():
    print("=" * 60)
    print("Phase 16: The Neural Execution Engine")
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
    z_ast = latents['ast']

    # Generate (function, args, result) triplets
    print("Generating execution data...")
    exec_data = []
    rng = np.random.RandomState(42)

    for i, d in enumerate(dataset):
        src = d['source']
        # Only numeric 2-arg and 1-arg functions
        ns = {}
        try:
            exec(compile(src, '<test>', 'exec'), ns)
        except:
            continue
        if 'f' not in ns:
            continue

        for _ in range(10):  # 10 random inputs per function
            try:
                if 'x, y' in src or 'a, b' in src or 'm, n' in src or 'p, q' in src:
                    a1 = float(rng.randint(1, 20))
                    a2 = float(rng.randint(1, 20))
                    result = ns['f'](a1, a2)
                    args = [a1/20.0, a2/20.0]  # Normalize
                elif any(v in src for v in ['(x)', '(a)', '(n)', '(v)', '(num)']):
                    a1 = float(rng.randint(-10, 20))
                    result = ns['f'](a1)
                    args = [a1/20.0, 0.0]
                else:
                    continue

                if isinstance(result, bool):
                    result = float(result)
                elif not isinstance(result, (int, float)):
                    continue

                result = float(result)
                if abs(result) > 1e6:
                    continue

                exec_data.append({
                    'func_idx': i, 'args': args,
                    'result': result, 'source': src,
                })
            except:
                continue

    print(f"Generated {len(exec_data)} execution samples")

    if len(exec_data) == 0:
        print("ERROR: No execution data generated. Check function formats.")
        return {'phase': 16, 'error': 'no data'}

    # Prepare tensors
    func_vecs = np.array([z_ast[d['func_idx']] for d in exec_data], dtype=np.float32)
    args_vecs = np.array([d['args'] for d in exec_data], dtype=np.float32)
    results_arr = np.array([d['result'] for d in exec_data], dtype=np.float32)

    # Normalize results
    r_mean, r_std = results_arr.mean(), results_arr.std() + 1e-8
    results_norm = (results_arr - r_mean) / r_std

    func_t = torch.tensor(func_vecs)
    args_t = torch.tensor(args_vecs)
    res_t = torch.tensor(results_norm)

    N = len(exec_data)
    n_train = int(N * 0.8)
    perm = torch.randperm(N)
    train_i, test_i = perm[:n_train], perm[n_train:]

    # Train Neural CPU
    cpu_model = NeuralCPU(64, 2, hidden=128).to(DEVICE)
    optimizer = torch.optim.Adam(cpu_model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 300)
    BATCH = 256
    losses = []

    for epoch in range(300):
        perm_e = torch.randperm(n_train)
        eloss, nb = 0, 0
        cpu_model.train()
        for i in range(0, n_train, BATCH):
            idx = train_i[perm_e[i:i+BATCH]]
            pred = cpu_model(func_t[idx].to(DEVICE), args_t[idx].to(DEVICE))
            loss = F.mse_loss(pred, res_t[idx].to(DEVICE))
            optimizer.zero_grad(); loss.backward()
            optimizer.step()
            eloss += loss.item(); nb += 1
        scheduler.step()
        losses.append(eloss / max(nb, 1))
        if (epoch+1) % 100 == 0:
            print(f"  Epoch {epoch+1}/300: loss={losses[-1]:.4f}")

    # Evaluate
    cpu_model.eval()
    with torch.no_grad():
        pred_test = cpu_model(func_t[test_i].to(DEVICE),
                              args_t[test_i].to(DEVICE)).cpu().numpy()
    true_test = res_t[test_i].numpy()

    # Denormalize
    pred_denorm = pred_test * r_std + r_mean
    true_denorm = true_test * r_std + r_mean

    # Metrics
    mse = float(np.mean((pred_denorm - true_denorm)**2))
    mae = float(np.mean(np.abs(pred_denorm - true_denorm)))
    # Accuracy within 10% tolerance
    close = np.abs(pred_denorm - true_denorm) < np.maximum(np.abs(true_denorm) * 0.1, 0.5)
    acc_10pct = float(np.mean(close))
    # Exact (within 0.5)
    exact = np.abs(pred_denorm - true_denorm) < 0.5
    acc_exact = float(np.mean(exact))

    from sklearn.metrics import r2_score
    r2 = float(r2_score(true_denorm, pred_denorm))

    print(f"\n--- Neural CPU Results ---")
    print(f"  MSE: {mse:.4f}, MAE: {mae:.4f}")
    print(f"  R2: {r2:.4f}")
    print(f"  Exact (within 0.5): {acc_exact:.1%}")
    print(f"  Close (within 10%): {acc_10pct:.1%}")

    # Show examples
    print(f"\n  Examples:")
    for k in range(min(10, len(test_i))):
        j = int(test_i[k])
        d = exec_data[j]
        a = [x*20 for x in d['args']]
        print(f"    {d['source'][:35]} f({a[0]:.0f},{a[1]:.0f})="
              f"{true_denorm[k]:.1f} -> pred={pred_denorm[k]:.1f}")

    elapsed = time.time() - t0
    results = {
        'phase': 16, 'name': 'Neural Execution Engine',
        'exec_samples': N, 'mse': mse, 'mae': mae, 'r2': r2,
        'acc_exact': acc_exact, 'acc_10pct': acc_10pct,
        'final_loss': float(losses[-1]),
        'elapsed': elapsed, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase16_neural_cpu.json'), 'w') as f:
        json.dump(results, f, indent=2)

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(losses, color='#E91E63', lw=1.5)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('Neural CPU Training', fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(true_denorm[:200], pred_denorm[:200], alpha=0.5, s=15, color='#2196F3')
    lim = max(abs(true_denorm[:200]).max(), abs(pred_denorm[:200]).max()) * 1.1
    axes[1].plot([-lim, lim], [-lim, lim], 'r--', lw=1)
    axes[1].set_xlabel('True Result'); axes[1].set_ylabel('Predicted')
    axes[1].set_title(f'Prediction (R2={r2:.3f})', fontweight='bold')

    bars = axes[2].bar(['Exact\n(+/-0.5)', 'Close\n(+/-10%)'],
                       [acc_exact, acc_10pct],
                       color=['#4CAF50','#FF9800'], edgecolor='black')
    for b, v in zip(bars, [acc_exact, acc_10pct]):
        axes[2].text(b.get_x()+b.get_width()/2, v+0.02, f'{v:.0%}',
                     ha='center', fontweight='bold', fontsize=14)
    axes[2].set_ylim(0, 1.1); axes[2].set_title('Accuracy', fontweight='bold')
    plt.suptitle('Phase 16: Neural CPU (Execute Code Without Running It)',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase16_neural_cpu.png'), dpi=150)
    plt.close()
    print(f"\nPhase 16 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
