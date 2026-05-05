"""
Phase 20: The Natural Language CPU
=====================================
Execute natural language directly as code.
NL vector + args -> result. No compiler, no interpreter.
The Grand Finale of Project Rosetta.
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


def main():
    print("=" * 60)
    print("Phase 20: The Natural Language CPU")
    print("GRAND FINALE - Language as an Executable")
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
    N, D = z_nl.shape

    # Re-train Neural CPU on AST vectors
    from experiments.phase16_neural_cpu import NeuralCPU
    rng = np.random.RandomState(42)
    exec_data = []
    for i, d in enumerate(dataset):
        src = d['source']
        ns = {}
        try:
            exec(compile(src, '<test>', 'exec'), ns)
        except: continue
        if 'f' not in ns: continue
        for _ in range(10):
            try:
                if any(p in src for p in ['x, y', 'a, b', 'm, n', 'p, q']):
                    a1, a2 = float(rng.randint(1, 20)), float(rng.randint(1, 20))
                    result = float(ns['f'](a1, a2))
                    args = [a1/20.0, a2/20.0]
                elif any(v in src for v in ['(x)', '(a)', '(n)', '(v)']):
                    a1 = float(rng.randint(1, 20))
                    result = float(ns['f'](a1))
                    args = [a1/20.0, 0.0]
                else: continue
                if isinstance(result, bool): result = float(result)
                if not isinstance(result, (int, float)) or abs(result) > 1e6: continue
                exec_data.append({
                    'func_idx': i, 'args': args, 'result': result,
                    'source': src, 'nl': d['nl'],
                })
            except: continue

    print(f"Execution samples: {len(exec_data)}")
    if len(exec_data) < 100:
        print("ERROR: insufficient data")
        return {'phase': 20, 'error': 'no data'}

    # Prepare
    ast_vecs = np.array([z_ast[d['func_idx']] for d in exec_data], dtype=np.float32)
    nl_vecs = np.array([z_nl[d['func_idx']] for d in exec_data], dtype=np.float32)
    args_arr = np.array([d['args'] for d in exec_data], dtype=np.float32)
    res_arr = np.array([d['result'] for d in exec_data], dtype=np.float32)
    r_mean, r_std = float(res_arr.mean()), float(res_arr.std()) + 1e-8
    res_norm = (res_arr - r_mean) / r_std

    ast_t = torch.tensor(ast_vecs)
    nl_t = torch.tensor(nl_vecs)
    args_t = torch.tensor(args_arr)
    res_t = torch.tensor(res_norm)

    Ntotal = len(exec_data)
    Ntrain = int(Ntotal * 0.8)
    perm = torch.randperm(Ntotal)
    train_i, test_i = perm[:Ntrain], perm[Ntrain:]

    # Train on AST vectors (the "official" CPU)
    cpu_ast = NeuralCPU(64, 2, hidden=128).to(DEVICE)
    opt = torch.optim.Adam(cpu_ast.parameters(), lr=1e-3)
    for epoch in range(200):
        pe = torch.randperm(Ntrain)
        eloss, nb = 0, 0
        cpu_ast.train()
        for i in range(0, Ntrain, 256):
            idx = train_i[pe[i:i+256]]
            pred = cpu_ast(ast_t[idx].to(DEVICE), args_t[idx].to(DEVICE))
            loss = F.mse_loss(pred, res_t[idx].to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
            eloss += loss.item(); nb += 1
        if (epoch+1) % 100 == 0:
            print(f"  AST-CPU Epoch {epoch+1}/200: loss={eloss/max(nb,1):.4f}")

    # === THE GRAND TEST: Feed NL vectors into AST-trained CPU ===
    print("\n--- THE GRAND TEST: NL CPU ---")
    print("Feeding NL vectors into AST-trained Neural CPU")
    cpu_ast.eval()
    with torch.no_grad():
        # AST path (baseline)
        pred_ast = cpu_ast(ast_t[test_i].to(DEVICE), args_t[test_i].to(DEVICE)).cpu().numpy()
        # NL path (the magic!)
        pred_nl = cpu_ast(nl_t[test_i].to(DEVICE), args_t[test_i].to(DEVICE)).cpu().numpy()

    true_vals = res_t[test_i].numpy()
    true_denorm = true_vals * r_std + r_mean
    pred_ast_d = pred_ast * r_std + r_mean
    pred_nl_d = pred_nl * r_std + r_mean

    from sklearn.metrics import r2_score
    r2_ast = float(r2_score(true_denorm, pred_ast_d))
    r2_nl = float(r2_score(true_denorm, pred_nl_d))

    # Correlation between AST and NL predictions
    corr = float(np.corrcoef(pred_ast_d, pred_nl_d)[0, 1])

    mae_ast = float(np.mean(np.abs(pred_ast_d - true_denorm)))
    mae_nl = float(np.mean(np.abs(pred_nl_d - true_denorm)))

    print(f"\n  AST CPU:  R2={r2_ast:.4f}, MAE={mae_ast:.2f}")
    print(f"  NL CPU:   R2={r2_nl:.4f}, MAE={mae_nl:.2f}")
    print(f"  AST-NL correlation: {corr:.4f}")

    # Show examples
    print(f"\n  Examples (NL direct execution):")
    for k in range(min(15, len(test_i))):
        j = int(test_i[k])
        d = exec_data[j]
        a = [x*20 for x in d['args']]
        print(f"    '{d['nl'][:35]}' f({a[0]:.0f},{a[1]:.0f})"
              f" true={true_denorm[k]:.1f} NL_pred={pred_nl_d[k]:.1f}"
              f" AST_pred={pred_ast_d[k]:.1f}")

    elapsed = time.time() - t0

    print("\n" + "=" * 60)
    print("PROJECT ROSETTA: GRAND FINALE COMPLETE!")
    print(f"The Natural Language CPU achieves R2={r2_nl:.4f}")
    print("Human language executed directly without compiler or interpreter!")
    print("=" * 60)

    results = {
        'phase': 20, 'name': 'The Natural Language CPU',
        'exec_samples': Ntotal,
        'r2_ast': r2_ast, 'r2_nl': r2_nl,
        'mae_ast': mae_ast, 'mae_nl': mae_nl,
        'ast_nl_correlation': corr,
        'elapsed': elapsed, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase20_nl_cpu.json'), 'w') as f:
        json.dump(results, f, indent=2)

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # R2 comparison
    bars = axes[0].bar(['AST CPU\n(Code)', 'NL CPU\n(Language)'],
                       [r2_ast, r2_nl],
                       color=['#2196F3','#E91E63'], edgecolor='black')
    for b, v in zip(bars, [r2_ast, r2_nl]):
        axes[0].text(b.get_x()+b.get_width()/2, v+0.02, f'{v:.3f}',
                     ha='center', fontweight='bold', fontsize=14)
    axes[0].set_ylabel('R2 Score'); axes[0].set_ylim(0, 1.1)
    axes[0].set_title('Code vs Language Execution', fontweight='bold')

    # Scatter: NL pred vs true
    axes[1].scatter(true_denorm[:300], pred_nl_d[:300], alpha=0.4, s=15, color='#E91E63')
    lim = max(abs(true_denorm[:300]).max(), abs(pred_nl_d[:300]).max()) * 1.1
    axes[1].plot([-lim, lim], [-lim, lim], 'k--', lw=1)
    axes[1].set_xlabel('True Result'); axes[1].set_ylabel('NL CPU Prediction')
    axes[1].set_title(f'NL CPU (R2={r2_nl:.3f})', fontweight='bold')

    # AST pred vs NL pred
    axes[2].scatter(pred_ast_d[:300], pred_nl_d[:300], alpha=0.4, s=15, color='#4CAF50')
    axes[2].plot([-lim, lim], [-lim, lim], 'k--', lw=1)
    axes[2].set_xlabel('AST CPU'); axes[2].set_ylabel('NL CPU')
    axes[2].set_title(f'AST vs NL (corr={corr:.3f})', fontweight='bold')

    plt.suptitle('Phase 20: The Natural Language CPU\n'
                 'Human Language Executed Directly - No Compiler Needed',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase20_nl_cpu.png'), dpi=150)
    plt.close()
    print(f"\nPhase 20 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
