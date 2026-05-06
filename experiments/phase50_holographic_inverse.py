"""
Phase 50: Holographic Inverse Synthesis
=========================================
LIMITATION BREAKER #2: Inverse Execution (P18 failed)

P18 failed because gradient descent explored all 64 dimensions,
falling into adversarial noise regions. Solution: constrain
optimization to only the 5 meaningful PCA dimensions (P40/P42).

The "Differentiable TDD" - synthesize code from I/O test cases
by optimizing ONLY on the 5D manifold where meaning exists.
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
    print("Phase 50: Holographic Inverse Synthesis")
    print("LIMITATION BREAKER #2: Inverse Execution")
    print("  Using P40's 5D manifold to constrain search")
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

    # Step 1: PCA to find the 5D manifold (from P40)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=10)
    z_pca = pca.fit_transform(z_ast)
    print(f"  PCA fit: 5D captures {sum(pca.explained_variance_ratio_[:5])*100:.1f}% variance")

    # Step 2: Build Neural CPU (from P16) in the 5D PCA space
    # This maps (function_5d_coords, input_args) -> output
    print("\n--- Training Neural CPU on 5D Manifold ---")

    # Generate execution data
    exec_data = []
    for i, d in enumerate(dataset):
        src = d['source']
        try:
            ns = {}
            exec(compile(src, '<string>', 'exec'), ns)
            func_name = [k for k in ns if not k.startswith('_')][0]
            func = ns[func_name]
            import inspect
            sig = inspect.signature(func)
            n_params = len(sig.parameters)

            # Generate test inputs
            test_inputs = []
            if n_params == 1:
                test_inputs = [(x,) for x in [-3, -1, 0, 1, 2, 5, 10]]
            elif n_params == 2:
                test_inputs = [(a, b) for a in [-2, 0, 1, 3, 5]
                              for b in [-1, 0, 2, 4]]
            elif n_params == 3:
                test_inputs = [(1, 2, 3), (0, -1, 1), (2, 3, 4)]

            for args in test_inputs:
                try:
                    # Skip functions that need string/list inputs
                    result = func(*args)
                    if isinstance(result, (int, float)) and not np.isnan(float(result)):
                        r = float(result)
                        if abs(r) < 1e6:
                            padded_args = list(args) + [0] * (3 - len(args))
                            exec_data.append({
                                'func_idx': i,
                                'z_5d': z_pca[i, :5].tolist(),
                                'args': padded_args[:3],
                                'result': r,
                            })
                except Exception:
                    pass
        except Exception:
            pass

    print(f"  Generated {len(exec_data)} execution samples")

    if len(exec_data) < 100:
        print("  WARNING: Too few execution samples, trying broader inputs")
        # Minimal fallback
        for i in range(min(N, 100)):
            src = sources[i]
            if 'x, y' in src or 'a, b' in src:
                for a, b in [(1,2),(3,4),(5,1),(-1,2),(0,0)]:
                    try:
                        ns = {}
                        exec(compile(src, '<string>', 'exec'), ns)
                        fn = [v for k,v in ns.items() if not k.startswith('_')][0]
                        r = float(fn(a, b))
                        if abs(r) < 1e6:
                            exec_data.append({
                                'func_idx': i, 'z_5d': z_pca[i, :5].tolist(),
                                'args': [a, b, 0], 'result': r,
                            })
                    except Exception:
                        pass
        print(f"  After fallback: {len(exec_data)} execution samples")

    # Build Neural CPU: 5D_coords + 3_args -> output
    X_cpu = np.array([[*d['z_5d'], *d['args']] for d in exec_data], dtype=np.float32)
    Y_cpu = np.array([d['result'] for d in exec_data], dtype=np.float32)

    # Normalize
    X_mean, X_std = X_cpu.mean(0), X_cpu.std(0) + 1e-8
    Y_mean, Y_std = Y_cpu.mean(), Y_cpu.std() + 1e-8
    X_norm = (X_cpu - X_mean) / X_std
    Y_norm = (Y_cpu - Y_mean) / Y_std

    X_t = torch.tensor(X_norm, dtype=torch.float32).to(DEVICE)
    Y_t = torch.tensor(Y_norm, dtype=torch.float32).to(DEVICE)

    # Simple MLP
    cpu_model = nn.Sequential(
        nn.Linear(8, 128), nn.ReLU(),
        nn.Linear(128, 128), nn.ReLU(),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, 1)
    ).to(DEVICE)

    opt = torch.optim.Adam(cpu_model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 500)

    for epoch in range(500):
        pred = cpu_model(X_t).squeeze()
        loss = F.mse_loss(pred, Y_t)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if (epoch+1) % 100 == 0:
            r2 = 1 - loss.item() / (Y_t.var().item() + 1e-8)
            print(f"  Epoch {epoch+1}/500: loss={loss.item():.4f}, R2={r2:.3f}")

    cpu_model.eval()
    with torch.no_grad():
        final_pred = cpu_model(X_t).squeeze()
        r2_5d = 1 - F.mse_loss(final_pred, Y_t).item() / (Y_t.var().item() + 1e-8)
    print(f"  5D Neural CPU R2 = {r2_5d:.4f}")

    # Step 3: INVERSE SYNTHESIS!
    # Given I/O test cases, find the 5D PCA coordinates that minimize error
    print("\n--- Holographic Inverse Synthesis ---")
    print("  Optimizing ONLY 5 PCA dimensions (not 64!)")

    # Load decoder for code generation
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

    def decode_from_5d(pca_5d_coeffs):
        """Convert 5D PCA coefficients -> 64D vector -> Python code."""
        full_pca = np.zeros(10)
        full_pca[:5] = pca_5d_coeffs
        z_64d = pca.inverse_transform(full_pca.reshape(1, -1))[0]
        with torch.no_grad():
            z_t = torch.tensor(z_64d.astype(np.float32)).unsqueeze(0).to(DEVICE)
            tokens = decoder(z_t)
            return decode_tokens(tokens[0].cpu().numpy(), idx2char)

    # Test targets: find functions from I/O
    test_targets = [
        {'name': 'add', 'io': [(1,2,0,3.0), (3,4,0,7.0), (-1,2,0,1.0), (5,5,0,10.0)]},
        {'name': 'sub', 'io': [(5,3,0,2.0), (1,1,0,0.0), (10,4,0,6.0)]},
        {'name': 'mul', 'io': [(2,3,0,6.0), (4,5,0,20.0), (-1,3,0,-3.0)]},
        {'name': 'neg', 'io': [(5,0,0,-5.0), (-3,0,0,3.0), (0,0,0,0.0)]},
        {'name': 'abs', 'io': [(-5,0,0,5.0), (3,0,0,3.0), (0,0,0,0.0)]},
        {'name': 'square', 'io': [(3,0,0,9.0), (-2,0,0,4.0), (5,0,0,25.0)]},
    ]

    synthesis_results = []
    for target in test_targets:
        # Optimize 5D PCA coefficients
        z_5d = torch.zeros(5, requires_grad=True, device=DEVICE, dtype=torch.float32)
        opt_inv = torch.optim.Adam([z_5d], lr=0.1)

        best_loss = float('inf')
        best_z = None
        for step in range(300):
            total_loss = 0
            for a1, a2, a3, expected in target['io']:
                args_norm = torch.tensor(
                    [(a1 - X_mean[5])/X_std[5],
                     (a2 - X_mean[6])/X_std[6],
                     (a3 - X_mean[7])/X_std[7]],
                    dtype=torch.float32, device=DEVICE)
                z_5d_norm = (z_5d - torch.tensor(X_mean[:5], device=DEVICE)) / \
                            torch.tensor(X_std[:5], device=DEVICE)
                inp = torch.cat([z_5d_norm, args_norm]).unsqueeze(0)
                pred = cpu_model(inp).squeeze()
                expected_norm = (expected - Y_mean) / Y_std
                total_loss += (pred - expected_norm) ** 2

            opt_inv.zero_grad()
            total_loss.backward()
            opt_inv.step()

            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_z = z_5d.detach().cpu().numpy().copy()

        # Decode the found vector
        code = decode_from_5d(best_z)

        # Verify
        verified = False
        try:
            ns = {}
            exec(compile(code, '<string>', 'exec'), ns)
            fn = [v for k,v in ns.items() if not k.startswith('_')][0]
            all_ok = True
            for a1, a2, a3, expected in target['io']:
                args = [a for a in [a1,a2,a3] if a != 0 or a1 == 0]
                try:
                    r = float(fn(*args[:2]) if a2 != 0 or a1 == 0 else fn(a1))
                    if abs(r - expected) > 0.01:
                        all_ok = False
                except Exception:
                    # Try single arg
                    try:
                        r = float(fn(a1))
                        if abs(r - expected) > 0.01:
                            all_ok = False
                    except Exception:
                        all_ok = False
            verified = all_ok
        except Exception:
            pass

        status = "SOLVED" if verified else "partial"
        print(f"  {target['name']:8s}: {code[:50]:50s} [{status}] (loss={best_loss:.4f})")
        synthesis_results.append({
            'target': target['name'], 'code': code,
            'verified': verified, 'loss': best_loss,
            'z_5d': best_z.tolist(),
        })

    n_solved = sum(1 for r in synthesis_results if r['verified'])
    n_total = len(synthesis_results)
    print(f"\n  === HOLOGRAPHIC INVERSE: {n_solved}/{n_total} targets solved! ===")

    # Compare: also try 64D optimization (P18 approach) as baseline
    print("\n--- Baseline: 64D Optimization (P18 approach) ---")
    # Build 64D Neural CPU for fair comparison
    X_64d = np.array([[*z_ast[d['func_idx']], *d['args']] for d in exec_data], dtype=np.float32)
    X64_mean, X64_std = X_64d.mean(0), X_64d.std(0) + 1e-8
    X64_norm = (X_64d - X64_mean) / X64_std
    X64_t = torch.tensor(X64_norm, dtype=torch.float32).to(DEVICE)

    cpu64 = nn.Sequential(
        nn.Linear(67, 128), nn.ReLU(),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, 1)
    ).to(DEVICE)

    opt64 = torch.optim.Adam(cpu64.parameters(), lr=1e-3)
    for epoch in range(300):
        pred = cpu64(X64_t).squeeze()
        loss = F.mse_loss(pred, Y_t)
        opt64.zero_grad(); loss.backward(); opt64.step()
    cpu64.eval()

    baseline_results = []
    for target in test_targets:
        z_64 = torch.zeros(64, requires_grad=True, device=DEVICE, dtype=torch.float32)
        opt_b = torch.optim.Adam([z_64], lr=0.1)
        best_loss_b = float('inf')
        best_z_b = None

        for step in range(300):
            total_loss = 0
            for a1, a2, a3, expected in target['io']:
                args_norm = torch.tensor(
                    [(a1 - X64_mean[64])/X64_std[64],
                     (a2 - X64_mean[65])/X64_std[65],
                     (a3 - X64_mean[66])/X64_std[66]],
                    dtype=torch.float32, device=DEVICE)
                z_norm = (z_64 - torch.tensor(X64_mean[:64], device=DEVICE)) / \
                          torch.tensor(X64_std[:64], device=DEVICE)
                inp = torch.cat([z_norm, args_norm]).unsqueeze(0)
                pred = cpu64(inp).squeeze()
                expected_norm = (expected - Y_mean) / Y_std
                total_loss += (pred - expected_norm) ** 2

            opt_b.zero_grad()
            total_loss.backward()
            opt_b.step()

            if total_loss.item() < best_loss_b:
                best_loss_b = total_loss.item()
                best_z_b = z_64.detach().cpu().numpy().copy()

        code_b = ""
        with torch.no_grad():
            z_t = torch.tensor(best_z_b.astype(np.float32)).unsqueeze(0).to(DEVICE)
            tokens = decoder(z_t)
            code_b = decode_tokens(tokens[0].cpu().numpy(), idx2char)

        verified_b = False
        try:
            ns = {}
            exec(compile(code_b, '<string>', 'exec'), ns)
            verified_b = True  # At least syntactically valid
        except Exception:
            pass

        print(f"  {target['name']:8s}: {code_b[:50]:50s} (loss={best_loss_b:.4f})")
        baseline_results.append({
            'target': target['name'], 'code': code_b,
            'loss': best_loss_b, 'valid': verified_b,
        })

    elapsed = time.time() - t0
    results = {
        'phase': 50, 'name': 'Holographic Inverse Synthesis',
        'limitation': 'Inverse Execution (P18 failed)',
        'n_exec_samples': len(exec_data),
        'neural_cpu_5d_r2': r2_5d,
        'synthesis_5d': synthesis_results,
        'synthesis_64d_baseline': baseline_results,
        'n_solved_5d': n_solved,
        'n_total': n_total,
        'solve_rate_5d': n_solved / max(n_total, 1),
        'elapsed': elapsed, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase50_holographic_inverse.json'), 'w',
              encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    # Figure
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. 5D vs 64D solve rates
    names = [r['target'] for r in synthesis_results]
    solved_5d = [1 if r['verified'] else 0 for r in synthesis_results]
    solved_64d = [1 if r.get('valid', False) else 0 for r in baseline_results]
    x = range(len(names))
    axes[0].bar([i-0.15 for i in x], solved_5d, 0.3, label='5D (Holographic)',
               color='#4CAF50', edgecolor='black')
    axes[0].bar([i+0.15 for i in x], solved_64d, 0.3, label='64D (P18 baseline)',
               color='#F44336', edgecolor='black')
    axes[0].set_xticks(list(x)); axes[0].set_xticklabels(names, rotation=45)
    axes[0].set_ylabel('Solved'); axes[0].set_title('Inverse Synthesis Success', fontweight='bold')
    axes[0].legend()

    # 2. Optimization loss comparison
    loss_5d = [r['loss'] for r in synthesis_results]
    loss_64d = [r['loss'] for r in baseline_results]
    axes[1].bar([i-0.15 for i in x], loss_5d, 0.3, label='5D', color='#4CAF50', edgecolor='black')
    axes[1].bar([i+0.15 for i in x], loss_64d, 0.3, label='64D', color='#F44336', edgecolor='black')
    axes[1].set_xticks(list(x)); axes[1].set_xticklabels(names, rotation=45)
    axes[1].set_ylabel('Final Loss'); axes[1].set_title('Optimization Loss', fontweight='bold')
    axes[1].legend()

    # 3. Summary
    axes[2].bar(['5D\n(Holographic)', '64D\n(P18 baseline)'],
               [n_solved/max(n_total,1)*100, sum(solved_64d)/max(len(solved_64d),1)*100],
               color=['#4CAF50', '#F44336'], edgecolor='black')
    axes[2].set_ylabel('% Solved'); axes[2].set_ylim(0, 110)
    axes[2].set_title('P18 Revenge:\nHolographic vs Unconstrained', fontweight='bold')
    for i, v in enumerate([n_solved/max(n_total,1)*100,
                           sum(solved_64d)/max(len(solved_64d),1)*100]):
        axes[2].text(i, v+3, f'{v:.0f}%', ha='center', fontweight='bold', fontsize=14)

    plt.suptitle('Phase 50: Holographic Inverse Synthesis\n'
                 'Limitation Breaker #2: Code from I/O via 5D Manifold',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase50_holographic_inverse.png'), dpi=150)
    plt.close()
    print(f"\nPhase 50 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
