"""
Phase 21: Manifold-Guided Inverse Synthesis
=============================================
Fix P18's adversarial collapse by adding a manifold prior.
L_total = L_io + lambda * ||V_f - V_centroid||^2
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
    print("Phase 21: Manifold-Guided Inverse Synthesis")
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
    N, D = z_ast.shape

    # Compute manifold statistics
    centroid = z_ast.mean(axis=0)
    cov = np.cov(z_ast.T)
    std_per_dim = np.sqrt(np.diag(cov))
    centroid_t = torch.tensor(centroid, dtype=torch.float32, device=DEVICE)
    std_t = torch.tensor(std_per_dim, dtype=torch.float32, device=DEVICE)

    # Build index for nearest-neighbor lookup
    z_ast_norm = z_ast / (np.linalg.norm(z_ast, axis=1, keepdims=True) + 1e-8)

    # Train Neural CPU
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
                exec_data.append({'func_idx': i, 'args': args, 'result': result, 'source': src})
            except: continue

    func_t = torch.tensor(np.array([z_ast[d['func_idx']] for d in exec_data]), dtype=torch.float32)
    args_t = torch.tensor(np.array([d['args'] for d in exec_data]), dtype=torch.float32)
    res_arr = np.array([d['result'] for d in exec_data], dtype=np.float32)
    r_mean, r_std = float(res_arr.mean()), float(res_arr.std()) + 1e-8
    res_t = torch.tensor((res_arr - r_mean) / r_std, dtype=torch.float32)

    cpu_model = NeuralCPU(64, 2, hidden=128).to(DEVICE)
    opt = torch.optim.Adam(cpu_model.parameters(), lr=1e-3)
    Ntrain = int(len(exec_data) * 0.8)
    for epoch in range(200):
        pe = torch.randperm(Ntrain)
        eloss, nb = 0, 0
        cpu_model.train()
        for i in range(0, Ntrain, 256):
            idx = pe[i:i+256]
            pred = cpu_model(func_t[idx].to(DEVICE), args_t[idx].to(DEVICE))
            loss = F.mse_loss(pred, res_t[idx].to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
            eloss += loss.item(); nb += 1
        if (epoch+1) % 100 == 0:
            print(f"  CPU Epoch {epoch+1}/200: loss={eloss/max(nb,1):.4f}")

    cpu_model.eval()
    for p in cpu_model.parameters():
        p.requires_grad_(False)

    # Load decoder
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

    def gen(z_vec):
        with torch.no_grad():
            z_t = torch.tensor(z_vec, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            tokens = decoder(z_t)
            return decode_tokens(tokens[0].cpu().numpy(), idx2char)

    # === Manifold-Guided Inverse ===
    test_specs = [
        ([(3, 5, 8), (7, 2, 9), (10, 4, 14), (1, 1, 2), (6, 3, 9)],
         "x + y", "def f(x, y): return x + y"),
        ([(3, 5, 15), (7, 2, 14), (4, 6, 24), (2, 8, 16), (5, 5, 25)],
         "x * y", "def f(x, y): return x * y"),
        ([(10, 3, 7), (8, 5, 3), (15, 6, 9), (20, 1, 19), (7, 7, 0)],
         "x - y", "def f(x, y): return x - y"),
        ([(5, 0, 10), (3, 0, 6), (8, 0, 16), (1, 0, 2), (10, 0, 20)],
         "x * 2", "def f(x): return x * 2"),
        ([(4, 0, 16), (3, 0, 9), (5, 0, 25), (2, 0, 4), (7, 0, 49)],
         "x^2", "def f(x): return x * x"),
    ]

    LAMBDA_MANIFOLD = 0.5
    inv_results = []
    print("\n--- Manifold-Guided Inverse Execution ---")

    for test_cases, desc, truth in test_specs:
        # Initialize from centroid (not random!)
        z_func = centroid_t.clone().unsqueeze(0).detach().requires_grad_(True)
        inv_opt = torch.optim.Adam([z_func], lr=0.05)

        best_loss = float('inf')
        best_z = None
        for step in range(1000):
            # I/O loss
            io_loss = torch.tensor(0.0, device=DEVICE)
            for a1, a2, expected in test_cases:
                args = torch.tensor([[a1/20.0, a2/20.0]], dtype=torch.float32, device=DEVICE)
                target = torch.tensor([(expected - r_mean)/r_std], dtype=torch.float32, device=DEVICE)
                pred = cpu_model(z_func, args)
                io_loss = io_loss + F.mse_loss(pred, target)

            # Manifold prior: stay close to data distribution
            manifold_loss = torch.mean(((z_func - centroid_t) / (std_t + 1e-8)) ** 2)

            total_loss = io_loss + LAMBDA_MANIFOLD * manifold_loss

            inv_opt.zero_grad()
            total_loss.backward()
            inv_opt.step()

            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_z = z_func.detach().cpu().numpy()[0].copy()

        # Decode best vector
        generated = gen(best_z)

        # Also find nearest known function
        sims = z_ast_norm @ (best_z / (np.linalg.norm(best_z) + 1e-8))
        nearest_idx = int(np.argmax(sims))
        nearest_src = dataset[nearest_idx]['source']
        nearest_code = gen(z_ast[nearest_idx])

        # Semantic check
        key_op = truth.split('return ')[-1].strip()
        gen_op = generated.split('return ')[-1].strip() if 'return ' in generated else ''
        for v in 'xyanmpqb':
            key_op = key_op.replace(v, '_')
            gen_op = gen_op.replace(v, '_')
        semantic = key_op.replace(' ','') == gen_op.replace(' ','')

        print(f"\n  Spec: {desc} | {len(test_cases)} I/O examples")
        print(f"    Generated:  {generated[:60]}")
        print(f"    Nearest:    {nearest_src[:60]}")
        print(f"    Nearest dec:{nearest_code[:60]}")
        print(f"    Truth:      {truth}")
        print(f"    Semantic:   {'OK' if semantic else 'X'} (loss={best_loss:.4f})")

        inv_results.append({
            'desc': desc, 'truth': truth, 'generated': generated,
            'nearest': nearest_src, 'nearest_decoded': nearest_code,
            'semantic': semantic, 'final_loss': best_loss,
        })

    n_correct = sum(1 for r in inv_results if r['semantic'])
    n_nearest_match = sum(1 for r in inv_results
                          if r['truth'].split('return ')[-1].strip() in r['nearest'])
    print(f"\n  Direct decode: {n_correct}/{len(inv_results)} semantic matches")
    print(f"  Nearest match: {n_nearest_match}/{len(inv_results)}")

    elapsed = time.time() - t0
    results = {
        'phase': 21, 'name': 'Manifold-Guided Inverse Synthesis',
        'total': len(inv_results), 'semantic_correct': n_correct,
        'nearest_correct': n_nearest_match,
        'lambda': LAMBDA_MANIFOLD,
        'details': inv_results,
        'elapsed': elapsed, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase21_manifold_inverse.json'), 'w') as f:
        json.dump(results, f, indent=2)

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    names = [r['desc'] for r in inv_results]
    # P18 vs P21
    axes[0].bar(['P18\n(No Prior)', 'P21\n(Manifold)'],
                [0, n_correct/len(inv_results)],
                color=['#F44336','#4CAF50'], edgecolor='black')
    axes[0].set_ylabel('Semantic Accuracy'); axes[0].set_ylim(0, 1.1)
    axes[0].set_title('Inverse Synthesis: P18 vs P21', fontweight='bold')

    colors = ['#4CAF50' if r['semantic'] else '#F44336' for r in inv_results]
    axes[1].barh(names, [1 if r['semantic'] else 0 for r in inv_results],
                color=colors, edgecolor='black')
    for i, r in enumerate(inv_results):
        axes[1].text(0.02, i, f"= {r['generated'][:35]}", va='center', fontsize=9,
                    fontfamily='monospace')
    axes[1].set_title('Per-Spec Results', fontweight='bold')
    plt.suptitle('Phase 21: Manifold-Guided Inverse Synthesis\n'
                 'TDD via Calculus + Manifold Prior', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase21_manifold_inverse.png'), dpi=150)
    plt.close()
    print(f"\nPhase 21 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
