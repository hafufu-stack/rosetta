"""
Phase 18: Neural Inverse Execution
=====================================
Given (input, expected_output), backprop through Neural CPU
to find the function vector, then decode it to source code.
TDD via calculus!
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
    print("Phase 18: Neural Inverse Execution")
    print("=" * 60)
    t0 = time.time()

    # Load dataset and latents
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

    # Load Neural CPU
    from experiments.phase16_neural_cpu import NeuralCPU
    cpu_model = NeuralCPU(64, 2, hidden=128).to(DEVICE)
    cpu_path = os.path.join(RESULTS_DIR, 'phase16_neural_cpu.json')

    # Re-train a quick Neural CPU if needed (with the fixed exec)
    print("Training Neural CPU for inverse execution...")
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

    if len(exec_data) < 100:
        print(f"Only {len(exec_data)} samples, not enough for inverse exec")
        return {'phase': 18, 'error': 'insufficient data'}

    func_t = torch.tensor(np.array([z_ast[d['func_idx']] for d in exec_data]), dtype=torch.float32)
    args_t = torch.tensor(np.array([d['args'] for d in exec_data]), dtype=torch.float32)
    res_arr = np.array([d['result'] for d in exec_data], dtype=np.float32)
    r_mean, r_std = float(res_arr.mean()), float(res_arr.std()) + 1e-8
    res_t = torch.tensor((res_arr - r_mean) / r_std, dtype=torch.float32)

    # Train
    Ntrain = int(len(exec_data) * 0.8)
    optimizer = torch.optim.Adam(cpu_model.parameters(), lr=1e-3)
    for epoch in range(200):
        perm = torch.randperm(Ntrain)
        eloss, nb = 0, 0
        cpu_model.train()
        for i in range(0, Ntrain, 256):
            idx = perm[i:i+256]
            pred = cpu_model(func_t[idx].to(DEVICE), args_t[idx].to(DEVICE))
            loss = F.mse_loss(pred, res_t[idx].to(DEVICE))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
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

    # === Inverse Execution: find function from I/O examples ===
    print("\n--- Inverse Execution ---")
    test_specs = [
        # (test_cases: [(args, expected_output)], description, ground_truth)
        ([(3, 5, 8), (7, 2, 9), (10, 4, 14)], "x + y", "def f(x, y): return x + y"),
        ([(3, 5, 15), (7, 2, 14), (4, 6, 24)], "x * y", "def f(x, y): return x * y"),
        ([(10, 3, 7), (8, 5, 3), (15, 6, 9)], "x - y", "def f(x, y): return x - y"),
        ([(3, 5, 0), (7, 2, 1), (4, 4, 0)], "x > y", "def f(x, y): return x > y"),
        ([(5, 0, 10), (3, 0, 6), (8, 0, 16)], "x * 2", "def f(x): return x * 2"),
    ]

    inv_results = []
    for test_cases, desc, truth in test_specs:
        # Initialize random function vector
        z_func = torch.randn(1, D, device=DEVICE, requires_grad=True)
        inv_opt = torch.optim.Adam([z_func], lr=0.1)

        for step in range(500):
            total_loss = torch.tensor(0.0, device=DEVICE)
            for a1, a2, expected in test_cases:
                args = torch.tensor([[a1/20.0, a2/20.0]], dtype=torch.float32, device=DEVICE)
                target = torch.tensor([(expected - r_mean)/r_std], dtype=torch.float32, device=DEVICE)
                pred = cpu_model(z_func, args)
                total_loss = total_loss + F.mse_loss(pred, target)

            inv_opt.zero_grad()
            total_loss.backward()
            inv_opt.step()

        z_found = z_func.detach().cpu().numpy()[0]
        generated = gen(z_found)

        # Also find nearest known function
        sims = z_ast @ z_found / (np.linalg.norm(z_ast, axis=1) * np.linalg.norm(z_found) + 1e-8)
        nearest_idx = int(np.argmax(sims))
        nearest_src = dataset[nearest_idx]['source']

        # Check if generated code is semantically correct
        key_op = truth.split('return ')[-1].strip()
        gen_op = generated.split('return ')[-1].strip() if 'return ' in generated else ''
        for v in 'xyanmpqb':
            key_op = key_op.replace(v, '_')
            gen_op = gen_op.replace(v, '_')
        semantic = key_op.replace(' ','') == gen_op.replace(' ','')

        print(f"\n  Spec: {desc} | Examples: {test_cases[0]}")
        print(f"    Generated: {generated[:60]}")
        print(f"    Nearest:   {nearest_src[:60]}")
        print(f"    Truth:     {truth}")
        print(f"    Semantic:  {'OK' if semantic else 'X'}")

        inv_results.append({
            'desc': desc, 'truth': truth, 'generated': generated,
            'nearest': nearest_src, 'semantic': semantic,
            'final_loss': float(total_loss.item()),
        })

    n_correct = sum(1 for r in inv_results if r['semantic'])
    print(f"\n  Inverse Exec: {n_correct}/{len(inv_results)} semantic matches")

    elapsed = time.time() - t0
    results = {
        'phase': 18, 'name': 'Neural Inverse Execution',
        'total_specs': len(inv_results),
        'semantic_correct': n_correct,
        'details': inv_results,
        'elapsed': elapsed, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase18_inverse_exec.json'), 'w') as f:
        json.dump(results, f, indent=2)

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    names = [r['desc'] for r in inv_results]
    colors = ['#4CAF50' if r['semantic'] else '#F44336' for r in inv_results]
    ax.barh(names, [1 if r['semantic'] else 0 for r in inv_results],
            color=colors, edgecolor='black')
    ax.set_xlabel('Correct'); ax.set_title('Phase 18: Neural Inverse Execution\n'
        'Find code from I/O examples via backprop', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase18_inverse_exec.png'), dpi=150)
    plt.close()
    print(f"\nPhase 18 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
