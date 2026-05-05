"""
Phase 39: The Compiler's Memory (Iterated Compilation)
========================================================
What happens if we compile -> decompile -> compile -> ...
infinitely? Does information degrade like photocopying?
Or does it converge to a fixed point?
"""
import os, json, time, sys
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    print("=" * 60)
    print("Phase 39: The Compiler's Memory")
    print("Compile -> Decompile -> Compile -> ... what survives?")
    print("=" * 60)
    t0 = time.time()

    lat_file = os.path.join(DATA_DIR, 'rosetta_latents_v2.npz')
    if not os.path.exists(lat_file):
        lat_file = os.path.join(DATA_DIR, 'rosetta_latents.npz')
    latents = np.load(lat_file)
    z_ast, z_bc = latents['ast'], latents['bc']

    ds_file = os.path.join(DATA_DIR, 'rosetta_dataset_v2.json')
    if not os.path.exists(ds_file):
        ds_file = os.path.join(DATA_DIR, 'rosetta_dataset.json')
    with open(ds_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)['dataset']

    # Build W_compile and W_decompile
    from sklearn.linear_model import Ridge
    W_compile = Ridge(alpha=1.0).fit(z_ast, z_bc)
    W_decompile = Ridge(alpha=1.0).fit(z_bc, z_ast)

    # The round-trip operator: T = W_decompile * W_compile
    # T maps AST -> BC -> AST
    def round_trip(v_ast):
        v_bc = W_compile.predict(v_ast.reshape(1, -1))[0]
        v_ast2 = W_decompile.predict(v_bc.reshape(1, -1))[0]
        return v_ast2

    # Load decoder for visualization
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

    def gen(z):
        with torch.no_grad():
            z_t = torch.tensor(z.astype(np.float32)).unsqueeze(0).to(DEVICE)
            tokens = decoder(z_t)
            return decode_tokens(tokens[0].cpu().numpy(), idx2char)

    # === Iterated compilation ===
    print("\n--- Iterated Compilation-Decompilation ---")
    src_to_idx = {}
    for i, d in enumerate(dataset):
        if d['source'] not in src_to_idx:
            src_to_idx[d['source']] = i

    test_srcs = list(src_to_idx.keys())[:10]
    N_ITERATIONS = 50

    trajectories = []
    for src in test_srcs:
        idx = src_to_idx[src]
        v = z_ast[idx].copy()
        v_orig = v.copy()

        cos_history = [1.0]
        norm_history = [float(np.linalg.norm(v))]
        code_history = [gen(v)]

        for it in range(N_ITERATIONS):
            v = round_trip(v)
            cos = float(np.dot(v, v_orig) /
                       (np.linalg.norm(v) * np.linalg.norm(v_orig) + 1e-8))
            cos_history.append(cos)
            norm_history.append(float(np.linalg.norm(v)))

        code_final = gen(v)
        code_history.append(code_final)

        # Convergence: did the vector stabilize?
        converged = abs(cos_history[-1] - cos_history[-2]) < 0.001
        fixed_point_cos = cos_history[-1]

        print(f"  {src[:35]}")
        print(f"    iter0: {code_history[0][:30]}")
        print(f"    iter{N_ITERATIONS}: {code_final[:30]}")
        print(f"    cos(orig, final): {fixed_point_cos:.4f}")
        print(f"    Converged: {converged}")

        trajectories.append({
            'src': src, 'cos_history': cos_history,
            'norm_history': norm_history,
            'code_start': code_history[0], 'code_end': code_final,
            'fixed_point_cos': float(fixed_point_cos),
            'converged': converged,
        })

    # === Analysis: eigenvalues of round-trip operator ===
    print("\n--- Round-Trip Operator Analysis ---")
    # T = W_decompile.coef_ @ W_compile.coef_ (as matrices)
    T = W_decompile.coef_ @ W_compile.coef_  # (64, 64)
    eigvals = np.linalg.eigvals(T)
    eigvals_sorted = sorted(eigvals, key=lambda x: -abs(x))

    print(f"  Top 10 eigenvalues of T = Decompile * Compile:")
    for i in range(10):
        ev = eigvals_sorted[i]
        print(f"    lambda_{i} = {ev.real:.4f} + {ev.imag:.4f}i "
              f"(|lambda| = {abs(ev):.4f})")

    # Eigenvalues |lambda| = 1 -> preserved forever
    # |lambda| < 1 -> decays to zero
    # |lambda| > 1 -> amplified (unstable)
    n_stable = sum(1 for ev in eigvals if abs(abs(ev) - 1) < 0.1)
    n_decay = sum(1 for ev in eigvals if abs(ev) < 0.9)
    n_grow = sum(1 for ev in eigvals if abs(ev) > 1.1)
    print(f"\n  Stable (|lambda| ~ 1): {n_stable}")
    print(f"  Decaying (|lambda| < 0.9): {n_decay}")
    print(f"  Growing (|lambda| > 1.1): {n_grow}")

    # Fixed point: what code does T^inf converge to?
    avg_final_cos = float(np.mean([t['fixed_point_cos'] for t in trajectories]))
    n_converged = sum(1 for t in trajectories if t['converged'])
    print(f"\n  Average final cos(orig): {avg_final_cos:.4f}")
    print(f"  Converged: {n_converged}/{len(trajectories)}")

    elapsed = time.time() - t0
    results = {
        'phase': 39, 'name': "The Compiler's Memory",
        'n_iterations': N_ITERATIONS,
        'avg_final_cos': avg_final_cos,
        'n_converged': n_converged,
        'n_stable': n_stable, 'n_decay': n_decay, 'n_grow': n_grow,
        'top_eigenvalues': [{'real': float(e.real), 'imag': float(e.imag),
                            'mag': float(abs(e))} for e in eigvals_sorted[:10]],
        'trajectories': [{'src': t['src'], 'cos': t['fixed_point_cos'],
                         'start': t['code_start'], 'end': t['code_end']}
                        for t in trajectories],
        'elapsed': elapsed, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase39_memory.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Cosine trajectories
    for t in trajectories[:6]:
        label = t['src'][:20].replace('def f(', '').replace('): return ', '->')
        axes[0].plot(t['cos_history'], label=label, linewidth=1.5)
    axes[0].set_xlabel('Iteration (compile->decompile cycles)')
    axes[0].set_ylabel('Cosine to original')
    axes[0].set_title("Iterated Compilation\n(what survives?)", fontweight='bold')
    axes[0].legend(fontsize=6)

    # 2. Eigenvalue spectrum of T
    axes[1].scatter([e.real for e in eigvals], [e.imag for e in eigvals],
                   c=[abs(e) for e in eigvals], cmap='hot', s=60, edgecolor='black')
    circle = plt.Circle((0,0), 1, fill=False, color='blue', ls='--')
    axes[1].add_patch(circle)
    axes[1].set_xlabel('Real'); axes[1].set_ylabel('Imaginary')
    axes[1].set_title('Eigenvalues of T\n(inside unit circle = forgotten)',
                     fontweight='bold')
    axes[1].set_aspect('equal')

    # 3. Final cos distribution
    final_cos = [t['fixed_point_cos'] for t in trajectories]
    axes[2].hist(final_cos, bins=15, color='#9C27B0', edgecolor='black', alpha=0.8)
    axes[2].axvline(avg_final_cos, color='red', ls='--',
                   label=f'Mean={avg_final_cos:.3f}')
    axes[2].set_xlabel('Final cosine to original')
    axes[2].set_ylabel('Count')
    axes[2].set_title("Compiler's Memory\n(how much survives 50 cycles?)",
                     fontweight='bold')
    axes[2].legend()

    plt.suptitle("Phase 39: The Compiler's Memory\n"
                 "What information survives infinite compilation?",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase39_memory.png'), dpi=150)
    plt.close()
    print(f"\nPhase 39 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
