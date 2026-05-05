"""
Phase 14: End-to-End Neural Compiler
======================================
NL text -> matrix multiply -> binary space -> decoder -> Python code.
No AST, no traditional compiler. Pure neural path.
"""
import os, json, time
import numpy as np
import torch
from sklearn.linear_model import Ridge

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    print("=" * 60)
    print("Phase 14: End-to-End Neural Compiler")
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
    z_nl, z_ast, z_bc = latents['nl'], latents['ast'], latents['bc']
    N, D = z_nl.shape

    with open(os.path.join(DATA_DIR, 'char_vocab.json'), 'r') as f:
        vd = json.load(f)
    idx2char = {int(i): c for c, i in vd['char2idx'].items()}
    V = len(vd['char2idx'])

    from experiments.phase9_generative_decompiler import CodeDecoder, decode_tokens
    decoder = CodeDecoder(64, V, hidden=128, max_len=80).to(DEVICE)
    decoder.load_state_dict(torch.load(os.path.join(DATA_DIR, 'decoder.pt'),
                                        map_location=DEVICE, weights_only=True))
    decoder.eval()

    # Learn translation matrices
    n_train = int(N * 0.8)
    perm = np.random.RandomState(42).permutation(N)
    train_i, test_i = perm[:n_train], perm[n_train:]

    # Path 1: NL -> AST (direct)
    W_nl_ast = Ridge(alpha=1.0).fit(z_nl[train_i], z_ast[train_i])
    # Path 2: NL -> Bin -> AST (two hops)
    W_nl_bin = Ridge(alpha=1.0).fit(z_nl[train_i], z_bc[train_i])
    W_bin_ast = Ridge(alpha=1.0).fit(z_bc[train_i], z_ast[train_i])

    def gen(z_vec):
        with torch.no_grad():
            z_t = torch.tensor(z_vec, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            tokens = decoder(z_t)
            return decode_tokens(tokens[0].cpu().numpy(), idx2char)

    # === Test: NL -> Code (direct path) ===
    print("\n--- Path 1: NL -> W_nl_ast -> Decoder -> Code ---")
    test_nls = []
    for i in test_i[:50]:
        nl = dataset[i]['nl']
        src = dataset[i]['source']
        if (nl, src) not in [(t['nl'], t['src']) for t in test_nls]:
            test_nls.append({'nl': nl, 'src': src, 'idx': i})
        if len(test_nls) >= 30:
            break

    path1_exact, path1_sem, total = 0, 0, 0
    path1_results = []
    for t in test_nls:
        z_nl_vec = z_nl[t['idx']]
        z_ast_pred = W_nl_ast.predict(z_nl_vec.reshape(1, -1))[0]
        gen_code = gen(z_ast_pred)
        exact = gen_code.strip() == t['src'].strip()
        # Semantic check
        true_op = t['src'].split('return ')[-1].strip() if 'return ' in t['src'] else ''
        gen_op = gen_code.split('return ')[-1].strip() if 'return ' in gen_code else ''
        for v in 'xyanmpqbvijkl':
            true_op = true_op.replace(v, '_')
            gen_op = gen_op.replace(v, '_')
        semantic = true_op == gen_op if true_op else False
        if exact: path1_exact += 1
        if semantic: path1_sem += 1
        total += 1
        path1_results.append({'nl': t['nl'], 'true': t['src'],
                              'generated': gen_code, 'exact': exact, 'semantic': semantic})
        if total <= 10:
            status = "EXACT" if exact else ("SEM" if semantic else "X")
            print(f"  [{status}] '{t['nl'][:40]}' -> {gen_code[:45]}")

    print(f"  Path 1: exact={path1_exact}/{total} ({path1_exact/total:.0%}), "
          f"semantic={path1_sem}/{total} ({path1_sem/total:.0%})")

    # === Path 2: NL -> Bin -> AST -> Decoder -> Code (two-hop) ===
    print("\n--- Path 2: NL -> W_nl_bin -> W_bin_ast -> Decoder -> Code ---")
    path2_exact, path2_sem = 0, 0
    for t in test_nls:
        z_nl_vec = z_nl[t['idx']]
        z_bin_pred = W_nl_bin.predict(z_nl_vec.reshape(1, -1))[0]
        z_ast_pred2 = W_bin_ast.predict(z_bin_pred.reshape(1, -1))[0]
        gen_code = gen(z_ast_pred2)
        exact = gen_code.strip() == t['src'].strip()
        true_op = t['src'].split('return ')[-1].strip() if 'return ' in t['src'] else ''
        gen_op = gen_code.split('return ')[-1].strip() if 'return ' in gen_code else ''
        for v in 'xyanmpqbvijkl':
            true_op = true_op.replace(v, '_')
            gen_op = gen_op.replace(v, '_')
        semantic = true_op == gen_op if true_op else False
        if exact: path2_exact += 1
        if semantic: path2_sem += 1

    print(f"  Path 2: exact={path2_exact}/{total} ({path2_exact/total:.0%}), "
          f"semantic={path2_sem}/{total} ({path2_sem/total:.0%})")

    # === Path 3: NL directly to Decoder (bypass all matrices) ===
    print("\n--- Path 3: NL vector -> Decoder (direct, no matrix) ---")
    path3_exact, path3_sem = 0, 0
    for t in test_nls:
        gen_code = gen(z_nl[t['idx']])
        exact = gen_code.strip() == t['src'].strip()
        true_op = t['src'].split('return ')[-1].strip() if 'return ' in t['src'] else ''
        gen_op = gen_code.split('return ')[-1].strip() if 'return ' in gen_code else ''
        for v in 'xyanmpqbvijkl':
            true_op = true_op.replace(v, '_')
            gen_op = gen_op.replace(v, '_')
        semantic = true_op == gen_op if true_op else False
        if exact: path3_exact += 1
        if semantic: path3_sem += 1

    print(f"  Path 3: exact={path3_exact}/{total} ({path3_exact/total:.0%}), "
          f"semantic={path3_sem}/{total} ({path3_sem/total:.0%})")

    elapsed = time.time() - t0
    results = {
        'phase': 14, 'name': 'End-to-End Neural Compiler',
        'total_tests': total,
        'path1_direct': {'exact': path1_exact/total, 'semantic': path1_sem/total},
        'path2_twohop': {'exact': path2_exact/total, 'semantic': path2_sem/total},
        'path3_raw_nl': {'exact': path3_exact/total, 'semantic': path3_sem/total},
        'examples': path1_results[:10],
        'elapsed': elapsed, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase14_neural_compiler.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Figure
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    paths = ['Path 1\nNL->AST->Dec', 'Path 2\nNL->Bin->AST->Dec', 'Path 3\nNL->Dec (raw)']
    exact_vals = [path1_exact/total, path2_exact/total, path3_exact/total]
    sem_vals = [path1_sem/total, path2_sem/total, path3_sem/total]
    x = np.arange(3)
    b1 = ax.bar(x-0.15, exact_vals, 0.3, label='Exact', color='#E91E63', edgecolor='black')
    b2 = ax.bar(x+0.15, sem_vals, 0.3, label='Semantic', color='#4CAF50', edgecolor='black')
    for b, v in zip(list(b1)+list(b2), exact_vals+sem_vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.02, f'{v:.0%}',
                ha='center', fontweight='bold', fontsize=12)
    ax.set_xticks(x); ax.set_xticklabels(paths, fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=12); ax.set_ylim(0, 1.1)
    ax.legend(fontsize=12)
    ax.set_title('Phase 14: End-to-End Neural Compiler\n'
                 'Human Language -> Matrix Multiply -> Python Code',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase14_neural_compiler.png'), dpi=150)
    plt.close()
    print(f"\nPhase 14 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
