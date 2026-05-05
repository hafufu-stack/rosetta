"""
Phase 10: Decoder-based Semantic Arithmetic
=============================================
P5 failed at retrieval-based analogy. Can the decoder GENERATE
correct code from analogy vectors in the void?
"""
import os, json, time
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    print("=" * 60)
    print("Phase 10: Decoder-based Semantic Arithmetic")
    print("=" * 60)
    t0 = time.time()

    # Load data
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

    # Load decoder
    with open(os.path.join(DATA_DIR, 'char_vocab.json'), 'r') as f:
        vocab_data = json.load(f)
    char2idx = vocab_data['char2idx']
    idx2char = {int(i): c for c, i in char2idx.items()}
    V = len(char2idx)

    from experiments.phase9_generative_decompiler import CodeDecoder, decode_tokens
    decoder = CodeDecoder(64, V, hidden=128, max_len=80).to(DEVICE)
    decoder.load_state_dict(torch.load(os.path.join(DATA_DIR, 'decoder.pt'),
                                        map_location=DEVICE, weights_only=True))
    decoder.eval()

    # Source index
    src_to_idx = {}
    for i, d in enumerate(dataset):
        if d['source'] not in src_to_idx:
            src_to_idx[d['source']] = i

    # Define analogies
    analogies = [
        ("def f(x, y): return x + y", "def f(x, y): return x - y",
         "def f(x, y): return x * y", "def f(x, y): return x / y",
         "add:sub :: mul:?div"),
        ("def f(x, y): return x + y", "def f(x, y): return x * y",
         "def f(x, y): return x - y", "def f(x, y): return x / y",
         "add:mul :: sub:?div"),
        ("def f(x, y): return x > y", "def f(x, y): return x < y",
         "def f(x, y): return x >= y", "def f(x, y): return x <= y",
         "gt:lt :: gte:?lte"),
        ("def f(x): return x + 1", "def f(x): return x - 1",
         "def f(x): return x * 2", "def f(x): return x / 2",
         "inc:dec :: double:?halve"),
        ("def f(x): return -x", "def f(x): return abs(x)",
         "def f(x): return x * x", "def f(x): return x * x * x",
         "neg:abs :: square:?cube"),
        ("def f(s): return s.upper()", "def f(s): return s.lower()",
         "def f(s): return len(s)", "def f(s): return s[::-1]",
         "upper:lower :: len:?reverse"),
    ]

    results_list = []
    n_semantic_correct = 0
    n_exact_correct = 0
    total = 0

    print("\n--- Generating code from analogy vectors ---")
    for src_a, src_b, src_c, src_d, desc in analogies:
        if not all(s in src_to_idx for s in [src_a, src_b, src_c, src_d]):
            print(f"  SKIP {desc}")
            continue
        total += 1
        ia, ib, ic, id_ = (src_to_idx[s] for s in [src_a, src_b, src_c, src_d])

        # Test on all three modalities
        for modal_name, z in [('AST', z_ast), ('NL', z_nl), ('BC', z_bc)]:
            v_result = z[ia] - z[ib] + z[ic]
            v_result = v_result / (np.linalg.norm(v_result) + 1e-8)

            with torch.no_grad():
                z_t = torch.tensor(v_result, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                gen_tokens = decoder(z_t)
                gen_code = decode_tokens(gen_tokens[0].cpu().numpy(), idx2char)

            exact = gen_code.strip() == src_d.strip()
            # Semantic check: does it contain the key operation?
            key_op = src_d.split('return ')[-1].strip() if 'return ' in src_d else ''
            semantic = key_op in gen_code if key_op else False

            if modal_name == 'AST':
                if exact: n_exact_correct += 1
                if semantic: n_semantic_correct += 1

            results_list.append({
                'analogy': desc, 'modality': modal_name,
                'expected': src_d, 'generated': gen_code,
                'exact': exact, 'semantic': semantic,
            })

        # Print results for this analogy
        ast_gen = [r for r in results_list if r['analogy']==desc and r['modality']=='AST']
        if ast_gen:
            r = ast_gen[0]
            status = "EXACT" if r['exact'] else ("SEM" if r['semantic'] else "X")
            print(f"\n  {desc}")
            print(f"    Expected: {src_d}")
            print(f"    AST gen:  {r['generated'][:60]} [{status}]")
            nl_r = [r2 for r2 in results_list if r2['analogy']==desc and r2['modality']=='NL']
            if nl_r:
                print(f"    NL gen:   {nl_r[0]['generated'][:60]}")

    # Also test: direct generation from known AST vectors (sanity check)
    print("\n--- Sanity check: direct AST vector decoding ---")
    sanity_srcs = list(src_to_idx.keys())[:10]
    sanity_exact = 0
    for src in sanity_srcs:
        idx = src_to_idx[src]
        with torch.no_grad():
            z_t = torch.tensor(z_ast[idx], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            gen = decoder(z_t)
            gen_code = decode_tokens(gen[0].cpu().numpy(), idx2char)
        if gen_code.strip() == src.strip():
            sanity_exact += 1

    print(f"  Sanity exact: {sanity_exact}/10")

    elapsed = time.time() - t0
    results = {
        'phase': 10, 'name': 'Decoder-based Semantic Arithmetic',
        'total_analogies': total,
        'ast_exact_correct': n_exact_correct,
        'ast_semantic_correct': n_semantic_correct,
        'sanity_exact': sanity_exact,
        'details': results_list,
        'elapsed': elapsed, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase10_semantic_arithmetic.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Figure
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    bars = axes[0].bar(['Exact Match', 'Semantic Match', 'Sanity Check'],
                       [n_exact_correct/max(total,1),
                        n_semantic_correct/max(total,1),
                        sanity_exact/10],
                       color=['#E91E63','#4CAF50','#2196F3'], edgecolor='black')
    for b, v in zip(bars, [n_exact_correct/max(total,1),
                           n_semantic_correct/max(total,1), sanity_exact/10]):
        axes[0].text(b.get_x()+b.get_width()/2, v+0.02, f'{v:.0%}',
                     ha='center', fontweight='bold', fontsize=14)
    axes[0].set_ylabel('Rate'); axes[0].set_ylim(0, 1.1)
    axes[0].set_title('Analogy via Generation', fontweight='bold')

    # Comparison: P5 retrieval vs P10 generation
    axes[1].bar(['P5 Retrieval\n(0%)', 'P10 Generation\n(Semantic)'],
                [0, n_semantic_correct/max(total,1)],
                color=['#F44336','#4CAF50'], edgecolor='black')
    axes[1].set_ylabel('Accuracy'); axes[1].set_ylim(0, 1.1)
    axes[1].set_title('Retrieval vs Generation', fontweight='bold')

    plt.suptitle('Phase 10: Decoder-based Semantic Arithmetic',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase10_semantic_arithmetic.png'), dpi=150)
    plt.close()
    print(f"\nPhase 10 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
