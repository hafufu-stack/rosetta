"""
Phase 49: The Topology of Control Flow
=========================================
LIMITATION BREAKER #1: Scale (single-expression -> multi-line)

Can the Rosetta Space encode if-else, loops, and stateful programs?
If so, does "control flow" appear as a new dimension in the manifold,
or does it fold into the existing 5 elements?
"""
import os, json, time, sys, dis, io, ast
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==============================================================
# Extended Function Templates (control flow, loops, state)
# ==============================================================
CONTROL_FLOW_FUNCTIONS = [
    # if-else (branching)
    ("def f(x): return x if x > 0 else -x", "absolute value via conditional",
     "conditional"),
    ("def f(x, y): return x if x > y else y", "maximum of two numbers",
     "conditional"),
    ("def f(x, y): return x if x < y else y", "minimum of two numbers",
     "conditional"),
    ("def f(x): return 'positive' if x > 0 else 'non-positive'",
     "classify sign as string", "conditional"),
    ("def f(x): return x * 2 if x > 0 else x * -1",
     "double if positive else negate", "conditional"),
    ("def f(x, y): return x + y if x > 0 else x - y",
     "add if positive else subtract", "conditional"),
    ("def f(x): return 1 if x > 0 else -1 if x < 0 else 0",
     "sign function", "conditional"),
    ("def f(x): return x ** 2 if x >= 0 else -(x ** 2)",
     "signed square", "conditional"),

    # Loops via comprehensions and builtins (single-expression encodable)
    ("def f(n): return sum(range(n))", "sum of 0 to n-1", "loop"),
    ("def f(n): return sum(i*i for i in range(n))", "sum of squares", "loop"),
    ("def f(n): return sum(1 for i in range(n) if i % 2 == 0)",
     "count even numbers", "loop"),
    ("def f(lst): return [x*2 for x in lst]", "double each element", "loop"),
    ("def f(lst): return [x for x in lst if x > 0]",
     "filter positive elements", "loop"),
    ("def f(lst): return sum(x for x in lst if x > 0)",
     "sum of positive elements", "loop"),
    ("def f(s): return ''.join(reversed(s))", "reverse a string", "loop"),
    ("def f(n): return list(range(1, n+1))", "list from 1 to n", "loop"),

    # Higher-order / functional (composition)
    ("def f(lst): return list(map(abs, lst))", "absolute value of each", "functional"),
    ("def f(lst): return list(filter(lambda x: x > 0, lst))",
     "filter positives via lambda", "functional"),
    ("def f(lst): return sorted(lst, reverse=True)",
     "sort descending", "functional"),
    ("def f(lst): return len([x for x in lst if x > 0])",
     "count positives", "functional"),

    # Multi-operation (state-like)
    ("def f(x, y): return (x + y) * (x - y)", "difference of squares", "multi_op"),
    ("def f(x): return (x + 1) * (x - 1)", "x squared minus one", "multi_op"),
    ("def f(a, b, c): return (-b + (b**2 - 4*a*c)**0.5) / (2*a)",
     "quadratic formula positive root", "multi_op"),
    ("def f(x, y, z): return x + y + z", "sum of three", "multi_op"),
    ("def f(x, y, z): return max(x, max(y, z))", "max of three", "multi_op"),
]


def main():
    print("=" * 60)
    print("Phase 49: The Topology of Control Flow")
    print("LIMITATION BREAKER #1: Scale")
    print("=" * 60)
    t0 = time.time()

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    # Load existing data
    ds_file = os.path.join(DATA_DIR, 'rosetta_dataset_v2.json')
    if not os.path.exists(ds_file):
        ds_file = os.path.join(DATA_DIR, 'rosetta_dataset.json')
    with open(ds_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)['dataset']

    lat_file = os.path.join(DATA_DIR, 'rosetta_latents_v2.npz')
    if not os.path.exists(lat_file):
        lat_file = os.path.join(DATA_DIR, 'rosetta_latents.npz')
    latents = np.load(lat_file)
    z_ast_orig = latents['ast']
    z_nl_orig = latents['nl']

    # Generate new triplets for control flow functions
    print("\n--- Generating Control Flow Triplets ---")
    new_triplets = []
    for src, nl_desc, category in CONTROL_FLOW_FUNCTIONS:
        try:
            # AST representation
            tree = ast.parse(src)
            ast_repr = ast.dump(tree)

            # Bytecode
            code = compile(src, '<string>', 'exec')
            # Execute to get function object
            ns = {}
            exec(code, ns)
            func_name = [k for k in ns if not k.startswith('_')][0]
            func = ns[func_name]
            bc_out = io.StringIO()
            dis.dis(func, file=bc_out)
            bytecode = bc_out.getvalue()

            new_triplets.append({
                'source': src, 'nl': nl_desc, 'ast_repr': ast_repr,
                'bytecode': bytecode, 'category': category,
            })
            print(f"  OK: {src[:50]} [{category}]")
        except Exception as e:
            print(f"  FAIL: {src[:50]}: {e}")

    print(f"  Generated {len(new_triplets)} control flow triplets")

    # Encode new functions using the same encoding scheme as original
    # Use TF-IDF-like bag-of-words on AST + bytecode, then project via
    # the same linear layer dimensions as original embeddings
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Collect all sources (original + new)
    orig_sources = [d['source'] for d in dataset]
    all_sources = orig_sources + [t['source'] for t in new_triplets]

    # Encode AST: use source code characters as proxy (same as original)
    # We embed new functions by finding their nearest neighbors in existing space
    # This is the most robust approach without retraining the encoder
    print("\n--- Encoding via Nearest-Neighbor Projection ---")

    from sklearn.metrics.pairwise import cosine_similarity

    # Create simple bag-of-ops features for matching
    def extract_features(src):
        ops = ['+', '-', '*', '/', '%', '**', '>', '<', '==', '!=', '>=', '<=',
               'if', 'else', 'for', 'in', 'range', 'sum', 'map', 'filter',
               'lambda', 'sorted', 'reversed', 'abs', 'len', 'max', 'min',
               '.upper', '.lower', '.strip', 'list', 'return']
        return np.array([float(op in src) for op in ops])

    orig_features = np.array([extract_features(s) for s in orig_sources])
    new_features = np.array([extract_features(t['source']) for t in new_triplets])

    # For each new function, find k nearest originals and interpolate
    K = 5
    new_z_ast = []
    new_z_nl = []
    for i, feat in enumerate(new_features):
        sims = cosine_similarity(feat.reshape(1, -1), orig_features)[0]
        top_k = np.argsort(sims)[-K:]
        weights = sims[top_k]
        weights = weights / (weights.sum() + 1e-8)
        z_new = np.sum(z_ast_orig[top_k] * weights[:, None], axis=0)
        z_nl_new = np.sum(z_nl_orig[top_k] * weights[:, None], axis=0)
        new_z_ast.append(z_new)
        new_z_nl.append(z_nl_new)
        nn_src = orig_sources[top_k[-1]]
        print(f"  {new_triplets[i]['source'][:40]} -> NN: {nn_src[:40]} (sim={sims[top_k[-1]]:.3f})")

    new_z_ast = np.array(new_z_ast)
    new_z_nl = np.array(new_z_nl)

    # Combine original + new
    z_combined = np.vstack([z_ast_orig, new_z_ast])
    labels_orig = ['single_expr'] * len(z_ast_orig)
    labels_new = [t['category'] for t in new_triplets]
    all_labels = labels_orig + labels_new

    # PCA on combined space
    from sklearn.decomposition import PCA
    print("\n--- PCA on Extended Space ---")
    pca = PCA(n_components=10)
    z_pca = pca.fit_transform(z_combined)

    print("  Variance explained:")
    for i in range(10):
        print(f"    PC{i}: {pca.explained_variance_ratio_[i]:.4f} "
              f"({sum(pca.explained_variance_ratio_[:i+1])*100:.1f}% cumul)")

    # Analyze: do control flow functions cluster differently?
    n_orig = len(z_ast_orig)
    z_pca_orig = z_pca[:n_orig]
    z_pca_new = z_pca[n_orig:]

    print("\n--- Control Flow vs Single-Expression ---")
    orig_mean = np.mean(z_pca_orig, axis=0)
    for cat in ['conditional', 'loop', 'functional', 'multi_op']:
        idxs = [i for i, l in enumerate(labels_new) if l == cat]
        if idxs:
            cat_mean = np.mean(z_pca_new[idxs], axis=0)
            shift = cat_mean - orig_mean
            dominant_pc = np.argmax(np.abs(shift[:6]))
            print(f"  {cat:15s}: shift from origin = [{', '.join(f'{s:.2f}' for s in shift[:6])}]")
            print(f"                  Dominant axis: PC{dominant_pc} (shift={shift[dominant_pc]:.3f})")

    # Does a 6th dimension emerge?
    pca_orig_only = PCA(n_components=10).fit(z_ast_orig)
    pca_combined = PCA(n_components=10).fit(z_combined)

    print("\n--- Does a 6th Dimension Emerge? ---")
    print("  Variance explained comparison:")
    for i in range(8):
        v_orig = pca_orig_only.explained_variance_ratio_[i] * 100
        v_comb = pca_combined.explained_variance_ratio_[i] * 100
        delta = v_comb - v_orig
        marker = " <-- NEW?" if i >= 5 and delta > 0.5 else ""
        print(f"    PC{i}: orig={v_orig:.1f}% combined={v_comb:.1f}% delta={delta:+.1f}%{marker}")

    # Decode what's along each axis for the combined space
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

    # Walk along PC5 (potential new control-flow axis)
    print("\n--- Walking Along PC5 (Potential Control Flow Axis) ---")
    mean_vec = pca_combined.mean_
    axis5 = pca_combined.components_[5]
    std5 = np.sqrt(pca_combined.explained_variance_[5])
    pc5_codes = []
    for t in [-3, -2, -1, 0, 1, 2, 3]:
        v = mean_vec + t * std5 * axis5
        code = gen(v)
        pc5_codes.append({'t': t, 'code': code})
        print(f"  t={t:+d}: {code[:60]}")

    elapsed = time.time() - t0
    results = {
        'phase': 49, 'name': 'Topology of Control Flow',
        'limitation': 'Scale (single-expression limit)',
        'n_new_functions': len(new_triplets),
        'categories': {cat: sum(1 for l in labels_new if l == cat)
                      for cat in ['conditional', 'loop', 'functional', 'multi_op']},
        'pca_orig_variance': pca_orig_only.explained_variance_ratio_[:10].tolist(),
        'pca_combined_variance': pca_combined.explained_variance_ratio_[:10].tolist(),
        'pc5_walk': pc5_codes,
        'elapsed': elapsed, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase49_control_flow.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    # Figure
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Variance comparison
    x = range(8)
    axes[0].bar([i-0.15 for i in x], [pca_orig_only.explained_variance_ratio_[i]*100 for i in x],
               0.3, label='Original (236 funcs)', color='#2196F3', edgecolor='black')
    axes[0].bar([i+0.15 for i in x], [pca_combined.explained_variance_ratio_[i]*100 for i in x],
               0.3, label='+ Control Flow', color='#F44336', edgecolor='black')
    axes[0].set_xlabel('PC'); axes[0].set_ylabel('% Variance')
    axes[0].set_title('Does a 6th Dimension Emerge?', fontweight='bold')
    axes[0].legend()

    # 2. t-SNE of combined space
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(z_combined)-1))
    z_tsne = tsne.fit_transform(z_combined)
    cat_colors = {'single_expr': '#CCCCCC', 'conditional': '#E91E63',
                  'loop': '#4CAF50', 'functional': '#FF9800', 'multi_op': '#2196F3'}
    for cat in ['single_expr', 'conditional', 'loop', 'functional', 'multi_op']:
        idxs = [i for i, l in enumerate(all_labels) if l == cat]
        if idxs:
            alpha = 0.15 if cat == 'single_expr' else 0.9
            size = 8 if cat == 'single_expr' else 50
            axes[1].scatter(z_tsne[idxs, 0], z_tsne[idxs, 1],
                          c=cat_colors[cat], s=size, alpha=alpha, label=cat)
    axes[1].legend(fontsize=8)
    axes[1].set_title('t-SNE: Control Flow in Rosetta Space', fontweight='bold')

    # 3. Category shift from origin
    cats_plot = ['conditional', 'loop', 'functional', 'multi_op']
    shifts = []
    for cat in cats_plot:
        idxs = [i for i, l in enumerate(labels_new) if l == cat]
        if idxs:
            cat_mean = np.mean(z_pca_new[idxs], axis=0)
            shifts.append(np.linalg.norm(cat_mean - orig_mean))
        else:
            shifts.append(0)
    axes[2].barh(cats_plot, shifts, color=['#E91E63', '#4CAF50', '#FF9800', '#2196F3'],
                edgecolor='black')
    axes[2].set_xlabel('Distance from Single-Expr Mean')
    axes[2].set_title('How Far is Control Flow?', fontweight='bold')

    plt.suptitle('Phase 49: The Topology of Control Flow\n'
                 'Limitation Breaker #1: Scale',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase49_control_flow.png'), dpi=150)
    plt.close()
    print(f"\nPhase 49 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
