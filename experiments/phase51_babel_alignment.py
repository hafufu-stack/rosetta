"""
Phase 51: Isomorphic Babel Alignment
=========================================
LIMITATION BREAKER #3: Cross-language (JS 3% wall)

P22's approach: train JS decoder from Python space (failed: 3%)
NEW approach: Build an INDEPENDENT JS Rosetta Space, then
align it to Python's space using Orthogonal Procrustes.

The insight: language difference is just a ROTATION.
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

# ==============================================================
# Python -> JavaScript function pairs
# ==============================================================
PYTHON_JS_PAIRS = [
    # (python_source, js_source, description)
    ("def f(x, y): return x + y", "function f(x, y) { return x + y; }", "add"),
    ("def f(x, y): return x - y", "function f(x, y) { return x - y; }", "sub"),
    ("def f(x, y): return x * y", "function f(x, y) { return x * y; }", "mul"),
    ("def f(x, y): return x / y", "function f(x, y) { return x / y; }", "div"),
    ("def f(x, y): return x % y", "function f(x, y) { return x % y; }", "mod"),
    ("def f(x, y): return x ** y", "function f(x, y) { return Math.pow(x, y); }", "pow"),
    ("def f(x, y): return x > y", "function f(x, y) { return x > y; }", "gt"),
    ("def f(x, y): return x < y", "function f(x, y) { return x < y; }", "lt"),
    ("def f(x, y): return x == y", "function f(x, y) { return x === y; }", "eq"),
    ("def f(x, y): return x != y", "function f(x, y) { return x !== y; }", "neq"),
    ("def f(x, y): return x >= y", "function f(x, y) { return x >= y; }", "gte"),
    ("def f(x, y): return x <= y", "function f(x, y) { return x <= y; }", "lte"),
    ("def f(x, y): return x and y", "function f(x, y) { return x && y; }", "and"),
    ("def f(x, y): return x or y", "function f(x, y) { return x || y; }", "or"),
    ("def f(x): return not x", "function f(x) { return !x; }", "not"),
    ("def f(x): return -x", "function f(x) { return -x; }", "neg"),
    ("def f(x): return abs(x)", "function f(x) { return Math.abs(x); }", "abs"),
    ("def f(x, y): return max(x, y)", "function f(x, y) { return Math.max(x, y); }", "max"),
    ("def f(x, y): return min(x, y)", "function f(x, y) { return Math.min(x, y); }", "min"),
    ("def f(x): return x + 1", "function f(x) { return x + 1; }", "inc"),
    ("def f(x): return x - 1", "function f(x) { return x - 1; }", "dec"),
    ("def f(x): return x * 2", "function f(x) { return x * 2; }", "double"),
    ("def f(x): return x * x", "function f(x) { return x * x; }", "square"),
    ("def f(x, y): return x + y + 1", "function f(x, y) { return x + y + 1; }", "add_inc"),
    ("def f(x, y): return (x + y) / 2", "function f(x, y) { return (x + y) / 2; }", "avg"),
    ("def f(s): return s.upper()", "function f(s) { return s.toUpperCase(); }", "upper"),
    ("def f(s): return s.lower()", "function f(s) { return s.toLowerCase(); }", "lower"),
    ("def f(s): return len(s)", "function f(s) { return s.length; }", "len"),
    ("def f(s): return s.strip()", "function f(s) { return s.trim(); }", "strip"),
    ("def f(x, y): return x // y", "function f(x, y) { return Math.floor(x / y); }", "floordiv"),
]


def main():
    print("=" * 60)
    print("Phase 51: Isomorphic Babel Alignment")
    print("LIMITATION BREAKER #3: Cross-language (JS 3%)")
    print("  Orthogonal Procrustes: Language diff = Rotation")
    print("=" * 60)
    t0 = time.time()

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    # Load Python Rosetta Space
    lat_file = os.path.join(DATA_DIR, 'rosetta_latents_v2.npz')
    if not os.path.exists(lat_file):
        lat_file = os.path.join(DATA_DIR, 'rosetta_latents.npz')
    latents = np.load(lat_file)
    z_ast_py = latents['ast']

    ds_file = os.path.join(DATA_DIR, 'rosetta_dataset_v2.json')
    if not os.path.exists(ds_file):
        ds_file = os.path.join(DATA_DIR, 'rosetta_dataset.json')
    with open(ds_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)['dataset']
    py_sources = [d['source'] for d in dataset]

    # Find matching Python functions in the dataset
    py_source_to_idx = {}
    for i, s in enumerate(py_sources):
        if s not in py_source_to_idx:
            py_source_to_idx[s] = i

    # Step 1: Build JS embeddings using character-level autoencoder
    print("\n--- Building JS Rosetta Space ---")
    js_sources = [pair[1] for pair in PYTHON_JS_PAIRS]
    all_js_chars = set()
    for s in js_sources:
        all_js_chars.update(s)
    js_vocab = ['<SOS>', '<EOS>', '<PAD>'] + sorted(all_js_chars)
    js_char2idx = {c: i for i, c in enumerate(js_vocab)}
    js_idx2char = {i: c for c, i in js_char2idx.items()}
    JS_V = len(js_vocab)
    JS_MAX_LEN = 60

    def encode_js(src):
        tokens = [js_char2idx.get('<SOS>', 0)]
        for c in src[:JS_MAX_LEN-2]:
            tokens.append(js_char2idx.get(c, 0))
        tokens.append(js_char2idx.get('<EOS>', 1))
        while len(tokens) < JS_MAX_LEN:
            tokens.append(js_char2idx.get('<PAD>', 2))
        return tokens

    # Train a small JS autoencoder to get 64-dim embeddings
    class JSEncoder(nn.Module):
        def __init__(self, vocab_size, embed_dim=32, hidden=128, latent=64):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.gru = nn.GRU(embed_dim, hidden, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(hidden * 2, latent)

        def forward(self, x):
            emb = self.embedding(x)
            _, h = self.gru(emb)
            h = torch.cat([h[0], h[1]], dim=-1)
            return self.fc(h)

    class JSDecoder(nn.Module):
        def __init__(self, vocab_size, latent=64, hidden=128, max_len=60):
            super().__init__()
            self.max_len = max_len
            self.vocab_size = vocab_size
            self.latent_to_hidden = nn.Linear(latent, hidden)
            self.embedding = nn.Embedding(vocab_size, 32)
            self.gru = nn.GRU(32, hidden, batch_first=True)
            self.fc_out = nn.Linear(hidden, vocab_size)

        def forward(self, z, targets=None):
            h = self.latent_to_hidden(z).unsqueeze(0)
            if targets is not None:
                emb = self.embedding(targets[:, :-1])
                out, _ = self.gru(emb, h)
                return self.fc_out(out)
            else:
                B = z.size(0)
                tokens = torch.zeros(B, 1, dtype=torch.long, device=z.device)
                outputs = []
                for _ in range(self.max_len):
                    emb = self.embedding(tokens[:, -1:])
                    out, h = self.gru(emb, h)
                    logits = self.fc_out(out[:, -1])
                    next_tok = logits.argmax(dim=-1)
                    outputs.append(next_tok)
                    tokens = torch.cat([tokens, next_tok.unsqueeze(1)], dim=1)
                return torch.stack(outputs, dim=1)

    # Augment JS data with variations (parameter name changes)
    augmented_js = []
    for py_src, js_src, desc in PYTHON_JS_PAIRS:
        augmented_js.append((py_src, js_src, desc))
        # Add variations
        for old, new in [('x', 'a'), ('y', 'b'), ('x', 'm'), ('y', 'n')]:
            variant = js_src.replace(old, new)
            if variant != js_src:
                augmented_js.append((py_src, variant, desc))

    print(f"  JS training samples: {len(augmented_js)} (from {len(PYTHON_JS_PAIRS)} pairs)")

    # Encode all JS sources
    all_js_for_training = [pair[1] for pair in augmented_js]
    # Rebuild vocab with augmented data
    all_js_chars = set()
    for s in all_js_for_training:
        all_js_chars.update(s)
    js_vocab = ['<SOS>', '<EOS>', '<PAD>'] + sorted(all_js_chars)
    js_char2idx = {c: i for i, c in enumerate(js_vocab)}
    js_idx2char = {i: c for c, i in js_char2idx.items()}
    JS_V = len(js_vocab)

    js_encoded = np.array([encode_js(s) for s in all_js_for_training])
    js_t = torch.tensor(js_encoded, dtype=torch.long).to(DEVICE)

    js_encoder = JSEncoder(JS_V).to(DEVICE)
    js_decoder = JSDecoder(JS_V, max_len=JS_MAX_LEN).to(DEVICE)

    opt_ae = torch.optim.Adam(list(js_encoder.parameters()) +
                              list(js_decoder.parameters()), lr=1e-3)
    pad_idx = js_char2idx['<PAD>']

    print("  Training JS autoencoder...")
    for epoch in range(500):
        z_js = js_encoder(js_t)
        logits = js_decoder(z_js, js_t)
        loss = F.cross_entropy(logits.reshape(-1, JS_V), js_t[:, 1:].reshape(-1),
                              ignore_index=pad_idx)
        opt_ae.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(list(js_encoder.parameters()) +
                                list(js_decoder.parameters()), 1.0)
        opt_ae.step()
        if (epoch+1) % 100 == 0:
            print(f"    Epoch {epoch+1}/500: loss={loss.item():.4f}")

    # Get JS embeddings
    js_encoder.eval()
    js_decoder.eval()
    with torch.no_grad():
        # Get embeddings for the original (non-augmented) pairs
        js_orig_encoded = np.array([encode_js(pair[1]) for pair in PYTHON_JS_PAIRS])
        js_orig_t = torch.tensor(js_orig_encoded, dtype=torch.long).to(DEVICE)
        z_js_space = js_encoder(js_orig_t).cpu().numpy()

    # Get corresponding Python embeddings
    z_py_matched = []
    matched_pairs = []
    for i, (py_src, js_src, desc) in enumerate(PYTHON_JS_PAIRS):
        if py_src in py_source_to_idx:
            idx = py_source_to_idx[py_src]
            z_py_matched.append(z_ast_py[idx])
            matched_pairs.append(i)

    z_py_matched = np.array(z_py_matched)
    z_js_matched = z_js_space[matched_pairs]
    n_matched = len(matched_pairs)
    print(f"\n  Matched {n_matched}/{len(PYTHON_JS_PAIRS)} Python-JS pairs")

    # Step 2: Orthogonal Procrustes alignment
    print("\n--- Orthogonal Procrustes Alignment ---")
    from scipy.linalg import orthogonal_procrustes

    # Center both spaces
    py_mean = z_py_matched.mean(axis=0)
    js_mean = z_js_matched.mean(axis=0)
    py_centered = z_py_matched - py_mean
    js_centered = z_js_matched - js_mean

    # Find rotation matrix Q: min ||py_centered @ Q - js_centered||
    Q, scale = orthogonal_procrustes(py_centered, js_centered)
    print(f"  Procrustes scale: {scale:.4f}")
    print(f"  Q shape: {Q.shape}, Q orthogonality: {np.max(np.abs(Q @ Q.T - np.eye(64))):.6f}")

    # Apply rotation
    py_rotated = py_centered @ Q

    # Measure alignment quality
    from sklearn.metrics.pairwise import cosine_similarity
    cos_before = np.mean([cosine_similarity(py_centered[i:i+1], js_centered[i:i+1])[0,0]
                         for i in range(n_matched)])
    cos_after = np.mean([cosine_similarity(py_rotated[i:i+1], js_centered[i:i+1])[0,0]
                        for i in range(n_matched)])
    print(f"  Cosine similarity (Python vs JS):")
    print(f"    Before Procrustes: {cos_before:.4f}")
    print(f"    After Procrustes:  {cos_after:.4f}")

    # Step 3: Zero-shot JS translation via rotation
    print("\n--- Zero-Shot JS Translation via Rotation ---")

    def decode_js_tokens(tokens):
        result = []
        for t in tokens:
            c = js_idx2char.get(int(t), '')
            if c == '<EOS>': break
            if c not in ('<SOS>', '<PAD>'):
                result.append(c)
        return ''.join(result)

    # Leave-one-out evaluation
    n_exact = 0
    n_semantic = 0
    translation_examples = []

    for leave_out in range(n_matched):
        # Train Procrustes on all EXCEPT leave_out
        train_idxs = [i for i in range(n_matched) if i != leave_out]
        py_train = py_centered[train_idxs]
        js_train = js_centered[train_idxs]
        Q_loo, _ = orthogonal_procrustes(py_train, js_train)

        # Rotate the left-out Python vector
        py_test = py_centered[leave_out:leave_out+1]
        py_rotated_test = py_test @ Q_loo + js_mean

        # Find nearest JS neighbor in JS space
        sims = cosine_similarity(py_rotated_test, z_js_space)[0]
        best_js_idx = np.argmax(sims)

        # Also try decoding directly
        with torch.no_grad():
            z_rot = torch.tensor(py_rotated_test, dtype=torch.float32).to(DEVICE)
            js_tokens = js_decoder(z_rot)
            decoded_js = decode_js_tokens(js_tokens[0].cpu().numpy())

        true_js = PYTHON_JS_PAIRS[matched_pairs[leave_out]][1]
        nn_js = PYTHON_JS_PAIRS[best_js_idx][1]
        true_desc = PYTHON_JS_PAIRS[matched_pairs[leave_out]][2]

        exact = decoded_js.strip() == true_js.strip()
        semantic = (best_js_idx == matched_pairs[leave_out])
        if exact: n_exact += 1
        if semantic: n_semantic += 1

        translation_examples.append({
            'python': PYTHON_JS_PAIRS[matched_pairs[leave_out]][0],
            'true_js': true_js,
            'decoded_js': decoded_js,
            'nn_js': nn_js,
            'exact_match': bool(exact),
            'semantic_match': bool(semantic),
            'desc': true_desc,
        })

    print(f"  Leave-one-out results ({n_matched} pairs):")
    print(f"    Exact decode match:    {n_exact}/{n_matched} ({n_exact/max(n_matched,1)*100:.1f}%)")
    print(f"    Semantic (NN) match:   {n_semantic}/{n_matched} ({n_semantic/max(n_matched,1)*100:.1f}%)")
    print(f"    P22 baseline:          3%")

    print("\n  Examples:")
    for ex in translation_examples[:10]:
        status = "OK" if ex['semantic_match'] else "X "
        print(f"    [{status}] {ex['python'][:30]} -> {ex['nn_js'][:35]}")
        if not ex['semantic_match']:
            print(f"         True: {ex['true_js'][:35]}")

    elapsed = time.time() - t0
    results = {
        'phase': 51, 'name': 'Isomorphic Babel Alignment',
        'limitation': 'Cross-language (JS 3% wall)',
        'n_matched_pairs': n_matched,
        'procrustes_scale': float(scale),
        'cosine_before_procrustes': float(cos_before),
        'cosine_after_procrustes': float(cos_after),
        'exact_decode_rate': n_exact / max(n_matched, 1),
        'semantic_nn_rate': n_semantic / max(n_matched, 1),
        'p22_baseline': 0.03,
        'improvement_over_p22': f"{n_semantic/max(n_matched,1)/0.03:.1f}x",
        'examples': translation_examples[:10],
        'elapsed': elapsed, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase51_babel_alignment.json'), 'w',
              encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    # Figure
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Before vs After Procrustes
    axes[0].bar(['Before\nProcrustes', 'After\nProcrustes'],
               [cos_before, cos_after],
               color=['#F44336', '#4CAF50'], edgecolor='black')
    axes[0].set_ylabel('Cosine Similarity')
    axes[0].set_title('Python-JS Alignment\n(Procrustes Rotation)', fontweight='bold')
    axes[0].set_ylim(0, 1.1)
    for i, v in enumerate([cos_before, cos_after]):
        axes[0].text(i, v+0.03, f'{v:.3f}', ha='center', fontweight='bold', fontsize=13)

    # 2. Translation accuracy
    axes[1].bar(['P22 Baseline\n(3%)', 'P51 Decoder\n(Procrustes)',
                 'P51 NN\n(Procrustes)'],
               [3, n_exact/max(n_matched,1)*100, n_semantic/max(n_matched,1)*100],
               color=['#9E9E9E', '#2196F3', '#4CAF50'], edgecolor='black')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Cross-Language Translation\nP22 vs P51', fontweight='bold')
    for i, v in enumerate([3, n_exact/max(n_matched,1)*100, n_semantic/max(n_matched,1)*100]):
        axes[1].text(i, v+2, f'{v:.0f}%', ha='center', fontweight='bold', fontsize=13)

    # 3. Per-pair cosine similarity after rotation
    per_pair_cos = [cosine_similarity(py_rotated[i:i+1], js_centered[i:i+1])[0,0]
                   for i in range(n_matched)]
    pair_names = [PYTHON_JS_PAIRS[matched_pairs[i]][2] for i in range(n_matched)]
    axes[2].barh(pair_names[:15], per_pair_cos[:15],
                color=['#4CAF50' if c > 0.5 else '#F44336' for c in per_pair_cos[:15]],
                edgecolor='black')
    axes[2].set_xlabel('Cosine Similarity (after rotation)')
    axes[2].set_title('Per-Function Alignment', fontweight='bold')
    axes[2].set_xlim(-0.2, 1.1)

    plt.suptitle('Phase 51: Isomorphic Babel Alignment\n'
                 'Limitation Breaker #3: Language = Rotation',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase51_babel_alignment.png'), dpi=150)
    plt.close()
    print(f"\nPhase 51 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
