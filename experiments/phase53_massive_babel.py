"""
Phase 53: Massive Isomorphic Babel Alignment
===============================================
P51 proved: Procrustes rotation works (cos 9x improvement)
but only 26 pairs = insufficient anchors.
Solution: Generate full JS dataset, retrain JS space, massive alignment.
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

# Python -> JS syntax conversion rules
def py_to_js(py_src):
    """Convert simple Python function to JavaScript."""
    js = py_src
    # def f(...) -> function f(...)
    js = js.replace('def ', 'function ')
    # : return ... -> { return ...; }
    if ': return ' in js:
        parts = js.split(': return ', 1)
        js = parts[0] + ' { return ' + parts[1] + '; }'
    # Python builtins -> JS equivalents
    js = js.replace('abs(', 'Math.abs(')
    js = js.replace('max(', 'Math.max(')
    js = js.replace('min(', 'Math.min(')
    js = js.replace('len(', '(').replace('.upper()', '.toUpperCase()')
    js = js.replace('.lower()', '.toLowerCase()')
    js = js.replace('.strip()', '.trim()')
    js = js.replace(' and ', ' && ').replace(' or ', ' || ')
    js = js.replace(' not ', ' !')
    js = js.replace('True', 'true').replace('False', 'false')
    js = js.replace('None', 'null')
    # x ** y -> Math.pow(x, y)  -- simplified
    if '**' in js:
        js = js.replace('**', ', ').replace('return ', 'return Math.pow(', 1)
        # Close the paren before ;
        js = js.replace('; }', '); }', 1)
    # x // y -> Math.floor(x / y)
    if '//' in js:
        idx = js.index('//')
        # Find operands around //
        js = js.replace('//', '/ ')
        js = js.replace('return ', 'return Math.floor(', 1)
        js = js.replace('; }', '); }', 1)
    return js


def main():
    print("=" * 60)
    print("Phase 53: Massive Isomorphic Babel Alignment")
    print("Full-scale Python<->JS space alignment")
    print("=" * 60)
    t0 = time.time()
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    # Load Python space
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

    # Generate JS equivalents for ALL Python functions
    print("\n--- Generating Full JS Dataset ---")
    unique_py = list(dict.fromkeys(py_sources))
    py_js_pairs = []
    failed = 0
    for py_src in unique_py:
        try:
            js_src = py_to_js(py_src)
            # Validate: must contain 'function' and 'return'
            if 'function' in js_src and 'return' in js_src:
                py_js_pairs.append((py_src, js_src))
            else:
                failed += 1
        except Exception:
            failed += 1

    print(f"  Converted {len(py_js_pairs)} / {len(unique_py)} functions (failed: {failed})")

    # Show some examples
    for py_s, js_s in py_js_pairs[:5]:
        print(f"    PY: {py_s[:40]}")
        print(f"    JS: {js_s[:40]}")
        print()

    # Build JS character vocab
    all_js = [js for _, js in py_js_pairs]
    js_chars = set()
    for s in all_js:
        js_chars.update(s)
    js_vocab = ['<SOS>', '<EOS>', '<PAD>'] + sorted(js_chars)
    js_c2i = {c: i for i, c in enumerate(js_vocab)}
    js_i2c = {i: c for c, i in js_c2i.items()}
    JS_V = len(js_vocab)
    JS_MAX = 80

    def enc_js(src):
        t = [js_c2i.get('<SOS>', 0)]
        for c in src[:JS_MAX-2]:
            t.append(js_c2i.get(c, 0))
        t.append(js_c2i.get('<EOS>', 1))
        while len(t) < JS_MAX:
            t.append(js_c2i.get('<PAD>', 2))
        return t

    # Augment: for each pair, also add parameter name variations
    augmented = []
    for py_s, js_s in py_js_pairs:
        augmented.append((py_s, js_s))
        for old_set, new_set in [
            (['x', 'y'], ['a', 'b']),
            (['x', 'y'], ['m', 'n']),
            (['x', 'y'], ['p', 'q']),
        ]:
            v = js_s
            for o, n in zip(old_set, new_set):
                v = v.replace(o, n)
            if v != js_s:
                augmented.append((py_s, v))

    print(f"  Augmented JS samples: {len(augmented)}")

    # Rebuild vocab with augmented
    all_js_aug = [js for _, js in augmented]
    js_chars = set()
    for s in all_js_aug:
        js_chars.update(s)
    js_vocab = ['<SOS>', '<EOS>', '<PAD>'] + sorted(js_chars)
    js_c2i = {c: i for i, c in enumerate(js_vocab)}
    js_i2c = {i: c for c, i in js_c2i.items()}
    JS_V = len(js_vocab)

    js_encoded = np.array([enc_js(s) for s in all_js_aug])
    js_t = torch.tensor(js_encoded, dtype=torch.long).to(DEVICE)

    # Train JS autoencoder (encoder + decoder)
    class JSEncoder(nn.Module):
        def __init__(self, vs, ed=32, hd=128, ld=64):
            super().__init__()
            self.emb = nn.Embedding(vs, ed)
            self.gru = nn.GRU(ed, hd, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(hd*2, ld)
        def forward(self, x):
            e = self.emb(x)
            _, h = self.gru(e)
            return self.fc(torch.cat([h[0], h[1]], -1))

    class JSDecoder(nn.Module):
        def __init__(self, vs, ld=64, hd=128, ml=80):
            super().__init__()
            self.ml = ml; self.vs = vs
            self.l2h = nn.Linear(ld, hd)
            self.emb = nn.Embedding(vs, 32)
            self.gru = nn.GRU(32, hd, batch_first=True)
            self.out = nn.Linear(hd, vs)
        def forward(self, z, tgt=None):
            h = self.l2h(z).unsqueeze(0)
            if tgt is not None:
                e = self.emb(tgt[:, :-1])
                o, _ = self.gru(e, h)
                return self.out(o)
            B = z.size(0)
            tok = torch.zeros(B, 1, dtype=torch.long, device=z.device)
            outs = []
            for _ in range(self.ml):
                e = self.emb(tok[:, -1:])
                o, h = self.gru(e, h)
                nx = self.out(o[:, -1]).argmax(-1)
                outs.append(nx)
                tok = torch.cat([tok, nx.unsqueeze(1)], 1)
            return torch.stack(outs, 1)

    js_enc = JSEncoder(JS_V).to(DEVICE)
    js_dec = JSDecoder(JS_V, ml=JS_MAX).to(DEVICE)
    pad = js_c2i['<PAD>']

    opt_ae = torch.optim.Adam(list(js_enc.parameters()) + list(js_dec.parameters()), lr=1e-3)
    sched_ae = torch.optim.lr_scheduler.CosineAnnealingLR(opt_ae, 800)

    print("\n--- Training JS Autoencoder ---")
    N_js = len(js_t)
    BATCH = min(128, N_js)
    for epoch in range(800):
        perm = torch.randperm(N_js)
        eloss, nb = 0, 0
        for bi in range(0, N_js, BATCH):
            idx = perm[bi:bi+BATCH]
            batch = js_t[idx]
            z = js_enc(batch)
            logits = js_dec(z, batch)
            loss = F.cross_entropy(logits.reshape(-1, JS_V), batch[:, 1:].reshape(-1),
                                  ignore_index=pad)
            opt_ae.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(list(js_enc.parameters()) + list(js_dec.parameters()), 1.0)
            opt_ae.step()
            eloss += loss.item(); nb += 1
        sched_ae.step()
        if (epoch+1) % 200 == 0:
            print(f"    Epoch {epoch+1}/800: loss={eloss/max(nb,1):.4f}")

    # Get JS embeddings for unique pairs
    js_enc.eval(); js_dec.eval()

    # Match Python vectors to JS vectors
    py_src_to_idx = {}
    for i, s in enumerate(py_sources):
        if s not in py_src_to_idx:
            py_src_to_idx[s] = i

    matched_py = []
    matched_js_src = []
    for py_s, js_s in py_js_pairs:
        if py_s in py_src_to_idx:
            matched_py.append(py_src_to_idx[py_s])
            matched_js_src.append(js_s)

    print(f"\n  Matched pairs for alignment: {len(matched_py)}")

    # Get embeddings
    with torch.no_grad():
        js_enc_input = torch.tensor(np.array([enc_js(s) for s in matched_js_src]),
                                    dtype=torch.long).to(DEVICE)
        # Process in batches to avoid OOM
        z_js_all = []
        for bi in range(0, len(js_enc_input), 256):
            z_js_all.append(js_enc(js_enc_input[bi:bi+256]).cpu().numpy())
        z_js = np.vstack(z_js_all)

    z_py = z_ast_py[matched_py]
    n_pairs = len(matched_py)

    # Orthogonal Procrustes
    from scipy.linalg import orthogonal_procrustes
    from sklearn.metrics.pairwise import cosine_similarity

    py_m = z_py.mean(0); js_m = z_js.mean(0)
    py_c = z_py - py_m; js_c = z_js - js_m

    # Split: 80% train, 20% test
    rng = np.random.RandomState(42)
    perm_idx = rng.permutation(n_pairs)
    n_train = int(n_pairs * 0.8)
    train_idx = perm_idx[:n_train]
    test_idx = perm_idx[n_train:]

    Q, scale = orthogonal_procrustes(py_c[train_idx], js_c[train_idx])
    print(f"\n--- Procrustes on {n_train} training pairs ---")
    print(f"  Q orthogonality: {np.max(np.abs(Q @ Q.T - np.eye(64))):.6f}")

    # Evaluate on test set
    py_rot_test = py_c[test_idx] @ Q
    cos_before = np.mean([cosine_similarity(py_c[i:i+1], js_c[i:i+1])[0,0] for i in test_idx])
    cos_after = np.mean([cosine_similarity(py_rot_test[j:j+1], js_c[test_idx[j]:test_idx[j]+1])[0,0]
                        for j in range(len(test_idx))])
    print(f"  TEST SET cosine similarity:")
    print(f"    Before: {cos_before:.4f}")
    print(f"    After:  {cos_after:.4f}")

    # NN-based translation evaluation on test set
    z_js_all_np = z_js  # All JS embeddings
    py_rot_all = py_c @ Q  # Rotate all Python vectors

    n_correct_nn = 0
    n_test = len(test_idx)
    examples = []

    def decode_js_tok(tokens):
        r = []
        for t in tokens:
            c = js_i2c.get(int(t), '')
            if c == '<EOS>': break
            if c not in ('<SOS>', '<PAD>'):
                r.append(c)
        return ''.join(r)

    for j, ti in enumerate(test_idx):
        # Nearest neighbor in JS space
        rot_vec = py_rot_all[ti:ti+1]
        sims = cosine_similarity(rot_vec + js_m, z_js_all_np)
        best_idx = np.argmax(sims[0])

        correct = (best_idx == ti)
        if correct: n_correct_nn += 1

        # Also try decoder
        with torch.no_grad():
            z_rot_t = torch.tensor((rot_vec + js_m).astype(np.float32)).to(DEVICE)
            dec_tok = js_dec(z_rot_t)
            decoded = decode_js_tok(dec_tok[0].cpu().numpy())

        if j < 15:
            true_js = matched_js_src[ti]
            nn_js = matched_js_src[best_idx] if best_idx < len(matched_js_src) else "OOB"
            examples.append({
                'python': py_sources[matched_py[ti]],
                'true_js': true_js,
                'nn_js': nn_js,
                'decoded_js': decoded,
                'correct': bool(correct),
            })

    nn_rate = n_correct_nn / max(n_test, 1)
    print(f"\n  TEST SET NN translation: {n_correct_nn}/{n_test} ({nn_rate*100:.1f}%)")
    print(f"  P51 baseline:            7.7%")
    print(f"  P22 baseline:            3%")

    print("\n  Examples:")
    for ex in examples[:10]:
        st = "OK" if ex['correct'] else "X "
        print(f"    [{st}] {ex['python'][:30]} -> {ex['nn_js'][:40]}")

    elapsed = time.time() - t0

    # Save rotation matrix for P54
    np.save(os.path.join(DATA_DIR, 'procrustes_Q.npy'), Q)
    np.save(os.path.join(DATA_DIR, 'py_center.npy'), py_m)
    np.save(os.path.join(DATA_DIR, 'js_center.npy'), js_m)
    torch.save(js_enc.state_dict(), os.path.join(DATA_DIR, 'js_encoder.pt'))
    torch.save(js_dec.state_dict(), os.path.join(DATA_DIR, 'js_decoder.pt'))
    with open(os.path.join(DATA_DIR, 'js_vocab.json'), 'w', encoding='utf-8') as f:
        json.dump({'char2idx': js_c2i, 'vocab_list': js_vocab,
                   'max_len': JS_MAX}, f)

    results = {
        'phase': 53, 'name': 'Massive Isomorphic Babel',
        'n_py_js_pairs': len(py_js_pairs),
        'n_augmented': len(augmented),
        'n_matched': n_pairs,
        'n_train': n_train, 'n_test': n_test,
        'cos_before': float(cos_before),
        'cos_after': float(cos_after),
        'nn_accuracy': float(nn_rate),
        'p51_baseline': 0.077,
        'p22_baseline': 0.03,
        'examples': examples[:10],
        'elapsed': elapsed, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase53_massive_babel.json'), 'w',
              encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    # Figure
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].bar(['Before\nProcrustes', 'After\nProcrustes'],
               [cos_before, cos_after],
               color=['#F44336', '#4CAF50'], edgecolor='black')
    axes[0].set_ylabel('Cosine Similarity')
    axes[0].set_title(f'Alignment Quality\n({n_pairs} pairs)', fontweight='bold')
    for i, v in enumerate([cos_before, cos_after]):
        axes[0].text(i, v+0.02, f'{v:.3f}', ha='center', fontweight='bold', fontsize=13)

    axes[1].bar(['P22\n(3%)', 'P51\n26 pairs', f'P53\n{n_pairs} pairs'],
               [3, 7.7, nn_rate*100],
               color=['#9E9E9E', '#FF9800', '#4CAF50'], edgecolor='black')
    axes[1].set_ylabel('NN Translation %')
    axes[1].set_title('Cross-Language Translation\nEvolution', fontweight='bold')
    for i, v in enumerate([3, 7.7, nn_rate*100]):
        axes[1].text(i, v+2, f'{v:.0f}%', ha='center', fontweight='bold', fontsize=13)

    # Per-function cosine after rotation
    per_cos = [cosine_similarity((py_c[i:i+1] @ Q), js_c[i:i+1])[0,0]
              for i in range(min(30, n_pairs))]
    axes[2].hist(per_cos, bins=20, color='#2196F3', edgecolor='black')
    axes[2].axvline(np.mean(per_cos), color='red', linestyle='--',
                   label=f'mean={np.mean(per_cos):.3f}')
    axes[2].set_xlabel('Cosine Similarity (after rotation)')
    axes[2].set_title('Distribution of Alignments', fontweight='bold')
    axes[2].legend()

    plt.suptitle('Phase 53: Massive Isomorphic Babel Alignment\n'
                 f'Python<->JS with {n_pairs} anchor pairs',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase53_massive_babel.png'), dpi=150)
    plt.close()
    print(f"\nPhase 53 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
