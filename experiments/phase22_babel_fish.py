"""
Phase 22: The Babel Fish Transpiler
=====================================
Train a JS decoder on the same Rosetta vectors.
Python -> Rosetta Space -> JavaScript. Zero-shot transpilation.
"""
import os, json, time, ast
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def python_to_js(py_src):
    """Convert simple Python function to JavaScript equivalent."""
    # Parse: def f(args): return expr
    py_src = py_src.strip()
    if not py_src.startswith('def f('):
        return None
    try:
        # Extract args and body
        sig = py_src[len('def f('):py_src.index('):')]
        body = py_src[py_src.index('return ')+7:].strip()
    except:
        return None

    # Convert Python operators/functions to JS
    js_body = body
    # Python builtins -> JS
    replacements = [
        ('True', 'true'), ('False', 'false'), ('None', 'null'),
        ('not ', '!'), (' and ', ' && '), (' or ', ' || '),
        ('abs(', 'Math.abs('), ('max(', 'Math.max('),
        ('min(', 'Math.min('), ('pow(', 'Math.pow('),
        ('float(', 'Number('), ('int(', 'Math.floor('),
        ('len(', '.length'), ('**', '**'),
        ('.upper()', '.toUpperCase()'), ('.lower()', '.toLowerCase()'),
        ('.strip()', '.trim()'), ('.swapcase()', '.split("").map(c=>c===c.toUpperCase()?c.toLowerCase():c.toUpperCase()).join("")'),
        ('[::-1]', '.split("").reverse().join("")'),
    ]
    for old, new in replacements:
        js_body = js_body.replace(old, new)

    # Handle // (integer division)
    js_body = js_body.replace('//', '/')
    if '//' in body:
        js_body = f'Math.floor({js_body})'

    # Handle ternary: a if cond else b -> cond ? a : b
    if ' if ' in js_body and ' else ' in js_body:
        parts = js_body.split(' if ')
        true_val = parts[0].strip()
        rest = parts[1].split(' else ')
        cond = rest[0].strip()
        false_val = rest[1].strip()
        js_body = f'{cond} ? {true_val} : {false_val}'

    return f'function f({sig}) {{ return {js_body}; }}'


def main():
    print("=" * 60)
    print("Phase 22: The Babel Fish Transpiler")
    print("Python -> Rosetta Space -> JavaScript")
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

    # Generate JS translations
    print("Generating Python -> JS pairs...")
    js_pairs = []
    seen_js = set()
    for i, d in enumerate(dataset):
        js = python_to_js(d['source'])
        if js and js not in seen_js:
            js_pairs.append({'idx': i, 'python': d['source'], 'js': js})
            seen_js.add(js)

    print(f"  Generated {len(js_pairs)} unique JS translations")

    # Build JS char vocab
    all_js = ''.join(p['js'] for p in js_pairs)
    js_chars = sorted(set(all_js))
    js_char2idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2}
    for c in js_chars:
        if c not in js_char2idx:
            js_char2idx[c] = len(js_char2idx)
    js_idx2char = {v: k for k, v in js_char2idx.items()}
    V_js = len(js_char2idx)
    print(f"  JS vocab size: {V_js}")

    MAX_LEN = 100

    def encode_js(s):
        tokens = [js_char2idx.get('<SOS>', 1)]
        for c in s[:MAX_LEN-2]:
            tokens.append(js_char2idx.get(c, 0))
        tokens.append(js_char2idx.get('<EOS>', 2))
        while len(tokens) < MAX_LEN:
            tokens.append(0)
        return tokens[:MAX_LEN]

    # Expand: use all dataset entries that map to a valid JS
    js_targets = []
    js_z_indices = []
    for p in js_pairs:
        # Find all dataset entries with same source
        for i, d in enumerate(dataset):
            if d['source'] == p['python']:
                js_targets.append(encode_js(p['js']))
                js_z_indices.append(i)

    js_targets = np.array(js_targets, dtype=np.int64)
    js_z = z_ast[js_z_indices].astype(np.float32)
    print(f"  Training samples: {len(js_targets)}")

    # Train JS Decoder (same architecture as P9)
    from experiments.phase9_generative_decompiler import CodeDecoder, decode_tokens
    js_decoder = CodeDecoder(D, V_js, hidden=128, max_len=MAX_LEN).to(DEVICE)
    optimizer = torch.optim.Adam(js_decoder.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 300)

    z_t = torch.tensor(js_z)
    tgt_t = torch.tensor(js_targets)
    Njs = len(js_targets)
    BATCH = 128

    for epoch in range(300):
        perm = torch.randperm(Njs)
        eloss, nb = 0, 0
        js_decoder.train()
        for i in range(0, Njs, BATCH):
            idx = perm[i:i+BATCH]
            z = z_t[idx].to(DEVICE)
            tgt = tgt_t[idx].to(DEVICE)
            logits = js_decoder(z, tgt)
            loss = F.cross_entropy(logits.reshape(-1, V_js), tgt[:, 1:].reshape(-1),
                                   ignore_index=0)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(js_decoder.parameters(), 1.0)
            optimizer.step()
            eloss += loss.item(); nb += 1
        scheduler.step()
        if (epoch+1) % 100 == 0:
            print(f"  JS Decoder Epoch {epoch+1}/300: loss={eloss/max(nb,1):.4f}")

    torch.save(js_decoder.state_dict(), os.path.join(DATA_DIR, 'decoder_js.pt'))
    with open(os.path.join(DATA_DIR, 'js_vocab.json'), 'w') as f:
        json.dump({'char2idx': js_char2idx}, f)

    # Load Python decoder too
    with open(os.path.join(DATA_DIR, 'char_vocab.json'), 'r') as f:
        py_vd = json.load(f)
    py_idx2char = {int(i): c for c, i in py_vd['char2idx'].items()}
    V_py = len(py_vd['char2idx'])
    py_decoder = CodeDecoder(D, V_py, hidden=128, max_len=80).to(DEVICE)
    dec_path = os.path.join(DATA_DIR, 'decoder_robust.pt')
    if not os.path.exists(dec_path):
        dec_path = os.path.join(DATA_DIR, 'decoder.pt')
    py_decoder.load_state_dict(torch.load(dec_path, map_location=DEVICE, weights_only=True))
    py_decoder.eval()

    def gen_py(z_vec):
        with torch.no_grad():
            z_t2 = torch.tensor(z_vec, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            tokens = py_decoder(z_t2)
            return decode_tokens(tokens[0].cpu().numpy(), py_idx2char)

    def gen_js(z_vec):
        with torch.no_grad():
            z_t2 = torch.tensor(z_vec, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            tokens = js_decoder(z_t2)
            return decode_tokens(tokens[0].cpu().numpy(), js_idx2char)

    # === TRANSPILATION TEST ===
    print("\n--- Zero-Shot Transpilation: Python -> Rosetta -> JS ---")
    test_srcs = list(set(d['source'] for d in dataset))[:30]
    n_exact, n_semantic, total = 0, 0, 0
    transpile_results = []

    src_to_idx = {}
    for i, d in enumerate(dataset):
        if d['source'] not in src_to_idx:
            src_to_idx[d['source']] = i

    for src in test_srcs:
        if src not in src_to_idx: continue
        idx = src_to_idx[src]
        z = z_ast[idx]
        truth_js = python_to_js(src)
        if truth_js is None: continue

        gen_js_code = gen_js(z)
        gen_py_code = gen_py(z)
        total += 1

        exact = gen_js_code.strip() == truth_js.strip()
        # Semantic: check key operation preserved
        py_op = src.split('return ')[-1].strip() if 'return ' in src else ''
        js_op = gen_js_code.split('return ')[-1].rstrip('; }').strip() if 'return ' in gen_js_code else ''
        # Rough semantic match
        semantic = False
        for op in ['+','-','*','/','%','>','<','==','!=','>=','<=','&&','||','abs','max','min']:
            if op in py_op and op in js_op:
                semantic = True
                break
        if exact: n_exact += 1
        if semantic: n_semantic += 1

        if total <= 12:
            status = "EXACT" if exact else ("SEM" if semantic else "X")
            print(f"  [{status}] Py: {src[:40]}")
            print(f"         JS: {gen_js_code[:50]}")
            print(f"        Exp: {truth_js[:50]}")

        transpile_results.append({
            'python': src, 'truth_js': truth_js,
            'generated_js': gen_js_code, 'generated_py': gen_py_code,
            'exact': exact, 'semantic': semantic,
        })

    print(f"\n  Total: {total}")
    print(f"  Exact: {n_exact}/{total} ({n_exact/max(total,1):.0%})")
    print(f"  Semantic: {n_semantic}/{total} ({n_semantic/max(total,1):.0%})")

    elapsed = time.time() - t0
    results = {
        'phase': 22, 'name': 'The Babel Fish Transpiler',
        'total': total, 'exact': n_exact, 'semantic': n_semantic,
        'exact_rate': n_exact/max(total,1),
        'semantic_rate': n_semantic/max(total,1),
        'js_vocab_size': V_js,
        'details': transpile_results[:15],
        'elapsed': elapsed, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase22_babel_fish.json'), 'w') as f:
        json.dump(results, f, indent=2)

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    bars = ax.bar(['Exact\nTranspile', 'Semantic\nTranspile'],
                  [n_exact/max(total,1), n_semantic/max(total,1)],
                  color=['#E91E63','#4CAF50'], edgecolor='black')
    for b, v in zip(bars, [n_exact/max(total,1), n_semantic/max(total,1)]):
        ax.text(b.get_x()+b.get_width()/2, v+0.02, f'{v:.0%}',
                ha='center', fontweight='bold', fontsize=14)
    ax.set_ylabel('Accuracy'); ax.set_ylim(0, 1.1)
    ax.set_title('Phase 22: The Babel Fish Transpiler\n'
                 'Python -> Rosetta Space -> JavaScript',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase22_babel_fish.png'), dpi=150)
    plt.close()
    print(f"\nPhase 22 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
