"""
Phase 54: Cross-Lingual Auto-Patching
=========================================
The ultimate application: combine P13's semantic bug repair
with P53's Procrustes rotation to fix JS bugs using Python knowledge.

"Fix bugs in a language you've never seen, using knowledge from another."
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
    print("Phase 54: Cross-Lingual Auto-Patching")
    print("Fix JS bugs with Python knowledge (zero-shot)")
    print("=" * 60)
    t0 = time.time()
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    # Load Python space
    lat_file = os.path.join(DATA_DIR, 'rosetta_latents_v2.npz')
    if not os.path.exists(lat_file):
        lat_file = os.path.join(DATA_DIR, 'rosetta_latents.npz')
    latents = np.load(lat_file)
    z_ast_py = latents['ast']
    z_nl_py = latents['nl']

    ds_file = os.path.join(DATA_DIR, 'rosetta_dataset_v2.json')
    if not os.path.exists(ds_file):
        ds_file = os.path.join(DATA_DIR, 'rosetta_dataset.json')
    with open(ds_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)['dataset']
    py_sources = [d['source'] for d in dataset]
    py_src_to_idx = {}
    for i, s in enumerate(py_sources):
        if s not in py_src_to_idx:
            py_src_to_idx[s] = i

    # Load Procrustes rotation Q from P53
    Q_path = os.path.join(DATA_DIR, 'procrustes_Q.npy')
    py_center_path = os.path.join(DATA_DIR, 'py_center.npy')
    js_center_path = os.path.join(DATA_DIR, 'js_center.npy')

    if not os.path.exists(Q_path):
        print("  ERROR: P53 rotation matrix not found. Run P53 first!")
        return None

    Q = np.load(Q_path)
    py_m = np.load(py_center_path)
    js_m = np.load(js_center_path)
    print(f"  Loaded Procrustes Q: {Q.shape}")

    # Load JS decoder
    js_vocab_path = os.path.join(DATA_DIR, 'js_vocab.json')
    with open(js_vocab_path, 'r', encoding='utf-8') as f:
        js_vd = json.load(f)
    js_c2i = js_vd['char2idx']
    js_i2c = {int(i): c for c, i in js_c2i.items()}
    JS_V = len(js_c2i)
    JS_MAX = js_vd.get('max_len', 80)

    # Rebuild JS decoder architecture (must match P53)
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

    js_dec = JSDecoder(JS_V, ml=JS_MAX).to(DEVICE)
    js_dec_path = os.path.join(DATA_DIR, 'js_decoder.pt')
    if os.path.exists(js_dec_path):
        js_dec.load_state_dict(torch.load(js_dec_path, map_location=DEVICE, weights_only=True))
    js_dec.eval()

    def decode_js(tokens):
        r = []
        for t in tokens:
            c = js_i2c.get(int(t), '')
            if c == '<EOS>': break
            if c not in ('<SOS>', '<PAD>'):
                r.append(c)
        return ''.join(r)

    # Also load Python decoder for comparison
    sys.path.insert(0, BASE_DIR)
    with open(os.path.join(DATA_DIR, 'char_vocab.json'), 'r') as f:
        py_vd = json.load(f)
    py_i2c = {int(i): c for c, i in py_vd['char2idx'].items()}
    PY_V = len(py_vd['char2idx'])
    from experiments.phase9_generative_decompiler import CodeDecoder, decode_tokens
    py_dec = CodeDecoder(64, PY_V, hidden=128, max_len=80).to(DEVICE)
    dec_path = os.path.join(DATA_DIR, 'decoder_robust.pt')
    if not os.path.exists(dec_path):
        dec_path = os.path.join(DATA_DIR, 'decoder.pt')
    py_dec.load_state_dict(torch.load(dec_path, map_location=DEVICE, weights_only=True))
    py_dec.eval()

    # Define bug scenarios (P13 style, but for cross-lingual)
    bug_scenarios = [
        {'bug_op': 'add', 'fix_op': 'sub',
         'bug_py': 'def f(x, y): return x + y',
         'fix_py': 'def f(x, y): return x - y',
         'bug_js': 'function f(x, y) { return x + y; }',
         'fix_js': 'function f(x, y) { return x - y; }'},
        {'bug_op': 'sub', 'fix_op': 'add',
         'bug_py': 'def f(x, y): return x - y',
         'fix_py': 'def f(x, y): return x + y',
         'bug_js': 'function f(x, y) { return x - y; }',
         'fix_js': 'function f(x, y) { return x + y; }'},
        {'bug_op': 'mul', 'fix_op': 'div',
         'bug_py': 'def f(x, y): return x * y',
         'fix_py': 'def f(x, y): return x / y',
         'bug_js': 'function f(x, y) { return x * y; }',
         'fix_js': 'function f(x, y) { return x / y; }'},
        {'bug_op': 'gt', 'fix_op': 'lt',
         'bug_py': 'def f(x, y): return x > y',
         'fix_py': 'def f(x, y): return x < y',
         'bug_js': 'function f(x, y) { return x > y; }',
         'fix_js': 'function f(x, y) { return x < y; }'},
        {'bug_op': 'eq', 'fix_op': 'neq',
         'bug_py': 'def f(x, y): return x == y',
         'fix_py': 'def f(x, y): return x != y',
         'bug_js': 'function f(x, y) { return x === y; }',
         'fix_js': 'function f(x, y) { return x !== y; }'},
        {'bug_op': 'max', 'fix_op': 'min',
         'bug_py': 'def f(x, y): return max(x, y)',
         'fix_py': 'def f(x, y): return min(x, y)',
         'bug_js': 'function f(x, y) { return Math.max(x, y); }',
         'fix_js': 'function f(x, y) { return Math.min(x, y); }'},
    ]

    print("\n--- Cross-Lingual Auto-Patching ---")
    print("  Method: Python patch vector @ Procrustes Q -> JS patch")

    results_list = []
    for scenario in bug_scenarios:
        # Step 1: Compute patch vector in Python space (P13 method)
        bug_idx = py_src_to_idx.get(scenario['bug_py'])
        fix_idx = py_src_to_idx.get(scenario['fix_py'])

        if bug_idx is None or fix_idx is None:
            print(f"  {scenario['bug_op']}->{scenario['fix_op']}: SKIP (not in dataset)")
            continue

        v_bug_py = z_ast_py[bug_idx]
        v_fix_py = z_ast_py[fix_idx]
        patch_py = v_fix_py - v_bug_py  # The semantic patch vector

        # Step 2: Rotate patch to JS space via Procrustes
        # patch_js = patch_py @ Q (rotation preserves vector operations!)
        patch_py_centered = patch_py  # Already centered relative to itself
        patch_js = patch_py_centered @ Q

        # Step 3: Apply patch to JS bug vector
        # First, project the Python bug vector to JS space
        v_bug_js_projected = (v_bug_py - py_m) @ Q + js_m

        # Apply the JS-space patch
        v_fixed_js = v_bug_js_projected + patch_js

        # Step 4: Decode the fixed JS vector
        with torch.no_grad():
            z_js_t = torch.tensor(v_fixed_js.astype(np.float32)).unsqueeze(0).to(DEVICE)
            js_tokens = js_dec(z_js_t)
            js_result = decode_js(js_tokens[0].cpu().numpy())

        # Also decode Python patch result for comparison
        v_fixed_py = v_bug_py + patch_py  # = v_fix_py
        with torch.no_grad():
            z_py_t = torch.tensor(v_fixed_py.astype(np.float32)).unsqueeze(0).to(DEVICE)
            py_tokens = py_dec(z_py_t)
            py_result = decode_tokens(py_tokens[0].cpu().numpy(), py_i2c)

        # Check if fix is semantically correct
        js_correct = False
        fix_op = scenario['fix_op']
        true_fix_js = scenario['fix_js']

        # Simple check: does the output contain the right operator?
        op_checks = {
            'sub': [' - ', '- '], 'add': [' + ', '+ '],
            'div': [' / ', '/ '], 'lt': [' < '],
            'neq': ['!==', '!='], 'min': ['Math.min', 'min'],
        }
        if fix_op in op_checks:
            for check in op_checks[fix_op]:
                if check in js_result:
                    js_correct = True
                    break

        # Also check exact match
        js_exact = js_result.strip() == true_fix_js.strip()

        print(f"  {scenario['bug_op']:4s}->{scenario['fix_op']:4s}: "
              f"JS out: {js_result[:45]:45s} "
              f"[{'EXACT' if js_exact else 'SEM' if js_correct else 'MISS'}]")
        print(f"         PY fix: {py_result[:45]}")
        print(f"         True:   {true_fix_js[:45]}")

        results_list.append({
            'bug_op': scenario['bug_op'],
            'fix_op': scenario['fix_op'],
            'js_output': js_result,
            'py_output': py_result,
            'true_js': true_fix_js,
            'js_exact': bool(js_exact),
            'js_semantic': bool(js_correct),
            'patch_norm': float(np.linalg.norm(patch_py)),
        })

    n_semantic = sum(1 for r in results_list if r['js_semantic'])
    n_exact = sum(1 for r in results_list if r['js_exact'])
    n_total = len(results_list)
    print(f"\n  === CROSS-LINGUAL PATCHING ===")
    print(f"  JS semantic fix: {n_semantic}/{n_total} ({n_semantic/max(n_total,1)*100:.0f}%)")
    print(f"  JS exact fix:    {n_exact}/{n_total} ({n_exact/max(n_total,1)*100:.0f}%)")
    print(f"  P13 Python-only: 4/7 (57%)")

    elapsed = time.time() - t0
    results = {
        'phase': 54, 'name': 'Cross-Lingual Auto-Patching',
        'method': 'Python patch vector @ Procrustes Q -> JS space',
        'n_semantic': n_semantic, 'n_exact': n_exact, 'n_total': n_total,
        'semantic_rate': n_semantic / max(n_total, 1),
        'exact_rate': n_exact / max(n_total, 1),
        'p13_python_rate': 4/7,
        'patches': results_list,
        'elapsed': elapsed, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase54_crosslingual_patch.json'), 'w',
              encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    # Figure
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Python vs JS patch success
    axes[0].bar(['P13\nPython->Python\n(same lang)', 'P54\nPython->JS\n(cross-lingual)'],
               [57, n_semantic/max(n_total,1)*100],
               color=['#2196F3', '#4CAF50'], edgecolor='black')
    axes[0].set_ylabel('Semantic Fix %')
    axes[0].set_title('Auto-Patching Success Rate', fontweight='bold')
    axes[0].set_ylim(0, 110)
    for i, v in enumerate([57, n_semantic/max(n_total,1)*100]):
        axes[0].text(i, v+3, f'{v:.0f}%', ha='center', fontweight='bold', fontsize=14)

    # 2. Per-bug results
    ops = [r['bug_op'] + '->' + r['fix_op'] for r in results_list]
    sem_vals = [1 if r['js_semantic'] else 0 for r in results_list]
    colors = ['#4CAF50' if s else '#F44336' for s in sem_vals]
    axes[1].barh(ops, sem_vals, color=colors, edgecolor='black')
    axes[1].set_xlabel('Fixed?')
    axes[1].set_title('Per-Bug Results\n(Cross-Lingual)', fontweight='bold')

    # 3. Patch vector norms
    norms = [r['patch_norm'] for r in results_list]
    colors_n = ['#4CAF50' if r['js_semantic'] else '#F44336' for r in results_list]
    axes[2].barh(ops, norms, color=colors_n, edgecolor='black')
    axes[2].set_xlabel('Patch Vector L2 Norm')
    axes[2].set_title('Patch Magnitude\nvs Success', fontweight='bold')

    plt.suptitle('Phase 54: Cross-Lingual Auto-Patching\n'
                 'Fix JS bugs using Python knowledge (zero-shot)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase54_crosslingual_patch.png'), dpi=150)
    plt.close()
    print(f"\nPhase 54 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
