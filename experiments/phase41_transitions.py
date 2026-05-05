"""
Phase 41: Program Phase Transitions
======================================
As we sweep through the 5D manifold, are there SHARP boundaries
where programs "jump" from one semantic category to another?
The phase diagram of software.
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
    print("Phase 41: Program Phase Transitions")
    print("Where does 'add' suddenly become 'subtract'?")
    print("=" * 60)
    t0 = time.time()

    lat_file = os.path.join(DATA_DIR, 'rosetta_latents_v2.npz')
    if not os.path.exists(lat_file):
        lat_file = os.path.join(DATA_DIR, 'rosetta_latents.npz')
    latents = np.load(lat_file)
    z_ast = latents['ast']

    ds_file = os.path.join(DATA_DIR, 'rosetta_dataset_v2.json')
    if not os.path.exists(ds_file):
        ds_file = os.path.join(DATA_DIR, 'rosetta_dataset.json')
    with open(ds_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)['dataset']

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

    src_to_idx = {}
    for i, d in enumerate(dataset):
        if d['source'] not in src_to_idx:
            src_to_idx[d['source']] = i

    # === Sweep between function pairs, find transition points ===
    pairs = [
        ("def f(x, y): return x + y", "def f(x, y): return x - y", "add -> sub"),
        ("def f(x, y): return x * y", "def f(x, y): return x / y", "mul -> div"),
        ("def f(x, y): return x + y", "def f(x, y): return x > y", "add -> gt"),
        ("def f(x, y): return x + y", "def f(x, y): return x * y", "add -> mul"),
        ("def f(x): return abs(x)", "def f(x): return -x", "abs -> neg"),
    ]

    N_STEPS = 50
    transitions = []

    for src_a, src_b, desc in pairs:
        if src_a not in src_to_idx or src_b not in src_to_idx:
            continue

        va = z_ast[src_to_idx[src_a]]
        vb = z_ast[src_to_idx[src_b]]

        print(f"\n--- {desc} ---")
        codes = []
        prev_code = None
        transition_points = []

        for step in range(N_STEPS + 1):
            t = step / N_STEPS
            v = (1 - t) * va + t * vb
            code = gen(v)
            codes.append({'t': float(t), 'code': code})

            if prev_code is not None and code.strip() != prev_code.strip():
                transition_points.append({
                    't': float(t), 'from': prev_code[:30], 'to': code[:30],
                })

            prev_code = code

        # Classify transition sharpness
        n_transitions = len(transition_points)
        if n_transitions > 0:
            # Average gap between transitions
            t_vals = [tp['t'] for tp in transition_points]
            gaps = [t_vals[i+1]-t_vals[i] for i in range(len(t_vals)-1)] if len(t_vals)>1 else [1.0]
            avg_gap = float(np.mean(gaps))
        else:
            avg_gap = 1.0

        print(f"  Transitions: {n_transitions}")
        for tp in transition_points[:5]:
            print(f"    t={tp['t']:.2f}: '{tp['from']}' -> '{tp['to']}'")

        transitions.append({
            'desc': desc, 'n_transitions': n_transitions,
            'avg_gap': avg_gap,
            'transition_points': transition_points[:10],
            'start_code': codes[0]['code'], 'end_code': codes[-1]['code'],
        })

    # === Phase diagram: 2D sweep ===
    print("\n--- 2D Phase Diagram ---")
    # Use 3 anchor functions, sweep a 2D triangle
    anchors = ["def f(x, y): return x + y",
               "def f(x, y): return x > y",
               "def f(s): return s.upper()"]

    grid_size = 15
    phase_grid = np.zeros((grid_size, grid_size), dtype=object)
    anchor_vecs = [z_ast[src_to_idx[a]] for a in anchors if a in src_to_idx]

    if len(anchor_vecs) >= 3:
        for i in range(grid_size):
            for j in range(grid_size):
                w1 = i / (grid_size - 1)
                w2 = j / (grid_size - 1) * (1 - w1)
                w3 = 1 - w1 - w2
                if w3 < 0:
                    phase_grid[i, j] = ""
                    continue
                v = w1 * anchor_vecs[0] + w2 * anchor_vecs[1] + w3 * anchor_vecs[2]
                code = gen(v)
                # Classify
                if '+' in code or '-' in code or '*' in code:
                    phase_grid[i, j] = "arith"
                elif '>' in code or '<' in code or '==' in code:
                    phase_grid[i, j] = "compare"
                elif 'upper' in code or 'lower' in code or 'strip' in code:
                    phase_grid[i, j] = "string"
                else:
                    phase_grid[i, j] = "other"

    elapsed = time.time() - t0
    results = {
        'phase': 41, 'name': 'Program Phase Transitions',
        'transitions': transitions,
        'elapsed': elapsed, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase41_transitions.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n_trans = len(transitions)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for ti, tr in enumerate(transitions[:5]):
        ax = axes[ti // 3, ti % 3]
        t_points = [tp['t'] for tp in tr['transition_points']]
        ax.eventplot([t_points], lineoffsets=0.5, linelengths=0.8, colors='red')
        ax.set_xlim(0, 1)
        ax.set_title(f"{tr['desc']}\n{tr['n_transitions']} transitions",
                    fontweight='bold', fontsize=10)
        ax.set_xlabel('t (interpolation)')

    # Phase diagram
    ax = axes[1, 2]
    cat_to_num = {'arith': 0, 'compare': 1, 'string': 2, 'other': 3, '': -1}
    grid_num = np.array([[cat_to_num.get(str(phase_grid[i,j]), -1)
                         for j in range(grid_size)] for i in range(grid_size)])
    im = ax.imshow(grid_num, cmap='Set1', aspect='auto', vmin=-1, vmax=3)
    ax.set_xlabel('Weight 2 (comparison)')
    ax.set_ylabel('Weight 1 (arithmetic)')
    ax.set_title('2D Phase Diagram\n(arith/compare/string)', fontweight='bold')

    plt.suptitle('Phase 41: Program Phase Transitions\n'
                 'Sharp boundaries in the semantic manifold',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase41_transitions.png'), dpi=150)
    plt.close()
    print(f"\nPhase 41 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
