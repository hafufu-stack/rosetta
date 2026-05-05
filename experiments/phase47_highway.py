"""
Phase 47: The Hidden Highway
================================
P41 showed add->sub passes through != and //.
Map ALL shortest paths. Find the "hub" functions
that everything passes through. The interstate system of code.
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
    print("Phase 47: The Hidden Highway")
    print("The interstate system of the code manifold")
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

    # Key functions to map highways between
    key_funcs = [
        "def f(x, y): return x + y",
        "def f(x, y): return x - y",
        "def f(x, y): return x * y",
        "def f(x, y): return x / y",
        "def f(x, y): return x > y",
        "def f(x, y): return x == y",
        "def f(x): return abs(x)",
        "def f(x): return -x",
        "def f(s): return s.upper()",
        "def f(x, y): return x % y",
    ]
    key_funcs = [f for f in key_funcs if f in src_to_idx]
    N = len(key_funcs)

    # === Map routes between all pairs ===
    N_STEPS = 30
    hub_counts = {}  # How often each intermediate function appears

    routes = []
    for ai in range(N):
        for bi in range(ai+1, N):
            va = z_ast[src_to_idx[key_funcs[ai]]]
            vb = z_ast[src_to_idx[key_funcs[bi]]]

            waypoints = []
            prev = None
            for step in range(N_STEPS + 1):
                t = step / N_STEPS
                v = (1-t) * va + t * vb
                code = gen(v).strip()
                if code != prev:
                    waypoints.append(code)
                    if code not in [gen(va).strip(), gen(vb).strip()]:
                        hub_counts[code] = hub_counts.get(code, 0) + 1
                    prev = code

            fa_short = key_funcs[ai].split('return')[-1].strip()[:12]
            fb_short = key_funcs[bi].split('return')[-1].strip()[:12]
            routes.append({
                'from': key_funcs[ai], 'to': key_funcs[bi],
                'n_waypoints': len(waypoints),
                'waypoints': [w[:35] for w in waypoints],
            })

    # === Find the HUB functions ===
    sorted_hubs = sorted(hub_counts.items(), key=lambda x: -x[1])

    print("\n--- Highway Hub Functions ---")
    print("  (Functions that appear on the most routes)")
    hub_results = []
    for code, count in sorted_hubs[:15]:
        print(f"    {count:3d} routes pass through: {code[:45]}")
        hub_results.append({'code': code[:50], 'count': count})

    # === Route complexity analysis ===
    print("\n--- Route Complexity ---")
    for r in routes:
        fa = r['from'].split('return')[-1].strip()[:10]
        fb = r['to'].split('return')[-1].strip()[:10]
        print(f"  {fa} -> {fb}: {r['n_waypoints']} stops via "
              f"{' -> '.join(w[:12] for w in r['waypoints'][:4])}")

    # Average waypoints
    avg_stops = float(np.mean([r['n_waypoints'] for r in routes]))
    print(f"\n  Average stops per route: {avg_stops:.1f}")

    elapsed = time.time() - t0
    results = {
        'phase': 47, 'name': 'The Hidden Highway',
        'hubs': hub_results,
        'n_routes': len(routes),
        'avg_stops': avg_stops,
        'routes': routes[:10],
        'elapsed': elapsed, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase47_highway.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Hub frequency
    if hub_results:
        names = [h['code'][:20] for h in hub_results[:10]]
        counts = [h['count'] for h in hub_results[:10]]
        axes[0].barh(names, counts, color='#FF5722', edgecolor='black')
        axes[0].set_xlabel('Number of Routes')
        axes[0].set_title('Highway Hubs\n(most-traversed intermediate functions)',
                         fontweight='bold')

    # 2. Route complexity matrix
    route_matrix = np.zeros((N, N))
    ri = 0
    for ai in range(N):
        for bi in range(ai+1, N):
            if ri < len(routes):
                route_matrix[ai, bi] = routes[ri]['n_waypoints']
                route_matrix[bi, ai] = routes[ri]['n_waypoints']
                ri += 1
    labels = [f.split('return')[-1].strip()[:10] for f in key_funcs]
    im = axes[1].imshow(route_matrix, cmap='YlOrRd', aspect='auto')
    axes[1].set_xticks(range(N)); axes[1].set_yticks(range(N))
    axes[1].set_xticklabels(labels, rotation=45, fontsize=7)
    axes[1].set_yticklabels(labels, fontsize=7)
    axes[1].set_title('Route Complexity Matrix\n(darker = more stops)', fontweight='bold')
    plt.colorbar(im, ax=axes[1], shrink=0.8)

    plt.suptitle('Phase 47: The Hidden Highway\n'
                 'The interstate system of the code manifold',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase47_highway.png'), dpi=150)
    plt.close()
    print(f"\nPhase 47 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
