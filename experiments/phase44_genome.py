"""
Phase 44: The Rosetta Genome (Phylogenetic Tree of Programs)
==============================================================
Treat programs as organisms with DNA (latent vectors).
Build a phylogenetic tree showing evolutionary relationships.
Which programs are siblings? Who is the common ancestor?
"""
import os, json, time
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')


def main():
    print("=" * 60)
    print("Phase 44: The Rosetta Genome")
    print("Phylogenetic tree of programs")
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

    src_to_idx = {}
    for i, d in enumerate(dataset):
        if d['source'] not in src_to_idx:
            src_to_idx[d['source']] = i
    unique_srcs = list(src_to_idx.keys())

    # Select representative functions (one per semantic type)
    rep_funcs = []
    seen_ops = set()
    for src in unique_srcs:
        # Extract the core operation
        if 'return' in src:
            op_part = src.split('return')[-1].strip()[:15]
        else:
            op_part = src[:15]
        if op_part not in seen_ops and len(rep_funcs) < 25:
            seen_ops.add(op_part)
            rep_funcs.append(src)

    rep_vecs = np.array([z_ast[src_to_idx[s]] for s in rep_funcs])
    N = len(rep_funcs)
    print(f"  Building tree for {N} representative functions")

    # === Hierarchical clustering (agglomerative) ===
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import pdist

    # Distance matrix using cosine distance
    dists = pdist(rep_vecs, metric='cosine')
    Z = linkage(dists, method='ward')

    # Extract tree structure
    print("\n--- Phylogenetic Relationships ---")

    # Find closest pairs (siblings)
    from scipy.spatial.distance import squareform
    dist_matrix = squareform(dists)

    siblings = []
    for i in range(N):
        for j in range(i+1, N):
            siblings.append((i, j, float(dist_matrix[i, j])))
    siblings.sort(key=lambda x: x[2])

    print("  Closest siblings (shared ancestor):")
    sibling_results = []
    for i, j, d in siblings[:8]:
        print(f"    d={d:.3f}: {rep_funcs[i][:30]} <-> {rep_funcs[j][:30]}")
        sibling_results.append({
            'func_a': rep_funcs[i], 'func_b': rep_funcs[j],
            'distance': d,
        })

    print("\n  Most distant (evolutionary outliers):")
    for i, j, d in siblings[-3:]:
        print(f"    d={d:.3f}: {rep_funcs[i][:30]} <-> {rep_funcs[j][:30]}")

    # === Common ancestor analysis ===
    print("\n--- Common Ancestors ---")
    # For each pair, the "ancestor" is their mean vector
    ancestor_results = []
    import sys, torch
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

    interesting_pairs = [
        ("def f(x, y): return x + y", "def f(x, y): return x - y"),
        ("def f(x, y): return x * y", "def f(x, y): return x / y"),
        ("def f(x, y): return x + y", "def f(x, y): return x * y"),
        ("def f(x): return abs(x)", "def f(x): return -x"),
        ("def f(x, y): return x > y", "def f(x, y): return x == y"),
    ]

    for src_a, src_b in interesting_pairs:
        if src_a in src_to_idx and src_b in src_to_idx:
            va = z_ast[src_to_idx[src_a]]
            vb = z_ast[src_to_idx[src_b]]
            ancestor = (va + vb) / 2
            ancestor_code = gen(ancestor)
            print(f"  {src_a[:25]} + {src_b[:25]}")
            print(f"    Ancestor: {ancestor_code[:40]}")
            ancestor_results.append({
                'func_a': src_a, 'func_b': src_b,
                'ancestor': ancestor_code,
            })

    # === Genetic distance matrix ===
    # How many "mutations" between functions?
    print("\n--- Genetic Distance Summary ---")
    avg_dist = float(np.mean(dists))
    min_dist = float(np.min(dists))
    max_dist = float(np.max(dists))
    print(f"  Avg genetic distance: {avg_dist:.3f}")
    print(f"  Min (closest relatives): {min_dist:.3f}")
    print(f"  Max (most divergent): {max_dist:.3f}")

    elapsed = time.time() - t0
    results = {
        'phase': 44, 'name': 'The Rosetta Genome',
        'n_species': N,
        'siblings': sibling_results,
        'ancestors': ancestor_results,
        'avg_genetic_distance': avg_dist,
        'min_distance': min_dist, 'max_distance': max_dist,
        'elapsed': elapsed, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase44_genome.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # 1. Dendrogram (phylogenetic tree)
    labels = [s.split('return')[-1].strip()[:18] if 'return' in s else s[:18]
             for s in rep_funcs]
    dendrogram(Z, labels=labels, ax=axes[0], leaf_rotation=90, leaf_font_size=7)
    axes[0].set_title('Phylogenetic Tree of Programs\n(Ward linkage on cosine distance)',
                     fontweight='bold')
    axes[0].set_ylabel('Distance')

    # 2. Distance heatmap
    im = axes[1].imshow(dist_matrix, cmap='viridis', aspect='auto')
    axes[1].set_xticks(range(N))
    axes[1].set_yticks(range(N))
    axes[1].set_xticklabels(labels, rotation=90, fontsize=6)
    axes[1].set_yticklabels(labels, fontsize=6)
    axes[1].set_title('Genetic Distance Matrix', fontweight='bold')
    plt.colorbar(im, ax=axes[1], shrink=0.8)

    plt.suptitle('Phase 44: The Rosetta Genome\n'
                 'Evolutionary relationships between programs',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase44_genome.png'), dpi=150)
    plt.close()
    print(f"\nPhase 44 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
