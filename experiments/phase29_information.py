"""
Phase 29: Information-Theoretic Compilation (Opus Bonus)
==========================================================
How many bits of information does each NL word carry about each
binary instruction? The true "information cost" of compilation.
Measures mutual information between NL and Binary dimensions.
"""
import os, json, time
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')


def mutual_info_discrete(x, y, n_bins=10):
    """Estimate MI between two continuous vars via histogram binning."""
    hist_2d, _, _ = np.histogram2d(x, y, bins=n_bins)
    pxy = hist_2d / hist_2d.sum()
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    mi = 0.0
    for i in range(n_bins):
        for j in range(n_bins):
            if pxy[i,j] > 0 and px[i] > 0 and py[j] > 0:
                mi += pxy[i,j] * np.log2(pxy[i,j] / (px[i] * py[j]))
    return mi


def main():
    print("=" * 60)
    print("Phase 29: Information-Theoretic Compilation")
    print("The true information cost of compilation (Opus Bonus)")
    print("=" * 60)
    t0 = time.time()

    lat_file = os.path.join(DATA_DIR, 'rosetta_latents_v2.npz')
    if not os.path.exists(lat_file):
        lat_file = os.path.join(DATA_DIR, 'rosetta_latents.npz')
    latents = np.load(lat_file)
    z_nl, z_ast, z_bc = latents['nl'], latents['ast'], latents['bc']
    D = z_nl.shape[1]  # 64

    # === MI(NL_dim_i, Binary_dim_j) ===
    print("  Computing MI(NL, Binary) matrix...")
    MI_nl_bc = np.zeros((D, D))
    for i in range(D):
        for j in range(D):
            MI_nl_bc[i, j] = mutual_info_discrete(z_nl[:, i], z_bc[:, j])
    total_mi_nl_bc = MI_nl_bc.sum()

    print("  Computing MI(AST, Binary) matrix...")
    MI_ast_bc = np.zeros((D, D))
    for i in range(D):
        for j in range(D):
            MI_ast_bc[i, j] = mutual_info_discrete(z_ast[:, i], z_bc[:, j])
    total_mi_ast_bc = MI_ast_bc.sum()

    print("  Computing MI(NL, AST) matrix...")
    MI_nl_ast = np.zeros((D, D))
    for i in range(D):
        for j in range(D):
            MI_nl_ast[i, j] = mutual_info_discrete(z_nl[:, i], z_ast[:, j])
    total_mi_nl_ast = MI_nl_ast.sum()

    print(f"\n  Total MI(NL, Binary):  {total_mi_nl_bc:.2f} bits")
    print(f"  Total MI(AST, Binary): {total_mi_ast_bc:.2f} bits")
    print(f"  Total MI(NL, AST):     {total_mi_nl_ast:.2f} bits")

    # Information loss during compilation
    info_loss = total_mi_nl_ast - total_mi_nl_bc
    print(f"\n  Information loss (NL->AST vs NL->Bin): {info_loss:.2f} bits")
    print(f"  Compilation preserves {total_mi_nl_bc/total_mi_nl_ast*100:.1f}% of NL info")

    # Top NL dims by total MI with Binary
    nl_mi_total = MI_nl_bc.sum(axis=1)  # Sum over binary dims
    top_nl = np.argsort(nl_mi_total)[::-1][:10]
    print(f"\n  Most informative NL dims for Binary: {top_nl.tolist()}")
    print(f"  Their MI: {[f'{nl_mi_total[d]:.3f}' for d in top_nl]}")

    # Top Binary dims by total MI with NL
    bc_mi_total = MI_nl_bc.sum(axis=0)
    top_bc = np.argsort(bc_mi_total)[::-1][:10]
    print(f"  Most informative Binary dims from NL: {top_bc.tolist()}")

    # Information bottleneck: how many dims carry 90% of MI?
    sorted_nl_mi = np.sort(nl_mi_total)[::-1]
    cumulative = np.cumsum(sorted_nl_mi) / sorted_nl_mi.sum()
    n90_nl = int(np.searchsorted(cumulative, 0.9)) + 1
    print(f"\n  90% of NL->Binary MI in {n90_nl} NL dims")

    elapsed = time.time() - t0
    results = {
        'phase': 29, 'name': 'Information-Theoretic Compilation',
        'total_mi_nl_bc': float(total_mi_nl_bc),
        'total_mi_ast_bc': float(total_mi_ast_bc),
        'total_mi_nl_ast': float(total_mi_nl_ast),
        'info_preservation': float(total_mi_nl_bc / total_mi_nl_ast),
        'n90_nl_dims': n90_nl,
        'top_nl_dims': top_nl.tolist(),
        'top_bc_dims': top_bc.tolist(),
        'elapsed': elapsed, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase29_information.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. MI(NL, Binary) heatmap
    im1 = axes[0,0].imshow(MI_nl_bc, aspect='auto', cmap='inferno')
    axes[0,0].set_xlabel('Binary Dimension')
    axes[0,0].set_ylabel('NL Dimension')
    axes[0,0].set_title('MI(NL, Binary)\nWord-to-Instruction Information',
                        fontweight='bold')
    plt.colorbar(im1, ax=axes[0,0], shrink=0.8)

    # 2. MI comparison bars
    labels = ['NL <-> AST', 'AST <-> Binary', 'NL <-> Binary']
    vals = [total_mi_nl_ast, total_mi_ast_bc, total_mi_nl_bc]
    colors = ['#4CAF50', '#2196F3', '#E91E63']
    bars = axes[0,1].bar(labels, vals, color=colors, edgecolor='black')
    for b, v in zip(bars, vals):
        axes[0,1].text(b.get_x()+b.get_width()/2, v+1, f'{v:.1f}',
                      ha='center', fontweight='bold')
    axes[0,1].set_ylabel('Total Mutual Information (bits)')
    axes[0,1].set_title('Information Flow Through Compilation',
                        fontweight='bold')

    # 3. Per-NL-dim MI
    axes[1,0].bar(range(D), nl_mi_total, color='#FF5722', alpha=0.7)
    axes[1,0].set_xlabel('NL Dimension')
    axes[1,0].set_ylabel('Total MI with Binary')
    axes[1,0].set_title('NL Dimension Informativeness\nfor Bytecode Prediction',
                        fontweight='bold')

    # 4. MI(AST, Binary) heatmap
    im4 = axes[1,1].imshow(MI_ast_bc, aspect='auto', cmap='inferno')
    axes[1,1].set_xlabel('Binary Dimension')
    axes[1,1].set_ylabel('AST Dimension')
    axes[1,1].set_title('MI(AST, Binary)\nCompilation Information', fontweight='bold')
    plt.colorbar(im4, ax=axes[1,1], shrink=0.8)

    plt.suptitle('Phase 29: Information-Theoretic Compilation\n'
                 'The true information cost of turning words into binary',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase29_information.png'), dpi=150)
    plt.close()
    print(f"\nPhase 29 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
