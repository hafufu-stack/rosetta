"""Phase 98: The Holographic Boundary - Programs live on a shell.
P96 showed packing fraction = 0.855 (shell-like).
Test if the surface of the hypersphere encodes all information.
"""
import os, json, sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(EXPERIMENT_DIR)
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')

def main():
    print("=" * 60)
    print("Phase 98: The Holographic Boundary")
    print("  Do programs live on a shell?")
    print("=" * 60)
    
    latents = np.load(os.path.join(BASE_DIR, 'data', 'rosetta_latents_v2.npz'))
    dataset = json.load(open(os.path.join(BASE_DIR, 'data', 'rosetta_dataset_v2.json'), 'r', encoding='utf-8'))
    ast_vectors = latents['ast']
    sources = [item['source'] for item in dataset['dataset']]
    
    func_to_vecs = {}
    for i, src in enumerate(sources):
        if src not in func_to_vecs: func_to_vecs[src] = []
        func_to_vecs[src].append(ast_vectors[i])
    func_means = {s: np.mean(v, axis=0) for s, v in func_to_vecs.items()}
    unique = list(func_means.keys())
    all_vecs = np.array([func_means[f] for f in unique])
    
    # Compute radial distribution
    centroid = np.mean(all_vecs, axis=0)
    radii = np.linalg.norm(all_vecs - centroid, axis=1)
    
    print(f"  Radii: mean={np.mean(radii):.4f}, std={np.std(radii):.4f}, max={np.max(radii):.4f}")
    print(f"  CV of radii: {np.std(radii)/np.mean(radii):.4f}")
    
    # Project onto unit sphere (normalize to unit vectors from centroid)
    centered = all_vecs - centroid
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    unit_vecs = centered / (norms + 1e-10)
    
    # Test: does NN accuracy change after projection to sphere?
    import inspect
    op_labels = {}
    for f in unique:
        if 'x + y' in f: op_labels[f] = 'add'
        elif 'x - y' in f: op_labels[f] = 'sub'
        elif 'x * y' in f: op_labels[f] = 'mul'
        elif 'x > y' in f or 'x < y' in f: op_labels[f] = 'cmp'
        elif 'max(' in f or 'min(' in f: op_labels[f] = 'minmax'
        else: op_labels[f] = 'other'
    
    labeled = [f for f in unique if op_labels[f] != 'other']
    lab_labels = [op_labels[f] for f in labeled]
    lab_idx = [unique.index(f) for f in labeled]
    
    def nn_accuracy(vecs, indices, labels):
        correct = 0
        for i, idx in enumerate(indices):
            v = vecs[idx]
            dists = np.linalg.norm(vecs[indices] - v, axis=1)
            dists[i] = float('inf')
            nn = np.argmin(dists)
            if labels[nn] == labels[i]: correct += 1
        return correct / len(indices)
    
    acc_original = nn_accuracy(all_vecs, lab_idx, lab_labels)
    acc_sphere = nn_accuracy(unit_vecs, lab_idx, lab_labels)
    
    print(f"\n--- Holographic Test ---")
    print(f"  NN accuracy (original 64D): {acc_original:.3f}")
    print(f"  NN accuracy (sphere projection): {acc_sphere:.3f}")
    print(f"  Info preserved: {acc_sphere/acc_original*100:.1f}%")
    print(f"  {'HOLOGRAPHIC: surface encodes volume!' if acc_sphere >= acc_original * 0.95 else 'Partial info loss'}")
    
    # Intrinsic dimensionality of the shell
    # Use correlation dimension estimate
    nn = NearestNeighbors(n_neighbors=20)
    nn.fit(unit_vecs)
    dists_nn, _ = nn.kneighbors(unit_vecs)
    
    # Estimate intrinsic dim via NN distances
    log_dists = np.log(dists_nn[:, 1:] + 1e-10)
    log_ratios = log_dists[:, -1:] - log_dists[:, 0:1]
    intrinsic_dim = (dists_nn.shape[1] - 2) / np.mean(log_ratios)
    
    print(f"\n--- Shell Dimensionality ---")
    print(f"  Estimated intrinsic dim of shell: {intrinsic_dim:.1f}")
    print(f"  Ambient dim: 64")
    print(f"  Codimension: {64 - intrinsic_dim:.1f}")
    
    # Angular clustering: do programs cluster by angle?
    cos_sims = unit_vecs @ unit_vecs.T
    np.fill_diagonal(cos_sims, 0)
    
    # For labeled functions, are same-class cosine sims higher?
    intra_sims = []
    inter_sims = []
    for i, fi in enumerate(labeled):
        for j, fj in enumerate(labeled):
            if i >= j: continue
            idx_i = unique.index(fi)
            idx_j = unique.index(fj)
            sim = cos_sims[idx_i, idx_j]
            if op_labels[fi] == op_labels[fj]:
                intra_sims.append(sim)
            else:
                inter_sims.append(sim)
    
    mean_intra = np.mean(intra_sims) if intra_sims else 0
    mean_inter = np.mean(inter_sims) if inter_sims else 0
    
    print(f"\n--- Angular Clustering ---")
    print(f"  Same-class cosine sim: {mean_intra:.4f}")
    print(f"  Diff-class cosine sim: {mean_inter:.4f}")
    print(f"  Ratio: {mean_intra/mean_inter:.2f}x")
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Phase 98: The Holographic Boundary', fontsize=14, fontweight='bold')
    
    axes[0, 0].hist(radii, bins=30, color='#2196F3', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(np.mean(radii), color='red', linestyle='--', label=f'mean={np.mean(radii):.3f}')
    axes[0, 0].set_xlabel('Distance from Centroid')
    axes[0, 0].set_title('Radial Distribution (Shell Structure)')
    axes[0, 0].legend()
    
    categories = ['Original 64D', 'Sphere Projection']
    accs = [acc_original*100, acc_sphere*100]
    axes[0, 1].bar(categories, accs, color=['#FF9800', '#4CAF50'], edgecolor='black')
    axes[0, 1].set_ylabel('NN Accuracy (%)')
    axes[0, 1].set_title(f'Holographic: {acc_sphere/acc_original*100:.0f}% info preserved')
    for i, v in enumerate(accs):
        axes[0, 1].text(i, v+1, f'{v:.1f}%', ha='center', fontweight='bold')
    
    axes[1, 0].hist(intra_sims, bins=30, alpha=0.6, color='#4CAF50', label='Same class', edgecolor='black')
    axes[1, 0].hist(inter_sims, bins=30, alpha=0.6, color='#F44336', label='Diff class', edgecolor='black')
    axes[1, 0].set_xlabel('Cosine Similarity')
    axes[1, 0].set_title('Angular Clustering on Shell')
    axes[1, 0].legend()
    
    axes[1, 1].axis('off')
    summary = f"""THE HOLOGRAPHIC PRINCIPLE

Programs live on a shell (rho={np.mean(radii)/np.max(radii):.3f})
Surface encodes {acc_sphere/acc_original*100:.0f}% of volume info

Shell intrinsic dim: {intrinsic_dim:.1f}
Angular clustering: {mean_intra/mean_inter:.2f}x

Same-class cosine: {mean_intra:.3f}
Diff-class cosine: {mean_inter:.3f}

Like black holes encode info on
their event horizon, programs encode
their meaning on the latent sphere."""
    axes[1, 1].text(0.05, 0.5, summary, fontsize=10, fontfamily='monospace',
                    verticalalignment='center', transform=axes[1, 1].transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase98_holographic.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    results = {
        'phase': 98, 'title': 'The Holographic Boundary',
        'mean_radius': float(np.mean(radii)), 'std_radius': float(np.std(radii)),
        'packing_fraction': float(np.mean(radii)/np.max(radii)),
        'nn_acc_original': float(acc_original), 'nn_acc_sphere': float(acc_sphere),
        'info_preserved_pct': float(acc_sphere/acc_original*100),
        'shell_intrinsic_dim': float(intrinsic_dim),
        'intra_class_cosine': float(mean_intra), 'inter_class_cosine': float(mean_inter),
        'angular_ratio': float(mean_intra/mean_inter) if mean_inter > 0 else 0,
        'law': f'Holographic: sphere projection preserves {acc_sphere/acc_original*100:.0f}% of info. Shell dim={intrinsic_dim:.1f}. Angular clustering={mean_intra/mean_inter:.2f}x.'
    }
    with open(os.path.join(RESULTS_DIR, 'phase98_holographic.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nPhase 98 complete!")
    return results

if __name__ == '__main__':
    main()
