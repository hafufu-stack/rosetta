"""
Phase 38: Semantic Entanglement
==================================
Are there pairs of dimensions that are "entangled"?
Knowing one tells you everything about the other.
If so, the TRUE dimensionality is even lower than 20.
"""
import os, json, time
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')


def main():
    print("=" * 60)
    print("Phase 38: Semantic Entanglement")
    print("Which dimensions are quantum-entangled?")
    print("=" * 60)
    t0 = time.time()

    lat_file = os.path.join(DATA_DIR, 'rosetta_latents_v2.npz')
    if not os.path.exists(lat_file):
        lat_file = os.path.join(DATA_DIR, 'rosetta_latents.npz')
    latents = np.load(lat_file)
    z_nl, z_ast, z_bc = latents['nl'], latents['ast'], latents['bc']

    # === Correlation matrix between all modality dimensions ===
    print("\n--- Cross-Modality Correlation ---")

    # NL-AST correlation
    corr_nl_ast = np.corrcoef(z_nl.T, z_ast.T)[:64, 64:]  # (64, 64)
    # NL-BC correlation
    corr_nl_bc = np.corrcoef(z_nl.T, z_bc.T)[:64, 64:]
    # AST-BC correlation
    corr_ast_bc = np.corrcoef(z_ast.T, z_bc.T)[:64, 64:]

    # Find highly entangled pairs (|corr| > 0.8)
    print("\n  Highly entangled NL-AST pairs (|r| > 0.8):")
    entangled_nl_ast = []
    for i in range(64):
        for j in range(64):
            r = float(corr_nl_ast[i, j])
            if abs(r) > 0.8:
                entangled_nl_ast.append({'nl_dim': i, 'ast_dim': j, 'r': r})
                print(f"    NL[{i}] <-> AST[{j}]: r={r:.3f}")

    print(f"  Total entangled NL-AST pairs: {len(entangled_nl_ast)}")

    print("\n  Highly entangled AST-BC pairs (|r| > 0.8):")
    entangled_ast_bc = []
    for i in range(64):
        for j in range(64):
            r = float(corr_ast_bc[i, j])
            if abs(r) > 0.8:
                entangled_ast_bc.append({'ast_dim': i, 'bc_dim': j, 'r': r})
                print(f"    AST[{i}] <-> BC[{j}]: r={r:.3f}")

    print(f"  Total entangled AST-BC pairs: {len(entangled_ast_bc)}")

    # === Effective dimensionality via PCA ===
    print("\n--- Effective Dimensionality ---")
    for name, z in [('NL', z_nl), ('AST', z_ast), ('Binary', z_bc)]:
        cov = np.cov(z.T)
        eigvals = np.linalg.eigvalsh(cov)[::-1]
        eigvals = eigvals[eigvals > 0]
        # Participation ratio (effective dim)
        pr = (np.sum(eigvals))**2 / np.sum(eigvals**2)
        # 90% variance dims
        cumvar = np.cumsum(eigvals) / np.sum(eigvals)
        n90 = int(np.searchsorted(cumvar, 0.9)) + 1
        n95 = int(np.searchsorted(cumvar, 0.95)) + 1
        print(f"  {name}: participation ratio={pr:.1f}, 90%var={n90} dims, 95%var={n95} dims")

    # === Entanglement graph ===
    # Build adjacency: which dims are connected across modalities?
    print("\n--- Entanglement Network ---")
    # Count how many cross-modality partners each dim has
    nl_partners = {}
    for e in entangled_nl_ast:
        nl_partners.setdefault(e['nl_dim'], []).append(e['ast_dim'])

    # Multi-entangled dims (connected to multiple partners)
    multi = {k: v for k, v in nl_partners.items() if len(v) > 1}
    print(f"  NL dims entangled with multiple AST dims: {len(multi)}")
    for dim, partners in list(multi.items())[:5]:
        print(f"    NL[{dim}] -> AST{partners}")

    # === Within-modality entanglement ===
    print("\n--- Within-Modality Entanglement ---")
    corr_ast = np.corrcoef(z_ast.T)
    np.fill_diagonal(corr_ast, 0)  # Ignore self-correlation
    max_corr = np.max(np.abs(corr_ast))
    n_high = np.sum(np.abs(corr_ast) > 0.7) // 2  # Symmetric
    print(f"  AST max within-corr: {max_corr:.3f}")
    print(f"  AST dim pairs with |r| > 0.7: {n_high}")

    # Top within-AST entangled pairs
    intra_pairs = []
    for i in range(64):
        for j in range(i+1, 64):
            r = float(corr_ast[i, j])
            if abs(r) > 0.6:
                intra_pairs.append({'dim_a': i, 'dim_b': j, 'r': r})

    intra_pairs.sort(key=lambda x: -abs(x['r']))
    print(f"  Top within-AST entangled pairs:")
    for p in intra_pairs[:5]:
        print(f"    AST[{p['dim_a']}] <-> AST[{p['dim_b']}]: r={p['r']:.3f}")

    elapsed = time.time() - t0
    results = {
        'phase': 38, 'name': 'Semantic Entanglement',
        'n_entangled_nl_ast': len(entangled_nl_ast),
        'n_entangled_ast_bc': len(entangled_ast_bc),
        'entangled_nl_ast': entangled_nl_ast[:20],
        'entangled_ast_bc': entangled_ast_bc[:20],
        'multi_entangled': {str(k): v for k, v in list(multi.items())[:10]},
        'intra_ast_pairs': intra_pairs[:10],
        'elapsed': elapsed, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase38_entanglement.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. NL-AST correlation heatmap
    im1 = axes[0].imshow(np.abs(corr_nl_ast), aspect='auto', cmap='hot', vmin=0, vmax=1)
    axes[0].set_xlabel('AST Dimension')
    axes[0].set_ylabel('NL Dimension')
    axes[0].set_title('|Correlation| NL <-> AST\nEntanglement Map', fontweight='bold')
    plt.colorbar(im1, ax=axes[0], shrink=0.8)

    # 2. AST-BC correlation heatmap
    im2 = axes[1].imshow(np.abs(corr_ast_bc), aspect='auto', cmap='hot', vmin=0, vmax=1)
    axes[1].set_xlabel('Binary Dimension')
    axes[1].set_ylabel('AST Dimension')
    axes[1].set_title('|Correlation| AST <-> Binary\nCompilation Entanglement',
                     fontweight='bold')
    plt.colorbar(im2, ax=axes[1], shrink=0.8)

    # 3. Within-AST correlation
    im3 = axes[2].imshow(np.abs(corr_ast), aspect='auto', cmap='inferno', vmin=0, vmax=1)
    axes[2].set_xlabel('AST Dimension')
    axes[2].set_ylabel('AST Dimension')
    axes[2].set_title('Within-AST Entanglement\n(internal redundancy)', fontweight='bold')
    plt.colorbar(im3, ax=axes[2], shrink=0.8)

    plt.suptitle('Phase 38: Semantic Entanglement\n'
                 'Which dimensions are quantum-linked?',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase38_entanglement.png'), dpi=150)
    plt.close()
    print(f"\nPhase 38 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
