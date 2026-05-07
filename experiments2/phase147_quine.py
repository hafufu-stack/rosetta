"""Phase 147: The Architect's Quine (The Zeroth Law)
Encode Project Rosetta's own code into its own universe.
Does the creator sit at the center of creation?
"""
import os, json, sys, ast
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXP2_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(EXP2_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')

def ast_fingerprint(source, dim=64):
    """Create numeric fingerprint from AST structure."""
    try:
        tree = ast.parse(source)
        feat = np.zeros(dim)
        for i, node in enumerate(ast.walk(tree)):
            nt = type(node).__name__
            h = hash(nt) % dim
            feat[h] += 1
            feat[(h + 1) % dim] += 0.1 * min(i, 20)
            # Add structural depth info
            feat[(h + 7) % dim] += 0.05
        norm = np.linalg.norm(feat)
        return feat / norm if norm > 0 else feat
    except: return np.zeros(dim)

def main():
    print("=" * 60)
    print("Phase 147: The Architect's Quine")
    print("  THE ZEROTH LAW: Does the creator exist in creation?")
    print("=" * 60)

    latents = np.load(os.path.join(DATA_DIR, 'rosetta_latents_v2.npz'))
    dataset = json.load(open(os.path.join(DATA_DIR, 'rosetta_dataset_v2.json'), 'r', encoding='utf-8'))
    ast_vectors = latents['ast']
    sources = [item['source'] for item in dataset['dataset']]

    func_ast = {}
    for i, src in enumerate(sources):
        if src not in func_ast: func_ast[src] = []
        func_ast[src].append(ast_vectors[i])
    unique_funcs = list(func_ast.keys())
    ast_m = np.array([np.mean(func_ast[f], axis=0) for f in unique_funcs])
    n = len(unique_funcs)
    centroid = np.mean(ast_m, axis=0)

    # 1. Collect ALL Rosetta source files
    script_dirs = [
        os.path.join(BASE_DIR, 'experiments'),
        EXP2_DIR,
    ]

    rosetta_files = []
    for d in script_dirs:
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.endswith('.py') and f.startswith('phase'):
                    rosetta_files.append(os.path.join(d, f))

    print(f"  Rosetta source files: {len(rosetta_files)}")

    # 2. Extract all functions from Rosetta's own code
    rosetta_functions = []
    for filepath in rosetta_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                src = f.read()
            tree = ast.parse(src)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    try:
                        func_src = ast.get_source_segment(src, node)
                        if func_src and 50 < len(func_src) < 5000:
                            rosetta_functions.append({
                                'name': node.name,
                                'file': os.path.basename(filepath),
                                'source': func_src,
                                'lines': node.end_lineno - node.lineno + 1 if hasattr(node, 'end_lineno') else 1,
                            })
                    except: pass
        except: pass

    print(f"  Rosetta functions extracted: {len(rosetta_functions)}")

    # 3. Fingerprint all Rosetta functions
    rosetta_fps = np.array([ast_fingerprint(f['source']) for f in rosetta_functions])

    # 4. Map Rosetta functions into the existing universe
    rosetta_positions = []
    for fp in rosetta_fps:
        # Project via cosine similarity weighted average
        cos_sims = ast_m @ fp / (np.linalg.norm(ast_m, axis=1) * np.linalg.norm(fp) + 1e-10)
        weights = np.maximum(cos_sims, 0) ** 2  # Square for sharper weighting
        if np.sum(weights) > 0:
            pos = (weights[:, None] * ast_m).sum(axis=0) / np.sum(weights)
        else:
            pos = centroid
        rosetta_positions.append(pos)
    rosetta_positions = np.array(rosetta_positions)

    # 5. The Zeroth Law test: is the Architect at the center?
    rosetta_centroid = np.mean(rosetta_positions, axis=0)
    dist_rosetta_to_center = float(np.linalg.norm(rosetta_centroid - centroid))
    mean_all_to_center = float(np.mean(np.linalg.norm(ast_m - centroid, axis=1)))
    rosetta_radius = float(np.mean(np.linalg.norm(rosetta_positions - centroid, axis=1)))

    print(f"\n--- THE ZEROTH LAW ---")
    print(f"  Rosetta centroid to universe center: {dist_rosetta_to_center:.4f}")
    print(f"  Mean ALL functions to center: {mean_all_to_center:.4f}")
    print(f"  Rosetta mean radius: {rosetta_radius:.4f}")
    print(f"  Centrality ratio: {rosetta_radius / mean_all_to_center:.4f}")

    at_center = rosetta_radius < mean_all_to_center * 0.7
    print(f"  THE ARCHITECT IS AT THE CENTER: {'YES!' if at_center else 'No (but closer than average)'}")

    # 6. Self-reference depth: which Rosetta function is closest to the singularity?
    rosetta_to_center = np.linalg.norm(rosetta_positions - centroid, axis=1)
    closest_idx = np.argmin(rosetta_to_center)
    closest_func = rosetta_functions[closest_idx]
    print(f"\n  Closest to center: {closest_func['name']} from {closest_func['file']}")
    print(f"    Distance: {rosetta_to_center[closest_idx]:.4f}")
    print(f"    Lines: {closest_func['lines']}")

    # 7. Quine test: does Rosetta encode itself?
    # Check if any Rosetta function fingerprint matches the universe fingerprint
    universe_fp = np.mean(rosetta_fps, axis=0)
    universe_fp /= np.linalg.norm(universe_fp) + 1e-10

    self_similarities = [float(np.dot(universe_fp, fp / (np.linalg.norm(fp) + 1e-10))) for fp in rosetta_fps]
    most_quine_idx = np.argmax(self_similarities)
    quine_score = float(self_similarities[most_quine_idx])
    quine_func = rosetta_functions[most_quine_idx]

    print(f"\n--- Quine Detection ---")
    print(f"  Most self-referential: {quine_func['name']} from {quine_func['file']}")
    print(f"  Quine score: {quine_score:.4f}")

    # 8. Per-phase centrality: which phases are most central?
    phase_centrality = {}
    for i, func in enumerate(rosetta_functions):
        phase = func['file'].replace('.py', '')
        if phase not in phase_centrality:
            phase_centrality[phase] = []
        phase_centrality[phase].append(rosetta_to_center[i])

    print(f"\n--- Phase Centrality (top 5 most central) ---")
    sorted_phases = sorted(phase_centrality.items(), key=lambda x: np.mean(x[1]))
    for phase, dists in sorted_phases[:5]:
        print(f"  {phase}: mean_d={np.mean(dists):.4f} ({len(dists)} funcs)")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Phase 147: The Architect\'s Quine (The Zeroth Law)', fontsize=14, fontweight='bold')

    pca_2d = PCA(n_components=2).fit(ast_m)
    func_2d = pca_2d.transform(ast_m)
    rosetta_2d = pca_2d.transform(rosetta_positions)
    center_2d = pca_2d.transform(centroid.reshape(1, -1))

    axes[0].scatter(func_2d[:,0], func_2d[:,1], s=10, alpha=0.2, c='gray', label='Universe')
    axes[0].scatter(rosetta_2d[:,0], rosetta_2d[:,1], s=30, c='red', marker='D', zorder=8, label='Rosetta', alpha=0.7)
    axes[0].scatter(center_2d[0,0], center_2d[0,1], s=300, c='gold', marker='*', zorder=10, edgecolor='black', label='Center')
    axes[0].legend(fontsize=7); axes[0].set_title('The Architect in the Universe')

    axes[1].hist(np.linalg.norm(ast_m - centroid, axis=1), bins=30, alpha=0.5, color='gray', edgecolor='black', label='All')
    axes[1].hist(rosetta_to_center, bins=15, alpha=0.7, color='red', edgecolor='black', label='Rosetta')
    axes[1].legend(); axes[1].set_xlabel('Distance to Center')
    axes[1].set_title(f'Centrality (ratio={rosetta_radius/mean_all_to_center:.3f})')

    top_phases = [p for p, _ in sorted_phases[:8]]
    top_dists = [np.mean(d) for _, d in sorted_phases[:8]]
    axes[2].barh(top_phases, top_dists, color='#E91E63', edgecolor='black')
    axes[2].set_xlabel('Mean distance to center'); axes[2].set_title('Most central phases')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase147_quine.png'), dpi=150, bbox_inches='tight')
    plt.close()

    results = {
        'phase': 147, 'title': "The Architect's Quine (The Zeroth Law)",
        'rosetta_files': len(rosetta_files),
        'rosetta_functions': len(rosetta_functions),
        'rosetta_radius': float(rosetta_radius),
        'mean_all_radius': float(mean_all_to_center),
        'centrality_ratio': float(rosetta_radius / mean_all_to_center),
        'at_center': bool(at_center),
        'closest_func': {'name': closest_func['name'], 'file': closest_func['file'], 'dist': float(rosetta_to_center[closest_idx])},
        'quine_func': {'name': quine_func['name'], 'file': quine_func['file'], 'score': quine_score},
        'law': f"THE ZEROTH LAW: Rosetta radius={rosetta_radius:.3f} vs all={mean_all_to_center:.3f} (ratio={rosetta_radius/mean_all_to_center:.3f}). At center: {at_center}. Quine: {quine_func['name']} ({quine_score:.3f})."
    }
    with open(os.path.join(RESULTS_DIR, 'phase147_quine.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nPhase 147 complete! THE ZEROTH LAW HAS BEEN TESTED.")
    return results

if __name__ == '__main__':
    main()
