"""Phase 131: The Rosetta Singularity - Encode Rosetta's own code
Map Project Rosetta's scripts into their own 64D space.
Does the code that built the universe appear at its center?
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

def extract_functions_from_file(filepath):
    """Extract all function sources from a Python file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()
        tree = ast.parse(source)
        funcs = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                try:
                    func_source = ast.get_source_segment(source, node)
                    if func_source and len(func_source) < 5000:
                        funcs.append({'name': node.name, 'source': func_source,
                                     'n_lines': node.end_lineno - node.lineno + 1 if hasattr(node, 'end_lineno') else 1})
                except Exception:
                    pass
        return funcs
    except Exception:
        return []

def simple_ast_hash(source):
    """Create a simple numeric fingerprint from AST structure."""
    try:
        tree = ast.parse(source)
        feature = np.zeros(64)
        for i, node in enumerate(ast.walk(tree)):
            node_type = type(node).__name__
            h = hash(node_type) % 64
            feature[h] += 1
            # Add depth info
            feature[(h + 1) % 64] += 0.1 * (i % 10)
        # Normalize
        norm = np.linalg.norm(feature)
        if norm > 0:
            feature /= norm
        return feature
    except Exception:
        return np.zeros(64)

def main():
    print("=" * 60)
    print("Phase 131: The Rosetta Singularity")
    print("  Does the universe-building code sit at the center?")
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
    
    # 1. Universe properties
    centroid = np.mean(ast_m, axis=0)
    
    # Find the function closest to the centroid
    dists_to_centroid = np.linalg.norm(ast_m - centroid, axis=1)
    center_idx = np.argmin(dists_to_centroid)
    center_func = unique_funcs[center_idx].split('return ')[-1].strip()[:20]
    
    print(f"  Universe centroid nearest function: '{center_func}'")
    print(f"  Distance to centroid: {dists_to_centroid[center_idx]:.4f}")
    
    # 2. Extract Rosetta's own functions from key scripts
    rosetta_scripts = [
        os.path.join(BASE_DIR, 'experiments', 'phase1_contrastive.py'),
        os.path.join(BASE_DIR, 'experiments', 'phase2_neural_cpu.py'),
        os.path.join(BASE_DIR, 'experiments', 'phase3_compile_matrix.py'),
        os.path.join(BASE_DIR, 'experiments', 'phase9_generative_decompiler.py'),
        os.path.join(EXP2_DIR, 'phase108_routing.py'),
        os.path.join(EXP2_DIR, 'phase118_multiverse.py'),
        os.path.join(EXP2_DIR, 'phase124_entanglement.py'),
    ]
    
    rosetta_funcs = []
    for script in rosetta_scripts:
        if os.path.exists(script):
            funcs = extract_functions_from_file(script)
            for f in funcs:
                f['script'] = os.path.basename(script)
            rosetta_funcs.extend(funcs)
    
    print(f"\n  Rosetta functions extracted: {len(rosetta_funcs)}")
    
    # 3. Create AST fingerprints for Rosetta's own code
    rosetta_vectors = []
    for func in rosetta_funcs:
        vec = simple_ast_hash(func['source'])
        rosetta_vectors.append(vec)
    rosetta_vectors = np.array(rosetta_vectors) if rosetta_vectors else np.zeros((0, 64))
    
    if len(rosetta_vectors) > 0:
        # Project Rosetta vectors into the existing space
        # Map via cosine similarity to existing functions
        rosetta_positions = []
        for rv in rosetta_vectors:
            # Find nearest existing function by AST hash similarity
            cos_sims = ast_m @ rv / (np.linalg.norm(ast_m, axis=1) * np.linalg.norm(rv) + 1e-10)
            # Weighted average position based on similarity
            weights = np.maximum(cos_sims, 0)
            if np.sum(weights) > 0:
                pos = (weights[:, None] * ast_m).sum(axis=0) / np.sum(weights)
            else:
                pos = centroid
            rosetta_positions.append(pos)
        
        rosetta_positions = np.array(rosetta_positions)
        
        # Distance from each Rosetta function to centroid
        rosetta_to_centroid = np.linalg.norm(rosetta_positions - centroid, axis=1)
        mean_rosetta_dist = np.mean(rosetta_to_centroid)
        mean_all_dist = np.mean(dists_to_centroid)
        
        print(f"\n--- Rosetta Code Position ---")
        print(f"  Mean Rosetta->centroid distance: {mean_rosetta_dist:.4f}")
        print(f"  Mean ALL->centroid distance: {mean_all_dist:.4f}")
        print(f"  Ratio: {mean_rosetta_dist/mean_all_dist:.4f}")
        
        at_center = mean_rosetta_dist < mean_all_dist * 0.5
        print(f"  Rosetta at center: {'YES!' if at_center else 'No'}")
        
        # Find closest Rosetta function to centroid
        closest_rosetta_idx = np.argmin(rosetta_to_centroid)
        closest_rosetta = rosetta_funcs[closest_rosetta_idx]
        print(f"  Closest Rosetta func to center: {closest_rosetta['name']} from {closest_rosetta['script']}")
        print(f"    Distance: {rosetta_to_centroid[closest_rosetta_idx]:.4f}")
        
        # Per-script analysis
        print(f"\n--- Per-script distance to centroid ---")
        script_dists = {}
        for i, func in enumerate(rosetta_funcs):
            script = func['script']
            if script not in script_dists: script_dists[script] = []
            script_dists[script].append(rosetta_to_centroid[i])
        
        for script, dists in sorted(script_dists.items(), key=lambda x: np.mean(x[1])):
            print(f"  {script}: mean_d={np.mean(dists):.4f} ({len(dists)} funcs)")
    else:
        mean_rosetta_dist = 0; at_center = False; rosetta_positions = np.zeros((0, 64))
    
    # 4. Self-reference test: does the space contain its own description?
    # Compute "information content" of the centroid
    eigenvalues = np.linalg.eigvalsh(np.cov(ast_m.T))
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    total_info = -np.sum(eigenvalues / np.sum(eigenvalues) * np.log2(eigenvalues / np.sum(eigenvalues) + 1e-15))
    
    print(f"\n--- Self-reference ---")
    print(f"  Total information content: {total_info:.4f} bits")
    print(f"  Effective dimensions: {np.sum(eigenvalues > 0.01 * max(eigenvalues))}")
    
    # 5. Simulation hypothesis: is the space a fixed point of its own mapping?
    # If F(space) = space, then the Rosetta code IS the universe
    # Test: project centroid through PCA round-trip
    pca = PCA(n_components=32).fit(ast_m)
    centroid_proj = pca.transform(centroid.reshape(1,-1))
    centroid_recon = pca.inverse_transform(centroid_proj)
    recon_error = np.linalg.norm(centroid - centroid_recon[0]) / np.linalg.norm(centroid)
    
    print(f"  Centroid reconstruction error: {recon_error:.6f}")
    print(f"  Fixed point: {'YES (error < 1%)' if recon_error < 0.01 else 'No'}")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Phase 131: The Rosetta Singularity', fontsize=14, fontweight='bold')
    
    pca_2d = PCA(n_components=2).fit_transform(ast_m)
    axes[0].scatter(pca_2d[:,0], pca_2d[:,1], s=15, alpha=0.3, c='gray', label='Dataset')
    c2d = PCA(n_components=2).fit(ast_m).transform(centroid.reshape(1,-1))
    axes[0].scatter(c2d[0,0], c2d[0,1], s=200, c='gold', marker='*', zorder=10, edgecolor='black', label='Centroid')
    if len(rosetta_positions) > 0:
        r2d = PCA(n_components=2).fit(ast_m).transform(rosetta_positions)
        axes[0].scatter(r2d[:,0], r2d[:,1], s=40, c='red', marker='D', zorder=8, label='Rosetta code')
    axes[0].legend(fontsize=8); axes[0].set_title('Rosetta code in its own universe')
    
    axes[1].hist(dists_to_centroid, bins=30, color='#9E9E9E', edgecolor='black', alpha=0.5, label='All funcs')
    if len(rosetta_positions) > 0:
        axes[1].hist(rosetta_to_centroid, bins=15, color='#F44336', edgecolor='black', alpha=0.7, label='Rosetta')
    axes[1].legend(); axes[1].set_xlabel('Distance to centroid')
    axes[1].set_title('Distance distribution')
    
    axes[2].semilogy(np.sort(eigenvalues)[::-1], 'o-', color='#2196F3', markersize=4)
    axes[2].set_xlabel('Eigenvalue index'); axes[2].set_ylabel('Eigenvalue (log)')
    axes[2].set_title(f'Spectrum: {total_info:.1f} bits')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase131_singularity.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    results = {
        'phase': 131, 'title': 'The Rosetta Singularity',
        'center_function': center_func,
        'center_distance': float(dists_to_centroid[center_idx]),
        'rosetta_funcs_extracted': len(rosetta_funcs),
        'mean_rosetta_to_centroid': float(mean_rosetta_dist) if len(rosetta_positions) > 0 else None,
        'mean_all_to_centroid': float(mean_all_dist),
        'rosetta_at_center': bool(at_center),
        'total_info_bits': float(total_info),
        'centroid_fixed_point_error': float(recon_error),
        'law': f'Center function: {center_func}. Rosetta at center: {at_center}. Info={total_info:.1f} bits. Fixed point error={recon_error:.5f}.'
    }
    with open(os.path.join(RESULTS_DIR, 'phase131_singularity.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nPhase 131 complete!")
    return results

if __name__ == '__main__':
    main()
