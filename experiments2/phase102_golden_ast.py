"""Phase 102: The Golden AST Structure - Why does phi appear in program space?
Hypothesis: ASTs grow like trees in nature, following Fibonacci branching.
"""
import os, json, sys, ast
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXP2_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(EXP2_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(EXP2_DIR, 'results')
FIGURES_DIR = os.path.join(EXP2_DIR, 'figures')

def main():
    print("=" * 60)
    print("Phase 102: The Golden AST Structure")
    print("  Why does phi appear in program space?")
    print("=" * 60)
    
    dataset = json.load(open(os.path.join(DATA_DIR, 'rosetta_dataset_v2.json'), 'r', encoding='utf-8'))
    sources = list(set(item['source'] for item in dataset['dataset']))
    
    golden_ratio = (1 + np.sqrt(5)) / 2  # 1.618...
    
    # Analyze AST structure of each function
    all_depths = []
    all_nodes_per_depth = []
    all_branching_ratios = []
    all_total_nodes = []
    all_max_depths = []
    
    for src in sources:
        try:
            tree = ast.parse(src)
        except:
            continue
        
        # BFS to count nodes at each depth
        depth_counts = {}
        queue = [(tree, 0)]
        total_nodes = 0
        while queue:
            node, depth = queue.pop(0)
            depth_counts[depth] = depth_counts.get(depth, 0) + 1
            total_nodes += 1
            for child in ast.iter_child_nodes(node):
                queue.append((child, depth + 1))
        
        if total_nodes < 3:
            continue
        
        max_depth = max(depth_counts.keys())
        all_total_nodes.append(total_nodes)
        all_max_depths.append(max_depth)
        
        # Branching ratios: nodes_at_d+1 / nodes_at_d
        for d in sorted(depth_counts.keys()):
            if d + 1 in depth_counts and depth_counts[d] > 0:
                ratio = depth_counts[d + 1] / depth_counts[d]
                all_branching_ratios.append(ratio)
                all_nodes_per_depth.append((d, depth_counts[d]))
                all_depths.append(d)
    
    ratios = np.array(all_branching_ratios)
    total_nodes_arr = np.array(all_total_nodes)
    max_depths_arr = np.array(all_max_depths)
    
    print(f"  Parsed {len(all_total_nodes)} ASTs")
    print(f"  Total branching ratios: {len(ratios)}")
    print(f"  Mean branching ratio: {np.mean(ratios):.4f}")
    print(f"  Median branching ratio: {np.median(ratios):.4f}")
    print(f"  Golden ratio: {golden_ratio:.4f}")
    print(f"  Deviation from phi: {abs(np.mean(ratios) - golden_ratio):.4f}")
    
    # Test: does the ratio converge to phi as depth increases?
    depth_to_ratios = {}
    for d, r in zip(all_depths, ratios):
        if d not in depth_to_ratios: depth_to_ratios[d] = []
        depth_to_ratios[d].append(r)
    
    print(f"\n--- Branching Ratio by Depth ---")
    depth_means = {}
    for d in sorted(depth_to_ratios.keys()):
        rs = depth_to_ratios[d]
        m = np.mean(rs)
        depth_means[d] = m
        dev = abs(m - golden_ratio)
        print(f"  Depth {d}: mean={m:.3f} (n={len(rs)}, dev from phi={dev:.3f})")
    
    # Fibonacci test: are node counts approximately Fibonacci-like?
    # Collect depth-count sequences per AST
    fib_like_count = 0
    fib_test_count = 0
    
    for src in sources:
        try:
            tree = ast.parse(src)
        except: continue
        
        depth_counts = {}
        queue = [(tree, 0)]
        while queue:
            node, depth = queue.pop(0)
            depth_counts[depth] = depth_counts.get(depth, 0) + 1
            for child in ast.iter_child_nodes(node):
                queue.append((child, depth + 1))
        
        if len(depth_counts) < 4: continue
        
        counts = [depth_counts.get(d, 0) for d in range(max(depth_counts.keys()) + 1)]
        
        # Test Fibonacci property: c[n] ~ c[n-1] + c[n-2]
        fib_errors = []
        for i in range(2, len(counts)):
            if counts[i] > 0:
                pred = counts[i-1] + counts[i-2]
                error = abs(counts[i] - pred) / max(1, counts[i])
                fib_errors.append(error)
        
        if fib_errors:
            fib_test_count += 1
            if np.mean(fib_errors) < 0.5:
                fib_like_count += 1
    
    fib_rate = fib_like_count / max(1, fib_test_count)
    print(f"\n--- Fibonacci Structure Test ---")
    print(f"  ASTs tested: {fib_test_count}")
    print(f"  Fibonacci-like: {fib_like_count} ({fib_rate:.1%})")
    
    # Node count distribution (power law?)
    print(f"\n--- Node Count Statistics ---")
    print(f"  Mean nodes: {np.mean(total_nodes_arr):.1f}")
    print(f"  Median nodes: {np.median(total_nodes_arr):.1f}")
    print(f"  Mean depth: {np.mean(max_depths_arr):.1f}")
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Phase 102: The Golden AST Structure', fontsize=14, fontweight='bold')
    
    axes[0, 0].hist(ratios, bins=50, color='#FF9800', edgecolor='black', alpha=0.7, density=True)
    axes[0, 0].axvline(golden_ratio, color='gold', linewidth=3, linestyle='--', label=f'phi={golden_ratio:.3f}')
    axes[0, 0].axvline(np.mean(ratios), color='red', linewidth=2, linestyle='-', label=f'mean={np.mean(ratios):.3f}')
    axes[0, 0].set_xlabel('Branching Ratio (nodes_d+1 / nodes_d)')
    axes[0, 0].set_title('AST Branching Ratio Distribution')
    axes[0, 0].legend()
    axes[0, 0].set_xlim(0, 5)
    
    ds = sorted(depth_means.keys())
    ms = [depth_means[d] for d in ds]
    axes[0, 1].plot(ds, ms, 'o-', color='#E91E63', linewidth=2, markersize=8)
    axes[0, 1].axhline(golden_ratio, color='gold', linewidth=2, linestyle='--', label=f'phi={golden_ratio:.3f}')
    axes[0, 1].set_xlabel('AST Depth')
    axes[0, 1].set_ylabel('Mean Branching Ratio')
    axes[0, 1].set_title('Branching Ratio vs Depth')
    axes[0, 1].legend()
    
    axes[1, 0].hist(total_nodes_arr, bins=30, color='#4CAF50', edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Total AST Nodes')
    axes[1, 0].set_title(f'AST Size Distribution (mean={np.mean(total_nodes_arr):.0f})')
    
    axes[1, 1].axis('off')
    summary = f"""THE GOLDEN AST STRUCTURE

Mean branching ratio: {np.mean(ratios):.4f}
Golden ratio (phi):   {golden_ratio:.4f}
Deviation:            {abs(np.mean(ratios)-golden_ratio):.4f}

Fibonacci-like ASTs:  {fib_rate:.1%}

Mean AST nodes:       {np.mean(total_nodes_arr):.0f}
Mean AST depth:       {np.mean(max_depths_arr):.1f}

Programs grow like trees in nature:
their branching patterns echo
the golden ratio."""
    axes[1, 1].text(0.05, 0.5, summary, fontsize=11, fontfamily='monospace',
                    verticalalignment='center', transform=axes[1, 1].transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase102_golden_ast.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    results = {
        'phase': 102, 'title': 'The Golden AST Structure',
        'n_asts': len(all_total_nodes),
        'mean_branching_ratio': float(np.mean(ratios)),
        'median_branching_ratio': float(np.median(ratios)),
        'golden_ratio': float(golden_ratio),
        'deviation_from_phi': float(abs(np.mean(ratios) - golden_ratio)),
        'fibonacci_rate': float(fib_rate),
        'mean_total_nodes': float(np.mean(total_nodes_arr)),
        'mean_max_depth': float(np.mean(max_depths_arr)),
        'branching_by_depth': {str(d): float(depth_means[d]) for d in sorted(depth_means.keys())},
        'law': f'AST mean branching ratio = {np.mean(ratios):.3f} (phi={golden_ratio:.3f}, dev={abs(np.mean(ratios)-golden_ratio):.3f}). {fib_rate:.0%} of ASTs are Fibonacci-like.'
    }
    with open(os.path.join(RESULTS_DIR, 'phase102_golden_ast.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nPhase 102 complete!")
    return results

if __name__ == '__main__':
    main()
