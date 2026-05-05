"""
Phase 1: The Rosetta Dataset
============================
Generate (NL, AST, Bytecode) triplets from simple Python functions.

- NL: Template-based natural language descriptions
- PL: AST (Abstract Syntax Tree) extracted via `ast` module -> graph
- Bin: Bytecode extracted via `compile()` -> byte sequence

Target: 1000+ triplets across 20+ function templates.
"""
import os, sys, json, time, ast, dis, io, types, random, hashlib
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ============================================================
# Function Templates: (source_code, nl_descriptions[])
# ============================================================
TEMPLATES = [
    # Arithmetic
    ("def f(x, y): return x + y", ["Add two numbers", "Compute the sum of x and y", "Return x plus y"]),
    ("def f(x, y): return x - y", ["Subtract y from x", "Compute the difference of x and y", "Return x minus y"]),
    ("def f(x, y): return x * y", ["Multiply two numbers", "Compute the product of x and y", "Return x times y"]),
    ("def f(x, y): return x / y", ["Divide x by y", "Compute the quotient of x and y", "Return x divided by y"]),
    ("def f(x, y): return x % y", ["Compute x modulo y", "Get the remainder of x divided by y", "Return x mod y"]),
    ("def f(x, y): return x ** y", ["Raise x to the power y", "Compute x to the y-th power", "Return x raised to y"]),
    ("def f(x, y): return x // y", ["Integer divide x by y", "Floor division of x by y", "Return x integer-divided by y"]),
    # Unary
    ("def f(x): return -x", ["Negate x", "Return the negative of x", "Flip the sign of x"]),
    ("def f(x): return abs(x)", ["Get absolute value of x", "Return the magnitude of x", "Compute abs of x"]),
    ("def f(x): return x * x", ["Square x", "Compute x squared", "Return x times x"]),
    ("def f(x): return x * x * x", ["Cube x", "Compute x cubed", "Return x to the third power"]),
    # Comparison
    ("def f(x, y): return max(x, y)", ["Get the larger of x and y", "Return the maximum of x and y", "Find the bigger value"]),
    ("def f(x, y): return min(x, y)", ["Get the smaller of x and y", "Return the minimum of x and y", "Find the smaller value"]),
    ("def f(x, y): return x > y", ["Check if x is greater than y", "Test whether x exceeds y", "Return true if x is larger"]),
    ("def f(x, y): return x == y", ["Check if x equals y", "Test whether x and y are the same", "Return true if equal"]),
    # Boolean
    ("def f(x, y): return x and y", ["Logical AND of x and y", "Return x and y", "Check both x and y are truthy"]),
    ("def f(x, y): return x or y", ["Logical OR of x and y", "Return x or y", "Check if either x or y is truthy"]),
    ("def f(x): return not x", ["Logical NOT of x", "Return the negation of x", "Invert the truth value of x"]),
    # Conditional
    ("def f(x): return x if x > 0 else -x", ["Return abs of x via conditional", "If x positive return x else negate", "Manual absolute value"]),
    ("def f(x, y): return x if x > y else y", ["Return the larger via conditional", "Manual max of x and y", "If x exceeds y return x else y"]),
    # String
    ("def f(s): return len(s)", ["Get the length of s", "Count characters in s", "Return how long s is"]),
    ("def f(s): return s.upper()", ["Convert s to uppercase", "Make s all caps", "Return uppercased s"]),
    ("def f(s): return s.lower()", ["Convert s to lowercase", "Make s all lowercase", "Return lowercased s"]),
    ("def f(s): return s[::-1]", ["Reverse the string s", "Return s backwards", "Flip s end to start"]),
    ("def f(a, b): return a + b", ["Concatenate a and b", "Join strings a and b", "Combine a and b"]),
    # List
    ("def f(lst): return sum(lst)", ["Sum all elements in lst", "Compute total of lst", "Add up all values in lst"]),
    ("def f(lst): return len(lst)", ["Get list length", "Count items in lst", "Return number of elements"]),
    ("def f(lst): return sorted(lst)", ["Sort the list", "Return lst in order", "Arrange lst ascending"]),
    ("def f(lst): return list(reversed(lst))", ["Reverse the list", "Return lst backwards", "Flip lst order"]),
    ("def f(lst): return max(lst)", ["Find maximum in lst", "Get largest element", "Return biggest value in lst"]),
    ("def f(lst): return min(lst)", ["Find minimum in lst", "Get smallest element", "Return smallest value in lst"]),
]

# Variable name variants for augmentation
VAR_PAIRS = [('x','y'), ('a','b'), ('m','n'), ('p','q'), ('i','j')]
VAR_SINGLES = ['x', 'a', 'n', 'val', 'num']

def ast_to_graph(tree):
    """Convert AST to graph: list of (node_type, parent_idx) pairs."""
    nodes = []
    edges = []
    node_types = set()

    def visit(node, parent_idx=-1):
        idx = len(nodes)
        ntype = type(node).__name__
        nodes.append(ntype)
        node_types.add(ntype)
        if parent_idx >= 0:
            edges.append((parent_idx, idx))
        for child in ast.iter_child_nodes(node):
            visit(child, idx)

    visit(tree)
    return {'nodes': nodes, 'edges': edges, 'n_nodes': len(nodes)}

def get_bytecode(source):
    """Compile source and extract bytecode bytes."""
    code = compile(source, '<rosetta>', 'exec')
    # Find the function code object
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            return list(const.co_code)
    return list(code.co_code)

def get_bytecode_ops(source):
    """Get human-readable bytecode ops."""
    code = compile(source, '<rosetta>', 'exec')
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            buf = io.StringIO()
            dis.dis(const, file=buf)
            return buf.getvalue()
    buf = io.StringIO()
    dis.dis(code, file=buf)
    return buf.getvalue()

def augment_varnames(source, nl, old_vars, new_vars):
    """Replace variable names for augmentation."""
    s, n = source, nl
    for old, new in zip(old_vars, new_vars):
        s = s.replace(f'({old}', f'({new}').replace(f' {old})', f' {new})')
        s = s.replace(f' {old} ', f' {new} ').replace(f' {old},', f' {new},')
        s = s.replace(f'({old},', f'({new},').replace(f' {old}:', f' {new}:')
        s = s.replace(f'return {old}', f'return {new}')
    return s, n  # NL stays the same (describes semantics, not variables)


def main():
    print("=" * 60)
    print("Phase 1: The Rosetta Dataset")
    print("=" * 60)
    t0 = time.time()

    dataset = []
    errors = 0

    for src, nl_list in TEMPLATES:
        for nl in nl_list:
            try:
                tree = ast.parse(src)
                graph = ast_to_graph(tree)
                bytecode = get_bytecode(src)
                bc_ops = get_bytecode_ops(src)

                triplet = {
                    'id': len(dataset),
                    'source': src,
                    'nl': nl,
                    'ast_graph': graph,
                    'bytecode': bytecode,
                    'bytecode_len': len(bytecode),
                    'bytecode_ops': bc_ops,
                    'n_ast_nodes': graph['n_nodes'],
                }
                dataset.append(triplet)
            except Exception as e:
                errors += 1
                print(f"  Error on '{src[:40]}': {e}")

    n_base = len(dataset)
    print(f"Base triplets: {n_base}")

    # Augment with random noise in NL (paraphrase via shuffling)
    aug_count = 0
    for src, nl_list in TEMPLATES:
        for nl in nl_list:
            # Add prefix/suffix variants
            for prefix in ["Please ", "Can you ", ""]:
                for suffix in ["", " and return the result", " for me"]:
                    if prefix == "" and suffix == "":
                        continue  # Skip the original
                    aug_nl = prefix + nl.lower() + suffix
                    try:
                        tree = ast.parse(src)
                        graph = ast_to_graph(tree)
                        bytecode = get_bytecode(src)
                        dataset.append({
                            'id': len(dataset),
                            'source': src,
                            'nl': aug_nl,
                            'ast_graph': graph,
                            'bytecode': bytecode,
                            'bytecode_len': len(bytecode),
                            'n_ast_nodes': graph['n_nodes'],
                        })
                        aug_count += 1
                    except:
                        pass

    print(f"Augmented: +{aug_count} -> Total: {len(dataset)}")

    # Collect all unique AST node types
    all_node_types = set()
    for d in dataset:
        all_node_types.update(d['ast_graph']['nodes'])
    node_type_vocab = sorted(all_node_types)
    print(f"AST node type vocabulary: {len(node_type_vocab)} types")

    # Collect all unique NL words
    all_words = set()
    for d in dataset:
        all_words.update(d['nl'].lower().split())
    nl_vocab = sorted(all_words)
    print(f"NL word vocabulary: {len(nl_vocab)} words")

    # Stats
    bc_lengths = [d['bytecode_len'] for d in dataset]
    ast_sizes = [d['n_ast_nodes'] for d in dataset]
    n_unique_sources = len(set(d['source'] for d in dataset))
    n_unique_nls = len(set(d['nl'] for d in dataset))

    print(f"\nDataset Statistics:")
    print(f"  Total triplets: {len(dataset)}")
    print(f"  Unique sources: {n_unique_sources}")
    print(f"  Unique NL descriptions: {n_unique_nls}")
    print(f"  Bytecode length: min={min(bc_lengths)}, max={max(bc_lengths)}, "
          f"mean={np.mean(bc_lengths):.1f}")
    print(f"  AST nodes: min={min(ast_sizes)}, max={max(ast_sizes)}, "
          f"mean={np.mean(ast_sizes):.1f}")

    # Save dataset
    save_data = {
        'dataset': dataset,
        'node_type_vocab': node_type_vocab,
        'nl_vocab': nl_vocab,
        'stats': {
            'total': len(dataset),
            'unique_sources': n_unique_sources,
            'unique_nls': n_unique_nls,
            'bc_len_mean': float(np.mean(bc_lengths)),
            'ast_nodes_mean': float(np.mean(ast_sizes)),
        }
    }
    with open(os.path.join(DATA_DIR, 'rosetta_dataset.json'), 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    # Save results
    elapsed = time.time() - t0
    results = {
        'phase': 1,
        'name': 'The Rosetta Dataset',
        'total_triplets': len(dataset),
        'unique_sources': n_unique_sources,
        'unique_nls': n_unique_nls,
        'node_type_vocab_size': len(node_type_vocab),
        'nl_vocab_size': len(nl_vocab),
        'bc_len_stats': {'min': min(bc_lengths), 'max': max(bc_lengths),
                         'mean': float(np.mean(bc_lengths))},
        'ast_stats': {'min': min(ast_sizes), 'max': max(ast_sizes),
                      'mean': float(np.mean(ast_sizes))},
        'errors': errors,
        'elapsed': elapsed,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase1_rosetta_dataset.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Visualization
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].hist(bc_lengths, bins=20, color='#2196F3', edgecolor='black')
    axes[0].set_xlabel('Bytecode Length (bytes)', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Bytecode Length Distribution', fontsize=13, fontweight='bold')

    axes[1].hist(ast_sizes, bins=15, color='#4CAF50', edgecolor='black')
    axes[1].set_xlabel('AST Node Count', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title('AST Size Distribution', fontsize=13, fontweight='bold')

    # NL word count distribution
    nl_lens = [len(d['nl'].split()) for d in dataset]
    axes[2].hist(nl_lens, bins=15, color='#FF9800', edgecolor='black')
    axes[2].set_xlabel('NL Word Count', fontsize=12)
    axes[2].set_ylabel('Count', fontsize=12)
    axes[2].set_title('NL Description Length', fontsize=13, fontweight='bold')

    plt.suptitle('Phase 1: The Rosetta Dataset', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase1_rosetta_dataset.png'), dpi=150)
    plt.close()

    print(f"\nPhase 1 complete in {elapsed:.1f}s")
    print(f"Dataset saved to data/rosetta_dataset.json")
    return results


if __name__ == '__main__':
    main()
