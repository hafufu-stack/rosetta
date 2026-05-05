"""
Phase 5: The Rosetta Scaling
==============================
Scale dataset from 837 to 10K+ triplets via combinatorial generation.
Re-train alignment model and re-test semantic arithmetic.
"""
import os, sys, json, time, ast, dis, io, types, random, itertools
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')

# ============================================================
# Combinatorial Function Generator
# ============================================================
BINARY_OPS = {
    '+': ['add', 'sum', 'plus'],
    '-': ['subtract', 'minus', 'difference'],
    '*': ['multiply', 'product', 'times'],
    '/': ['divide', 'quotient', 'ratio'],
    '//': ['floor divide', 'integer divide', 'floor division'],
    '%': ['modulo', 'remainder', 'mod'],
    '**': ['power', 'exponentiate', 'raise to'],
}

COMPARE_OPS = {
    '>': ['greater than', 'exceeds', 'larger than'],
    '<': ['less than', 'smaller than', 'below'],
    '==': ['equals', 'is equal to', 'same as'],
    '!=': ['not equal to', 'differs from', 'unequal to'],
    '>=': ['at least', 'greater or equal', 'no less than'],
    '<=': ['at most', 'less or equal', 'no more than'],
}

UNARY_FUNCS = {
    'abs({x})': ['absolute value of {x}', 'magnitude of {x}', 'abs of {x}'],
    '-{x}': ['negate {x}', 'negative of {x}', 'flip sign of {x}'],
    '{x} * {x}': ['square {x}', '{x} squared', '{x} times itself'],
    '{x} * {x} * {x}': ['cube {x}', '{x} cubed', '{x} to the third'],
    'int({x})': ['integer part of {x}', 'truncate {x}', 'int of {x}'],
    'float({x})': ['float of {x}', 'convert {x} to float', '{x} as decimal'],
    'bool({x})': ['truth value of {x}', 'bool of {x}', 'is {x} truthy'],
}

STR_FUNCS = {
    '{s}.upper()': ['uppercase {s}', '{s} to caps', 'capitalize all of {s}'],
    '{s}.lower()': ['lowercase {s}', '{s} to lower', 'decapitalize {s}'],
    '{s}.strip()': ['strip {s}', 'trim whitespace from {s}', 'clean {s}'],
    '{s}[::-1]': ['reverse {s}', '{s} backwards', 'flip {s}'],
    'len({s})': ['length of {s}', 'count chars in {s}', 'size of {s}'],
    '{s}.title()': ['title case {s}', 'capitalize words in {s}', 'title of {s}'],
    '{s}.swapcase()': ['swap case of {s}', 'invert case of {s}', 'toggle case'],
}

LIST_FUNCS = {
    'sum({L})': ['sum of {L}', 'total of {L}', 'add all in {L}'],
    'len({L})': ['length of {L}', 'count items in {L}', 'size of {L}'],
    'max({L})': ['maximum of {L}', 'largest in {L}', 'biggest in {L}'],
    'min({L})': ['minimum of {L}', 'smallest in {L}', 'least in {L}'],
    'sorted({L})': ['sort {L}', '{L} in order', 'arrange {L}'],
    'list(reversed({L}))': ['reverse {L}', '{L} backwards', 'flip {L} order'],
}

VAR_PAIRS = [('x','y'), ('a','b'), ('m','n'), ('p','q')]
VAR_SINGLES = ['x', 'a', 'n', 'v']
STR_VARS = ['s', 't', 'w', 'text']
LIST_VARS = ['lst', 'arr', 'nums', 'items']

def ast_to_graph(tree):
    nodes, edges = [], []
    def visit(node, parent_idx=-1):
        idx = len(nodes)
        nodes.append(type(node).__name__)
        if parent_idx >= 0: edges.append((parent_idx, idx))
        for child in ast.iter_child_nodes(node):
            visit(child, idx)
    visit(tree)
    return {'nodes': nodes, 'edges': edges, 'n_nodes': len(nodes)}

def get_bytecode(source):
    code = compile(source, '<rosetta>', 'exec')
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            return list(const.co_code)
    return list(code.co_code)

def make_triplet(source, nl, tid):
    tree = ast.parse(source)
    graph = ast_to_graph(tree)
    bytecode = get_bytecode(source)
    return {
        'id': tid, 'source': source, 'nl': nl,
        'ast_graph': graph, 'bytecode': bytecode,
        'bytecode_len': len(bytecode), 'n_ast_nodes': graph['n_nodes'],
    }

def generate_all():
    dataset = []
    seen = set()

    def add(src, nl):
        key = (src, nl)
        if key not in seen:
            seen.add(key)
            try:
                dataset.append(make_triplet(src, nl, len(dataset)))
            except:
                pass

    # Binary ops with variable name variants
    for op, nls in BINARY_OPS.items():
        for v1, v2 in VAR_PAIRS:
            src = f"def f({v1}, {v2}): return {v1} {op} {v2}"
            for nl in nls:
                add(src, f"{nl} {v1} and {v2}")
                add(src, f"Compute {nl} of {v1} and {v2}")
                add(src, f"Return {v1} {op} {v2}")
            # Composed: op then abs
            src2 = f"def f({v1}, {v2}): return abs({v1} {op} {v2})"
            add(src2, f"absolute {nl[0]} of {v1} and {v2}")

    # Comparisons
    for op, nls in COMPARE_OPS.items():
        for v1, v2 in VAR_PAIRS:
            src = f"def f({v1}, {v2}): return {v1} {op} {v2}"
            for nl in nls:
                add(src, f"Check if {v1} is {nl} {v2}")
                add(src, f"Test {v1} {nl} {v2}")

    # Unary functions
    for expr, nls in UNARY_FUNCS.items():
        for v in VAR_SINGLES:
            src = f"def f({v}): return {expr.format(x=v)}"
            for nl in nls:
                add(src, nl.format(x=v))

    # String functions
    for expr, nls in STR_FUNCS.items():
        for v in STR_VARS:
            src = f"def f({v}): return {expr.format(s=v)}"
            for nl in nls:
                add(src, nl.format(s=v))

    # List functions
    for expr, nls in LIST_FUNCS.items():
        for v in LIST_VARS:
            src = f"def f({v}): return {expr.format(L=v)}"
            for nl in nls:
                add(src, nl.format(L=v))

    # Conditionals
    for v1, v2 in VAR_PAIRS:
        add(f"def f({v1}, {v2}): return {v1} if {v1} > {v2} else {v2}",
            f"Return the larger of {v1} and {v2}")
        add(f"def f({v1}, {v2}): return {v1} if {v1} < {v2} else {v2}",
            f"Return the smaller of {v1} and {v2}")
        add(f"def f({v1}): return {v1} if {v1} > 0 else -{v1}",
            f"Manual absolute value of {v1}")
        add(f"def f({v1}): return {v1} if {v1} >= 0 else 0",
            f"Clamp {v1} to non-negative")

    # Boolean
    for v1, v2 in VAR_PAIRS:
        add(f"def f({v1}, {v2}): return {v1} and {v2}", f"Logical AND of {v1} and {v2}")
        add(f"def f({v1}, {v2}): return {v1} or {v2}", f"Logical OR of {v1} and {v2}")
        add(f"def f({v1}): return not {v1}", f"Logical NOT of {v1}")

    # Compositions (2-deep)
    for op1, nls1 in list(BINARY_OPS.items())[:4]:
        for op2, nls2 in list(BINARY_OPS.items())[:4]:
            if op1 != op2:
                for v1, v2 in VAR_PAIRS[:2]:
                    src = f"def f({v1}, {v2}): return ({v1} {op1} {v2}) {op2} {v1}"
                    add(src, f"{nls1[0]} {v1},{v2} then {nls2[0]} with {v1}")

    # Multi-arg functions
    for v in VAR_SINGLES:
        add(f"def f({v}): return {v} + 1", f"Increment {v}")
        add(f"def f({v}): return {v} - 1", f"Decrement {v}")
        add(f"def f({v}): return {v} * 2", f"Double {v}")
        add(f"def f({v}): return {v} / 2", f"Halve {v}")
        add(f"def f({v}): return {v} % 2 == 0", f"Is {v} even")
        add(f"def f({v}): return {v} % 2 != 0", f"Is {v} odd")

    # NL augmentation (prefix/suffix variants)
    base = list(dataset)
    for d in base:
        for prefix in ["Please ", "Can you ", "I want to "]:
            add(d['source'], prefix + d['nl'].lower())

    return dataset


def main():
    print("=" * 60)
    print("Phase 5: The Rosetta Scaling")
    print("=" * 60)
    t0 = time.time()

    dataset = generate_all()
    print(f"Generated {len(dataset)} triplets")

    # Collect vocabs
    all_node_types = set()
    all_words = set()
    for d in dataset:
        all_node_types.update(d['ast_graph']['nodes'])
        all_words.update(d['nl'].lower().split())
    node_type_vocab = sorted(all_node_types)
    nl_vocab = sorted(all_words)
    n_unique_src = len(set(d['source'] for d in dataset))

    print(f"Unique sources: {n_unique_src}")
    print(f"AST vocab: {len(node_type_vocab)}, NL vocab: {len(nl_vocab)}")

    # Save
    save_data = {
        'dataset': dataset,
        'node_type_vocab': node_type_vocab,
        'nl_vocab': nl_vocab,
        'stats': {'total': len(dataset), 'unique_sources': n_unique_src}
    }
    with open(os.path.join(DATA_DIR, 'rosetta_dataset_v2.json'), 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    # ---- Re-train Phase 2 model with scaled data ----
    print("\n--- Re-training Tri-Modal Alignment on scaled data ---")
    sys.path.insert(0, os.path.dirname(__file__))

    # Import Phase 2 encoding functions
    from experiments.phase2_latent_alignment import (
        encode_nl_bow, encode_ast_features, encode_bytecode,
        NLEncoder, ASTEncoder, BytecodeEncoder, clip_loss, LATENT_DIM
    )
    import torch

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NL_DIM = min(len(nl_vocab), 200)
    AST_FEAT = 30
    BC_LEN = 100

    nl_data, ast_f, ast_a, ast_n, bc_data = [], [], [], [], []
    unique_sources = list(set(d['source'] for d in dataset))
    src2id = {s: i for i, s in enumerate(unique_sources)}
    source_ids = []

    for d in dataset:
        nl_data.append(encode_nl_bow(d['nl'], nl_vocab, NL_DIM))
        nf, adj, n = encode_ast_features(d['ast_graph'], node_type_vocab,
                                          feat_dim=AST_FEAT)
        ast_f.append(nf); ast_a.append(adj); ast_n.append(n)
        bc_data.append(encode_bytecode(d['bytecode'], BC_LEN))
        source_ids.append(src2id[d['source']])

    nl_t = torch.tensor(np.array(nl_data), dtype=torch.float32)
    af_t = torch.tensor(np.array(ast_f), dtype=torch.float32)
    aa_t = torch.tensor(np.array(ast_a), dtype=torch.float32)
    an_t = torch.tensor(np.array(ast_n), dtype=torch.long)
    bc_t = torch.tensor(np.array(bc_data), dtype=torch.float32)

    N = len(dataset)
    nl_enc = NLEncoder(NL_DIM, LATENT_DIM).to(DEVICE)
    ast_enc = ASTEncoder(AST_FEAT, LATENT_DIM).to(DEVICE)
    bc_enc = BytecodeEncoder(BC_LEN, LATENT_DIM).to(DEVICE)

    optimizer = torch.optim.Adam(
        list(nl_enc.parameters()) + list(ast_enc.parameters()) +
        list(bc_enc.parameters()), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 300)

    BATCH = 128
    losses = []
    for epoch in range(300):
        perm = torch.randperm(N)
        eloss, nb = 0, 0
        for i in range(0, N, BATCH):
            idx = perm[i:i+BATCH]
            if len(idx) < 4: continue
            z_nl = nl_enc(nl_t[idx].to(DEVICE))
            z_ast = ast_enc(af_t[idx].to(DEVICE), aa_t[idx].to(DEVICE),
                           an_t[idx].to(DEVICE))
            z_bc = bc_enc(bc_t[idx].to(DEVICE))
            loss = (clip_loss(z_nl, z_ast) + clip_loss(z_nl, z_bc) +
                    clip_loss(z_ast, z_bc)) / 3
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(nl_enc.parameters()) + list(ast_enc.parameters()) +
                list(bc_enc.parameters()), 1.0)
            optimizer.step()
            eloss += loss.item(); nb += 1
        scheduler.step()
        losses.append(eloss / max(nb, 1))
        if (epoch+1) % 100 == 0:
            print(f"  Epoch {epoch+1}/300: loss={losses[-1]:.4f}")

    # Extract latents
    nl_enc.eval(); ast_enc.eval(); bc_enc.eval()
    with torch.no_grad():
        all_nl = nl_enc(nl_t.to(DEVICE)).cpu().numpy()
        all_ast = ast_enc(af_t.to(DEVICE), aa_t.to(DEVICE),
                          an_t.to(DEVICE)).cpu().numpy()
        all_bc = bc_enc(bc_t.to(DEVICE)).cpu().numpy()

    np.savez(os.path.join(DATA_DIR, 'rosetta_latents_v2.npz'),
             nl=all_nl, ast=all_ast, bc=all_bc,
             labels=np.array(source_ids))

    # ---- Re-test Phase 4 analogies ----
    print("\n--- Re-testing Semantic Arithmetic ---")
    src_to_idx = {}
    for i, d in enumerate(dataset):
        if d['source'] not in src_to_idx:
            src_to_idx[d['source']] = i

    analogies = [
        ("def f(x, y): return x + y", "def f(x, y): return x - y",
         "def f(x, y): return x * y", "def f(x, y): return x / y",
         "add:sub :: mul:?div"),
        ("def f(x, y): return x + y", "def f(x, y): return x * y",
         "def f(x, y): return x - y", "def f(x, y): return x / y",
         "add:mul :: sub:?div"),
        ("def f(x, y): return x > y", "def f(x, y): return x < y",
         "def f(x, y): return x >= y", "def f(x, y): return x <= y",
         "gt:lt :: gte:?lte"),
        ("def f(x): return x + 1", "def f(x): return x - 1",
         "def f(x): return x * 2", "def f(x): return x / 2",
         "inc:dec :: double:?halve"),
        ("def f(s): return s.upper()", "def f(s): return s.lower()",
         "def f(s): return len(s)", "def f(s): return s[::-1]",
         "upper:lower :: len:?reverse"),
    ]

    correct_counts = {'NL': 0, 'AST': 0, 'BC': 0}
    total_valid = 0

    for src_a, src_b, src_c, src_d, desc in analogies:
        if not all(s in src_to_idx for s in [src_a, src_b, src_c, src_d]):
            continue
        total_valid += 1
        ia, ib, ic, id_ = (src_to_idx[s] for s in [src_a, src_b, src_c, src_d])

        for name, z in [('NL', all_nl), ('AST', all_ast), ('BC', all_bc)]:
            v_pred = z[ia] - z[ib] + z[ic]
            v_pred /= (np.linalg.norm(v_pred) + 1e-8)
            best_sim, best_src = -1, None
            for src, idx in src_to_idx.items():
                sim = float(np.dot(v_pred, z[idx]) /
                           (np.linalg.norm(z[idx]) + 1e-8))
                if sim > best_sim:
                    best_sim, best_src = sim, src
            if best_src == src_d:
                correct_counts[name] += 1

        target_sim_nl = float(np.dot(
            (all_nl[ia]-all_nl[ib]+all_nl[ic]) /
            (np.linalg.norm(all_nl[ia]-all_nl[ib]+all_nl[ic])+1e-8),
            all_nl[id_]))
        print(f"  {desc}: NL_sim={target_sim_nl:.3f}")

    for name in ['NL', 'AST', 'BC']:
        acc = correct_counts[name] / max(total_valid, 1)
        print(f"  {name} analogy accuracy: {correct_counts[name]}/{total_valid} = {acc:.0%}")

    elapsed = time.time() - t0
    results = {
        'phase': 5, 'name': 'The Rosetta Scaling',
        'dataset_size': len(dataset), 'unique_sources': n_unique_src,
        'final_loss': float(losses[-1]),
        'analogy_accuracy': {k: v/max(total_valid,1) for k,v in correct_counts.items()},
        'elapsed': elapsed,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase5_rosetta_scaling.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Figure
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(losses, color='#E91E63', lw=1.5)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title(f'Training Loss ({N} samples)', fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    accs = [correct_counts[k]/max(total_valid,1) for k in ['NL','AST','BC']]
    bars = axes[1].bar(['NL','AST','Bytecode'], accs,
                       color=['#2196F3','#4CAF50','#FF9800'], edgecolor='black')
    for b, v in zip(bars, accs):
        axes[1].text(b.get_x()+b.get_width()/2, v+0.02, f'{v:.0%}',
                     ha='center', fontweight='bold', fontsize=13)
    axes[1].set_ylabel('Accuracy'); axes[1].set_ylim(0, 1.1)
    axes[1].set_title('Semantic Arithmetic (Scaled)', fontweight='bold')
    plt.suptitle('Phase 5: The Rosetta Scaling', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase5_rosetta_scaling.png'), dpi=150)
    plt.close()

    print(f"\nPhase 5 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
