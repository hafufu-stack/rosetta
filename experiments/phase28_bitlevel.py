"""
Phase 28: Bit-Level Rosetta Stone
====================================
The ultimate grounding: natural language -> individual bits (0/1).
Which bits flip when you say "add" vs "subtract"?
"""
import os, json, time, dis, io
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')


def source_to_bits(src, max_bytes=64):
    """Convert Python source to raw bit string via bytecode."""
    try:
        code = compile(src, '<test>', 'exec')
        # Get the inner function's bytecode
        for const in code.co_consts:
            if hasattr(const, 'co_code'):
                raw = const.co_code
                break
        else:
            raw = code.co_code

        # Convert to bits
        bits = []
        for byte_val in raw[:max_bytes]:
            for bit in range(8):
                bits.append((byte_val >> (7 - bit)) & 1)

        # Pad to fixed length
        target_len = max_bytes * 8
        while len(bits) < target_len:
            bits.append(0)
        return bits[:target_len]
    except:
        return [0] * (max_bytes * 8)


def main():
    print("=" * 60)
    print("Phase 28: Bit-Level Rosetta Stone")
    print("Natural language -> individual bits (0/1)")
    print("=" * 60)
    t0 = time.time()

    ds_file = os.path.join(DATA_DIR, 'rosetta_dataset_v2.json')
    if not os.path.exists(ds_file):
        ds_file = os.path.join(DATA_DIR, 'rosetta_dataset.json')
    with open(ds_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)['dataset']

    lat_file = os.path.join(DATA_DIR, 'rosetta_latents_v2.npz')
    if not os.path.exists(lat_file):
        lat_file = os.path.join(DATA_DIR, 'rosetta_latents.npz')
    latents = np.load(lat_file)
    z_nl = latents['nl']

    # Convert all sources to bit arrays
    MAX_BYTES = 32  # 256 bits
    N_BITS = MAX_BYTES * 8
    print(f"  Converting to {N_BITS}-bit representations...")

    bits_matrix = np.zeros((len(dataset), N_BITS), dtype=np.float32)
    for i, d in enumerate(dataset):
        bits_matrix[i] = source_to_bits(d['source'], MAX_BYTES)

    print(f"  Bit matrix: {bits_matrix.shape}")
    print(f"  Avg bits set: {bits_matrix.mean():.3f}")

    # Train: NL -> each bit (logistic regression)
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        z_nl, bits_matrix, test_size=0.2, random_state=42)

    print("\n  Training per-bit predictors (NL -> bit)...")
    bit_accuracies = np.zeros(N_BITS)
    bit_coefs = np.zeros((N_BITS, 64))  # Coefficients for each bit

    # Only train on bits that have variance
    active_bits = []
    for b in range(N_BITS):
        if y_train[:, b].std() > 0.01:
            active_bits.append(b)

    print(f"  Active bits (with variance): {len(active_bits)}/{N_BITS}")

    for b in active_bits:
        try:
            lr = LogisticRegression(max_iter=200, C=1.0, solver='lbfgs')
            lr.fit(X_train, y_train[:, b])
            acc = lr.score(X_test, y_test[:, b])
            bit_accuracies[b] = acc
            bit_coefs[b] = lr.coef_[0]
        except:
            bit_accuracies[b] = 0.5

    active_accs = bit_accuracies[active_bits]
    mean_acc = float(np.mean(active_accs))
    max_acc = float(np.max(active_accs))
    n_above_80 = int(np.sum(active_accs > 0.8))
    n_above_90 = int(np.sum(active_accs > 0.9))
    print(f"\n  Mean bit accuracy: {mean_acc:.3f}")
    print(f"  Max bit accuracy: {max_acc:.3f}")
    print(f"  Bits with >80% accuracy: {n_above_80}")
    print(f"  Bits with >90% accuracy: {n_above_90}")

    # Which NL dimensions most strongly predict each bit?
    print("\n--- Top NL dimensions per bit cluster ---")
    # Average absolute coefficients across all bits
    mean_coef = np.abs(bit_coefs[active_bits]).mean(axis=0)
    top_nl_dims = np.argsort(mean_coef)[::-1][:10]
    print(f"  Most influential NL dims: {top_nl_dims.tolist()}")
    print(f"  Their mean |coef|: {[f'{mean_coef[d]:.3f}' for d in top_nl_dims]}")

    # Differential: which bits differ between add and subtract?
    print("\n--- Bit-Level Differentials ---")
    diff_pairs = [
        ("def f(x, y): return x + y", "def f(x, y): return x - y", "add vs sub"),
        ("def f(x, y): return x > y", "def f(x, y): return x < y", "gt vs lt"),
        ("def f(s): return s.upper()", "def f(s): return len(s)", "upper vs len"),
    ]

    diff_results = []
    src_to_idx = {}
    for i, d in enumerate(dataset):
        if d['source'] not in src_to_idx:
            src_to_idx[d['source']] = i

    for src_a, src_b, desc in diff_pairs:
        if src_a in src_to_idx and src_b in src_to_idx:
            bits_a = bits_matrix[src_to_idx[src_a]]
            bits_b = bits_matrix[src_to_idx[src_b]]
            diff = bits_a - bits_b  # Which bits flipped?
            n_flipped = int(np.sum(np.abs(diff) > 0.5))
            flip_positions = np.where(np.abs(diff) > 0.5)[0].tolist()
            print(f"  {desc}: {n_flipped} bits flipped at positions {flip_positions[:10]}")
            diff_results.append({
                'desc': desc, 'n_flipped': n_flipped,
                'positions': flip_positions[:20],
            })

    elapsed = time.time() - t0
    results = {
        'phase': 28, 'name': 'Bit-Level Rosetta Stone',
        'n_bits': N_BITS, 'active_bits': len(active_bits),
        'mean_accuracy': mean_acc, 'max_accuracy': max_acc,
        'bits_above_80': n_above_80, 'bits_above_90': n_above_90,
        'top_nl_dims': top_nl_dims.tolist(),
        'differentials': diff_results,
        'elapsed': elapsed, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase28_bitlevel.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Bit accuracy distribution
    axes[0].hist(active_accs, bins=30, color='#9C27B0', edgecolor='black', alpha=0.8)
    axes[0].axvline(mean_acc, color='red', ls='--', label=f'Mean={mean_acc:.3f}')
    axes[0].set_xlabel('Prediction Accuracy')
    axes[0].set_ylabel('Number of Bits')
    axes[0].set_title('Per-Bit Prediction from NL\n(logistic regression)', fontweight='bold')
    axes[0].legend()

    # 2. NL dimension importance
    sorted_coef = np.sort(mean_coef)[::-1][:20]
    axes[1].bar(range(20), sorted_coef, color='#FF5722', edgecolor='black')
    axes[1].set_xlabel('NL Dimension (ranked)')
    axes[1].set_ylabel('Mean |Coefficient|')
    axes[1].set_title('Most Influential NL Dimensions\nfor Bit Prediction', fontweight='bold')

    # 3. Bit accuracy heatmap (reshape to byte grid)
    bit_grid = bit_accuracies[:MAX_BYTES*8].reshape(MAX_BYTES, 8)
    im = axes[2].imshow(bit_grid.T, aspect='auto', cmap='hot', vmin=0.5, vmax=1.0)
    axes[2].set_xlabel('Byte Position')
    axes[2].set_ylabel('Bit Position (MSB->LSB)')
    axes[2].set_title('Bit Prediction Accuracy\n(language -> 0/1)', fontweight='bold')
    plt.colorbar(im, ax=axes[2], shrink=0.8)

    plt.suptitle('Phase 28: Bit-Level Rosetta Stone\n'
                 'The ultimate grounding: words -> individual bits',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase28_bitlevel.png'), dpi=150)
    plt.close()
    print(f"\nPhase 28 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
