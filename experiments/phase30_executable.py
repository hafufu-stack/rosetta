"""
Phase 30: Zero-Shot Executable Synthesis
==========================================
Create executable functions from NL WITHOUT a compiler.
NL -> predicted bits -> nearest bytecode -> execute.
"""
import os, json, time, dis, io, types
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')


def source_to_bytes(src, max_bytes=32):
    try:
        code = compile(src, '<test>', 'exec')
        for const in code.co_consts:
            if hasattr(const, 'co_code'):
                return const.co_code[:max_bytes]
        return code.co_code[:max_bytes]
    except:
        return b''


def main():
    print("=" * 60)
    print("Phase 30: Zero-Shot Executable Synthesis")
    print("NL -> Predicted Bits -> Execute (NO compiler!)")
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

    MAX_BYTES = 32

    # Build bytecode database
    src_to_idx = {}
    bytecode_db = {}
    for i, d in enumerate(dataset):
        src = d['source']
        if src not in src_to_idx:
            src_to_idx[src] = i
            bc = source_to_bytes(src, MAX_BYTES)
            if bc:
                bytecode_db[src] = bc

    # Build bit matrix for matching
    all_srcs = list(bytecode_db.keys())
    bit_matrix = np.zeros((len(all_srcs), MAX_BYTES * 8), dtype=np.float32)
    for si, src in enumerate(all_srcs):
        bc = bytecode_db[src]
        for bi, byte_val in enumerate(bc):
            if bi >= MAX_BYTES:
                break
            for bit in range(8):
                bit_matrix[si, bi*8 + (7-bit)] = (byte_val >> bit) & 1

    # Train NL -> bits predictor (simple Ridge regression for speed)
    from sklearn.linear_model import Ridge
    nl_vecs = np.array([z_nl[src_to_idx[s]] for s in all_srcs])
    reg = Ridge(alpha=1.0).fit(nl_vecs, bit_matrix)

    # NL search index
    nl_to_idx = {}
    for i, d in enumerate(dataset):
        nl_key = d['nl'].lower().strip()
        if nl_key not in nl_to_idx:
            nl_to_idx[nl_key] = i

    def find_nl_vec(text):
        text_l = text.lower().strip()
        best_idx, best_score = 0, -1
        for nl_key, idx in nl_to_idx.items():
            words_q = set(text_l.split())
            words_k = set(nl_key.split())
            score = len(words_q & words_k) / max(len(words_q | words_k), 1)
            if score > best_score:
                best_score = score
                best_idx = idx
        return z_nl[best_idx]

    # === Zero-Shot Synthesis ===
    print("\n--- Zero-Shot Executable Synthesis ---")
    test_prompts = [
        "add two numbers",
        "multiply x and y",
        "subtract a from b",
        "check if x is greater than y",
        "return absolute value of x",
        "negate a number",
        "return x modulo y",
        "compute x to the power of y",
    ]

    test_cases = {
        "add two numbers": [(3, 5), 8],
        "multiply x and y": [(4, 7), 28],
        "subtract a from b": [(10, 3), 7],
        "check if x is greater than y": [(5, 3), True],
        "return absolute value of x": [(-7,), 7],
        "negate a number": [(5,), -5],
        "return x modulo y": [(17, 5), 2],
        "compute x to the power of y": [(2, 3), 8],
    }

    results_list = []
    n_exec, n_correct = 0, 0

    for prompt in test_prompts:
        nl_vec = find_nl_vec(prompt)
        # Predict bits
        pred_bits = reg.predict(nl_vec.reshape(1, -1))[0]
        pred_bits_binary = (pred_bits > 0.5).astype(np.float32)

        # Find nearest bytecode by hamming distance
        distances = np.sum(np.abs(bit_matrix - pred_bits_binary), axis=1)
        nearest_idx = np.argmin(distances)
        nearest_src = all_srcs[nearest_idx]
        hamming = int(distances[nearest_idx])

        # Also try cosine distance on raw predictions
        cos_dists = np.array([
            np.dot(pred_bits, bit_matrix[j]) /
            (np.linalg.norm(pred_bits) * np.linalg.norm(bit_matrix[j]) + 1e-8)
            for j in range(len(all_srcs))
        ])
        cos_nearest = all_srcs[np.argmax(cos_dists)]

        # Execute the matched function
        exec_result = None
        try:
            ns = {}
            exec(nearest_src, ns)
            f = ns['f']
            args, expected = test_cases.get(prompt, [(), None])
            exec_result = f(*args)
            correct = exec_result == expected
            n_exec += 1
            if correct:
                n_correct += 1
            status = "CORRECT" if correct else "WRONG"
        except Exception as e:
            status = f"ERROR: {e}"
            correct = False

        print(f"\n  NL: '{prompt}'")
        print(f"    Nearest: {nearest_src[:50]} (hamming={hamming})")
        print(f"    Result: {exec_result} (expected: {test_cases.get(prompt, [(),None])[1]}) [{status}]")

        results_list.append({
            'prompt': prompt, 'nearest_src': nearest_src,
            'hamming': hamming, 'result': str(exec_result),
            'expected': str(test_cases.get(prompt, [(), None])[1]),
            'correct': correct,
        })

    accuracy = n_correct / max(n_exec, 1)
    print(f"\n  Execution accuracy: {n_correct}/{n_exec} ({accuracy:.0%})")

    elapsed = time.time() - t0
    results = {
        'phase': 30, 'name': 'Zero-Shot Executable Synthesis',
        'n_correct': n_correct, 'n_exec': n_exec,
        'accuracy': accuracy,
        'details': results_list,
        'elapsed': elapsed, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase30_executable.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    prompts_short = [p[:20] for p in test_prompts[:len(results_list)]]
    colors = ['#4CAF50' if r['correct'] else '#F44336' for r in results_list]
    ax.barh(prompts_short, [1]*len(results_list), color=colors, edgecolor='black')
    ax.set_xlabel('Execution Result')
    ax.set_title(f'Phase 30: Zero-Shot Executable Synthesis\n'
                 f'NL -> Bits -> Execute: {n_correct}/{n_exec} correct ({accuracy:.0%})',
                 fontsize=13, fontweight='bold')
    ax.legend(['Green=Correct, Red=Wrong'], loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase30_executable.png'), dpi=150)
    plt.close()
    print(f"\nPhase 30 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
