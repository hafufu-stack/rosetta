"""
Phase 27: Sequential Semantic Crystallization
================================================
Watch meaning emerge step-by-step as bytecode instructions are read.
At which instruction does "meaningless binary" become "add"?
"""
import os, json, time, dis, io
import numpy as np
import torch
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def extract_opcodes(src):
    """Extract opcode sequence from Python source."""
    try:
        code = compile(src, '<test>', 'exec')
        ops = []
        for instr in dis.get_instructions(code):
            ops.append(instr.opname)
            # Also get function body instructions
        # Get the inner function
        for const in code.co_consts:
            if hasattr(const, 'co_code'):
                for instr in dis.get_instructions(const):
                    ops.append(instr.opname)
        return ops
    except:
        return []


class OpcodeRNN(nn.Module):
    """RNN that processes opcodes one by one, outputting hidden states."""
    def __init__(self, vocab_size, hidden_dim=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 32)
        self.rnn = nn.GRU(32, hidden_dim, batch_first=True)
        self.proj = nn.Linear(hidden_dim, 64)  # Project to Rosetta dim

    def forward(self, x):
        # x: (batch, seq_len)
        emb = self.embed(x)
        outputs, _ = self.rnn(emb)  # (batch, seq_len, hidden)
        projected = self.proj(outputs)  # (batch, seq_len, 64)
        return projected


def main():
    print("=" * 60)
    print("Phase 27: Sequential Semantic Crystallization")
    print("When does binary become meaning?")
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

    # Build opcode vocab
    all_ops = set()
    src_ops = {}
    for d in dataset:
        ops = extract_opcodes(d['source'])
        src_ops[d['source']] = ops
        all_ops.update(ops)

    op2idx = {'<PAD>': 0}
    for op in sorted(all_ops):
        op2idx[op] = len(op2idx)
    V_op = len(op2idx)
    print(f"  Opcode vocab: {V_op} unique opcodes")

    # Prepare training data
    MAX_SEQ = 30
    X_seqs, Y_targets = [], []
    for i, d in enumerate(dataset):
        ops = src_ops.get(d['source'], [])
        if len(ops) < 3:
            continue
        seq = [op2idx.get(op, 0) for op in ops[:MAX_SEQ]]
        while len(seq) < MAX_SEQ:
            seq.append(0)
        X_seqs.append(seq)
        Y_targets.append(z_nl[i])

    X_seqs = torch.tensor(X_seqs, dtype=torch.long)
    Y_targets = torch.tensor(np.array(Y_targets), dtype=torch.float32)
    N = len(X_seqs)
    print(f"  Training samples: {N}")

    # Train RNN
    model = OpcodeRNN(V_op, hidden_dim=64).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    BATCH = 128

    for epoch in range(200):
        perm = torch.randperm(N)
        eloss, nb = 0, 0
        model.train()
        for i in range(0, N, BATCH):
            idx = perm[i:i+BATCH]
            x = X_seqs[idx].to(DEVICE)
            y = Y_targets[idx].to(DEVICE)
            proj = model(x)  # (batch, seq, 64)
            # Loss: final step should match NL vector
            final = proj[:, -1, :]  # Use last step
            loss = nn.functional.mse_loss(final, y)
            optimizer.zero_grad(); loss.backward()
            optimizer.step()
            eloss += loss.item(); nb += 1
        if (epoch+1) % 50 == 0:
            print(f"  Epoch {epoch+1}/200: loss={eloss/max(nb,1):.4f}")

    # === Crystallization Analysis ===
    print("\n--- Semantic Crystallization Trajectories ---")
    model.eval()

    # Pick representative functions
    test_funcs = {}
    for d in dataset:
        src = d['source']
        if src not in test_funcs and len(src_ops.get(src, [])) >= 5:
            test_funcs[src] = d['nl']
        if len(test_funcs) >= 8:
            break

    trajectories = []
    for src, nl in test_funcs.items():
        ops = src_ops[src][:MAX_SEQ]
        seq = [op2idx.get(op, 0) for op in ops]
        while len(seq) < MAX_SEQ:
            seq.append(0)

        # Find the NL target
        idx = None
        for i, d in enumerate(dataset):
            if d['source'] == src:
                idx = i; break
        if idx is None:
            continue

        target_nl = z_nl[idx]
        x = torch.tensor([seq], dtype=torch.long).to(DEVICE)

        with torch.no_grad():
            proj = model(x)[0].cpu().numpy()  # (seq_len, 64)

        # Cosine similarity at each step
        cos_traj = []
        for step in range(len(ops)):
            vec = proj[step]
            cos = float(np.dot(vec, target_nl) /
                       (np.linalg.norm(vec) * np.linalg.norm(target_nl) + 1e-8))
            cos_traj.append(cos)

        # Find the crystallization point (max increase)
        max_jump, max_jump_idx = 0, 0
        for j in range(1, len(cos_traj)):
            jump = cos_traj[j] - cos_traj[j-1]
            if jump > max_jump:
                max_jump = jump
                max_jump_idx = j

        crystal_op = ops[max_jump_idx] if max_jump_idx < len(ops) else "?"
        print(f"  {src[:40]}")
        print(f"    Crystallization at step {max_jump_idx}: '{crystal_op}' "
              f"(cos jump: {max_jump:.3f})")
        print(f"    Final cos: {cos_traj[-1]:.3f}")

        trajectories.append({
            'source': src, 'nl': nl, 'opcodes': ops[:15],
            'cos_trajectory': cos_traj, 'crystal_step': max_jump_idx,
            'crystal_opcode': crystal_op, 'max_jump': float(max_jump),
            'final_cos': float(cos_traj[-1]),
        })

    elapsed = time.time() - t0
    results = {
        'phase': 27, 'name': 'Sequential Semantic Crystallization',
        'opcode_vocab_size': V_op, 'n_samples': N,
        'trajectories': trajectories,
        'elapsed': elapsed, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase27_crystallization.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n_traj = min(len(trajectories), 6)
    cols = min(3, n_traj)
    rows = (n_traj + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows))
    if n_traj == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for ti, traj in enumerate(trajectories[:n_traj]):
        ax = axes[ti]
        cos = traj['cos_trajectory']
        ops = traj['opcodes']
        ax.plot(range(len(cos)), cos, 'b-o', markersize=4, linewidth=2)
        ax.axvline(traj['crystal_step'], color='red', ls='--', alpha=0.7,
                  label=f"Crystal: {traj['crystal_opcode']}")
        ax.set_xlabel('Bytecode Step')
        ax.set_ylabel('Cosine to NL target')
        src_short = traj['source'].replace('def f(', '').replace('): return ', ' -> ')[:25]
        ax.set_title(f'{src_short}\n(jump={traj["max_jump"]:.3f})',
                    fontsize=9, fontweight='bold')
        ax.legend(fontsize=7)
        ax.set_ylim(-1, 1)

    for ti in range(n_traj, len(axes)):
        axes[ti].set_visible(False)

    plt.suptitle('Phase 27: Sequential Semantic Crystallization\n'
                 'When does binary become meaning?',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase27_crystallization.png'), dpi=150)
    plt.close()
    print(f"\nPhase 27 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
