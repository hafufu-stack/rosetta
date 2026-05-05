"""
Phase 15: SVD-Bottleneck Decoder
==================================
Re-train decoder with noise injection to ignore null-space dimensions.
Then re-test P12's null-space obfuscation.
"""
import os, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    print("=" * 60)
    print("Phase 15: SVD-Bottleneck Decoder")
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
    z_ast, z_bc = latents['ast'], latents['bc']
    N, D = z_ast.shape

    with open(os.path.join(DATA_DIR, 'char_vocab.json'), 'r') as f:
        vd = json.load(f)
    char2idx = vd['char2idx']
    idx2char = {int(i): c for c, i in char2idx.items()}
    V = len(char2idx)

    from experiments.phase9_generative_decompiler import (
        CodeDecoder, decode_tokens, encode_source
    )

    # Get SVD basis for noise injection
    perm = np.random.RandomState(42).permutation(N)
    reg = Ridge(alpha=1.0).fit(z_ast[perm[:int(N*0.8)]], z_bc[perm[:int(N*0.8)]])
    _, S, Vt = np.linalg.svd(reg.coef_)
    K_SIGNAL = 6  # Keep top 6 (covers 95% energy)

    # Re-train decoder with null-space noise injection
    sources = [d['source'] for d in dataset]
    MAX_LEN = 80
    pad_idx = char2idx['<PAD>']
    all_targets = np.array([encode_source(s, char2idx, MAX_LEN) for s in sources])
    targets_t = torch.tensor(all_targets, dtype=torch.long)
    ast_t = torch.tensor(z_ast, dtype=torch.float32)
    Vt_t = torch.tensor(Vt, dtype=torch.float32).to(DEVICE)

    decoder = CodeDecoder(64, V, hidden=128, max_len=MAX_LEN).to(DEVICE)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 400)
    BATCH = 128
    losses = []

    for epoch in range(400):
        perm_e = torch.randperm(N)
        eloss, nb = 0, 0
        decoder.train()
        for i in range(0, N, BATCH):
            idx = perm_e[i:i+BATCH]
            z = ast_t[idx].to(DEVICE)
            tgt = targets_t[idx].to(DEVICE)

            # BOTTLENECK: project to SVD, add noise to null-space, project back
            proj = z @ Vt_t.T  # (B, D)
            noise = torch.randn_like(proj) * 2.0  # Strong noise
            noise[:, :K_SIGNAL] = 0  # Protect signal dimensions
            z_noisy = (proj + noise) @ Vt_t  # Back to original space

            logits = decoder(z_noisy, tgt)
            loss = F.cross_entropy(logits.reshape(-1, V), tgt[:, 1:].reshape(-1),
                                   ignore_index=pad_idx)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            optimizer.step()
            eloss += loss.item(); nb += 1
        scheduler.step()
        losses.append(eloss / max(nb, 1))
        if (epoch+1) % 100 == 0:
            print(f"  Epoch {epoch+1}/400: loss={losses[-1]:.4f}")

    torch.save(decoder.state_dict(), os.path.join(DATA_DIR, 'decoder_robust.pt'))

    # Re-test P12: Null-space obfuscation
    decoder.eval()
    src_to_idx = {}
    for i, d in enumerate(dataset):
        if d['source'] not in src_to_idx:
            src_to_idx[d['source']] = i
    test_srcs = list(src_to_idx.keys())[:20]
    test_idx = [src_to_idx[s] for s in test_srcs]

    def gen(z_vec):
        with torch.no_grad():
            z_t = torch.tensor(z_vec, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            tokens = decoder(z_t)
            return decode_tokens(tokens[0].cpu().numpy(), idx2char)

    noise_levels = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
    results_grid = []
    print(f"\n--- Re-testing Null-Space Obfuscation ---")

    for noise_std in noise_levels:
        exact, semantic, total = 0, 0, 0
        for ti in test_idx:
            z_orig = z_ast[ti].copy()
            true_src = dataset[ti]['source']
            if noise_std > 0:
                proj = z_orig @ Vt.T
                noise = np.zeros(D, dtype=np.float32)
                noise[K_SIGNAL:] = np.random.randn(D - K_SIGNAL).astype(np.float32) * noise_std
                z_noisy = (proj + noise) @ Vt
            else:
                z_noisy = z_orig
            gen_code = gen(z_noisy)
            if gen_code.strip() == true_src.strip(): exact += 1
            key_op = true_src.split('return ')[-1].strip().replace(' ','') if 'return ' in true_src else ''
            gen_op = gen_code.split('return ')[-1].strip().replace(' ','') if 'return ' in gen_code else ''
            for v in 'xyanmpqbvtextswlstarrnumsitems':
                key_op = key_op.replace(v, '_')
                gen_op = gen_op.replace(v, '_')
            if key_op and key_op == gen_op: semantic += 1
            total += 1
        results_grid.append({'noise_std': noise_std, 'exact': exact/total,
                             'semantic': semantic/total})
        print(f"  Noise={noise_std:5.1f}: exact={exact}/{total} ({exact/total:.0%}), "
              f"semantic={semantic}/{total} ({semantic/total:.0%})")

    elapsed = time.time() - t0
    results = {
        'phase': 15, 'name': 'SVD-Bottleneck Decoder',
        'signal_dims': K_SIGNAL, 'noise_results': results_grid,
        'final_loss': float(losses[-1]), 'elapsed': elapsed,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase15_bottleneck.json'), 'w') as f:
        json.dump(results, f, indent=2)

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(losses, color='#E91E63', lw=1.5)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('Bottleneck Decoder Training', fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    ns = [r['noise_std'] for r in results_grid]
    axes[1].plot(ns, [r['semantic'] for r in results_grid], 's-', color='#4CAF50',
                lw=2, ms=8, label='Robust (P15)')
    # Load P12 for comparison
    try:
        with open(os.path.join(RESULTS_DIR, 'phase12_null_space.json')) as f:
            p12 = json.load(f)
        axes[1].plot([r['noise_std'] for r in p12['noise_results']],
                    [r['semantic'] for r in p12['noise_results']], 'o--',
                    color='#F44336', lw=2, ms=8, label='Original (P12)')
    except: pass
    axes[1].set_xlabel('Null-Space Noise'); axes[1].set_ylabel('Semantic Accuracy')
    axes[1].set_title('Noise Robustness: P12 vs P15', fontweight='bold')
    axes[1].legend(fontsize=12); axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-0.05, 1.05)
    plt.suptitle('Phase 15: SVD-Bottleneck Decoder', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase15_bottleneck.png'), dpi=150)
    plt.close()
    print(f"\nPhase 15 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
