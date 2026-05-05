"""
Phase 26: LLM-to-Binary Translation Matrix
=============================================
Can GPT-2's hidden states be linearly mapped to bytecode vectors?
Proving that LLM code generation is fundamentally a matrix multiply.
"""
import os, json, time
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    print("=" * 60)
    print("Phase 26: LLM-to-Binary Translation Matrix")
    print("GPT-2 Hidden States -> Bytecode Vectors")
    print("=" * 60)
    t0 = time.time()

    # Load dataset and latents
    ds_file = os.path.join(DATA_DIR, 'rosetta_dataset_v2.json')
    if not os.path.exists(ds_file):
        ds_file = os.path.join(DATA_DIR, 'rosetta_dataset.json')
    with open(ds_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)['dataset']

    lat_file = os.path.join(DATA_DIR, 'rosetta_latents_v2.npz')
    if not os.path.exists(lat_file):
        lat_file = os.path.join(DATA_DIR, 'rosetta_latents.npz')
    latents = np.load(lat_file)
    z_bc = latents['bc']
    z_nl = latents['nl']
    z_ast = latents['ast']

    # Load GPT-2
    print("  Loading GPT-2...")
    from transformers import GPT2Model, GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    gpt2 = GPT2Model.from_pretrained('gpt2', local_files_only=True).to(DEVICE)
    gpt2.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Extract GPT-2 hidden states for all NL descriptions
    print("  Extracting GPT-2 hidden states...")
    N = len(dataset)
    gpt2_hidden = np.zeros((N, 768), dtype=np.float32)
    BATCH = 32
    for i in range(0, N, BATCH):
        batch_nl = [d['nl'] for d in dataset[i:i+BATCH]]
        enc = tokenizer(batch_nl, return_tensors='pt', padding=True,
                       truncation=True, max_length=64).to(DEVICE)
        with torch.no_grad():
            out = gpt2(**enc)
            # Mean pool over sequence
            mask = enc['attention_mask'].unsqueeze(-1).float()
            h = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
            gpt2_hidden[i:i+len(batch_nl)] = h.cpu().numpy()
        if (i // BATCH) % 20 == 0:
            print(f"    {i}/{N}")

    print(f"  GPT-2 hidden states: shape={gpt2_hidden.shape}")

    # Train: GPT-2 -> Bytecode
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train_bc, y_test_bc = train_test_split(
        gpt2_hidden, z_bc, test_size=0.2, random_state=42)
    _, _, y_train_ast, y_test_ast = train_test_split(
        gpt2_hidden, z_ast, test_size=0.2, random_state=42)
    _, _, y_train_nl, y_test_nl = train_test_split(
        gpt2_hidden, z_nl, test_size=0.2, random_state=42)

    print("\n  Training W_llm_bin (GPT-2 -> Binary)...")
    reg_bc = Ridge(alpha=10.0).fit(X_train, y_train_bc)
    r2_bc = reg_bc.score(X_test, y_test_bc)
    pred_bc = reg_bc.predict(X_test)
    cos_bc = float(np.mean([
        np.dot(p, t) / (np.linalg.norm(p) * np.linalg.norm(t) + 1e-8)
        for p, t in zip(pred_bc, y_test_bc)
    ]))
    print(f"  GPT-2 -> Binary: R2={r2_bc:.4f}, cos={cos_bc:.4f}")

    print("  Training W_llm_ast (GPT-2 -> AST)...")
    reg_ast = Ridge(alpha=10.0).fit(X_train, y_train_ast)
    r2_ast = reg_ast.score(X_test, y_test_ast)
    pred_ast = reg_ast.predict(X_test)
    cos_ast = float(np.mean([
        np.dot(p, t) / (np.linalg.norm(p) * np.linalg.norm(t) + 1e-8)
        for p, t in zip(pred_ast, y_test_ast)
    ]))
    print(f"  GPT-2 -> AST:    R2={r2_ast:.4f}, cos={cos_ast:.4f}")

    print("  Training W_llm_nl (GPT-2 -> Rosetta NL)...")
    reg_nl = Ridge(alpha=10.0).fit(X_train, y_train_nl)
    r2_nl = reg_nl.score(X_test, y_test_nl)
    print(f"  GPT-2 -> NL:     R2={r2_nl:.4f}")

    # SVD of W_llm_bin
    W = reg_bc.coef_  # (64, 768)
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    energy = np.cumsum(S**2) / np.sum(S**2)
    n90 = int(np.searchsorted(energy, 0.9)) + 1
    print(f"\n  SVD of W_llm_bin: 90% energy in {n90} dims")
    print(f"  Top 5 singular values: {[f'{s:.3f}' for s in S[:5]]}")

    elapsed = time.time() - t0
    results = {
        'phase': 26, 'name': 'LLM-to-Binary Translation Matrix',
        'r2_gpt2_binary': float(r2_bc), 'cos_gpt2_binary': cos_bc,
        'r2_gpt2_ast': float(r2_ast), 'cos_gpt2_ast': cos_ast,
        'r2_gpt2_nl': float(r2_nl),
        'svd_90pct': n90,
        'top_singular': [float(s) for s in S[:10]],
        'elapsed': elapsed, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase26_llm_binary.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. R2 comparison
    labels = ['GPT-2 -> Binary', 'GPT-2 -> AST', 'GPT-2 -> NL']
    r2s = [r2_bc, r2_ast, r2_nl]
    colors = ['#E91E63', '#2196F3', '#4CAF50']
    bars = axes[0].bar(labels, r2s, color=colors, edgecolor='black')
    for b, v in zip(bars, r2s):
        axes[0].text(b.get_x()+b.get_width()/2, v+0.02, f'{v:.3f}',
                    ha='center', fontweight='bold')
    axes[0].set_ylabel('R2 Score')
    axes[0].set_ylim(0, 1.1)
    axes[0].set_title('GPT-2 -> Rosetta Space\n(Linear Translation)', fontweight='bold')

    # 2. SVD spectrum
    axes[1].bar(range(min(30, len(S))), S[:30], color='#9C27B0', alpha=0.8)
    axes[1].axvline(n90-1, color='red', ls='--', label=f'90% energy ({n90} dims)')
    axes[1].set_xlabel('Singular Value Index')
    axes[1].set_ylabel('Magnitude')
    axes[1].set_title('SVD of W_llm_bin (768 -> 64)', fontweight='bold')
    axes[1].legend()

    # 3. Predicted vs actual (scatter)
    axes[2].scatter(y_test_bc[:, 0], pred_bc[:, 0], alpha=0.3, s=10, c='#FF5722')
    lims = [min(y_test_bc[:, 0].min(), pred_bc[:, 0].min()),
            max(y_test_bc[:, 0].max(), pred_bc[:, 0].max())]
    axes[2].plot(lims, lims, 'k--', alpha=0.5)
    axes[2].set_xlabel('Actual Binary dim[0]')
    axes[2].set_ylabel('Predicted Binary dim[0]')
    axes[2].set_title(f'GPT-2 -> Binary (dim 0)\nR2={r2_bc:.3f}', fontweight='bold')

    plt.suptitle('Phase 26: LLM-to-Binary Translation Matrix\n'
                 'Can GPT-2 reach bytecode in one matrix multiply?',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase26_llm_binary.png'), dpi=150)
    plt.close()
    print(f"\nPhase 26 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
