"""
Phase 9: Generative Decompiler
================================
Train a GRU decoder: 64-dim latent vector -> Python source code string.
Complete pipeline: Binary -> Ridge inverse -> AST vector -> Decoder -> Code
"""
import os, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================
# Character-level Decoder
# ============================================================
class CodeDecoder(nn.Module):
    """GRU decoder: latent vector -> source code characters."""
    def __init__(self, latent_dim, vocab_size, hidden=128, max_len=80):
        super().__init__()
        self.hidden_dim = hidden
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.latent_to_hidden = nn.Linear(latent_dim, hidden)
        self.embedding = nn.Embedding(vocab_size, 32)
        self.gru = nn.GRU(32, hidden, batch_first=True)
        self.fc_out = nn.Linear(hidden, vocab_size)

    def forward(self, z, targets=None):
        """z: (B, latent_dim), targets: (B, L) char indices."""
        h = self.latent_to_hidden(z).unsqueeze(0)  # (1, B, H)
        if targets is not None:
            emb = self.embedding(targets[:, :-1])  # (B, L-1, 32)
            out, _ = self.gru(emb, h)
            logits = self.fc_out(out)  # (B, L-1, V)
            return logits
        else:
            return self.generate(z, h)

    def generate(self, z, h):
        B = z.size(0)
        tokens = torch.zeros(B, 1, dtype=torch.long, device=z.device)  # SOS=0
        outputs = []
        for _ in range(self.max_len):
            emb = self.embedding(tokens[:, -1:])
            out, h = self.gru(emb, h)
            logits = self.fc_out(out[:, -1])
            next_token = logits.argmax(dim=-1)
            outputs.append(next_token)
            tokens = torch.cat([tokens, next_token.unsqueeze(1)], dim=1)
        return torch.stack(outputs, dim=1)  # (B, max_len)


def build_char_vocab(sources):
    chars = set()
    for s in sources:
        chars.update(s)
    vocab = ['<SOS>', '<EOS>', '<PAD>'] + sorted(chars)
    return {c: i for i, c in enumerate(vocab)}, vocab

def encode_source(source, char2idx, max_len=80):
    tokens = [char2idx.get('<SOS>', 0)]
    for c in source[:max_len-2]:
        tokens.append(char2idx.get(c, 0))
    tokens.append(char2idx.get('<EOS>', 1))
    while len(tokens) < max_len:
        tokens.append(char2idx.get('<PAD>', 2))
    return tokens

def decode_tokens(tokens, idx2char):
    result = []
    for t in tokens:
        c = idx2char.get(int(t), '')
        if c == '<EOS>': break
        if c not in ('<SOS>', '<PAD>'):
            result.append(c)
    return ''.join(result)


def main():
    print("=" * 60)
    print("Phase 9: Generative Decompiler")
    print("=" * 60)
    t0 = time.time()

    # Load data
    ds_file = os.path.join(DATA_DIR, 'rosetta_dataset_v2.json')
    if not os.path.exists(ds_file):
        ds_file = os.path.join(DATA_DIR, 'rosetta_dataset.json')
    with open(ds_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)['dataset']

    lat_file = os.path.join(DATA_DIR, 'rosetta_latents_v2.npz')
    if not os.path.exists(lat_file):
        lat_file = os.path.join(DATA_DIR, 'rosetta_latents.npz')
    latents = np.load(lat_file)
    z_ast = latents['ast']
    z_bc = latents['bc']
    N = len(z_ast)

    sources = [d['source'] for d in dataset]
    char2idx, vocab_list = build_char_vocab(sources)
    idx2char = {i: c for c, i in char2idx.items()}
    V = len(vocab_list)
    MAX_LEN = 80
    print(f"Char vocab: {V}, Samples: {N}")

    # Encode all sources
    all_targets = np.array([encode_source(s, char2idx, MAX_LEN) for s in sources])
    targets_t = torch.tensor(all_targets, dtype=torch.long)
    ast_t = torch.tensor(z_ast, dtype=torch.float32)

    # Train decoder on AST vectors -> source code
    decoder = CodeDecoder(64, V, hidden=128, max_len=MAX_LEN).to(DEVICE)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 300)

    BATCH = 128
    losses = []
    pad_idx = char2idx['<PAD>']

    for epoch in range(300):
        perm = torch.randperm(N)
        eloss, nb = 0, 0
        decoder.train()
        for i in range(0, N, BATCH):
            idx = perm[i:i+BATCH]
            z = ast_t[idx].to(DEVICE)
            tgt = targets_t[idx].to(DEVICE)
            logits = decoder(z, tgt)  # (B, L-1, V)
            loss = F.cross_entropy(logits.reshape(-1, V), tgt[:, 1:].reshape(-1),
                                   ignore_index=pad_idx)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            optimizer.step()
            eloss += loss.item(); nb += 1
        scheduler.step()
        losses.append(eloss / max(nb, 1))
        if (epoch+1) % 100 == 0:
            print(f"  Epoch {epoch+1}/300: loss={losses[-1]:.4f}")

    # Evaluate: generate from AST vectors
    decoder.eval()
    n_test = min(50, N)
    test_idx = np.random.RandomState(42).choice(N, n_test, replace=False)
    exact, partial = 0, 0
    examples = []

    with torch.no_grad():
        z_test = ast_t[test_idx].to(DEVICE)
        generated = decoder(z_test)  # (n_test, max_len)
        for i in range(n_test):
            true_src = sources[test_idx[i]]
            gen_src = decode_tokens(generated[i].cpu().numpy(), idx2char)
            match = true_src.strip() == gen_src.strip()
            if match: exact += 1
            if true_src[:15] in gen_src[:20]: partial += 1
            if i < 10:
                examples.append({'true': true_src, 'generated': gen_src, 'match': match})

    print(f"\n--- AST -> Code Generation ---")
    print(f"  Exact match: {exact}/{n_test} ({exact/n_test:.1%})")
    print(f"  Partial match: {partial}/{n_test} ({partial/n_test:.1%})")
    for ex in examples[:5]:
        status = "OK" if ex['match'] else "X"
        print(f"  [{status}] True: {ex['true'][:50]}")
        print(f"       Gen:  {ex['generated'][:50]}")

    # Full pipeline: Binary -> Ridge inverse -> AST -> Decoder -> Code
    print(f"\n--- Full Pipeline: Binary -> Code ---")
    from sklearn.linear_model import Ridge
    n_train = int(N * 0.8)
    perm_split = np.random.RandomState(42).permutation(N)
    train_i, test_i2 = perm_split[:n_train], perm_split[n_train:]

    reg_inv = Ridge(alpha=1.0).fit(z_bc[train_i], z_ast[train_i])
    z_ast_from_bc = reg_inv.predict(z_bc[test_i2[:20]])

    pipeline_exact = 0
    with torch.no_grad():
        z_dec = torch.tensor(z_ast_from_bc, dtype=torch.float32).to(DEVICE)
        gen_pipe = decoder(z_dec)
        for i in range(min(20, len(test_i2))):
            true_src = sources[test_i2[i]]
            gen_src = decode_tokens(gen_pipe[i].cpu().numpy(), idx2char)
            match = true_src.strip() == gen_src.strip()
            if match: pipeline_exact += 1
            if i < 5:
                status = "OK" if match else "X"
                print(f"  [{status}] True: {true_src[:50]}")
                print(f"       Gen:  {gen_src[:50]}")

    print(f"  Pipeline exact: {pipeline_exact}/20 ({pipeline_exact/20:.1%})")

    # Save decoder
    torch.save(decoder.state_dict(), os.path.join(DATA_DIR, 'decoder.pt'))
    with open(os.path.join(DATA_DIR, 'char_vocab.json'), 'w') as f:
        json.dump({'char2idx': char2idx, 'vocab_list': vocab_list}, f)

    elapsed = time.time() - t0
    results = {
        'phase': 9, 'name': 'Generative Decompiler',
        'char_vocab_size': V, 'decoder_params': sum(p.numel() for p in decoder.parameters()),
        'final_loss': float(losses[-1]),
        'ast_to_code_exact': exact/n_test,
        'ast_to_code_partial': partial/n_test,
        'pipeline_exact': pipeline_exact/20,
        'examples': examples[:5],
        'elapsed': elapsed, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase9_generative_decompiler.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Figure
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(losses, color='#E91E63', lw=1.5)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('Decoder Training Loss', fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    bars = axes[1].bar(['AST->Code\n(Exact)', 'AST->Code\n(Partial)',
                        'Full Pipeline\n(Bin->Code)'],
                       [exact/n_test, partial/n_test, pipeline_exact/20],
                       color=['#4CAF50','#2196F3','#FF9800'], edgecolor='black')
    for b, v in zip(bars, [exact/n_test, partial/n_test, pipeline_exact/20]):
        axes[1].text(b.get_x()+b.get_width()/2, v+0.02, f'{v:.0%}',
                     ha='center', fontweight='bold', fontsize=14)
    axes[1].set_ylabel('Accuracy'); axes[1].set_ylim(0, 1.1)
    axes[1].set_title('Code Generation Accuracy', fontweight='bold')
    plt.suptitle('Phase 9: Generative Decompiler', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase9_generative_decompiler.png'), dpi=150)
    plt.close()
    print(f"\nPhase 9 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
