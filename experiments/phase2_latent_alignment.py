"""
Phase 2: Tri-Modal Latent Alignment
====================================
Train 3 encoders (NL, AST, Bytecode) to map to a shared 64-dim
latent space using CLIP-style contrastive learning.

- NL Encoder: Bag-of-words + MLP
- AST Encoder: GNN (GlassBox heritage!) on AST graph
- Bin Encoder: 1D-CNN on bytecode sequence
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
LATENT_DIM = 64
EPOCHS = 200
LR = 1e-3
BATCH = 64

# ============================================================
# Data Loading & Encoding
# ============================================================
def load_dataset():
    with open(os.path.join(DATA_DIR, 'rosetta_dataset.json'), 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['dataset'], data['nl_vocab'], data['node_type_vocab']

def encode_nl_bow(nl_text, vocab, max_dim=200):
    """Bag-of-words encoding."""
    word2idx = {w: i for i, w in enumerate(vocab)}
    vec = np.zeros(min(len(vocab), max_dim), dtype=np.float32)
    for w in nl_text.lower().split():
        idx = word2idx.get(w, -1)
        if 0 <= idx < max_dim:
            vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec

def encode_ast_features(graph, node_vocab, max_nodes=50, feat_dim=30):
    """Encode AST graph as (node_features, adjacency)."""
    type2idx = {t: i for i, t in enumerate(node_vocab)}
    n = min(len(graph['nodes']), max_nodes)
    # Node features: one-hot of type (truncated)
    node_feat = np.zeros((max_nodes, feat_dim), dtype=np.float32)
    for i in range(n):
        idx = type2idx.get(graph['nodes'][i], 0)
        if idx < feat_dim:
            node_feat[i, idx] = 1.0
    # Adjacency matrix
    adj = np.zeros((max_nodes, max_nodes), dtype=np.float32)
    for src, dst in graph['edges']:
        if src < max_nodes and dst < max_nodes:
            adj[src, dst] = 1.0
            adj[dst, src] = 1.0  # Undirected
    # Add self-loops
    for i in range(n):
        adj[i, i] = 1.0
    # Degree normalization
    deg = adj.sum(axis=1, keepdims=True)
    deg = np.maximum(deg, 1.0)
    adj = adj / deg
    return node_feat, adj, n

def encode_bytecode(bytecode, max_len=100):
    """Encode bytecode as normalized float sequence."""
    bc = np.zeros(max_len, dtype=np.float32)
    for i, b in enumerate(bytecode[:max_len]):
        bc[i] = b / 255.0
    return bc

# ============================================================
# Encoders
# ============================================================
class NLEncoder(nn.Module):
    """Bag-of-words -> MLP -> latent."""
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)

class ASTEncoder(nn.Module):
    """GNN on AST graph -> latent (GlassBox heritage!)."""
    def __init__(self, feat_dim, latent_dim, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(feat_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc_out = nn.Linear(hidden, latent_dim)
    def forward(self, node_feat, adj, n_nodes):
        # 2-layer GNN message passing
        h = F.relu(self.fc1(node_feat))  # (B, N, H)
        h = torch.bmm(adj, h)           # Message passing
        h = F.relu(self.fc2(h))
        h = torch.bmm(adj, h)           # Second layer
        # Global mean pooling (over valid nodes)
        mask = torch.arange(h.size(1), device=h.device).unsqueeze(0) < n_nodes.unsqueeze(1)
        mask = mask.unsqueeze(-1).float()
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return F.normalize(self.fc_out(pooled), dim=-1)

class BytecodeEncoder(nn.Module):
    """1D-CNN on bytecode sequence -> latent."""
    def __init__(self, max_len, latent_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, 5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, 5, padding=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, latent_dim)
    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, L)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return F.normalize(self.fc(x), dim=-1)

# ============================================================
# CLIP-style Contrastive Loss
# ============================================================
def clip_loss(z1, z2, temperature=0.07):
    """Symmetric contrastive loss between two sets of embeddings."""
    logits = z1 @ z2.T / temperature
    labels = torch.arange(len(z1), device=z1.device)
    loss_12 = F.cross_entropy(logits, labels)
    loss_21 = F.cross_entropy(logits.T, labels)
    return (loss_12 + loss_21) / 2

def main():
    print("=" * 60)
    print("Phase 2: Tri-Modal Latent Alignment")
    print("=" * 60)
    t0 = time.time()

    # Load data
    dataset, nl_vocab, node_vocab = load_dataset()
    print(f"Loaded {len(dataset)} triplets")
    print(f"NL vocab: {len(nl_vocab)}, AST vocab: {len(node_vocab)}")

    # Encode all data
    NL_DIM = min(len(nl_vocab), 200)
    AST_FEAT = 30
    BC_LEN = 100

    nl_data, ast_feat_data, ast_adj_data, ast_n_data, bc_data = [], [], [], [], []
    source_ids = []  # Track which source each sample came from

    unique_sources = list(set(d['source'] for d in dataset))
    src2id = {s: i for i, s in enumerate(unique_sources)}

    for d in dataset:
        nl_data.append(encode_nl_bow(d['nl'], nl_vocab, NL_DIM))
        nf, adj, n = encode_ast_features(d['ast_graph'], node_vocab,
                                          feat_dim=AST_FEAT)
        ast_feat_data.append(nf)
        ast_adj_data.append(adj)
        ast_n_data.append(n)
        bc_data.append(encode_bytecode(d['bytecode'], BC_LEN))
        source_ids.append(src2id[d['source']])

    nl_t = torch.tensor(np.array(nl_data), dtype=torch.float32)
    ast_f_t = torch.tensor(np.array(ast_feat_data), dtype=torch.float32)
    ast_a_t = torch.tensor(np.array(ast_adj_data), dtype=torch.float32)
    ast_n_t = torch.tensor(np.array(ast_n_data), dtype=torch.long)
    bc_t = torch.tensor(np.array(bc_data), dtype=torch.float32)

    N = len(dataset)
    print(f"Encoded: NL({NL_DIM}), AST({AST_FEAT}x50), BC({BC_LEN})")

    # Models
    nl_enc = NLEncoder(NL_DIM, LATENT_DIM).to(DEVICE)
    ast_enc = ASTEncoder(AST_FEAT, LATENT_DIM).to(DEVICE)
    bc_enc = BytecodeEncoder(BC_LEN, LATENT_DIM).to(DEVICE)

    total_params = sum(sum(p.numel() for p in m.parameters())
                       for m in [nl_enc, ast_enc, bc_enc])
    print(f"Total parameters: {total_params:,}")

    optimizer = torch.optim.Adam(
        list(nl_enc.parameters()) + list(ast_enc.parameters()) +
        list(bc_enc.parameters()), lr=LR
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)

    # Training
    losses = []
    best_loss = float('inf')

    for epoch in range(EPOCHS):
        perm = torch.randperm(N)
        epoch_loss = 0
        n_batch = 0

        for i in range(0, N, BATCH):
            idx = perm[i:i+BATCH]
            if len(idx) < 4:
                continue

            nl_b = nl_t[idx].to(DEVICE)
            af_b = ast_f_t[idx].to(DEVICE)
            aa_b = ast_a_t[idx].to(DEVICE)
            an_b = ast_n_t[idx].to(DEVICE)
            bc_b = bc_t[idx].to(DEVICE)

            z_nl = nl_enc(nl_b)
            z_ast = ast_enc(af_b, aa_b, an_b)
            z_bc = bc_enc(bc_b)

            # Three-way contrastive: NL<->AST, NL<->BC, AST<->BC
            loss = (clip_loss(z_nl, z_ast) +
                    clip_loss(z_nl, z_bc) +
                    clip_loss(z_ast, z_bc)) / 3

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(nl_enc.parameters()) + list(ast_enc.parameters()) +
                list(bc_enc.parameters()), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batch += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batch, 1)
        losses.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS}: loss={avg_loss:.4f} "
                  f"(best={best_loss:.4f})")

    # Evaluation: compute all latent vectors
    nl_enc.eval(); ast_enc.eval(); bc_enc.eval()
    with torch.no_grad():
        all_nl = nl_enc(nl_t.to(DEVICE)).cpu().numpy()
        all_ast = ast_enc(ast_f_t.to(DEVICE), ast_a_t.to(DEVICE),
                          ast_n_t.to(DEVICE)).cpu().numpy()
        all_bc = bc_enc(bc_t.to(DEVICE)).cpu().numpy()

    # Retrieval accuracy: for each NL, find nearest AST/BC
    def retrieval_acc(query, gallery, labels):
        correct = 0
        for i in range(len(query)):
            sims = query[i] @ gallery.T
            best = np.argmax(sims)
            if labels[best] == labels[i]:
                correct += 1
        return correct / len(query)

    labels = np.array(source_ids)
    acc_nl_ast = retrieval_acc(all_nl, all_ast, labels)
    acc_nl_bc = retrieval_acc(all_nl, all_bc, labels)
    acc_ast_bc = retrieval_acc(all_ast, all_bc, labels)

    # Cosine similarity within same source vs different
    same_sims, diff_sims = [], []
    for i in range(min(500, N)):
        for j in range(i+1, min(500, N)):
            sim = float(all_nl[i] @ all_ast[j])
            if labels[i] == labels[j]:
                same_sims.append(sim)
            else:
                diff_sims.append(sim)

    print(f"\n--- Alignment Results ---")
    print(f"Retrieval NL->AST: {acc_nl_ast:.1%}")
    print(f"Retrieval NL->BC:  {acc_nl_bc:.1%}")
    print(f"Retrieval AST->BC: {acc_ast_bc:.1%}")
    print(f"Same-source cosine sim:  {np.mean(same_sims):.3f}")
    print(f"Diff-source cosine sim:  {np.mean(diff_sims):.3f}")

    # Save latent vectors for Phase 3-4
    np.savez(os.path.join(DATA_DIR, 'rosetta_latents.npz'),
             nl=all_nl, ast=all_ast, bc=all_bc,
             labels=labels, source_ids=source_ids)

    elapsed = time.time() - t0
    results = {
        'phase': 2,
        'name': 'Tri-Modal Latent Alignment',
        'latent_dim': LATENT_DIM,
        'total_params': total_params,
        'epochs': EPOCHS,
        'final_loss': float(losses[-1]),
        'best_loss': float(best_loss),
        'retrieval_nl_ast': float(acc_nl_ast),
        'retrieval_nl_bc': float(acc_nl_bc),
        'retrieval_ast_bc': float(acc_ast_bc),
        'same_source_sim': float(np.mean(same_sims)),
        'diff_source_sim': float(np.mean(diff_sims)),
        'elapsed': elapsed,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase2_latent_alignment.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Visualization
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(losses, color='#E91E63', lw=1.5)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Contrastive Loss', fontsize=12)
    axes[0].set_title('Training Loss', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Retrieval accuracy bar
    accs = [acc_nl_ast, acc_nl_bc, acc_ast_bc]
    labels_bar = ['NL->AST', 'NL->BC', 'AST->BC']
    bars = axes[1].bar(labels_bar, accs, color=['#2196F3','#4CAF50','#FF9800'],
                       edgecolor='black')
    for b, a in zip(bars, accs):
        axes[1].text(b.get_x()+b.get_width()/2, a+0.02 if a < 0.9 else a-0.08,
                     f'{a:.1%}', ha='center', fontweight='bold', fontsize=13)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Cross-Modal Retrieval', fontsize=13, fontweight='bold')
    axes[1].set_ylim(0, 1.1)

    # Similarity distribution
    axes[2].hist(same_sims[:500], bins=30, alpha=0.7, color='#4CAF50', label='Same source')
    axes[2].hist(diff_sims[:500], bins=30, alpha=0.7, color='#F44336', label='Different source')
    axes[2].set_xlabel('Cosine Similarity', fontsize=12)
    axes[2].set_ylabel('Count', fontsize=12)
    axes[2].set_title('Latent Space Separation', fontsize=13, fontweight='bold')
    axes[2].legend(fontsize=11)

    plt.suptitle('Phase 2: Tri-Modal Latent Alignment (Rosetta Space)',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase2_latent_alignment.png'), dpi=150)
    plt.close()

    print(f"\nPhase 2 complete in {elapsed:.1f}s")
    return results


if __name__ == '__main__':
    main()
