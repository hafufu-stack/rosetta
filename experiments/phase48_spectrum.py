"""
Phase 48: The Syntax-Semantics Spectrum
=========================================
P40: dims 1-5 = 86% variance (semantics)
P42: need 32 dims for 100% code match
WHAT do dims 6-32 encode? At which dim does MEANING stop
and mere SYNTAX begin?
"""
import os, json, time, sys
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    print("=" * 60)
    print("Phase 48: The Syntax-Semantics Spectrum")
    print("Where does meaning end and syntax begin?")
    print("=" * 60)
    t0 = time.time()

    lat_file = os.path.join(DATA_DIR, 'rosetta_latents_v2.npz')
    if not os.path.exists(lat_file):
        lat_file = os.path.join(DATA_DIR, 'rosetta_latents.npz')
    latents = np.load(lat_file)
    z_ast = latents['ast']

    ds_file = os.path.join(DATA_DIR, 'rosetta_dataset_v2.json')
    if not os.path.exists(ds_file):
        ds_file = os.path.join(DATA_DIR, 'rosetta_dataset.json')
    with open(ds_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)['dataset']

    sys.path.insert(0, BASE_DIR)
    with open(os.path.join(DATA_DIR, 'char_vocab.json'), 'r') as f:
        vd = json.load(f)
    idx2char = {int(i): c for c, i in vd['char2idx'].items()}
    V = len(vd['char2idx'])
    from experiments.phase9_generative_decompiler import CodeDecoder, decode_tokens
    decoder = CodeDecoder(64, V, hidden=128, max_len=80).to(DEVICE)
    dec_path = os.path.join(DATA_DIR, 'decoder_robust.pt')
    if not os.path.exists(dec_path):
        dec_path = os.path.join(DATA_DIR, 'decoder.pt')
    decoder.load_state_dict(torch.load(dec_path, map_location=DEVICE, weights_only=True))
    decoder.eval()

    def gen(z):
        with torch.no_grad():
            z_t = torch.tensor(z.astype(np.float32)).unsqueeze(0).to(DEVICE)
            tokens = decoder(z_t)
            return decode_tokens(tokens[0].cpu().numpy(), idx2char)

    src_to_idx = {}
    for i, d in enumerate(dataset):
        if d['source'] not in src_to_idx:
            src_to_idx[d['source']] = i
    unique_srcs = list(src_to_idx.keys())

    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(64, z_ast.shape[1]))
    z_pca = pca.fit_transform(z_ast)

    # === For each number of dims, measure semantic match vs syntactic match ===
    dims_to_test = list(range(1, 11)) + [12, 16, 20, 32, 64]

    print("\n--- Semantic vs Syntactic Recovery ---")
    spectrum = []

    test_srcs = unique_srcs[:30]
    for n_dim in dims_to_test:
        if n_dim > z_ast.shape[1]:
            continue

        pca_n = PCA(n_components=n_dim)
        z_comp = pca_n.fit_transform(z_ast)
        z_recon = pca_n.inverse_transform(z_comp)

        semantic_match = 0  # Same operation (e.g., both are addition)
        syntactic_match = 0  # Exact string match
        var_names_match = 0  # Same var names

        for src in test_srcs:
            idx = src_to_idx[src]
            orig = gen(z_ast[idx])
            recon = gen(z_recon[idx])

            # Exact match
            if orig.strip() == recon.strip():
                syntactic_match += 1
                semantic_match += 1
                var_names_match += 1
                continue

            # Semantic: does the core operation match?
            def get_op(code):
                if 'return' in code:
                    ret = code.split('return')[-1].strip()
                    for op in ['+', '-', '*', '/', '%', '**', '>', '<', '==',
                              '!=', '>=', '<=', '.upper', '.lower', 'abs(',
                              'len(', 'not ', ' and ', ' or ']:
                        if op in ret:
                            return op
                return '?'

            if get_op(orig) == get_op(recon):
                semantic_match += 1

            # Variable names
            def get_vars(code):
                if 'def f(' in code:
                    params = code.split('def f(')[1].split(')')[0]
                    return params
                return ''
            if get_vars(orig) == get_vars(recon):
                var_names_match += 1

        n = len(test_srcs)
        sem_rate = semantic_match / n
        syn_rate = syntactic_match / n
        var_rate = var_names_match / n

        print(f"  {n_dim:2d}D: semantic={sem_rate:.0%}, syntax={syn_rate:.0%}, "
              f"vars={var_rate:.0%}")

        spectrum.append({
            'dims': n_dim, 'semantic': sem_rate, 'syntactic': syn_rate,
            'var_names': var_rate,
            'var_explained': float(sum(pca_n.explained_variance_ratio_)),
        })

    # === Find the transition point ===
    semantic_90 = None
    syntactic_90 = None
    for sp in spectrum:
        if sp['semantic'] >= 0.9 and semantic_90 is None:
            semantic_90 = sp['dims']
        if sp['syntactic'] >= 0.9 and syntactic_90 is None:
            syntactic_90 = sp['dims']

    print(f"\n  Semantic 90% threshold: {semantic_90} dims")
    print(f"  Syntactic 90% threshold: {syntactic_90} dims")
    if semantic_90 and syntactic_90:
        print(f"  Syntax-Semantics Gap: {syntactic_90 - semantic_90} dims")
        print(f"  -> Dims {semantic_90+1}-{syntactic_90} encode PURE SYNTAX")

    elapsed = time.time() - t0
    results = {
        'phase': 48, 'name': 'The Syntax-Semantics Spectrum',
        'spectrum': spectrum,
        'semantic_90': semantic_90, 'syntactic_90': syntactic_90,
        'gap': (syntactic_90 - semantic_90) if semantic_90 and syntactic_90 else None,
        'elapsed': elapsed, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase48_spectrum.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    dims = [sp['dims'] for sp in spectrum]
    ax.plot(dims, [sp['semantic'] for sp in spectrum], 'g-o', linewidth=2,
           markersize=6, label='Semantic (operation)')
    ax.plot(dims, [sp['var_names'] for sp in spectrum], 'b-s', linewidth=2,
           markersize=6, label='Variable names')
    ax.plot(dims, [sp['syntactic'] for sp in spectrum], 'r-^', linewidth=2,
           markersize=6, label='Exact syntax')

    ax.axhline(0.9, color='gray', ls='--', alpha=0.5)
    if semantic_90:
        ax.axvline(semantic_90, color='green', ls=':', alpha=0.5,
                  label=f'Semantic @{semantic_90}D')
    if syntactic_90:
        ax.axvline(syntactic_90, color='red', ls=':', alpha=0.5,
                  label=f'Syntax @{syntactic_90}D')
    if semantic_90 and syntactic_90:
        ax.axvspan(semantic_90, syntactic_90, alpha=0.1, color='yellow',
                  label='Pure syntax zone')

    ax.set_xlabel('Number of Dimensions', fontsize=12)
    ax.set_ylabel('Recovery Rate', fontsize=12)
    ax.set_title('Phase 48: The Syntax-Semantics Spectrum\n'
                 'Where does meaning end and syntax begin?',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase48_spectrum.png'), dpi=150)
    plt.close()
    print(f"\nPhase 48 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
