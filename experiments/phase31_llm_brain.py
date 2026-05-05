"""
Phase 31: Decompiling the LLM Brain
======================================
Read GPT-2's mind: extract hidden states during inference,
project through W_llm_ast, decode to Python source code.
What is the LLM "thinking in code"?
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
    print("Phase 31: Decompiling the LLM Brain")
    print("Reading GPT-2's mind via Rosetta Space")
    print("=" * 60)
    t0 = time.time()

    # Load decoder
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

    # Load latents for fitting
    lat_file = os.path.join(DATA_DIR, 'rosetta_latents_v2.npz')
    if not os.path.exists(lat_file):
        lat_file = os.path.join(DATA_DIR, 'rosetta_latents.npz')
    latents = np.load(lat_file)
    z_ast = latents['ast']
    z_nl = latents['nl']

    ds_file = os.path.join(DATA_DIR, 'rosetta_dataset_v2.json')
    if not os.path.exists(ds_file):
        ds_file = os.path.join(DATA_DIR, 'rosetta_dataset.json')
    with open(ds_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)['dataset']

    # Load GPT-2
    print("  Loading GPT-2...")
    from transformers import GPT2Model, GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    gpt2 = GPT2Model.from_pretrained('gpt2', local_files_only=True).to(DEVICE)
    gpt2.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Fit W_llm_ast: GPT-2 hidden -> AST space
    print("  Extracting training hidden states...")
    N = len(dataset)
    gpt2_hidden = np.zeros((N, 768), dtype=np.float32)
    BATCH = 32
    for i in range(0, N, BATCH):
        batch_nl = [d['nl'] for d in dataset[i:i+BATCH]]
        enc = tokenizer(batch_nl, return_tensors='pt', padding=True,
                       truncation=True, max_length=64).to(DEVICE)
        with torch.no_grad():
            out = gpt2(**enc)
            mask = enc['attention_mask'].unsqueeze(-1).float()
            h = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
            gpt2_hidden[i:i+len(batch_nl)] = h.cpu().numpy()

    from sklearn.linear_model import Ridge
    W_llm_ast = Ridge(alpha=10.0).fit(gpt2_hidden, z_ast)

    def gen_code(z_vec):
        with torch.no_grad():
            z_t = torch.tensor(z_vec, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            tokens = decoder(z_t)
            return decode_tokens(tokens[0].cpu().numpy(), idx2char)

    # === Mind Reading ===
    print("\n--- Decompiling GPT-2's Brain ---")
    prompts = [
        "Calculate the sum of two numbers",
        "Find the product of x and y",
        "Check if the first number is larger",
        "Get the absolute value",
        "Subtract b from a",
        "Compute the remainder of division",
        "Return the negative of a number",
        "Convert text to uppercase",
        "Raise x to the power y",
        "Divide x by y",
    ]

    mind_reads = []
    for prompt in prompts:
        enc = tokenizer(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out = gpt2(**enc, output_hidden_states=True)

        # Mean pool of last hidden state
        h_mean = out.last_hidden_state.mean(dim=1).cpu().numpy()[0]  # (768,)

        # Per-token hidden states (for trajectory)
        h_per_token = out.last_hidden_state[0].cpu().numpy()  # (seq_len, 768)
        tokens = tokenizer.tokenize(prompt)

        # Project to AST space
        z_pred = W_llm_ast.predict(h_mean.reshape(1, -1))[0]

        # Decode to Python
        code = gen_code(z_pred)

        # Per-token trajectory
        token_codes = []
        for ti in range(len(tokens)):
            h_t = h_per_token[ti]
            z_t = W_llm_ast.predict(h_t.reshape(1, -1))[0]
            c_t = gen_code(z_t)
            token_codes.append({'token': tokens[ti], 'code': c_t})

        print(f"\n  Prompt: '{prompt}'")
        print(f"    LLM thinks (mean): {code}")
        print(f"    Token-by-token mind reading:")
        for tc in token_codes[:6]:
            tok_safe = tc['token'].encode('ascii', 'replace').decode('ascii')
            print(f"      '{tok_safe}' -> {tc['code'][:40]}")

        mind_reads.append({
            'prompt': prompt, 'decoded_mean': code,
            'token_trajectory': token_codes[:8],
        })

    elapsed = time.time() - t0
    results = {
        'phase': 31, 'name': 'Decompiling the LLM Brain',
        'n_prompts': len(prompts),
        'mind_reads': mind_reads,
        'elapsed': elapsed, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase31_llm_brain.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n_show = min(6, len(mind_reads))
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for mi, mr in enumerate(mind_reads[:n_show]):
        ax = axes[mi]
        tokens = [tc['token'][:8] for tc in mr['token_trajectory']]
        codes = [tc['code'][:20] for tc in mr['token_trajectory']]
        y_pos = range(len(tokens))
        ax.barh(y_pos, [1]*len(tokens), color='#2196F3', alpha=0.6)
        for yi, (tok, cod) in enumerate(zip(tokens, codes)):
            ax.text(0.05, yi, f"'{tok}' -> {cod}", va='center', fontsize=7)
        ax.set_title(f"{mr['prompt'][:30]}\n= {mr['decoded_mean'][:35]}",
                    fontsize=8, fontweight='bold')
        ax.set_yticks([])
        ax.set_xlim(0, 1.2)
    for mi in range(n_show, 6):
        axes[mi].set_visible(False)

    plt.suptitle('Phase 31: Decompiling the LLM Brain\n'
                 'GPT-2 Hidden States -> Rosetta Space -> Python Code',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase31_llm_brain.png'), dpi=150)
    plt.close()
    print(f"\nPhase 31 complete in {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
