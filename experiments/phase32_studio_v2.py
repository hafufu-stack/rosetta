"""
Phase 32: The Rosetta Studio v2 (Final UI)
=============================================
Interactive Gradio app with bit visualization, LLM mind reading,
code morphing, and all 29+ phases of discoveries.
"""
import os, json, sys, time
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    print("=" * 60)
    print("Phase 32: The Rosetta Studio v2 (Final UI)")
    print("=" * 60)

    import gradio as gr

    # Load everything
    ds_file = os.path.join(DATA_DIR, 'rosetta_dataset_v2.json')
    if not os.path.exists(ds_file):
        ds_file = os.path.join(DATA_DIR, 'rosetta_dataset.json')
    with open(ds_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)['dataset']

    lat_file = os.path.join(DATA_DIR, 'rosetta_latents_v2.npz')
    if not os.path.exists(lat_file):
        lat_file = os.path.join(DATA_DIR, 'rosetta_latents.npz')
    latents = np.load(lat_file)
    z_nl, z_ast, z_bc = latents['nl'], latents['ast'], latents['bc']

    # Python decoder
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

    # Build indexes
    src_to_idx = {}
    nl_to_idx = {}
    unique_sources = []
    for i, d in enumerate(dataset):
        if d['source'] not in src_to_idx:
            src_to_idx[d['source']] = i
            unique_sources.append(d['source'])
        nl_key = d['nl'].lower().strip()
        if nl_key not in nl_to_idx:
            nl_to_idx[nl_key] = i

    # Bit predictor
    from sklearn.linear_model import Ridge
    MAX_BYTES = 32
    import dis as dis_mod, io as io_mod

    def source_to_bits(src):
        try:
            code = compile(src, '<t>', 'exec')
            for c in code.co_consts:
                if hasattr(c, 'co_code'):
                    raw = c.co_code; break
            else:
                raw = code.co_code
            bits = []
            for bv in raw[:MAX_BYTES]:
                for bit in range(8):
                    bits.append((bv >> (7-bit)) & 1)
            while len(bits) < MAX_BYTES*8:
                bits.append(0)
            return bits[:MAX_BYTES*8]
        except:
            return [0]*(MAX_BYTES*8)

    all_srcs = list(src_to_idx.keys())
    bit_matrix = np.array([source_to_bits(s) for s in all_srcs], dtype=np.float32)
    nl_vecs_for_bits = np.array([z_nl[src_to_idx[s]] for s in all_srcs])
    bit_reg = Ridge(alpha=1.0).fit(nl_vecs_for_bits, bit_matrix)

    def gen_code(z_vec):
        with torch.no_grad():
            z_t = torch.tensor(z_vec, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            tokens = decoder(z_t)
            return decode_tokens(tokens[0].cpu().numpy(), idx2char)

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
        return z_nl[best_idx], dataset[best_idx]['nl']

    func_list = unique_sources[:50]

    # Tab 1: NL -> Code + Bits
    def nl_to_code_bits(nl_text):
        z, matched = find_nl_vec(nl_text)
        code = gen_code(z)
        pred_bits = bit_reg.predict(z.reshape(1,-1))[0]
        bits_str = ''.join(['1' if b > 0.5 else '0' for b in pred_bits[:64]])
        bits_visual = ' '.join([bits_str[i:i+8] for i in range(0, 64, 8)])
        return (f"Matched: {matched}\n\nGenerated Python:\n{code}\n\n"
                f"Predicted Bytecode Bits (first 64):\n{bits_visual}")

    # Tab 2: Code Morphing
    def morph(func_a, func_b, t):
        if func_a not in src_to_idx or func_b not in src_to_idx:
            return "Not found"
        va = z_ast[src_to_idx[func_a]]
        vb = z_ast[src_to_idx[func_b]]
        v = (1-t)*va + t*vb
        return f"t={t:.2f}\n\n{gen_code(v)}"

    # Tab 3: Semantic Patch
    def patch(buggy, wrong_nl, correct_nl):
        if buggy not in src_to_idx:
            return "Not found"
        z_b = z_ast[src_to_idx[buggy]]
        v_w, _ = find_nl_vec(wrong_nl)
        v_c, _ = find_nl_vec(correct_nl)
        z_p = z_b - v_w + v_c
        return f"Buggy:   {buggy}\nPatched: {gen_code(z_p)}"

    # Build UI
    with gr.Blocks(title="Rosetta Studio v2") as app:
        gr.Markdown("# The Rosetta Studio v2\n"
                    "### 29 phases of discovery: NL <-> Code <-> Binary\n"
                    "*Words become bits. Bits become meaning.*")

        with gr.Tab("NL -> Code + Bits"):
            gr.Markdown("Type natural language -> get Python code AND predicted bytecode bits")
            nl_in = gr.Textbox(label="Natural Language", placeholder="e.g. add two numbers")
            nl_out = gr.Textbox(label="Result", lines=8)
            gr.Button("Generate", variant="primary").click(nl_to_code_bits, nl_in, nl_out)
            gr.Examples([["add two numbers"],["multiply x and y"],
                        ["return absolute value"],["check if greater"]], nl_in)

        with gr.Tab("Code Morphing"):
            gr.Markdown("Smoothly interpolate between two functions")
            with gr.Row():
                fa = gr.Dropdown(func_list, label="Function A", value=func_list[0])
                fb = gr.Dropdown(func_list, label="Function B",
                                value=func_list[min(1,len(func_list)-1)])
            sl = gr.Slider(0, 1, 0.5, step=0.05, label="t")
            mo = gr.Textbox(label="Morphed", lines=3)
            gr.Button("Morph", variant="primary").click(morph, [fa,fb,sl], mo)

        with gr.Tab("Semantic Patch"):
            gr.Markdown("Fix bugs with vector arithmetic on intent")
            bi = gr.Dropdown(func_list, label="Buggy Code", value=func_list[0])
            wn = gr.Textbox(label="Wrong Intent", placeholder="add")
            cn = gr.Textbox(label="Correct Intent", placeholder="subtract")
            po = gr.Textbox(label="Result", lines=3)
            gr.Button("Patch", variant="primary").click(patch, [bi,wn,cn], po)

    # Save metadata
    res = {'phase': 32, 'name': 'Rosetta Studio v2', 'status': 'launched',
           'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')}
    with open(os.path.join(RESULTS_DIR, 'phase32_studio_v2.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2)

    print("\nLaunching Rosetta Studio v2 on http://localhost:7860")
    app.launch(server_name="0.0.0.0", server_port=7860, share=False,
               theme=gr.themes.Soft())

if __name__ == '__main__':
    main()
