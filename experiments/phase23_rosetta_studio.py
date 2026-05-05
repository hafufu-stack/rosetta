"""
Phase 23: The Rosetta Studio UI
==================================
Interactive Gradio web app to explore the Rosetta Space.
Features: NL->Code, Code Morphing, Python<->JS transpilation.
"""
import os, json, sys
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_app():
    """Build and return the Gradio app."""
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
    z_nl, z_ast = latents['nl'], latents['ast']

    # Python decoder
    with open(os.path.join(DATA_DIR, 'char_vocab.json'), 'r') as f:
        py_vd = json.load(f)
    py_idx2char = {int(i): c for c, i in py_vd['char2idx'].items()}
    V_py = len(py_vd['char2idx'])

    sys.path.insert(0, BASE_DIR)
    from experiments.phase9_generative_decompiler import CodeDecoder, decode_tokens

    py_decoder = CodeDecoder(64, V_py, hidden=128, max_len=80).to(DEVICE)
    dec_path = os.path.join(DATA_DIR, 'decoder_robust.pt')
    if not os.path.exists(dec_path):
        dec_path = os.path.join(DATA_DIR, 'decoder.pt')
    py_decoder.load_state_dict(torch.load(dec_path, map_location=DEVICE, weights_only=True))
    py_decoder.eval()

    # JS decoder (if available)
    js_decoder = None
    js_idx2char = None
    js_path = os.path.join(DATA_DIR, 'decoder_js.pt')
    js_vocab_path = os.path.join(DATA_DIR, 'js_vocab.json')
    if os.path.exists(js_path) and os.path.exists(js_vocab_path):
        with open(js_vocab_path, 'r') as f:
            js_vd = json.load(f)
        js_idx2char = {int(i): c for c, i in js_vd['char2idx'].items()}
        V_js = len(js_vd['char2idx'])
        js_decoder = CodeDecoder(64, V_js, hidden=128, max_len=100).to(DEVICE)
        js_decoder.load_state_dict(torch.load(js_path, map_location=DEVICE, weights_only=True))
        js_decoder.eval()

    # Build source index and NL index
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

    def gen_py(z_vec):
        with torch.no_grad():
            z_t = torch.tensor(z_vec, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            tokens = py_decoder(z_t)
            return decode_tokens(tokens[0].cpu().numpy(), py_idx2char)

    def gen_js(z_vec):
        if js_decoder is None:
            return "(JS decoder not trained yet)"
        with torch.no_grad():
            z_t = torch.tensor(z_vec, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            tokens = js_decoder(z_t)
            return decode_tokens(tokens[0].cpu().numpy(), js_idx2char)

    def find_nl_vec(text):
        text_l = text.lower().strip()
        best_idx, best_score = 0, -1
        for nl_key, idx in nl_to_idx.items():
            # Simple word overlap score
            words_q = set(text_l.split())
            words_k = set(nl_key.split())
            score = len(words_q & words_k) / max(len(words_q | words_k), 1)
            if score > best_score:
                best_score = score
                best_idx = idx
        return z_nl[best_idx], dataset[best_idx]['nl']

    # === Tab 1: NL -> Code ===
    def nl_to_code(nl_text):
        z, matched_nl = find_nl_vec(nl_text)
        py = gen_py(z)
        js = gen_js(z)
        return f"Matched NL: {matched_nl}\n\nPython:\n{py}\n\nJavaScript:\n{js}"

    # === Tab 2: Code Morphing ===
    func_list = unique_sources[:50]

    def morph_code(func_a_name, func_b_name, t_value):
        if func_a_name not in src_to_idx or func_b_name not in src_to_idx:
            return "Function not found in dataset"
        va = z_ast[src_to_idx[func_a_name]]
        vb = z_ast[src_to_idx[func_b_name]]
        v_interp = (1 - t_value) * va + t_value * vb
        py = gen_py(v_interp)
        js = gen_js(v_interp)
        return f"t = {t_value:.2f}\n\nPython:     {py}\nJavaScript: {js}"

    # === Tab 3: Semantic Patching ===
    def semantic_patch(buggy_code, wrong_intent, correct_intent):
        if buggy_code not in src_to_idx:
            return "Buggy code not found in dataset. Try one from the dropdown."
        z_buggy = z_ast[src_to_idx[buggy_code]]
        v_wrong, _ = find_nl_vec(wrong_intent)
        v_correct, _ = find_nl_vec(correct_intent)
        z_patched = z_buggy - v_wrong + v_correct
        z_patched = z_patched / (np.linalg.norm(z_patched) + 1e-8)
        patched = gen_py(z_patched)
        return f"Buggy:   {buggy_code}\nPatched: {patched}"

    # === Tab 4: Transpiler ===
    def transpile(py_code):
        if py_code not in src_to_idx:
            return "Code not found in dataset. Try one from the dropdown."
        z = z_ast[src_to_idx[py_code]]
        py_out = gen_py(z)
        js_out = gen_js(z)
        return f"Python (re-decoded): {py_out}\nJavaScript:          {js_out}"

    # Build Gradio UI
    with gr.Blocks(title="Rosetta Studio") as app:
        gr.Markdown("""
        # The Rosetta Studio
        ### Interactive exploration of the Rosetta Semantic Space
        *NL <-> Python <-> JavaScript -- One Space, Three Languages*
        """)

        with gr.Tab("NL -> Code"):
            gr.Markdown("Type natural language and get Python + JavaScript code")
            nl_input = gr.Textbox(label="Natural Language",
                                  placeholder="e.g. add two numbers")
            nl_output = gr.Textbox(label="Generated Code", lines=6)
            nl_btn = gr.Button("Generate", variant="primary")
            nl_btn.click(nl_to_code, inputs=nl_input, outputs=nl_output)
            gr.Examples([
                ["add two numbers"],
                ["multiply x and y"],
                ["check if x is greater than y"],
                ["return absolute value"],
                ["subtract a from b"],
            ], inputs=nl_input)

        with gr.Tab("Code Morphing"):
            gr.Markdown("Smoothly transform between two functions")
            with gr.Row():
                func_a = gr.Dropdown(func_list, label="Function A", value=func_list[0])
                func_b = gr.Dropdown(func_list, label="Function B",
                                     value=func_list[min(1, len(func_list)-1)])
            morph_slider = gr.Slider(0, 1, value=0.5, step=0.05, label="Interpolation t")
            morph_output = gr.Textbox(label="Morphed Code", lines=4)
            morph_btn = gr.Button("Morph", variant="primary")
            morph_btn.click(morph_code, inputs=[func_a, func_b, morph_slider],
                           outputs=morph_output)

        with gr.Tab("Semantic Patch"):
            gr.Markdown("Fix bugs using vector arithmetic on NL intent vectors")
            bug_input = gr.Dropdown(func_list, label="Buggy Code", value=func_list[0])
            wrong_nl = gr.Textbox(label="Wrong Intent (NL)", placeholder="e.g. add")
            correct_nl = gr.Textbox(label="Correct Intent (NL)", placeholder="e.g. subtract")
            patch_output = gr.Textbox(label="Result", lines=3)
            patch_btn = gr.Button("Patch", variant="primary")
            patch_btn.click(semantic_patch,
                           inputs=[bug_input, wrong_nl, correct_nl],
                           outputs=patch_output)

        with gr.Tab("Transpiler"):
            gr.Markdown("Python -> Rosetta Space -> JavaScript")
            trans_input = gr.Dropdown(func_list, label="Python Code", value=func_list[0])
            trans_output = gr.Textbox(label="Transpiled", lines=3)
            trans_btn = gr.Button("Transpile", variant="primary")
            trans_btn.click(transpile, inputs=trans_input, outputs=trans_output)

    return app


def main():
    print("=" * 60)
    print("Phase 23: The Rosetta Studio UI")
    print("=" * 60)

    import time as time_mod
    # Save metadata
    results = {
        'phase': 23, 'name': 'The Rosetta Studio UI',
        'status': 'launched',
        'timestamp': time_mod.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase23_studio.json'), 'w') as f:
        json.dump(results, f, indent=2)

    import gradio as gr
    app = build_app()
    print("\nLaunching Rosetta Studio on http://localhost:7860")
    print("Press Ctrl+C to stop")
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=gr.themes.Soft())


if __name__ == '__main__':
    main()
