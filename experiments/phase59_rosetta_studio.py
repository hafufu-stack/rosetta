"""
Phase 59: The Rosetta Studio - GRAND FINALE
=============================================
Interactive Web UI combining all Rosetta discoveries:
  1. Semantic Code Search (P56)
  2. Code Morphing slider (P17)
  3. Latent Linter (P58)

The culmination of 59 experimental phases.
"""
import os, json, sys, inspect
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ============================================================
# Load all models and data at startup
# ============================================================
print("Loading Rosetta Space...")

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
sources = [d['source'] for d in dataset]

from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
pca = PCA(n_components=10).fit(z_ast)
z_5d = pca.transform(z_ast)[:, :5]

# Build unique index
unique = {}
for i, src in enumerate(sources):
    if src not in unique:
        unique[src] = {'idx': i, 'z_ast': z_ast[i], 'z_nl': z_nl[i],
                      'z_5d': z_5d[i], 'nl': dataset[i].get('nl', '')}
func_list = list(unique.items())
db_ast = np.array([f[1]['z_ast'] for f in func_list])
db_5d = np.array([f[1]['z_5d'] for f in func_list])
func_names = [f[0] for f in func_list]

# Load decoder
sys.path.insert(0, BASE_DIR)
with open(os.path.join(DATA_DIR, 'char_vocab.json'), 'r') as f:
    vd = json.load(f)
idx2char = {int(i): c for c, i in vd['char2idx'].items()}
V_size = len(vd['char2idx'])
from experiments.phase9_generative_decompiler import CodeDecoder, decode_tokens
decoder = CodeDecoder(64, V_size, hidden=128, max_len=80).to(DEVICE)
dec_path = os.path.join(DATA_DIR, 'decoder_robust.pt')
if not os.path.exists(dec_path):
    dec_path = os.path.join(DATA_DIR, 'decoder.pt')
decoder.load_state_dict(torch.load(dec_path, map_location=DEVICE, weights_only=True))
decoder.eval()

print(f"Loaded {len(func_list)} functions in 5D Rosetta Space")


# ============================================================
# Core functions
# ============================================================
def gen_code(z64):
    """Decode 64-dim vector to Python code."""
    with torch.no_grad():
        zt = torch.tensor(z64.astype(np.float32)).unsqueeze(0).to(DEVICE)
        tokens = decoder(zt)
        return decode_tokens(tokens[0].cpu().numpy(), idx2char)


def search_by_io(io_text):
    """Search by I/O examples. Format: 'f(1,2)=3; f(5,3)=8'"""
    io_pairs = []
    try:
        for part in io_text.split(';'):
            part = part.strip()
            if '=' not in part:
                continue
            lhs, rhs = part.split('=', 1)
            rhs = float(rhs.strip())
            # Parse f(a,b) or f(a)
            paren_start = lhs.index('(')
            paren_end = lhs.rindex(')')
            args_str = lhs[paren_start+1:paren_end]
            args = [float(a.strip()) for a in args_str.split(',')]
            io_pairs.append((*args, rhs))
    except Exception as e:
        return f"Parse error: {e}\nFormat: f(1,2)=3; f(5,3)=8"

    if not io_pairs:
        return "No valid I/O pairs found"

    # Score each function
    results = []
    for src, info in func_list:
        try:
            ns = {}
            exec(compile(src, '<string>', 'exec'), ns)
            fn = [v for k, v in ns.items()
                  if callable(v) and not k.startswith('_')][0]
            sig = inspect.signature(fn)
            n_p = len(sig.parameters)
            n_match = 0
            for io in io_pairs:
                try:
                    if n_p == 1:
                        r = float(fn(io[0]))
                        target = io[1]
                    elif n_p == 2:
                        r = float(fn(io[0], io[1]))
                        target = io[2]
                    else:
                        continue
                    if abs(r - target) < 0.01:
                        n_match += 1
                except Exception:
                    pass
            if n_match > 0:
                results.append((src, n_match, len(io_pairs)))
        except Exception:
            pass

    results.sort(key=lambda x: x[1], reverse=True)

    output = f"## Search Results ({len(io_pairs)} I/O constraints)\n\n"
    for i, (src, matched, total) in enumerate(results[:10]):
        status = "PERFECT" if matched == total else f"{matched}/{total}"
        output += f"{i+1}. `{src}` [{status}]\n"
    if not results:
        output += "No matching functions found.\n"
    return output


def search_by_text(query):
    """Search by text similarity (keyword matching in source)."""
    query_lower = query.lower()
    scored = []
    for src, info in func_list:
        score = 0
        src_lower = src.lower()
        for word in query_lower.split():
            if word in src_lower:
                score += 2
            # Semantic keyword mapping
            kw_map = {
                'add': ['+', 'sum'], 'subtract': ['-'],
                'multiply': ['*'], 'divide': ['/'],
                'absolute': ['abs'], 'negate': ['-x', '-n', '-a'],
                'compare': ['>', '<', '=='], 'greater': ['>'],
                'less': ['<'], 'equal': ['=='], 'maximum': ['max'],
                'minimum': ['min'], 'square': ['* x', '** 2', 'x * x'],
                'double': ['* 2'], 'modulo': ['%'],
            }
            for kw, patterns in kw_map.items():
                if kw in word:
                    for p in patterns:
                        if p in src:
                            score += 1
        if score > 0:
            scored.append((src, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    output = f"## Text Search: '{query}'\n\n"
    for i, (src, score) in enumerate(scored[:10]):
        output += f"{i+1}. `{src}` (relevance: {score})\n"
    if not scored:
        output += "No matches. Try keywords like: add, multiply, compare\n"
    return output


def morph_code(func_a_src, func_b_src, t):
    """Morph between two functions in latent space."""
    z_a = unique.get(func_a_src, {}).get('z_ast')
    z_b = unique.get(func_b_src, {}).get('z_ast')

    if z_a is None:
        return f"Function A not found in database."
    if z_b is None:
        return f"Function B not found in database."

    # Linear interpolation in 64D space
    z_morph = z_a * (1 - t) + z_b * t
    code = gen_code(z_morph)

    # Generate full morph sequence for display
    output = f"## Code Morphing: t={t:.2f}\n\n"
    output += f"**A** (t=0.0): `{func_a_src}`\n\n"

    steps = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for step in steps:
        z_s = z_a * (1 - step) + z_b * step
        c = gen_code(z_s)
        marker = " <-- current" if abs(step - t) < 0.05 else ""
        output += f"t={step:.1f}: `{c}`{marker}\n\n"

    output += f"**B** (t=1.0): `{func_b_src}`\n"

    # 5D coordinates
    z5_a = unique[func_a_src]['z_5d']
    z5_b = unique[func_b_src]['z_5d']
    dist = np.linalg.norm(z5_a - z5_b)
    output += f"\n5D Distance: {dist:.4f}\n"
    return output


def lint_code(code_src, intent_src):
    """Compute semantic alignment between code and intent."""
    z_code = unique.get(code_src, {}).get('z_ast')
    z_intent = unique.get(intent_src, {}).get('z_ast')

    if z_code is None:
        return "Code not found in Rosetta database. Try a function from the dropdown."
    if z_intent is None:
        return "Intent function not found. Try a function from the dropdown."

    cos = float(cosine_similarity(
        z_code.reshape(1, -1), z_intent.reshape(1, -1))[0, 0])

    z5_code = unique[code_src]['z_5d']
    z5_intent = unique[intent_src]['z_5d']
    dist_5d = np.linalg.norm(z5_code - z5_intent)

    if cos > 0.95:
        verdict = "PASS - Identical meaning"
        emoji = "##"
    elif cos > 0.7:
        verdict = "WARN - Similar but not exact"
        emoji = "##"
    elif cos > 0.4:
        verdict = "SUSPICIOUS - Semantic drift detected"
        emoji = "##"
    else:
        verdict = "REJECT - Likely hallucination!"
        emoji = "##"

    output = f"{emoji} Latent Linter Report\n\n"
    output += f"**Code**: `{code_src}`\n\n"
    output += f"**Intent**: `{intent_src}`\n\n"
    output += f"---\n\n"
    output += f"| Metric | Value |\n"
    output += f"|--------|-------|\n"
    output += f"| Cosine Similarity | {cos:.4f} |\n"
    output += f"| 5D Distance | {dist_5d:.4f} |\n"
    output += f"| Verdict | **{verdict}** |\n\n"

    # Show what's nearby in 5D
    dists = np.linalg.norm(db_5d - z5_code, axis=1)
    nearest = np.argsort(dists)[1:4]
    output += "**Nearest neighbors** (in 5D space):\n"
    for ni in nearest:
        output += f"- `{func_names[ni]}` (d={dists[ni]:.4f})\n"

    return output


# ============================================================
# Gradio UI
# ============================================================
def build_ui():
    import gradio as gr

    # Dropdown choices
    func_choices = func_names[:100]  # Limit for UI performance

    with gr.Blocks(
        title="The Rosetta Studio",
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="blue",
        )
    ) as demo:
        gr.Markdown("""
# The Rosetta Studio
### Project Rosetta: 59 Phases of Discovery
*The Physics of Software - Interactive Explorer*

Programs live in a **5-dimensional manifold**. This tool lets you explore it.
        """)

        with gr.Tabs():
            # ==========================================
            # Tab 1: Semantic Search
            # ==========================================
            with gr.Tab("Semantic Search"):
                gr.Markdown("### Find code by meaning, not by text")
                with gr.Row():
                    with gr.Column():
                        search_mode = gr.Radio(
                            ["I/O Examples", "Text Query"],
                            label="Search Mode", value="I/O Examples")
                        io_input = gr.Textbox(
                            label="I/O Examples",
                            placeholder="f(1,2)=3; f(5,3)=8; f(-1,1)=0",
                            lines=2)
                        text_input = gr.Textbox(
                            label="Text Query",
                            placeholder="add two numbers",
                            lines=1)
                        search_btn = gr.Button("Search", variant="primary")
                    with gr.Column():
                        search_output = gr.Markdown(label="Results")

                def do_search(mode, io_text, text_query):
                    if mode == "I/O Examples":
                        return search_by_io(io_text)
                    else:
                        return search_by_text(text_query)

                search_btn.click(
                    do_search,
                    inputs=[search_mode, io_input, text_input],
                    outputs=search_output)

            # ==========================================
            # Tab 2: Code Morphing
            # ==========================================
            with gr.Tab("Code Morphing"):
                gr.Markdown("### Watch code transform smoothly in latent space")
                with gr.Row():
                    func_a = gr.Dropdown(
                        choices=func_choices, label="Function A",
                        value=func_choices[0] if func_choices else None)
                    func_b = gr.Dropdown(
                        choices=func_choices, label="Function B",
                        value=func_choices[5] if len(func_choices) > 5 else None)
                morph_slider = gr.Slider(
                    0.0, 1.0, value=0.5, step=0.05,
                    label="Morphing: A <-----> B")
                morph_btn = gr.Button("Morph!", variant="primary")
                morph_output = gr.Markdown(label="Morphing Result")

                morph_btn.click(
                    morph_code,
                    inputs=[func_a, func_b, morph_slider],
                    outputs=morph_output)

            # ==========================================
            # Tab 3: Latent Linter
            # ==========================================
            with gr.Tab("Latent Linter"):
                gr.Markdown("### Does this code match the intent? (No execution needed)")
                with gr.Row():
                    lint_code_input = gr.Dropdown(
                        choices=func_choices,
                        label="Code to Check",
                        value=func_choices[0] if func_choices else None)
                    lint_intent_input = gr.Dropdown(
                        choices=func_choices,
                        label="Intended Behavior (reference)",
                        value=func_choices[0] if func_choices else None)
                lint_btn = gr.Button("Lint!", variant="primary")
                lint_output = gr.Markdown(label="Linter Report")

                lint_btn.click(
                    lint_code,
                    inputs=[lint_code_input, lint_intent_input],
                    outputs=lint_output)

            # ==========================================
            # Tab 4: About
            # ==========================================
            with gr.Tab("About"):
                gr.Markdown("""
### Project Rosetta: The Physics of Software

**59 experimental phases** exploring the mathematical structure of code.

#### Key Discoveries:
- **5-Dimensional Theory (P40)**: All programs live in 5 dimensions
- **Holographic Principle (P42)**: Meaning concentrates in 5D, syntax in 32D
- **Semantic Gravity (P43)**: `list` operations are gravitational attractors
- **5D Invariance (P49)**: Even loops and control flow fold into 5D
- **I/O Search 100% (P56)**: Find any function from its behavior

#### Paper
[The Physics of Software](https://doi.org/10.5281/zenodo.19808285)
by Hiroto Funasaki

*Built with Rosetta Space embeddings and Gradio*
                """)

    return demo


def main():
    print("=" * 60)
    print("Phase 59: The Rosetta Studio - GRAND FINALE")
    print("=" * 60)

    # First run P57 and P58
    print("\n--- Running P57 (Rosetta-RAG) ---")
    try:
        sys.path.insert(0, os.path.join(BASE_DIR, 'experiments'))
        from experiments.phase57_rosetta_rag import main as p57_main
        p57_main()
    except Exception as e:
        print(f"  P57 error: {e}")

    print("\n--- Running P58 (Latent Linter) ---")
    try:
        from experiments.phase58_latent_linter import main as p58_main
        p58_main()
    except Exception as e:
        print(f"  P58 error: {e}")

    # Save completion record
    results = {
        'phase': 59, 'name': 'The Rosetta Studio',
        'n_functions': len(func_list),
        'features': ['Semantic Search', 'Code Morphing', 'Latent Linter'],
        'timestamp': __import__('time').strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase59_rosetta_studio.json'), 'w',
              encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    # Beep
    try:
        import winsound, time
        for _ in range(5):
            winsound.Beep(1000, 500)
            time.sleep(0.3)
    except Exception:
        pass

    # Launch Gradio
    print("\n" + "=" * 60)
    print("  LAUNCHING THE ROSETTA STUDIO!")
    print("  Open your browser to the URL below.")
    print("=" * 60)

    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == '__main__':
    main()
