# Project Rosetta 🏛️

**Linear Algebraic Compilation, Neural Decompilation, and the Physics of Software in a 5-Dimensional Latent Space**

[![Paper](https://img.shields.io/badge/Paper-Zenodo-blue)](https://doi.org/10.5281/zenodo.20036684)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20036684.svg)](https://doi.org/10.5281/zenodo.20036684)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)

## Overview

Project Rosetta demonstrates that **compilation is a linear operator**, **decompilation is its regularized inverse**, and **semantic code manipulation is vector arithmetic** — all within a shared 64-dimensional latent space aligning natural language (NL), Python AST, and compiled bytecode.

Beyond linearity, the extended investigation (48 phases) reveals that programs reside on a **~5-dimensional manifold**, compilation is an **information concentrator** (not destroyer), and the code manifold exhibits **phase transitions**, **gravitational wells**, and **hidden highways**.

### Key Discoveries

| Discovery | Result |
|-----------|--------|
| **Compilation = Matrix Multiply** | AST→Binary: R²=0.965, 90% energy in 4 dimensions |
| **Generative Decompilation** | GRU decoder: 100% semantic accuracy |
| **Binary Surgery** | SVD-axis interventions alter semantics in 64% of cases |
| **Neural CPU** | Predict execution from vectors: R²=0.924 |
| **Bit-Level Grounding** | NL vectors predict bytecode bits at 98.4% accuracy |
| **Information Preservation** | 100% mutual information preserved across compilation |
| **Neural OS** | Zero-shot executable synthesis: 100% accuracy |
| **True Dimensionality** | Programs live on a ~5-dimensional manifold |
| **Five Elements of Code** | 5 principal axes explain 86.3% of all program variance |
| **Phase Transitions** | `x + y` becomes `x != y` at just 8% interpolation |
| **Holographic Principle** | 5D captures meaning; 32D achieves 100% code reconstruction |
| **Semantic Gravity** | `min`/`max`/`len` are gravitational wells; `x + y` is isolated |
| **Hidden Highway** | `x >= y` is the hub — 22/45 routes pass through it |

## Architecture

```
Natural Language ──→ [Encoder_NL] ──→ ┐
                                       ├──→ 64-dim Rosetta Space
Python AST ────────→ [Encoder_AST] ──→ ┤     (true dim ≈ 5)
                                       ├──→ W_compile (linear!) ──→ Binary Space
Bytecode ──────────→ [Encoder_BC] ───→ ┘
```

## Project Structure

```
rosetta/
├── data/                    # Dataset and trained models
├── experiments/             # All 48 phase scripts
│   ├── phase1–4             # Foundation: dataset, alignment, translation matrix
│   ├── phase5–8             # Scaling, decompilation, SVD anatomy, clustering
│   ├── phase9–11            # Generative decompiler, semantic arithmetic, surgery
│   ├── phase12–14           # Null-space, auto-patching, neural compiler
│   ├── phase15–17           # Robust decoder, Neural CPU, code morphing
│   ├── phase18–23           # Inverse exec, composition, NL CPU, studio
│   ├── phase24–29           # Grounding, universality, LLM-binary, bit-level, MI
│   ├── phase30–33           # Neural assembler, LLM brain, eigenfunctions
│   ├── phase34–36           # Null space, evolution, uncertainty principle
│   ├── phase37–39           # Rosetta paradox, entanglement, compiler memory
│   ├── phase40–42           # Five elements, phase transitions, holographic
│   ├── phase43–45           # Gravity, genome, arithmetic completeness
│   └── phase46–48           # Isolation paradox, hidden highway, syntax spectrum
├── figures/                 # Generated visualizations (46 figures)
├── papers/                  # LaTeX paper (V1: P1-23, V2: P1-48)
├── results/                 # JSON result files
├── runner.py                # Chapter I runner (P1-4)
├── runner_ch2.py – ch7.py   # Chapters II–VII (P5-23)
├── runner_ch8.py – ch14.py  # Chapters VIII–XIV (P24-48)
└── README.md
```

## Quick Start

### Requirements

```bash
pip install torch numpy scikit-learn matplotlib transformers scipy gradio
```

### Run All Experiments

```bash
# Foundation (Chapters I–VII, Phases 1-23)
python runner.py
python runner_ch2.py
# ... through runner_ch7.py

# Beyond Linearity (Chapters VIII–XIV, Phases 24-48)
python runner_ch8.py
python runner_ch9.py
# ... through runner_ch14.py
```

### Interactive Demo

```bash
python experiments/phase23_rosetta_studio.py
# Open http://localhost:7860
```

## Experiment Phases

### Chapter I: The Rosetta Stone (P1-4)
- **P1**: Tri-modal dataset generation (236 functions × ~12 NL variants)
- **P2**: Contrastive latent alignment (same-source sim: 0.964)
- **P3**: Translation matrices (compile R²=0.965)
- **P4**: Semantic arithmetic exploration

### Chapter II: The Linear Archaeologist (P5-8)
- **P5**: Dataset scaling to 2,736 triplets
- **P6**: Linear decompiler via Ridge regression (R²=0.862)
- **P7**: Mechanistic compiler anatomy via SVD probes
- **P8**: Rosetta Compass — semantic clustering (purity: 99.2%)

### Chapter III: The Latent Hacker (P9-11)
- **P9**: GRU-based generative decompiler (100% semantic accuracy)
- **P10**: Semantic arithmetic in void (analogy by generation)
- **P11**: Binary surgery via SVD-axis manipulation (64% change rate)

### Chapter IV: The Cybernetic Alchemist (P12-14)
- **P12**: Null-space robustness analysis
- **P13**: Semantic auto-patching via NL vectors (57% repair)
- **P14**: End-to-end neural compiler (63% semantic accuracy)

### Chapter V: Semantic Execution (P15-17)
- **P15**: SVD-Bottleneck robust decoder (70% at noise=2.0)
- **P16**: Neural CPU — predict execution results (R²=0.924)
- **P17**: Infinite code morphing in latent space

### Chapter VI: Ultimate Abstraction (P18-20)
- **P18**: Inverse execution via backpropagation
- **P19**: Latent function composition
- **P20**: Natural Language CPU (R²=0.435, corr=0.731)

### Chapter VII: The Rosetta Legacy (P21-23)
- **P21**: Manifold-guided inverse synthesis
- **P22**: Babel Fish transpiler (Python→JS)
- **P23**: Rosetta Studio interactive UI

### Chapter VIII: The Genesis of Binary Meaning (P24-29)
- **P24**: Semantic-binary grounding — the "fMRI of the compiler"
- **P25**: Metric-space universality test (100% triangle inequality)
- **P26**: LLM-to-binary matrix — GPT-2 hidden states → bytecode (R²=0.815)
- **P27**: Semantic crystallization — training dynamics analysis
- **P28**: Bit-level grounding — predict individual bits at 98.4% accuracy
- **P29**: Information preservation law — MI ≈ 5300 bits preserved

### Chapter IX: The Neural Operating System (P30-33)
- **P30**: Zero-shot executable synthesis (100% functional accuracy)
- **P31**: LLM brain decompilation — read GPT-2's "mind" as Python
- **P32**: Rosetta Studio v2 — expanded interactive tooling
- **P33**: Compilation eigenfunctions — programs unchanged by compilation

### Chapter X: The Deeper Laws (P34-36)
- **P34**: Compiler's null space — 44/64 dimensions discarded
- **P35**: Latent-space evolutionary programming — evolve code without text
- **P36**: Software uncertainty principle — precision vs manipulation tradeoff

### Chapter XI: The Information Paradox (P37-39)
- **P37**: The Rosetta Paradox — signal dims carry 2.2x info density
- **P38**: Semantic entanglement — true dimensionality ≈ 5.5
- **P39**: Compiler's memory — only 4 "immortal" dimensions survive infinite cycles

### Chapter XII: The Five Elements (P40-42)
- **P40**: Five principal axes of the software manifold (86.3% variance)
- **P41**: Program phase transitions — sharp semantic boundaries
- **P42**: Holographic principle — 5D for meaning, 32D for syntax

### Chapter XIII: The Topology of Software (P43-45)
- **P43**: Semantic gravity — gravitational wells of the code manifold
- **P44**: The Rosetta Genome — phylogenetic tree of programs
- **P45**: Latent arithmetic completeness — can 5 functions span all software?

### Chapter XIV: The Cross-Connections (P46-48)
- **P46**: The isolation paradox — lonely functions have 15x larger basins
- **P47**: The hidden highway — `x >= y` is the hub of all code routes
- **P48**: Syntax-semantics spectrum — meaning and form are inseparable

## Citation

```bibtex
@article{funasaki2026rosetta,
  title={Project Rosetta: Linear Algebraic Compilation, Neural Decompilation, 
         and the Physics of Software in a 5-Dimensional Latent Space},
  author={Funasaki, Hiroto},
  year={2026},
  doi={10.5281/zenodo.20036684}
}
```

## License

MIT License

## Acknowledgments

This research was conducted entirely independently. The author is actively seeking community sponsorship at [GitHub Sponsors](https://github.com/sponsors/hafufu-stack).

**AI Collaboration Statement**: This research was conducted as a collaborative effort between the human author and AI research assistants. All experimental decisions, research direction, and final interpretation were made by the human author.
