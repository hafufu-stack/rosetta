# Project Rosetta 🏛️

**Linear Algebraic Compilation, Neural Decompilation, and Semantic Code Surgery in a Unified Latent Space**

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](papers/paper_v1.pdf)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)

## Overview

Project Rosetta demonstrates that **compilation is a linear operator**, **decompilation is its regularized inverse**, and **semantic code manipulation is vector arithmetic** — all within a shared 64-dimensional latent space aligning natural language (NL), Python AST, and compiled bytecode.

### Key Discoveries

| Discovery | Result |
|-----------|--------|
| **Compilation = Matrix Multiply** | AST→Binary: R²=0.965, rank 27, 90% energy in 4 dimensions |
| **Linear Decompilation** | Binary→AST via Ridge: R²=0.862, round-trip cos=0.956 |
| **Generative Decompilation** | GRU decoder: 100% semantic accuracy (variable names differ) |
| **Compiler Anatomy** | SVD probes: 93-100% accuracy identifying arithmetic/comparison axes |
| **Semantic Clustering** | t-SNE cluster purity: 99.2% |
| **Binary Surgery** | SVD-axis interventions alter semantics in 64% of cases |
| **Auto-Patching** | NL vector arithmetic fixes 57% of operator bugs |
| **Neural Compiler** | NL→Code via matrix: 63% semantic accuracy |
| **Neural CPU** | Predict execution from vectors: R²=0.924 |
| **Code Morphing** | Smooth interpolation between programs in latent space |
| **NL CPU** | Execute natural language directly: R²=0.435, correlation=0.731 |

## Architecture

```
Natural Language ──→ [Encoder_NL] ──→ ┐
                                       ├──→ 64-dim Rosetta Space
Python AST ────────→ [Encoder_AST] ──→ ┤
                                       ├──→ W_compile (linear!) ──→ Binary Space
Bytecode ──────────→ [Encoder_BC] ───→ ┘
```

## Project Structure

```
rosetta/
├── data/                    # Dataset and trained models
├── experiments/             # All 23 phase scripts
│   ├── phase1_rosetta_dataset.py    # Tri-modal dataset generation
│   ├── phase2_latent_alignment.py   # Contrastive alignment
│   ├── phase3_translation_matrix.py # Compilation = linear algebra
│   ├── phase6_linear_decompiler.py  # Ridge-based decompilation
│   ├── phase7_compiler_anatomy.py   # SVD mechanistic analysis
│   ├── phase9_generative_decompiler.py  # GRU code decoder
│   ├── phase11_binary_surgery.py    # SVD-axis interventions
│   ├── phase13_auto_patching.py     # NL vector bug repair
│   ├── phase16_neural_cpu.py        # Predict execution results
│   ├── phase17_code_morphing.py     # Latent space interpolation
│   ├── phase20_nl_cpu.py            # Natural language execution
│   ├── phase23_rosetta_studio.py    # Interactive Gradio UI
│   └── ...                          # All other phases
├── figures/                 # Generated visualizations
├── papers/                  # LaTeX paper
├── results/                 # JSON result files
├── runner.py                # Chapter I runner (P1-4)
├── runner_ch2.py            # Chapter II runner (P5-8)
├── runner_ch3.py            # Chapter III runner (P9-11)
├── runner_ch4.py            # Chapter IV runner (P12-14)
├── runner_ch5.py            # Chapter V runner (P15-17)
├── runner_ch6.py            # Chapter VI runner (P18-20)
└── runner_ch7.py            # Chapter VII runner (P21-23)
```

## Quick Start

### Requirements

```bash
pip install torch numpy scikit-learn matplotlib gradio
```

### Run All Experiments

```bash
# Chapter I: Foundation (Phases 1-4)
python runner.py

# Chapter II: Scaling & Analysis (Phases 5-8)
python runner_ch2.py

# ... through Chapter VII
python runner_ch7.py
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

## Citation

```bibtex
@article{funasaki2026rosetta,
  title={Project Rosetta: Linear Algebraic Compilation, Neural Decompilation, 
         and Semantic Code Surgery in a Unified Latent Space},
  author={Funasaki, Hiroto},
  year={2026}
}
```

## License

MIT License

## Acknowledgments

This research was conducted entirely independently. The author is actively seeking community sponsorship at [GitHub Sponsors](https://github.com/sponsors/hafufu-stack).

**AI Collaboration Statement**: This research was conducted as a collaborative effort between the human author and AI research assistants. All experimental decisions, research direction, and final interpretation were made by the human author.
