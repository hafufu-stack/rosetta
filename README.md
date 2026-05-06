# Project Rosetta 🏛️

**Linear Algebraic Compilation, Neural Decompilation, and the 10 Laws of Software Physics in a Unified 5D Latent Space**

[![Paper](https://img.shields.io/badge/Paper-Zenodo-blue)](https://doi.org/10.5281/zenodo.20036684)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20036684.svg)](https://doi.org/10.5281/zenodo.20036684)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)

## Overview

Project Rosetta is a systematic **86-phase** investigation demonstrating that **compilation is a linear operator**, **decompilation is its regularized inverse**, and **semantic code manipulation is vector arithmetic** — all within a shared 64-dimensional latent space aligning natural language (NL), Python AST, and compiled bytecode.

The extended investigation reveals that programs reside on a **~5-dimensional manifold** that is **invariant under Turing-complete extensions**, obeys **gauge symmetries and conservation laws**, supports **semantic malware detection at 100% precision**, enables **direct silicon translation**, and functions as a **neural calculator at R²=0.97**. Together, these 86 phases establish the **10 Laws of Software Physics** with a unified **Rosetta Score of 90.8/100**.

### Key Discoveries

| Discovery | Phase | Result |
|-----------|-------|--------|
| **Compilation = Matrix Multiply** | P3 | AST→Binary: R²=0.965, 90% energy in 4 dims |
| **Generative Decompilation** | P9 | GRU decoder: 100% semantic accuracy |
| **Binary Surgery** | P11 | SVD-axis interventions: 64% semantic change |
| **Neural CPU** | P16 | Predict execution from vectors: R²=0.924 |
| **Information Preservation** | P29 | 100% MI preserved across compilation |
| **True Dimensionality = 5** | P38-40 | ~5D manifold explains 86.3% variance |
| **5D Turing Invariance** | P49 | Adding if/for changes PCA by <±0.1% |
| **Latent Antivirus** | P60 | Malware detection: 100% precision |
| **Silicon Translation** | P62 | Python→x86 via 5D: 76.7% accuracy |
| **14 Species of Code** | P64 | DBSCAN: 14 clusters, 0% noise |
| **Operator Algebra** | P66 | Non-commutative composition: 64% acc |
| **Gauge Symmetry** | P74 | Variable renaming: cos=1.000 |
| **Noether's Theorem** | P82 | 6/6 charges perfectly conserved |
| **Latent Calculator** | P84 | Predict f(x,y) from 5D: R²=0.97 |

## Architecture

```
Natural Language ──→ [Encoder_NL] ──→ ┐
                                       ├──→ 64-dim Rosetta Space
Python AST ────────→ [Encoder_AST] ──→ ┤     (true dim ≈ 5)
                                       ├──→ W_compile (linear!) ──→ Binary Space
Bytecode ──────────→ [Encoder_BC] ───→ ┘

10 Laws of Software Physics │ Rosetta Score: 90.8/100
```

## Project Structure

```
rosetta/
├── data/                    # Dataset and trained models
├── experiments/             # All 86 phase scripts
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
│   ├── phase43–48           # Gravity, genome, completeness, isolation, highway
│   ├── phase49–59           # Turing invariance, I/O search, linter, studio v3
│   ├── phase60–68           # Antivirus, NVM, silicon, taxonomy, operator algebra
│   └── phase69–86           # Symmetry, conservation, periodic table, calculator
├── figures/                 # Generated visualizations (86 figures)
├── papers/                  # LaTeX paper (V1: P1-23, V2: P1-48, V3: P1-86)
├── results/                 # JSON result files
├── runner.py                # Chapter I runner (P1-4)
├── runner_ch2.py – ch23.py  # Chapters II–XXIII
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
# ... through runner_ch14.py

# Ultimate Grounding (Chapters XV–XIX, Phases 49-68)
python runner_ch15.py
# ... through runner_ch19.py

# The Deeper Universe (Chapters XX–XXIII, Phases 69-86)
python runner_ch20.py
# ... through runner_ch23.py
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
- **P26**: LLM-to-binary matrix — GPT-2 → bytecode (R²=0.815)
- **P28**: Bit-level grounding — predict individual bits at 98.4%
- **P29**: Information preservation law — MI ≈ 5300 bits preserved

### Chapter IX: The Neural Operating System (P30-33)
- **P30**: Zero-shot executable synthesis (100% functional accuracy)
- **P31**: LLM brain decompilation — read GPT-2's "mind" as Python
- **P33**: Compilation eigenfunctions — programs unchanged by compilation

### Chapter X–XIV: The Physics of Software (P34-48)
- **P34-36**: Null space, evolution, uncertainty principle
- **P37-39**: Rosetta Paradox (2.2× info density), entanglement, compiler memory
- **P40-42**: Five Elements (86.3% variance), phase transitions, holographic principle
- **P43-45**: Semantic gravity, genome, arithmetic completeness
- **P46-48**: Isolation paradox, hidden highway, syntax-semantics spectrum

### Chapter XV–XIX: The Ultimate Grounding (P49-68) 🆕
- **P49**: 5D invariance under Turing completeness (±0.1% change)
- **P55**: Inverse execution via gradient descent (40% first-try)
- **P56**: I/O search — 100% accuracy on program-from-examples
- **P58**: Latent Linter — add vs sub: cos=-0.770 (opposite directions)
- **P60**: Latent Antivirus — 100% precision, sees through obfuscation
- **P61**: Neural Virtual Machine — execute programs in latent space
- **P62**: Silicon Translation — Python→x86 at 76.7% via 5D embedding
- **P64**: 14 Species of Code — DBSCAN taxonomy with 0% noise
- **P66**: Operator Algebra — non-commutative composition (64% vs 9%)
- **P67**: Attractor-Stabilized NVM — 99.88% drift reduction
- **P68**: Grand Finale — integrated demonstration

### Chapter XX–XXIII: The Deeper Universe (P69-86) 🆕
- **P69**: Adversarial robustness — cos=1.000 for renaming, detects mutations
- **P71**: Fractal hypothesis — dimension 0.22 (not fractal, discrete islands)
- **P72**: Genetic programming — evolve abs(x-y) in 4 generations
- **P74**: Gauge symmetry — variable naming is a perfect symmetry (cos=1.000)
- **P76**: Information bottleneck — 10.2× compression (193→19 bits)
- **P78**: Structure-behavior independence — correlation only r=0.034
- **P79**: Program analogies — (x+y)-(a+b)+(x*y)=(a*b) at distance 0.000
- **P80**: Semantic gradient field — 3 attractors, ground state = abs(x)
- **P81**: Periodic table of programs — 7 periods, 10 groups, 21 types
- **P82**: Noether's theorem — 6/6 charges perfectly conserved
- **P83**: The 10 Laws of Software Physics — Rosetta Score: 90.8/100
- **P84**: Latent Calculator — predict f(x,y) from 5D at R²=0.97
- **P85**: Latent debugging — bug vectors are multi-dimensional
- **P86**: Quantum entanglement mapping of program correlations

## The 10 Laws of Software Physics

1. **The 5-Dimensional Theorem**: Programs reside on a 5D manifold (87% variance)
2. **The Variable Symmetry Law**: Variable naming is a perfect gauge symmetry
3. **Noether's Software Theorem**: 6/6 charges conserved under renaming
4. **The Operator Algebra Law**: Composition is non-commutative (64% vs 9%)
5. **The Taxonomy Theorem**: 14 natural species with 0% noise
6. **The Independence Principle**: Structure and behavior are orthogonal (r=0.034)
7. **The Compression Theorem**: 10.2× compression (193→19 bits)
8. **The Continuity Theorem**: Interpolation passes through meaningful intermediates
9. **The Semantic Invariance Law**: Malware detection at 100% precision
10. **The Rosetta Principle**: Source, behavior, and machine code are projections of one 5D object

## Citation

```bibtex
@article{funasaki2026rosetta,
  title={Project Rosetta: Linear Algebraic Compilation, Neural Decompilation,
         and the 10 Laws of Software Physics in a Unified 5D Latent Space},
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
