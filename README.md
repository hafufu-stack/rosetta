# Project Rosetta 🏛️

**Linear Algebraic Compilation, Neural Decompilation, and the 12 Laws of Software Physics in a Unified Latent Space**

[![Paper](https://img.shields.io/badge/Paper-Zenodo-blue)](https://doi.org/10.5281/zenodo.20036684)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20036684.svg)](https://doi.org/10.5281/zenodo.20036684)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)

## Overview

Project Rosetta is a systematic **117-phase** investigation demonstrating that **compilation is a linear operator**, **decompilation is its regularized inverse**, and **semantic code manipulation is vector arithmetic** — all within a shared 64-dimensional latent space aligning natural language (NL), Python AST, and compiled bytecode.

Programs reside on a **~5-dimensional manifold** governed by **gauge symmetries**, **conservation laws**, and **gravitational dynamics** with inverse-cube force ($d^{-3.40}$). The space is **holographic** (angles retain 99.9% accuracy), **classical** (commutator = 0), and **curved** (geodesics 3.66× Euclidean). Bug repair breaks the 0% barrier via **Cosmic Web routing (50%)**. Together, these 117 phases establish the **12 Laws of Software Physics** with a **Grand Rosetta Score of 8.4/10**.

### Key Discoveries

| Discovery | Phase | Result |
|-----------|-------|--------|
| **Compilation = Matrix Multiply** | P3 | AST→Binary: R²=0.965, 90% energy in 4 dims |
| **Generative Decompilation** | P9 | GRU decoder: 100% semantic accuracy |
| **Binary Surgery** | P11 | SVD-axis interventions: 64% semantic change |
| **Neural CPU** | P16 | Predict execution from vectors: R²=0.924 |
| **True Dimensionality = 5** | P38-40 | ~5D manifold explains 86.3% variance |
| **5D Turing Invariance** | P49 | Adding if/for changes PCA by <±0.1% |
| **Latent Antivirus** | P60 | Malware detection: 100% precision |
| **14 Species of Code** | P64 | DBSCAN: 14 clusters, 0% noise |
| **Gauge Symmetry** | P74 | Variable renaming: cos=1.000 |
| **Noether's Theorem** | P82 | 6/6 charges perfectly conserved |
| **Spectral Gap** | P95 | PC5/PC6 = 2.93, confirms 5D hypothesis |
| **Golden Ratio** | P96 | φ=1.621 in eigenvalue spectrum |
| **Holographic Decoder** | P101 | Angles-only CPU: R²=0.9983 |
| **Cosmic Web Routing** | P108 | 50% bug repair (broke 0% barrier!) |
| **Space Curvature** | P115 | Geodesics 3.66× Euclidean |
| **Time Reversal** | P116 | 100% simplification success |

## Architecture

```
Natural Language ──→ [Encoder_NL] ──→ ┐
                                       ├──→ 64-dim Rosetta Space
Python AST ────────→ [Encoder_AST] ──→ ┤     (true dim ≈ 5)
                                       ├──→ W_compile (linear!) ──→ Binary Space
Bytecode ──────────→ [Encoder_BC] ───→ ┘

12 Laws of Software Physics │ Grand Rosetta Score: 8.4/10
```

## Project Structure

```
rosetta/
├── data/                    # Dataset and trained models
├── experiments/             # Phase 1-86 scripts
├── experiments2/            # Phase 101-117 scripts (Season 2)
├── figures/                 # Generated visualizations (117 figures)
├── papers/                  # LaTeX paper (V1-V4)
├── results/                 # JSON result files
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

# Season 2: Grand Unification & Applied Physics (Phases 87-117)
# See experiments2/ directory
```

### Interactive Demo

```bash
python experiments/phase23_rosetta_studio.py
# Open http://localhost:7860
```

## Experiment Phases

### Season 1: Foundation to Laws (P1-86)

<details>
<summary>Chapter I: The Rosetta Stone (P1-4)</summary>

- **P1**: Tri-modal dataset generation (236 functions × ~12 NL variants)
- **P2**: Contrastive latent alignment (same-source sim: 0.964)
- **P3**: Translation matrices (compile R²=0.965)
- **P4**: Semantic arithmetic exploration
</details>

<details>
<summary>Chapter II: The Linear Archaeologist (P5-8)</summary>

- **P5**: Dataset scaling to 2,736 triplets
- **P6**: Linear decompiler via Ridge regression (R²=0.862)
- **P7**: Mechanistic compiler anatomy via SVD probes
- **P8**: Rosetta Compass — semantic clustering (purity: 99.2%)
</details>

<details>
<summary>Chapter III: The Latent Hacker (P9-11)</summary>

- **P9**: GRU-based generative decompiler (100% semantic accuracy)
- **P10**: Semantic arithmetic in void (analogy by generation)
- **P11**: Binary surgery via SVD-axis manipulation (64% change rate)
</details>

<details>
<summary>Chapter IV: The Cybernetic Alchemist (P12-14)</summary>

- **P12**: Null-space robustness analysis
- **P13**: Semantic auto-patching via NL vectors (57% repair)
- **P14**: End-to-end neural compiler (63% semantic accuracy)
</details>

<details>
<summary>Chapter V–VII: Semantic Execution & Beyond (P15-23)</summary>

- **P15-17**: Robust decoder, Neural CPU (R²=0.924), code morphing
- **P18-20**: Inverse execution, composition, NL CPU (R²=0.435)
- **P21-23**: Manifold synthesis, Babel Fish, Rosetta Studio
</details>

<details>
<summary>Chapter VIII–XIV: The Physics of Software (P24-48)</summary>

- **P24-29**: Symbol grounding, universality, LLM-binary, bit-level (98.4%), MI preservation
- **P30-33**: Neural assembler, LLM brain decompilation, eigenfunctions
- **P34-42**: Null space, evolution, Rosetta Paradox, entanglement, Five Elements, phase transitions, holographic principle
- **P43-48**: Semantic gravity, genome, completeness, isolation paradox, hidden highway, syntax-semantics spectrum
</details>

<details>
<summary>Chapter XV–XIX: The Ultimate Grounding (P49-68)</summary>

- **P49**: 5D invariance under Turing completeness (±0.1% change)
- **P55-56**: Inverse execution (40%), I/O search (100%)
- **P58**: Latent Linter — add vs sub: cos=-0.770
- **P60**: Latent Antivirus — 100% precision
- **P62**: Silicon Translation — Python→x86 at 76.7%
- **P64**: 14 Species of Code — 0% noise
- **P66-67**: Operator Algebra (64%), Attractor-Stabilized NVM (99.88% drift reduction)
</details>

<details>
<summary>Chapter XX–XXIII: The Deeper Universe (P69-86)</summary>

- **P74**: Gauge symmetry — variable naming is a perfect symmetry (cos=1.000)
- **P76**: Information bottleneck — 10.2× compression (193→19 bits)
- **P82**: Noether's theorem — 6/6 charges perfectly conserved
- **P83**: The 10 Laws of Software Physics — Rosetta Score: 90.8/100
- **P84**: Latent Calculator — predict f(x,y) from 5D at R²=0.97
</details>

### Season 2: Grand Unification (P87-100) 🆕

- **P87**: Molecular orbitals — bonding/antibonding energy differences (3.10)
- **P90**: Three-body problem — zero residual (~10⁻⁸), purely pairwise interactions
- **P93**: Memetic engine — Lamarckian evolution discovers 2/7 functions
- **P95**: Spectral gap — PC5/PC6 ratio 2.93 confirms 5D hypothesis
- **P96**: Rosetta constants — golden ratio φ=1.621, mass ratio α=1.000
- **P97**: Classical space — commutator [AST,BC] = 0 (not quantum)
- **P98**: Holographic boundary — 108% nearest-neighbor info on unit sphere
- **P99**: Renormalization group — 3 fixed points (CV < 12%)
- **P100**: Grand Rosetta Score — 8.4/10 across 12 Laws

### Season 2: Applied Physics (P101-117) 🆕

- **P101**: Holographic decoder — angles-only CPU: R²=0.9983 (99.9% accuracy)
- **P102**: Golden AST structure — φ emerges from collective spectra, not individual trees
- **P104**: Dark matter census — 100% of space is void
- **P105**: Cosmic web — MST: 194 leaves, 38 hubs, x+y is supernode (degree 12)
- **P106**: Gravity equation — F ∝ d⁻³·⁴⁰ (inverse-cube, not inverse-square)
- **P107**: Entropy of code — CCA=1.000 (perfect AST-BC duality)
- **P108**: 🎯 **Cosmic Web Routing — 50% bug repair, breaking the 0% barrier!**
- **P109**: Black hole spaghettification — particle capture by x+y singularity
- **P110**: Dark matter abiogenesis — void programs are computationally chaotic
- **P111**: Planck length — confusion distance = 0.765
- **P112**: Arrow of time — PC2 axis (r=0.56), monotonic complexity
- **P114**: Periodic table — 9 algebraic element types (commutative, identity, etc.)
- **P115**: Space curvature — geodesics 3.66× longer than Euclidean
- **P116**: Time reversal — 100% simplification success (fully reversible!)
- **P117**: Dual-space wormhole — BC routing outperforms AST (50% vs 33%)

## The 12 Laws of Software Physics

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
11. **The Holographic Principle**: Angular information retains 99.9% accuracy 🆕
12. **The Spectral Gap Law**: Phase transition at PC5–PC6 (ratio 2.93) separates meaning from noise 🆕

## Citation

```bibtex
@article{funasaki2026rosetta,
  title={Project Rosetta: Linear Algebraic Compilation, Neural Decompilation,
         and the 12 Laws of Software Physics in a Unified Latent Space},
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
