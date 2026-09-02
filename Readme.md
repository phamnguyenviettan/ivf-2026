# EmbryoStream: A Causal Online Spatiotemporal Framework for Real Time Embryo Stage Classification in Time Lapse Monitoring

[![Project Page](https://img.shields.io/badge/Project%20Page-EmbryoStream-orange?style=flat-square&logo=googlechrome&logoColor=white)](https://phamnguyenviettan.github.io/ivf-2026/)
[![DS-MED LAB](https://img.shields.io/badge/Research-DS--MED%20LAB-blue?style=flat-square)](https://sites.google.com/view/ds-medlab/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)

> **📢 Notice on Code and Data Availability:**  
> The full source code, pre-trained model weights/checkpoints, and data preprocessing pipelines will be made publicly available upon the formal acceptance and publication of our paper.  
> An interactive web demo, Grad-CAM interpretability gallery, and benchmark comparisons are available on our [**Project Page**](https://phamnguyenviettan.github.io/ivf-2026/).

---

## 🔬 Overview

Time-Lapse Imaging (TLI) technology provides continuous non-invasive observations of embryo morphokinetics. **EmbryoStream** is an online, causal two-stage deep learning framework designed for real-time automated 7-stage human embryo developmental stage classification in In Vitro Fertilization (IVF).

Our framework models both local 3D spatio-temporal morphological transitions and long-term longitudinal developmental trajectories while eliminating patient identity leakage through a strict embryo-level 5-fold stratified cross-validation protocol across **697 independent embryos** (235,928 frames).

### 🏆 Key Benchmark Results (5-Fold Cross-Validation, Mean ± SD)

| Metric | Score |
| :--- | :--- |
| **Top-1 Accuracy** | **80.18% ± 1.30%** |
| **Top-3 Accuracy** | **98.30% ± 0.29%** |
| **Macro Precision** | **79.19% ± 1.63%** |
| **Macro Recall** | **78.68% ± 1.48%** |
| **Macro F1-Score** | **0.7886 ± 0.0154** |
| **Weighted F1-Score** | **0.8004 ± 0.0136** |
| **Cohen's Kappa ($\kappa$)** | **0.7645 ± 0.0137** |

---

## 🏗️ Model Architecture

The system comprises two integrated phases (~47.2M total parameters):

1. **Phase 1: MambaVisionMorph (~36.8M params)**
   - 3D spatio-temporal hierarchical backbone processing local 5-frame clips $[t-4, t-3, t-2, t-1, t]$ (~1.25 hours).
   - Integrates 3D convolutional residual stages, Factorized Hybrid Mamba-Attention blocks, Spatial Attention Pooling, and Target-Aware Temporal Cross-Attention (TATA) to output a 640D feature representation at frame $t$.

2. **Phase 2: EmbryoTemporalNet (~10.4M params)**
   - Causal sequence refiner operating on full developmental trajectories $(T \times 640)$.
   - Employs 2× Selective State Space Model (SSM) layers for cumulative global historical memory ($0 \to t$), followed by 2× Causal Sliding-Window Attention blocks ($W=128$ frames ≈ 32 hours) with Log-ALiBi relative positional bias and residual skip connections.

---

## 📁 Repository Structure

```text
ivf-2026/
├── docs/                                  # Interactive project page & demo website
│   ├── index.html                         # Interactive web application
│   ├── assets/                            # Bundled CSS, JS, and Three.js 3D viewer
│   └── static/                            # Morphokinetic clips, Grad-CAM XAI & charts
├── train.py                               # Model training & pipeline entrypoint
└── Readme.md                              # Project documentation
```

---

## 📊 Explainable AI (XAI) & Downstream Transfer

- **3D Grad-CAM Clinical Interpretability:** Provides transparent, localized morphological evidence tracking biological cleavage events rather than optical artifacts or culture dish debris across challenging conditions (rapid cleavage, visual noise, spatial occlusion).
- **Zero-Shot Downstream Transfer:** Validated on Gardner blastocyst morphological grading (Good vs. Poor quality) without retraining, confirming clinical generalization.

---

## ✍️ Authors

* **Thien Bao Nguyen-Tat** — *University of Information Technology, Vietnam National University, Ho Chi Minh City* — [ORCID](https://orcid.org/0000-0002-4809-7126)
* **Viet Tan Pham-Nguyen** — *University of Information Technology, Vietnam National University, Ho Chi Minh City* — [ORCID](https://orcid.org/0009-0009-2605-449X)

---

## 📖 Citation

If you find this work or our interactive platform useful in your research, please cite our paper:

```bibtex
@article{nguyen2026embryostream,
  title={EmbryoStream: A Causal Online Spatiotemporal Framework for Real Time Embryo Stage Classification in Time Lapse Monitoring},
  author={Nguyen-Tat, Thien Bao and Pham-Nguyen, Viet Tan},
  journal={arXiv preprint},
  year={2026}
}
```
