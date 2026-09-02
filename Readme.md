# EmbryoStream: A Causal Online Spatiotemporal Framework for Real Time Embryo Stage Classification in Time Lapse Monitoring

[![DS-MED LAB](https://img.shields.io/badge/Research-DS--MED%20LAB-orange?style=flat-square)](https://sites.google.com/view/ds-medlab/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?style=flat-square)](https://pytorch.org/)

This repository contains the official implementation of **EmbryoStream**, an online causal two-stage deep learning framework for automated 7-stage human embryo morphokinetic classification in In Vitro Fertilization (IVF).

## 🔬 Overview

Time-Lapse Imaging (TLI) technology provides continuous non-invasive observations of embryo morphokinetics. Our framework models both local spatio-temporal dynamics and long-term longitudinal trajectories while eliminating patient identity leakage through a rigorous embryo-level 5-fold stratified cross-validation protocol across **697 independent embryos** (235,928 frames).

### Key Performance (5-Fold Cross-Validation, Mean ± SD)
*   **Top-1 Accuracy:** 80.18% ± 1.30%
*   **Top-3 Accuracy:** 98.30% ± 0.29%
*   **Macro Precision:** 79.19% ± 1.63%
*   **Macro Recall:** 78.68% ± 1.48%
*   **Macro F1-Score:** 0.7886 ± 0.0154
*   **Weighted F1-Score:** 0.8004 ± 0.0136
*   **Cohen's Kappa (κ):** 0.7645 ± 0.0137

---

## 🏗️ Model Architecture

The system comprises two integrated phases (~49.06M total parameters):

1.  **Phase 1: MambaVisionMorph (~36.8M params)**
    *   3D spatio-temporal hierarchical backbone processing local 5-frame clips $[t-4, t-3, t-2, t-1, t]$.
    *   Integrates Conv3D residual stages, Factorized Hybrid Mamba-Attention blocks, Spatial Attention Pooling, and Target-Aware Temporal Cross-Attention (TATA).
2.  **Phase 2: EmbryoTemporalNet (~10.4M params)**
    *   Causal sequence refiner taking full-sequence 640D feature embeddings $T \times 640$.
    *   Combines 2× Selective State Space Model (SSM) layers (for global memory $0 \to t$), 2× Causal Sliding-Window Attention blocks ($W=128$) with Log-ALiBi relative positional bias, and global skip connections.

---

## 📁 Repository Structure

```text
ivf-2026/
├── training/                # Core ML training and inference logic
│   ├── models/              # Standardized architecture implementations
│   │   ├── ivf_3d_mamba2.py           # MambaVisionMorph (Phase 1)
│   │   └── ivf_phase2_mamba2MIX...    # EmbryoTemporalNet (Phase 2)
│   ├── train.py             # Phase 1 training script
│   ├── train_phase2.py      # Phase 2 sequence training script
│   ├── run_visualization_2d.py  # 2D rollout visualization
│   └── run_visualization_3d.py  # 3D rollout visualization
└── docs/                    # Interactive demo website & Grad-CAM gallery
```

---

## 🚀 Getting Started

### 1. Installation
The project requires PyTorch, `mamba_ssm`, and `causal_conv1d`. We recommend using a Conda environment:

```bash
pip install -r requirements.txt
pip install mamba-ssm causal-conv1d
```

### 2. Training

#### Phase 1: MambaVisionMorph (3D Backbone)
```bash
python3 training/train.py \
    --data_dir /path/to/dataset_3d \
    --model MambaVisionMorph \
    --batch-size 64 \
    --epochs 70 \
    --use-3d --num-frames 5 \
    --amp --output ./output/phase1
```

#### Phase 2: EmbryoTemporalNet (Sequential Refiner)
```bash
python training/train_phase2.py \
    --phase1_ckpt ./output/phase1/model_best.pth.tar \
    --data_root /path/to/embryo/images \
    --ann_root /path/to/annotations \
    --splits_json splits.json \
    --output_dir ./output/phase2 \
    --num_frames 5 --amp
```

### 3. Visualization & Rollout
To generate developmental charts and stage transition visualizations:
```bash
python3 training/run_visualization_3d.py \
    --model ./output/phase1/model_best.pth.tar \
    --phase2_ckpt ./output/phase2/phase2_best.pth \
    --embryo_id CO119-8 \
    --output_dir ./output/viz
```

---

## 📊 Visualizations (3D Grad-CAM & Attention Maps)

The system integrates 3D Grad-CAM and Temporal Attention weights to provide transparent clinical interpretability across challenging scenarios (rapid cleavage transitions, visual noise/debris, and multi-layer spatial occlusions), as well as downstream morphological quality assessment (Gardner grading).

---

## ✍️ Authors

*   **Thien Bao Nguyen-Tat** - *University of Information Technology (VNU-HCM)* - [ORCID](https://orcid.org/0000-0002-4809-7126)
*   **Viet Tan Pham-Nguyen** - *University of Information Technology (VNU-HCM)* - [ORCID](https://orcid.org/0009-0009-2605-449X)
