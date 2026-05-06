# Enhancing Time-Lapse Embryo Monitoring in IVF with Spatio-Temporal Deep Learning

[![DS-MED LAB](https://img.shields.io/badge/Research-DS--MED%20LAB-orange?style=flat-square)](https://sites.google.com/view/ds-medlab/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?style=flat-square)](https://pytorch.org/)

This repository contains the official implementation of the two-stage deep learning system for automated human embryo cell division stage classification in In Vitro Fertilization (IVF).

## 🔬 Overview

Time-Lapse Imaging (TLI) technology provides abundant non-invasive data on embryo morphokinetics. Our study proposes a system specifically designed for recognizing seven human embryo cell division stages, ensuring objective assessments and eliminating data leakage through a rigorous patient-wise data splitting strategy.

### Key Performance
*   **Top-1 Accuracy:** 81.26%
*   **Top-3 Accuracy:** 98.81%
*   **Macro F1-Score:** 0.798
*   **Weighted F1-Score:** 0.811

---

## 🏗️ Model Architecture

The system comprises two integrated phases:

1.  **Phase 1: Mamba2VisionMorph**
    *   A 3D spatio-temporal backbone for high-resolution feature extraction.
    *   Processes 5-frame clips to capture local temporal dynamics.
2.  **Phase 2: EmbryoTemporalNet**
    *   A causal temporal refiner using **Mamba2** SSM (State Space Model) and sliding-window attention.
    *   Refines stage transitions and maintains sequence consistency over the entire developmental history.

---

## 📁 Repository Structure

```text
ivf-public/
├── training/                # Core ML training and inference logic
│   ├── models/              # Standardized architecture implementations
│   │   ├── ivf_3d_mamba2.py           # Mamba2VisionMorph (Phase 1)
│   │   └── ivf_phase2_mamba2MIX...    # EmbryoTemporalNet (Phase 2)
│   ├── train.py             # Phase 1 training script
│   ├── train_phase2.py      # Phase 2 sequence training script
│   ├── run_visualization_2d.py  # 2D rollout visualization
│   └── run_visualization_3d.py  # 3D rollout visualization
└── docs/
```

---

## 🚀 Getting Started

### 1. Installation
The project requires `mamba_ssm` and `causal_conv1d`. We recommend using a Conda environment:

```bash
pip install -r requirements.txt
# Additional requirements for Mamba2
pip install mamba-ssm causal-conv1d
```

### 2. Training

#### Phase 1: Mamba2VisionMorph (3D Backbone)
```bash
python3 training/train.py \
    --data_dir /path/to/dataset_3d \
    --model Mamba2VisionMorph \
    --batch-size 64 \
    --epochs 100 \
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

### 4. Website Development
The demo website is built with **Vite**, **Three.js**, and **Tailwind CSS**.
```bash
cd src-site
npm install
npm run dev   # Start local development server
npm run build # Build for production (outputs to /docs)
```

---

## 📊 Visualizations (Grad-CAM)

The system utilizes Grad-CAM to identify core morphological changes, ensuring inference transparency and filtering out optical noise during clinical assessment.

---

## ✍️ Authors

*   **Thien Bao Nguyen-Tat** - *University of Information Technology (VNU-HCM)* - [ORCID](https://orcid.org/0000-0002-4809-7126)
*   **Viet Tan Pham-Nguyen** - *University of Information Technology (VNU-HCM)* - [ORCID](https://orcid.org/0009-0009-2605-449X)
