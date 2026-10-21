# 1D-CNN on PTB-XL: Early Fusion vs. Lead-Wise Fusion

| Notebook | Link |
| :--- | :--- |
| **01. Data Loading & Inspection** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HafizaRimah/1D-CNN-on-PTB-XL/blob/main/notebooks/01_data_loading_inspection.ipynb) |
| **02. Training & Fair Comparison** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HafizaRimah/1D-CNN-on-PTB-XL/blob/main/notebooks/02_training_and_comparison.ipynb) |

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/PhysioNet-PTB--XL-blue.svg)](https://physionet.org/content/ptb-xl/1.0.3/)

This repository implements and benchmarks **1D Convolutional Neural Networks (CNN)** on the [PTB-XL ECG dataset](https://physionet.org/content/ptb-xl/1.0.3/), comparing a standard **Early-Fusion 1D CNN baseline** against a **Lead-Wise Attention Fusion variant**.

---

## 🎯 Task & Dataset Setup

* **Dataset**: PTB-XL (100 Hz version, `records100`), 21,799 12-lead ECG records of 10 seconds each ($12 \times 1000$).
* **Task**: Multi-label classification across the **5 Diagnostic Superclasses**:
  * `NORM`: Normal ECG
  * `MI`: Myocardial Infarction
  * `STTC`: ST/T Change
  * `CD`: Conduction Disturbance
  * `HYP`: Hypertrophy
* **Split Strategy**: Patient-aware `strat_fold` column to prevent data leakage:
  * **Train**: Folds 1–8 (~17,400 records)
  * **Validation**: Fold 9 (~2,200 records)
  * **Test**: Fold 10 (~2,200 records)
* **Standardization**: Per-lead zero-mean and unit-variance computed strictly from the training set.

---

## 🧠 Architectural Concepts

### 1. Baseline Model: Early Fusion 1D CNN
* All 12 leads are passed into the input layer as channels: `in_channels = 12`.
* 1D convolutional filters mix temporal and spatial lead representations simultaneously from the very first layer.

### 2. Lead-Fusion Variant: Lead-Wise Attention Fusion
* Clinically, each of the 12 leads records cardiac activity from a distinct 3D projection angle (frontal limb leads $I, II, III, aVR, aVL, aVF$ vs. horizontal precordial leads $V_1 - V_6$).
* In lead fusion, **per-lead feature representations** are extracted independently first.
* A **learnable attention module** computes an importance weight $\alpha_i$ for each lead $i \in \{1, \dots, 12\}$:
  $$\text{Representation} = \sum_{i=1}^{12} \alpha_i \mathbf{v}_i$$
* **Interpretability**: Enables plotting the model's average attention per lead across diagnostic conditions (e.g., verifying if anterior MI emphasizes leads $V_1-V_4$).

---

## 📊 Evaluation & Fair Comparison Protocol

* **Metrics**: Macro AUROC on the test fold + per-class AUROCs.
* **Controlled Conditions**: Identical seeds, epochs, optimizer (AdamW), learning rate, and batch size.
* **Multi-Seed Evaluation**: Each model is trained across 3 random seeds; reported as $\text{mean} \pm \text{std}$.
* **Artifacts**: Results comparison table and per-lead attention weight plots.

---

## 📁 Repository Structure

```
1D-CNN-on-PTB-XL/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── 01_data_loading_inspection.ipynb   # Download, preprocess, and visualize
│   └── 02_training_and_comparison.ipynb   # Train Baseline vs Lead-Fusion across seeds
└── src/
    ├── __init__.py
    ├── dataset.py                          # Metadata mapping, signal loading, standardization
    ├── models.py                           # Baseline 1D CNN and Lead-Wise Attention CNN
    ├── train.py                            # Training loop, evaluation, and AUROC computation
    └── utils.py                            # Multi-seed runner, results table, visualization
```

---

## 🚀 Quick Start

### 1. Local Installation
```bash
git clone https://github.com/HafizaRimah/1D-CNN-on-PTB-XL.git
cd 1D-CNN-on-PTB-XL
pip install -r requirements.txt
```

### 2. Run in Google Colab
Click the badge at the top of the README to open any notebook directly in Google Colab with a free GPU!
