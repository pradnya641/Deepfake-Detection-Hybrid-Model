# Deepfake Detection using Lightweight Hybrid Spatial–Temporal Framework

## 🚀 Overview
This project implements a lightweight deepfake detection system that combines spatial features from pretrained CNNs with statistical temporal modeling. The goal is to achieve high accuracy while maintaining low computational complexity.

## 🧠 Pipeline

1. Frame extraction
2. EfficientNet feature extraction
3. Temporal β feature computation
4. Hanning window aggregation
5. PCA + SVM classification

## 🧠 Methodology
- Spatial feature extraction using EfficientNet-B0
- Temporal modeling using frame-wise β-features
- Hanning window-based aggregation
- Dimensionality reduction using PCA
- Classification using Support Vector Machine (SVM)

## 📊 Results

| Metric | Value |
|--------|------|
| Accuracy | 94.84% |
| AUC | 0.94 |


## 🛠️ Tech Stack
- Python
- PyTorch
- OpenCV
- NumPy
- Scikit-learn

## ⚙️ How to Run
1. Open the notebook `DEEPFAKE_FINAL.ipynb`
2. Run all cells sequentially

## 📁 Dataset
Dataset is not included due to size constraints.  
Download Celeb-DF-v2 from:  
https://github.com/yuezunli/celeb-deepfakeforensics

## 📌 Highlights
- Lightweight alternative to heavy deep learning models
- Combines spatial + temporal features efficiently
- Suitable for real-world deployment

## 📎 Note
This repository contains the implementation corresponding to a research paper submitted to IEEE conference (IGNITE 2026).
