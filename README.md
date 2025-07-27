# Breast Cancer Detection Using Deep Learning

This repository contains the prototype implementation for a final-year BSc Computer Science project at the University of London. The project explores the use of Convolutional Neural Networks (CNNs) to classify mammogram images for breast cancer screening using the CBIS-DDSM dataset.

## Project Overview

- **Goal:** Automatically classify mammograms into benign or malignant categories using deep learning.
- **Approach:** Transfer learning with MobileNetV2, evaluated using standard classification metrics.
- **Dataset:** Curated Breast Imaging Subset of DDSM (CBIS-DDSM).
- **Prototype Output:** End-to-end pipeline from data preprocessing to model evaluation.

## Features

- Image path resolution from DICOM to JPEG format
- Data cleaning and binary label assignment
- CNN training using MobileNetV2 with transfer learning
- Evaluation metrics: Accuracy, Precision, Recall, F1 Score, AUC
- Confusion matrix and ROC curve visualization

## Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- PIL (Pillow)
- OpenCV (optional, for preprocessing)

To install dependencies:

pip install -r requirements.txt

## Usage

Open and run the main prototype notebook:
02_mobilenetv2_prototype_development.ipynb


Make sure the CBIS-DDSM dataset is correctly downloaded and that all image paths are accessible. See comments inside the notebook for guidance on adjusting paths if needed.

## Results Summary

- **Test Accuracy:** ~66%
- **AUC Score:** 0.7231
- **Precision / Recall (Malignant):** ~57%
- **Model Used:** MobileNetV2 (ImageNet weights, base frozen)

These results serve as a baseline and will be extended with improved architectures and training strategies in future stages of the project.

## Author

**Ramiro Muniz**  
University of London – BSc Computer Science  
Final Year Project 2025  
Supervised by Dr. Foaad Haddod

## License

This project is for academic and research purposes only. Not intended for clinical use or deployment in diagnostic environments.
