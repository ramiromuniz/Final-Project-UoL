# Breast Cancer Detection using Deep Learning

A University of London final project comparing deep learning models for automated breast cancer detection from mammogram images.

## Project Overview

This project compares three deep learning architectures for classifying mammograms as benign or malignant:

- **MobileNetV2**: Lightweight model for efficient processing
- **DenseNet121**: Dense connectivity for better feature learning
- **Grad-CAM**: Visual explanations showing which image regions influenced predictions

## Dataset

- **CBIS-DDSM** (Curated Breast Imaging Subset of Digital Database for Screening Mammography)
- Binary classification: Benign vs Malignant tumors
- Mammogram images in JPEG format

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/ramiromuniz/Final-Project-UoL.git
cd Final-Project-UoL
```

### 2. Set Up Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Download Dataset

- Download the CBIS-DDSM dataset from Kaggle
- Extract to `data/kaggle/jpeg/` folder
- The CSV files should go in `data/csv/` folder

### 4. Run the Notebooks

```bash
jupyter notebook
```

Open and run in this order:

1. `02_mobilenetv2_experiments.ipynb` - Train MobileNetV2 model
2. `03_densenet121_experiments.ipynb` - Train DenseNet121 model
3. `04_resnet50.ipynb` - Train ResNet50V2 model
4. `05_grad-cam.ipynb` - Generate visual explanations

## Project Structure

```
├── notebooks/           # Jupyter notebooks for experiments
├── data/               # Dataset (not included - download separately)
├── figures/            # Generated plots and visualizations
├── artifacts/          # Model metrics and predictions
└── requirements.txt    # Python dependencies
```

## Key Features

- **Data preprocessing** pipeline for mammogram images
- **Transfer learning** from ImageNet pretrained weights
- **Comprehensive evaluation** with multiple metrics
- **Grad-CAM visualizations** for model interpretability
- **Model comparison** analysis

## Dependencies

### Core Libraries

- **Python 3.8+**
- **TensorFlow 2.12.0+** - Deep learning framework
- **Keras** - High-level neural networks API (included with TensorFlow)
- **NumPy 1.21.0+** - Numerical computing
- **Pandas 1.3.0+** - Data manipulation and analysis

### Machine Learning & Evaluation

- **Scikit-learn 1.0.0+** - Machine learning metrics and utilities

### Visualization

- **Matplotlib 3.5.0+** - Plotting and visualization
- **Seaborn 0.11.0+** - Statistical data visualization

### Development Environment

- **Jupyter 1.0.0+** - Interactive development environment
- **Notebook 6.4.0+** - Jupyter notebook interface

### Additional Libraries

- **Pillow 8.3.0+** - Image processing
- **Pathlib** - Object-oriented filesystem paths

## System Requirements

- **RAM**: Minimum 8GB (16GB recommended for training)
- **Storage**: At least 10GB free space for dataset and models
- **GPU**: CUDA-compatible GPU recommended but not required
- **Training Time**: 1-3 hours per model depending on hardware

## Installation

Install all dependencies using:

```bash
pip install -r requirements.txt
```

## Academic Context

This project was developed as a final project for the University of London Computer Science program. The work focuses on applying state-of-the-art deep learning techniques to medical image analysis, specifically mammogram classification for breast cancer detection.

## License

MIT License

Copyright (c) 2025 Final Project - University of London
