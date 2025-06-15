
# 🧾 PROJECT CODING REQUEST (PCR)

## 📌 Project Title:
**Breast Cancer Detection using CNNs and Transfer Learning (CBIS-DDSM Dataset)**

## 🧠 Context:
You are building a **mammogram classification pipeline** using **deep learning**. The model will use **transfer learning with MobileNetV2**, trained and evaluated on the **CBIS-DDSM dataset**, with **Grad-CAM explainability**. The final deliverable is a functioning prototype in **Python**, developed using **Jupyter Notebooks**, organized for clarity and modularity.

## 🧑‍💻 AI ROLE PROMPT:
**You are a senior AI/ML developer assigned to create a deep learning-based diagnostic tool to classify mammogram images. You must implement each feature step-by-step, verifying success before continuing. Build using best MLOps and clean code practices. Write each development step in modular Python or Jupyter Notebook cells.**

## 🧰 Tech Stack

- **Language:** Python 3.10+
- **IDE:** Jupyter Notebook
- **Libraries:**
  - TensorFlow / Keras
  - NumPy
  - Pandas
  - Scikit-learn
  - OpenCV
  - PIL (Pillow)
  - Matplotlib / Seaborn
  - Grad-CAM (via tf-keras-vis or custom wrapper)
- **Logging:** TensorBoard or CSV
- **Directory structure:**

```bash
breast-cancer-cnn/
│
├── notebooks/
│   ├── 01_setup_environment.ipynb
│   ├── 02_load_preprocess_data.ipynb
│   ├── 03_model_build_train.ipynb
│   ├── 04_evaluation_metrics.ipynb
│   ├── 05_explainability_gradcam.ipynb
│   └── 06_experiment_logs.ipynb
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── mobilenet_cbis_ddsm.h5
│
├── logs/
│   ├── metrics.csv
│   └── tensorboard/
│
├── utils/
│   ├── data_utils.py
│   ├── model_utils.py
│   └── viz_utils.py
│
├── README.md
└── requirements.txt
```

## 🎯 Development Process

> 🔄 Proceed **step-by-step**, ticket-by-ticket. **DO NOT** skip ahead without validating previous steps.

### 🎫 TICKET 1: Environment Setup
- Install all libraries.
- Verify versions and GPU access.

### 🎫 TICKET 2: Data Loader and Preprocessing
- Load, resize, and augment CBIS-DDSM images.
- Normalize and split data.

### 🎫 TICKET 3: Model Construction
- Load MobileNetV2 with ImageNet weights.
- Add classifier head and compile model.

### 🎫 TICKET 4: Model Training
- Train model with early stopping and checkpointing.

### 🎫 TICKET 5: Evaluation
- Compute Accuracy, Precision, Recall, F1, AUC.
- Display confusion matrix and classification report.

### 🎫 TICKET 6: Explainability
- Run Grad-CAM and overlay heatmaps.

### 🎫 TICKET 7: Logging
- Log experiments to CSV or TensorBoard.

### 🎫 TICKET 8: Contingency Mode
- Try binary classification or smaller subsets if needed.

## ✅ Completion Criteria
- All notebooks executable end-to-end.
- Test accuracy > 65%.
- Interpretable Grad-CAM output.
- Proper logs and documentation.
