# AI Diagnostic Companion - Models Directory

This directory contains all machine learning models and related files organized by medical analysis type.

## Directory Structure

```
models/
├── mri/                    # MRI Brain Scan Analysis
│   ├── mri_classifier.py
│   ├── mri_classifier_simple.py
│   ├── mri_improved_classifier.py
│   ├── mri_random_forest.py
│   ├── mri_improved_model.pkl
│   ├── mri_random_forest.pkl
│   ├── MRI_FINAL_REPORT.md
│   └── MRI_MODEL_REPORT.md
├── xray/                   # Chest X-Ray Analysis
│   ├── xray_pneumonia_classifier.py
│   └── xray_pneumonia_model.pkl
└── diabetes/               # Diabetes Prediction
    ├── diabetes_classifier.py
    └── diabetes_model.pkl
```

## Model Descriptions

### MRI Analysis (`models/mri/`)
- **Purpose**: Brain tumor classification (Glioma, Meningioma, Pituitary, No Tumor)
- **Model**: Ensemble classifier combining RandomForest, GradientBoosting, ExtraTrees, SVM, KNN, LogisticRegression
- **Accuracy**: High accuracy across all four categories
- **Input**: MRI brain scan images (JPEG, PNG)
- **Output**: Tumor type classification with confidence score

### X-Ray Analysis (`models/xray/`)
- **Purpose**: Chest X-ray pneumonia detection (Normal vs Pneumonic)
- **Model**: GradientBoosting classifier
- **Accuracy**: Very high accuracy for binary classification
- **Input**: Chest X-ray images (JPEG, PNG)
- **Output**: Normal or Pneumonic classification with confidence score

### Diabetes Analysis (`models/diabetes/`)
- **Purpose**: Diabetes prediction based on health parameters
- **Model**: RandomForest classifier
- **Accuracy**: Good accuracy for binary classification
- **Input**: Health parameters (Glucose, Blood Pressure, BMI, etc.)
- **Output**: Diabetic or Non-Diabetic classification with confidence score

## Usage

All models are integrated into the Next.js API routes:
- `/api/mri-analysis` - Uses `models/mri/mri_improved_classifier.py`
- `/api/xray-analysis` - Uses `models/xray/xray_pneumonia_classifier.py`
- `/api/diabetes-analysis` - Uses `models/diabetes/diabetes_classifier.py`

## File Types

- **`.py` files**: Model implementation and training scripts
- **`.pkl` files**: Trained model weights (pickle format)
- **`.md` files**: Model documentation and reports

## Dependencies

All models require:
- Python 3.8+
- scikit-learn
- OpenCV (cv2)
- NumPy
- Pandas (for diabetes model)

## Training Data

Models are trained on datasets located in the `datasets/` directory:
- `datasets/mri scan/` - MRI brain scan images
- `datasets/chest_xray/` - Chest X-ray images
- `datasets/diabetes.csv` - Diabetes health parameters

## Notes

- All models operate locally without external API dependencies
- Models are optimized for accuracy and performance
- Regular retraining recommended as new data becomes available
- Model files are version controlled for reproducibility
