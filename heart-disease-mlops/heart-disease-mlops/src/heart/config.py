"""Central configuration: paths, column groups and training constants."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw" / "heart.csv"
DATA_CLEAN = ROOT / "data" / "processed" / "heart_clean.csv"
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "heart_pipeline.joblib"))
METADATA_PATH = Path(os.getenv("METADATA_PATH", ROOT / "models" / "metadata.json"))
FIGURES_DIR = ROOT / "reports" / "figures"
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
MLFLOW_EXPERIMENT = "heart-disease-risk"

TARGET = "target"  # 1 = heart disease present, 0 = absent (after cleaning)
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Continuous measurements -> imputed with the median and standardised
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
# Engineered continuous features (created inside the pipeline, see features.py)
ENGINEERED = ["pct_max_hr", "st_depression_with_angina"]
# Coded / categorical variables -> imputed with the mode and one-hot encoded
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
FEATURES = NUMERIC + CATEGORICAL

# Data dictionary (coding of the public GitHub mirror of the Cleveland data)
DATA_DICTIONARY = {
    "age": "Age in years",
    "sex": "1 = male, 0 = female",
    "cp": "Chest pain type (0-3)",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl (1 = true)",
    "restecg": "Resting ECG result (0-2)",
    "thalach": "Maximum heart rate achieved",
    "exang": "Exercise-induced angina (1 = yes)",
    "oldpeak": "ST depression induced by exercise relative to rest",
    "slope": "Slope of the peak-exercise ST segment (0-2)",
    "ca": "Number of major vessels coloured by fluoroscopy (0-3)",
    "thal": "Thalassemia: 1 = fixed defect, 2 = normal, 3 = reversible defect",
    "target": "1 = heart disease present, 0 = absent",
}
