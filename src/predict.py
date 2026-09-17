"""
Inference Module for Heart Disease Prediction.
Loads the serialized Scikit-Learn pipeline and generates predictions and
probabilities for new patient clinical profiles.
"""

import os
import joblib
import pandas as pd
from typing import Dict, Any, Union

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "heart_disease_model.pkl")

# Expected feature columns in canonical order
EXPECTED_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]


class HeartDiseasePredictor:
    """
    Wrapper for loading the trained pipeline and performing safe inference.
    """

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at: {model_path}. "
                "Please run 'python src/train_model.py' to generate and serialize the model."
            )
        self.model_path = model_path
        self.pipeline = joblib.load(model_path)

    def predict(self, patient_data: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
        """
        Executes prediction on input patient profile.

        Parameters:
            patient_data (dict or pd.DataFrame): Feature mapping matching EXPECTED_COLUMNS.

        Returns:
            dict: {
                'predicted_class': int (0 or 1),
                'prediction_label': str ('Heart Disease Absent' or 'Heart Disease Present'),
                'predicted_probability': float (0.0 to 1.0, probability of disease presence),
                'predicted_probability_percent': float (0.0 to 100.0)
            }
        """
        if isinstance(patient_data, dict):
            # Ensure all required features are present
            missing_cols = [c for c in EXPECTED_COLUMNS if c not in patient_data]
            if missing_cols:
                raise ValueError(f"Missing required clinical features: {missing_cols}")
            df = pd.DataFrame([patient_data])[EXPECTED_COLUMNS]
        elif isinstance(patient_data, pd.DataFrame):
            df = patient_data[EXPECTED_COLUMNS]
        else:
            raise TypeError("patient_data must be a dictionary or pandas DataFrame.")

        # Ensure numeric dtypes
        df = df.astype(float)

        # Inference
        pred_class = int(self.pipeline.predict(df)[0])
        pred_proba_pos = float(self.pipeline.predict_proba(df)[0, 1])

        label = "Heart Disease Present" if pred_class == 1 else "Heart Disease Absent"

        return {
            "predicted_class": pred_class,
            "prediction_label": label,
            "predicted_probability": pred_proba_pos,
            "predicted_probability_percent": round(pred_proba_pos * 100, 2),
        }


def predict_sample():
    """
    Demonstration function using sample patient data.
    """
    predictor = HeartDiseasePredictor()

    # Sample 1: Low-risk patient profile (younger, asymptomatic chest pain absent, normal vessels/thal)
    sample_low_risk = {
        "age": 41.0,
        "sex": 0.0,      # Female
        "cp": 2.0,       # Atypical angina
        "trestbps": 120.0,
        "chol": 190.0,
        "fbs": 0.0,
        "restecg": 0.0,  # Normal
        "thalach": 172.0,
        "exang": 0.0,    # No
        "oldpeak": 0.0,
        "slope": 1.0,    # Upsloping
        "ca": 0.0,
        "thal": 3.0,     # Normal
    }

    # Sample 2: High-risk patient profile (older male, asymptomatic chest pain, high ST depression, fluoroscopy vessels)
    sample_high_risk = {
        "age": 67.0,
        "sex": 1.0,      # Male
        "cp": 4.0,       # Asymptomatic
        "trestbps": 160.0,
        "chol": 286.0,
        "fbs": 0.0,
        "restecg": 2.0,  # LVH
        "thalach": 108.0,
        "exang": 1.0,    # Yes
        "oldpeak": 1.5,
        "slope": 2.0,    # Flat
        "ca": 3.0,       # 3 vessels colored
        "thal": 7.0,     # Reversible defect
    }

    print("\n--- SAMPLE INFERENCE DEMONSTRATION ---")
    res1 = predictor.predict(sample_low_risk)
    print("\nPatient 1 (Sample Profile A):")
    print(f"  Prediction: {res1['prediction_label']}")
    print(f"  Predicted Probability: {res1['predicted_probability_percent']}%")

    res2 = predictor.predict(sample_high_risk)
    print("\nPatient 2 (Sample Profile B):")
    print(f"  Prediction: {res2['prediction_label']}")
    print(f"  Predicted Probability: {res2['predicted_probability_percent']}%")


if __name__ == "__main__":
    predict_sample()
