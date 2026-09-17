"""
Inference Module for Heart Disease Prediction.
Loads the trained Scikit-Learn pipeline and generates predictions.
"""

import os
from typing import Any, Dict, Union

import joblib
import pandas as pd


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "app",
    "models",
    "heart_disease_model.pkl"
)


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

    def __init__(self, model_path: str = MODEL_PATH):

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at: {model_path}. "
                "Please run the training script first."
            )

        self.model_path = model_path
        self.pipeline = joblib.load(model_path)

    def predict(
        self,
        patient_data: Union[Dict[str, Any], pd.DataFrame]
    ) -> Dict[str, Any]:

        if isinstance(patient_data, dict):

            missing_columns = [
                column
                for column in EXPECTED_COLUMNS
                if column not in patient_data
            ]

            if missing_columns:
                raise ValueError(
                    f"Missing required features: {missing_columns}"
                )

            df = pd.DataFrame(
                [patient_data]
            )

            df = df[EXPECTED_COLUMNS]

        elif isinstance(patient_data, pd.DataFrame):

            missing_columns = [
                column
                for column in EXPECTED_COLUMNS
                if column not in patient_data.columns
            ]

            if missing_columns:
                raise ValueError(
                    f"Missing required features: {missing_columns}"
                )

            df = patient_data[
                EXPECTED_COLUMNS
            ].copy()

        else:
            raise TypeError(
                "patient_data must be a dictionary "
                "or pandas DataFrame."
            )

        df = df.astype(float)

        predicted_class = int(
            self.pipeline.predict(df)[0]
        )

        predicted_probability = float(
            self.pipeline.predict_proba(df)[0, 1]
        )

        if predicted_class == 1:
            prediction_label = "Heart Disease Present"
        else:
            prediction_label = "Heart Disease Absent"

        return {
            "predicted_class": predicted_class,
            "prediction_label": prediction_label,
            "predicted_probability": predicted_probability,
            "predicted_probability_percent": round(
                predicted_probability * 100,
                2
            ),
        }


def predict_sample():

    predictor = HeartDiseasePredictor()

    sample_profile_a = {
        "age": 41,
        "sex": 0,
        "cp": 2,
        "trestbps": 120,
        "chol": 190,
        "fbs": 0,
        "restecg": 0,
        "thalach": 172,
        "exang": 0,
        "oldpeak": 0.0,
        "slope": 1,
        "ca": 0,
        "thal": 3,
    }

    sample_profile_b = {
        "age": 67,
        "sex": 1,
        "cp": 4,
        "trestbps": 160,
        "chol": 286,
        "fbs": 0,
        "restecg": 2,
        "thalach": 108,
        "exang": 1,
        "oldpeak": 1.5,
        "slope": 2,
        "ca": 3,
        "thal": 7,
    }

    result_a = predictor.predict(
        sample_profile_a
    )

    result_b = predictor.predict(
        sample_profile_b
    )

    print("Sample Profile A:")
    print(result_a)

    print("\nSample Profile B:")
    print(result_b)


if __name__ == "__main__":
    predict_sample()