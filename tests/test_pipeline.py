"""
Unit and Integration Tests for Heart Disease Prediction Pipeline.
Validates dataset integrity, missing value imputation, preprocessing,
training, serialization, and inference.
"""

import os
import sys
import tempfile
import pytest
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression

# Set up module path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from data_loader import (
    load_raw_dataset,
    prepare_dataset,
    COLUMN_NAMES,
    PROCESSED_DATA_PATH,
)
from preprocessing import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    get_preprocessor,
    create_full_pipeline,
)
from predict import HeartDiseasePredictor, EXPECTED_COLUMNS


def test_1_dataset_structure():
    """
    Test 1 — Dataset structure:
    Verifies that the dataset has 303 rows, expected column names, and binary target.
    """
    df = prepare_dataset(PROCESSED_DATA_PATH)
    assert df.shape[0] == 303, f"Expected 303 rows, got {df.shape[0]}"
    assert df.shape[1] == 14, f"Expected 14 columns, got {df.shape[1]}"
    for col in FEATURE_COLUMNS:
        assert col in df.columns, f"Missing feature column: {col}"
    assert "target" in df.columns, "Missing 'target' column"
    assert set(df["target"].unique()).issubset({0, 1}), "Target must be strictly binary (0 or 1)"


def test_2_missing_values_handling():
    """
    Test 2 — Missing values:
    Verifies that '?' values are correctly parsed as NaNs and that the
    pipeline handles rows containing NaNs without throwing an error.
    """
    raw_df = load_raw_dataset()
    # The authentic Cleveland dataset has 4 NaNs in 'ca' and 2 in 'thal'
    assert raw_df["ca"].isna().sum() == 4, "Expected 4 missing values in 'ca'"
    assert raw_df["thal"].isna().sum() == 2, "Expected 2 missing values in 'thal'"

    # Extract features from processed dataset (which retain NaNs before imputer)
    df = prepare_dataset(PROCESSED_DATA_PATH)
    X_with_nan = df[FEATURE_COLUMNS]

    # Preprocessor should fit and transform without error
    preprocessor = get_preprocessor()
    X_transformed = preprocessor.fit_transform(X_with_nan)
    assert not np.isnan(X_transformed).any(), "Transformed output still contains NaNs"


def test_3_preprocessor_output():
    """
    Test 3 — Preprocessor:
    Verifies that the ColumnTransformer transforms data into a 2D float array
    with no NaNs and consistent feature length.
    """
    df = prepare_dataset(PROCESSED_DATA_PATH)
    X = df[FEATURE_COLUMNS]
    preprocessor = get_preprocessor()
    X_trans = preprocessor.fit_transform(X)

    assert isinstance(X_trans, np.ndarray), "Preprocessed output must be a NumPy ndarray"
    assert X_trans.shape[0] == 303, "Number of rows must remain 303"
    assert X_trans.shape[1] > len(FEATURE_COLUMNS), "OHE should expand categorical dimensions"
    assert not np.isnan(X_trans).any(), "Preprocessed data contains NaN values"


def test_4_training():
    """
    Test 4 — Training:
    Verifies that a full pipeline can be initialized, fitted on training data,
    and output predictions.
    """
    df = prepare_dataset(PROCESSED_DATA_PATH)
    X = df[FEATURE_COLUMNS]
    y = df["target"]

    pipeline = create_full_pipeline(LogisticRegression(max_iter=1000, random_state=42))
    pipeline.fit(X, y)

    preds = pipeline.predict(X)
    assert len(preds) == len(y), "Predictions length must match target length"
    assert set(preds).issubset({0, 1}), "Predictions must only be 0 or 1"


def test_5_serialization():
    """
    Test 5 — Serialization:
    Verifies that the trained pipeline can be safely saved and reloaded using joblib.
    """
    df = prepare_dataset(PROCESSED_DATA_PATH)
    X = df[FEATURE_COLUMNS]
    y = df["target"]

    pipeline = create_full_pipeline(LogisticRegression(max_iter=1000, random_state=42))
    pipeline.fit(X, y)

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
        tmp_path = tmp_file.name

    try:
        joblib.dump(pipeline, tmp_path)
        assert os.path.exists(tmp_path), "Serialized model file does not exist"

        loaded_pipeline = joblib.load(tmp_path)
        original_preds = pipeline.predict(X.iloc[:5])
        loaded_preds = loaded_pipeline.predict(X.iloc[:5])
        np.testing.assert_array_equal(original_preds, loaded_preds)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_6_prediction_interface():
    """
    Test 6 — Prediction:
    Verifies that a valid patient profile input dictionary produces a valid prediction
    and a valid predicted probability between 0.0 and 1.0.
    """
    sample_input = {
        "age": 55.0,
        "sex": 1.0,
        "cp": 2.0,
        "trestbps": 130.0,
        "chol": 240.0,
        "fbs": 0.0,
        "restecg": 0.0,
        "thalach": 150.0,
        "exang": 0.0,
        "oldpeak": 1.0,
        "slope": 1.0,
        "ca": 0.0,
        "thal": 3.0,
    }

    predictor = HeartDiseasePredictor()
    result = predictor.predict(sample_input)

    assert "predicted_class" in result
    assert result["predicted_class"] in [0, 1]
    assert "prediction_label" in result
    assert result["prediction_label"] in ["Heart Disease Absent", "Heart Disease Present"]
    assert "predicted_probability" in result
    assert 0.0 <= result["predicted_probability"] <= 1.0
    assert 0.0 <= result["predicted_probability_percent"] <= 100.0
