"""
Feature Typing and Preprocessing Module for Heart Disease Prediction.
Builds zero-leakage Scikit-Learn pipelines using ColumnTransformer.
"""

from typing import List, Tuple
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Exact feature definitions according to UCI Cleveland clinical specifications
NUMERICAL_FEATURES: List[str] = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
]

CATEGORICAL_FEATURES: List[str] = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]

FEATURE_COLUMNS: List[str] = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


def get_numerical_pipeline() -> Pipeline:
    """
    Creates preprocessing pipeline for continuous/numerical features:
      1. Median imputation for robust central tendency against outliers
      2. Standard scaling (mean=0, variance=1)
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def get_categorical_pipeline() -> Pipeline:
    """
    Creates preprocessing pipeline for categorical/discrete features:
      1. Most-frequent (mode) imputation for discrete categories
      2. One-hot encoding with handle_unknown='ignore' to prevent inference crashes
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )


def get_preprocessor() -> ColumnTransformer:
    """
    Assembles numerical and categorical pipelines into a unified ColumnTransformer.
    Ensures zero data leakage when embedded in an sklearn Pipeline.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", get_numerical_pipeline(), NUMERICAL_FEATURES),
            ("cat", get_categorical_pipeline(), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def create_full_pipeline(classifier) -> Pipeline:
    """
    Binds the ColumnTransformer preprocessor and the specified classifier
    into an end-to-end Scikit-Learn Pipeline.
    """
    return Pipeline(
        steps=[
            ("preprocessor", get_preprocessor()),
            ("classifier", classifier),
        ]
    )


if __name__ == "__main__":
    import pandas as pd
    from data_loader import prepare_dataset

    df = prepare_dataset()
    X = df[FEATURE_COLUMNS]
    preprocessor = get_preprocessor()
    X_transformed = preprocessor.fit_transform(X)
    print("Preprocessing verification:")
    print(f"  Input feature matrix shape: {X.shape}")
    print(f"  Transformed feature matrix shape: {X_transformed.shape}")
    print(f"  Contains any NaN after transformation: {pd.isna(X_transformed).any()}")
