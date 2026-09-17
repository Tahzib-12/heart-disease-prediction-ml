"""
Data Ingestion and Loading Module for Heart Disease Prediction.
Acquires the authentic UCI Cleveland dataset, handles missing values ('?'),
transforms the multi-class diagnosis target into a binary classification target,
and manages raw and processed data paths.
"""

import os
import urllib.request
import pandas as pd
import numpy as np

UCI_CLEVELAND_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
)

COLUMN_NAMES = [
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
    "num",
]

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "processed.cleveland.data")
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "heart_disease.csv")


def fetch_raw_data(raw_path: str = RAW_DATA_PATH, force_download: bool = False) -> str:
    """
    Downloads authentic UCI Cleveland dataset if not already present.

    Parameters:
        raw_path (str): Destination path for raw file.
        force_download (bool): If True, re-downloads even if file exists.

    Returns:
        str: Absolute path to the raw data file.
    """
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    if not os.path.exists(raw_path) or force_download:
        print(f"Downloading authentic UCI Cleveland dataset from:\n  {UCI_CLEVELAND_URL}")
        req = urllib.request.Request(UCI_CLEVELAND_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response, open(raw_path, "wb") as out_file:
            out_file.write(response.read())
        print(f"Raw data saved to: {raw_path}")
    else:
        print(f"Raw data file already exists at: {raw_path}")
    return raw_path


def load_raw_dataset(raw_path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Reads the raw UCI Cleveland data, parsing '?' characters as NaN.

    Parameters:
        raw_path (str): Path to raw data file.

    Returns:
        pd.DataFrame: Raw dataframe with 14 columns and NaN for missing entries.
    """
    if not os.path.exists(raw_path):
        fetch_raw_data(raw_path)

    df = pd.read_csv(
        raw_path,
        header=None,
        names=COLUMN_NAMES,
        na_values="?",
        sep=",",
        skipinitialspace=True,
    )
    return df


def prepare_dataset(
    raw_path: str = RAW_DATA_PATH,
    save_path: str = PROCESSED_DATA_PATH,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Loads raw data, performs binary target transformation, saves clean CSV,
    and returns the prepared DataFrame.

    Target conversion:
      num == 0 -> 0 (No Heart Disease)
      num in [1, 2, 3, 4] -> 1 (Heart Disease Present)

    Parameters:
        raw_path (str): Raw data file path.
        save_path (str): Processed data CSV path.
        force_refresh (bool): If True, forces re-creation of processed CSV.

    Returns:
        pd.DataFrame: Cleaned dataframe ready for exploratory analysis and ML.
    """
    if os.path.exists(save_path) and not force_refresh:
        print(f"Loading existing processed dataset from: {save_path}")
        return pd.read_csv(save_path)

    print("Preparing dataset from raw file...")
    df = load_raw_dataset(raw_path)

    # Explicit binary target transformation
    # 0 -> Absence (0), 1, 2, 3, 4 -> Presence (1)
    df["target"] = (df["num"] > 0).astype(int)
    # Drop original multi-class column 'num'
    df = df.drop(columns=["num"])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Cleaned dataset saved successfully to: {save_path}")
    print(f"Dataset Shape: {df.shape} (Rows: {df.shape[0]}, Columns: {df.shape[1]})")
    print(f"Target Distribution:\n{df['target'].value_counts().to_dict()}")

    return df


if __name__ == "__main__":
    fetch_raw_data()
    df = prepare_dataset(force_refresh=True)
    print("\nDataset Info Summary:")
    print(df.info())
    print("\nMissing values per column:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
