"""
Model Training, Cross-Validation, Evaluation, and Serialization Module.
Strictly adheres to zero data leakage:
  - 80/20 Stratified train-test split
  - 5-fold Stratified Cross-Validation performed solely on the training set
  - Untouched test set evaluated only after final model selection
  - Artifacts generated: model_comparison.csv, confusion_matrix.png, roc_curves.png, heart_disease_model.pkl
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)

# Ensure src modules are resolvable
sys.path.insert(0, os.path.dirname(__file__))
from data_loader import prepare_dataset, PROCESSED_DATA_PATH
from preprocessing import FEATURE_COLUMNS, create_full_pipeline

# Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
SAVED_MODEL_PATH = os.path.join(MODELS_DIR, "heart_disease_model.pkl")
MODEL_COMPARISON_PATH = os.path.join(RESULTS_DIR, "model_comparison.csv")
CONFUSION_MATRIX_PATH = os.path.join(RESULTS_DIR, "confusion_matrix.png")
ROC_CURVES_PATH = os.path.join(RESULTS_DIR, "roc_curves.png")

RANDOM_STATE = 42


def get_candidate_models():
    """
    Instantiates the 4 required standard machine learning classifiers.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=4, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=RANDOM_STATE
        ),
    }


def perform_stratified_cv(models_dict, X_train, y_train):
    """
    Executes 5-fold Stratified Cross-Validation on the training set.
    Calculates Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
    Returns:
      cv_results_df (pd.DataFrame): Sorted by mean ROC-AUC
      fitted_cv_pipelines (dict): Pipelines for candidate models
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    records = []
    pipelines = {}

    print("\n" + "=" * 70)
    print("--- 5-FOLD STRATIFIED CROSS-VALIDATION ON TRAINING SET (N=242) ---")
    print("=" * 70)

    for name, clf in models_dict.items():
        pipeline = create_full_pipeline(clf)
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
        )

        acc_mean, acc_std = scores["test_accuracy"].mean(), scores["test_accuracy"].std()
        prec_mean, prec_std = scores["test_precision"].mean(), scores["test_precision"].std()
        rec_mean, rec_std = scores["test_recall"].mean(), scores["test_recall"].std()
        f1_mean, f1_std = scores["test_f1"].mean(), scores["test_f1"].std()
        auc_mean, auc_std = scores["test_roc_auc"].mean(), scores["test_roc_auc"].std()

        records.append({
            "Model": name,
            "CV_Accuracy_Mean": acc_mean,
            "CV_Accuracy_Std": acc_std,
            "CV_Precision_Mean": prec_mean,
            "CV_Precision_Std": prec_std,
            "CV_Recall_Mean": rec_mean,
            "CV_Recall_Std": rec_std,
            "CV_F1_Mean": f1_mean,
            "CV_F1_Std": f1_std,
            "CV_ROC_AUC_Mean": auc_mean,
            "CV_ROC_AUC_Std": auc_std,
        })
        pipelines[name] = pipeline

        print(
            f"[{name}]\n"
            f"  ROC-AUC:   {auc_mean:.4f} (+/- {auc_std:.4f})\n"
            f"  Recall:    {rec_mean:.4f} (+/- {rec_std:.4f})\n"
            f"  F1-Score:  {f1_mean:.4f} (+/- {f1_std:.4f})\n"
            f"  Accuracy:  {acc_mean:.4f} (+/- {acc_std:.4f})\n"
            f"  Precision: {prec_mean:.4f} (+/- {prec_std:.4f})\n"
        )

    df_comparison = pd.DataFrame(records).sort_values(
        by="CV_ROC_AUC_Mean", ascending=False
    ).reset_index(drop=True)

    return df_comparison, pipelines


def evaluate_on_test_set(best_pipeline, X_train, y_train, X_test, y_test):
    """
    Fits the winning pipeline on full X_train and evaluates on untouched X_test.
    Computes Accuracy, Precision, Recall, Specificity, F1, ROC-AUC, Confusion Matrix.
    """
    best_pipeline.fit(X_train, y_train)
    y_pred = best_pipeline.predict(X_test)
    y_proba = best_pipeline.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)  # TP / (TP + FN)
    specificity = tn / (tn + fp)           # TN / (TN + FP)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall (Sensitivity)": recall,
        "Specificity": specificity,
        "F1-Score": f1,
        "ROC-AUC": roc_auc,
        "True Negatives (TN)": tn,
        "False Positives (FP)": fp,
        "False Negatives (FN)": fn,
        "True Positives (TP)": tp,
    }

    return metrics, y_pred, y_proba, tn, fp, fn, tp


def plot_confusion_matrix(tn, fp, fn, tp, model_name, save_path=CONFUSION_MATRIX_PATH):
    """
    Generates and saves a clearly labeled 2x2 confusion matrix heatmap:
    True Negative, False Positive, False Negative, True Positive.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    matrix = np.array([[tn, fp], [fn, tp]])
    labels = np.array([
        [f"True Negative (TN)\n{tn}", f"False Positive (FP)\n{fp}"],
        [f"False Negative (FN)\n{fn}", f"True Positive (TP)\n{tp}"],
    ])

    plt.figure(figsize=(6.5, 5.5), dpi=300)
    sns.heatmap(
        matrix,
        annot=labels,
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=["Predicted Negative (0)", "Predicted Positive (1)"],
        yticklabels=["Actual Negative (0)", "Actual Positive (1)"],
        annot_kws={"fontsize": 11, "fontweight": "bold"},
    )
    plt.title(f"Confusion Matrix — {model_name}\n(Untouched Test Set, N={tn+fp+fn+tp})", fontsize=12, pad=12)
    plt.xlabel("Predicted Diagnosis", fontsize=11, labelpad=8)
    plt.ylabel("Actual Diagnosis", fontsize=11, labelpad=8)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix plot saved to: {save_path}")


def plot_roc_curves(candidate_pipelines, X_train, y_train, X_test, y_test, save_path=ROC_CURVES_PATH):
    """
    Plots and compares ROC curves for all candidate models on the test set.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(7.5, 6), dpi=300)

    for name, pipeline in candidate_pipelines.items():
        pipeline.fit(X_train, y_train)
        if hasattr(pipeline.named_steps["classifier"], "predict_proba"):
            y_scores = pipeline.predict_proba(X_test)[:, 1]
        else:
            y_scores = pipeline.decision_function(X_test)

        fpr, tpr, _ = roc_curve(y_test, y_scores)
        auc_val = roc_auc_score(y_test, y_scores)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc_val:.4f})")

    plt.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.5000)")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11)
    plt.title("ROC Curves Comparison (Test Set Evaluation)", fontsize=13, pad=12)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"ROC curves comparison saved to: {save_path}")


def run_training_pipeline():
    """
    Full reproducible pipeline orchestration.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Load Data
    df = prepare_dataset(PROCESSED_DATA_PATH)
    X = df[FEATURE_COLUMNS]
    y = df["target"]

    # 2. 80/20 Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    print(f"\nDataset Splitting Summary:")
    print(f"  Total samples: {len(df)}")
    print(f"  Training set (80%): {len(X_train)} samples (Target counts: {dict(y_train.value_counts())})")
    print(f"  Test set (20%):     {len(X_test)} samples (Target counts: {dict(y_test.value_counts())})")
    print("  Note: Test set is strictly untouched during model selection.")

    # 3. 5-Fold Stratified Cross-Validation on Training Data
    models = get_candidate_models()
    cv_comparison, candidate_pipelines = perform_stratified_cv(models, X_train, y_train)

    # Save Cross-Validation comparison table
    cv_comparison.to_csv(MODEL_COMPARISON_PATH, index=False)
    print(f"\nModel comparison results saved to: {MODEL_COMPARISON_PATH}")

    # 4. Final Model Selection
    # Select best model based on Mean CV ROC-AUC
    best_model_name = cv_comparison.iloc[0]["Model"]
    best_auc = cv_comparison.iloc[0]["CV_ROC_AUC_Mean"]
    print("\n" + "=" * 70)
    print(f"FINAL MODEL SELECTED VIA CV: {best_model_name} (Mean CV ROC-AUC: {best_auc:.4f})")
    print("=" * 70)

    best_pipeline = candidate_pipelines[best_model_name]

    # 5. Untouched Test Set Evaluation
    test_metrics, y_pred, y_proba, tn, fp, fn, tp = evaluate_on_test_set(
        best_pipeline, X_train, y_train, X_test, y_test
    )

    print("\n" + "=" * 70)
    print(f"UNTOUCHED TEST SET PERFORMANCE: {best_model_name} (N={len(y_test)})")
    print("=" * 70)
    for k, v in test_metrics.items():
        if isinstance(v, float):
            print(f"  {k:25s}: {v:.4f} ({v*100:.2f}%)")
        else:
            print(f"  {k:25s}: {v}")

    # 6. Save Visualizations
    plot_confusion_matrix(tn, fp, fn, tp, best_model_name, CONFUSION_MATRIX_PATH)
    plot_roc_curves(candidate_pipelines, X_train, y_train, X_test, y_test, ROC_CURVES_PATH)

    # 7. Serialize Winning Pipeline
    joblib.dump(best_pipeline, SAVED_MODEL_PATH)
    print(f"\nWinning pipeline saved to: {SAVED_MODEL_PATH}")

    return best_model_name, test_metrics, cv_comparison


if __name__ == "__main__":
    run_training_pipeline()
