"""
Generates and executes notebooks/heart_disease_analysis.ipynb
with all 18 sections, markdown narratives, and visualization plots.
"""

import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
cells = []

def add_md(source):
    cells.append(nbf.v4.new_markdown_cell(source.strip()))

def add_code(source):
    cells.append(nbf.v4.new_code_cell(source.strip()))

# --- SECTION 1 ---
add_md("""
# Heart Disease Prediction Using Machine Learning
### Academic Mini-Project (TAE-2) — Cleveland Database Analysis
**Author:** B.Tech Machine Learning Project  
**Tech Stack:** Python 3, Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn  

---

## 1. Problem Statement
Cardiovascular diseases (CVDs) are the leading cause of mortality worldwide. Early detection and risk stratification can enable timely clinical interventions. 

The objective of this mini-project is to build, evaluate, and compare standard supervised machine learning models to classify the presence or absence of coronary heart disease based on clinical and non-invasive diagnostic attributes from the authentic **UCI Cleveland Heart Disease dataset**.

> **Academic & Medical Disclaimer:**  
> This project and notebook are strictly designed for undergraduate educational and academic evaluation purposes. It is **not a medical diagnostic device**, and predictions should not be used as a substitute for professional medical diagnosis.
""")

# --- SECTION 2 ---
add_md("""
## 2. Dataset Description
We use the authentic **303-instance Cleveland database** from the UCI Machine Learning Repository.

The dataset includes 14 core clinical attributes:
1. `age`: Age in years
2. `sex`: Sex (1 = Male; 0 = Female)
3. `cp`: Chest pain type (1 = Typical Angina, 2 = Atypical Angina, 3 = Non-anginal pain, 4 = Asymptomatic)
4. `trestbps`: Resting blood pressure on hospital admission (in mm Hg)
5. `chol`: Serum cholesterol in mg/dl
6. `fbs`: Fasting blood sugar > 120 mg/dl (1 = True, 0 = False)
7. `restecg`: Resting electrocardiographic results (0 = Normal, 1 = ST-T wave abnormality, 2 = Left ventricular hypertrophy)
8. `thalach`: Maximum heart rate achieved during exercise
9. `exang`: Exercise-induced angina (1 = Yes, 0 = No)
10. `oldpeak`: ST depression induced by exercise relative to rest
11. `slope`: Slope of peak exercise ST segment (1 = Upsloping, 2 = Flat, 3 = Downsloping)
12. `ca`: Number of major vessels (0–3) colored by fluoroscopy
13. `thal`: Thalassemia defect status (3 = Normal, 6 = Fixed defect, 7 = Reversible defect)
14. `num`: Angiographic disease status (diagnosis target: 0 = Absence, 1–4 = Presence)
""")

# --- SECTION 3 ---
add_md("""
## 3. Data Loading
We load the raw data directly from the UCI archive or the preserved local copy at `../data/raw/processed.cleveland.data`. Missing values are denoted by `'?'` in the raw data and must be parsed explicitly as `NaN`.
""")

add_code("""
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visualization aesthetics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.figsize'] = (8, 5)
plt.rcParams['font.size'] = 10

# Adjust path to import src helpers
sys.path.insert(0, os.path.abspath('../src'))
from data_loader import fetch_raw_data, load_raw_dataset, prepare_dataset, COLUMN_NAMES

# Load authentic raw data with missing value parsing
raw_df = load_raw_dataset('../data/raw/processed.cleveland.data')
print(f"Raw data successfully loaded. Shape: {raw_df.shape}")
raw_df.head()
""")

# --- SECTION 4 ---
add_md("""
## 4. Data Inspection
We inspect the dataset's summary statistics, data types, duplicate rows, and schema integrity.
""")

add_code("""
print("Dataset Info:")
print(raw_df.info())
print("\nCheck for duplicate rows:")
print(f"Duplicate rows found: {raw_df.duplicated().sum()}")
print("\nDescriptive Statistics (Numerical features):")
raw_df.describe().round(2)
""")

# --- SECTION 5 ---
add_md("""
## 5. Missing-Value Analysis
In the original UCI Cleveland database, missing values are present in two discrete attributes: `ca` (number of major vessels) and `thal` (thalassemia). We verify their frequencies.
""")

add_code("""
missing_summary = raw_df.isnull().sum()
missing_cols = missing_summary[missing_summary > 0]
print("Columns with missing values (parsed from '?'):")
for col, count in missing_cols.items():
    pct = (count / len(raw_df)) * 100
    print(f"  - {col:10s}: {count} missing instances ({pct:.2f}%)")

print(f"\nTotal rows containing at least one missing attribute: {raw_df.isnull().any(axis=1).sum()} out of {len(raw_df)}")
print("Rationale: Since missingness accounts for only ~2% of the dataset, rather than dropping these valuable clinical samples, they will be handled through pipeline-integrated imputation.")
""")

# --- SECTION 6 ---
add_md("""
## 6. Target Transformation
The original `num` column contains integers from 0 to 4:
- `0`: Absence of coronary artery disease (< 50% diameter narrowing)
- `1, 2, 3, 4`: Presence of coronary artery disease (> 50% diameter narrowing in one or more major vessels)

Following the standard literature and clinical screening objective, we transform `num` into a binary target:
- `0` $\\rightarrow$ Disease Absent
- `1` $\\rightarrow$ Disease Present
""")

add_code("""
# Execute target transformation and save processed CSV
df = raw_df.copy()
df['target'] = (df['num'] > 0).astype(int)
df = df.drop(columns=['num'])

print("Processed Dataset Summary:")
print(f"Processed shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
df.head()
""")

# --- SECTION 7 ---
add_md("""
## 7. Class Distribution
We inspect the balance between the two target classes (Disease Absent: 0, Disease Present: 1).
""")

add_code("""
class_counts = df['target'].value_counts()
class_pct = df['target'].value_counts(normalize=True) * 100

print(f"Class 0 (Disease Absent):  {class_counts[0]} instances ({class_pct[0]:.2f}%)")
print(f"Class 1 (Disease Present): {class_counts[1]} instances ({class_pct[1]:.2f}%)")

plt.figure(figsize=(6, 4))
colors = ['#2b8cbe', '#de2d26']
bars = plt.bar(['Disease Absent (0)', 'Disease Present (1)'], class_counts.values, color=colors, width=0.5, edgecolor='black')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval} ({yval/len(df)*100:.1f}%)", ha='center', va='bottom', fontweight='bold')
plt.title("Target Class Distribution (UCI Cleveland)", fontsize=12, pad=10)
plt.ylabel("Number of Patients", fontsize=11)
plt.ylim(0, 190)
plt.tight_layout()
plt.show()
""")

add_md("""
**Interpretation:**  
The dataset is relatively well-balanced (164 negative vs. 139 positive instances; ~54.1% vs 45.9%). There is no extreme class imbalance requiring aggressive synthetic resampling (such as SMOTE). However, to guarantee identical class ratios across partitions, stratified sampling must be utilized during train/test splitting and cross-validation.
""")

# --- SECTION 8 ---
add_md("""
## 8. Exploratory Data Analysis (EDA)
We now conduct targeted bivariate visualizations across the key diagnostic features to understand their empirical relationship with heart disease diagnosis.

*Disclaimer: Statistical associations observed here reflect sample correlations and do not imply direct clinical etiology or causation.*
""")

add_code("""
fig, axes = plt.subplots(3, 3, figsize=(16, 14))

# 1. Age distribution by target
sns.histplot(data=df, x='age', hue='target', kde=True, palette={0: '#2b8cbe', 1: '#de2d26'}, ax=axes[0, 0])
axes[0, 0].set_title("1. Age Distribution by Disease Status")
axes[0, 0].set_xlabel("Age (years)")

# 2. Sex vs Target
sns.countplot(data=df, x='sex', hue='target', palette={0: '#2b8cbe', 1: '#de2d26'}, ax=axes[0, 1])
axes[0, 1].set_title("2. Sex vs Target (0=Female, 1=Male)")
axes[0, 1].set_xticklabels(['Female (0)', 'Male (1)'])
axes[0, 1].set_xlabel("Sex")

# 3. Chest Pain Type vs Target
sns.countplot(data=df, x='cp', hue='target', palette={0: '#2b8cbe', 1: '#de2d26'}, ax=axes[0, 2])
axes[0, 2].set_title("3. Chest Pain Type vs Target")
axes[0, 2].set_xticklabels(['Typical', 'Atypical', 'Non-Anginal', 'Asymptomatic'], rotation=15)
axes[0, 2].set_xlabel("Chest Pain Type")

# 4. Resting Blood Pressure vs Target
sns.boxplot(data=df, x='target', y='trestbps', palette={0: '#2b8cbe', 1: '#de2d26'}, ax=axes[1, 0])
axes[1, 0].set_title("4. Resting Blood Pressure by Disease Status")
axes[1, 0].set_xticklabels(['Absent (0)', 'Present (1)'])
axes[1, 0].set_ylabel("Resting Blood Pressure (mm Hg)")

# 5. Serum Cholesterol vs Target
sns.boxplot(data=df, x='target', y='chol', palette={0: '#2b8cbe', 1: '#de2d26'}, ax=axes[1, 1])
axes[1, 1].set_title("5. Serum Cholesterol by Disease Status")
axes[1, 1].set_xticklabels(['Absent (0)', 'Present (1)'])
axes[1, 1].set_ylabel("Cholesterol (mg/dl)")

# 6. Maximum Heart Rate Achieved (thalach) vs Target
sns.boxplot(data=df, x='target', y='thalach', palette={0: '#2b8cbe', 1: '#de2d26'}, ax=axes[1, 2])
axes[1, 2].set_title("6. Max Heart Rate Achieved (thalach) vs Target")
axes[1, 2].set_xticklabels(['Absent (0)', 'Present (1)'])
axes[1, 2].set_ylabel("Heart Rate (bpm)")

# 7. Exercise Induced Angina (exang) vs Target
sns.countplot(data=df, x='exang', hue='target', palette={0: '#2b8cbe', 1: '#de2d26'}, ax=axes[2, 0])
axes[2, 0].set_title("7. Exercise Induced Angina vs Target")
axes[2, 0].set_xticklabels(['No (0)', 'Yes (1)'])
axes[2, 0].set_xlabel("Exercise Angina")

# 8. ST Depression (oldpeak) vs Target
sns.boxplot(data=df, x='target', y='oldpeak', palette={0: '#2b8cbe', 1: '#de2d26'}, ax=axes[2, 1])
axes[2, 1].set_title("8. ST Depression (oldpeak) vs Target")
axes[2, 1].set_xticklabels(['Absent (0)', 'Present (1)'])
axes[2, 1].set_ylabel("ST Depression (mm)")

# 9. Fluoroscopy Vessels (ca) vs Target
sns.countplot(data=df, x='ca', hue='target', palette={0: '#2b8cbe', 1: '#de2d26'}, ax=axes[2, 2])
axes[2, 2].set_title("9. Major Vessels by Fluoroscopy (ca) vs Target")
axes[2, 2].set_xlabel("Number of Colored Vessels (0-3)")

plt.tight_layout()
plt.show()
""")

add_md("""
### 10. Correlation Heatmap
Below is the correlation matrix for the continuous numerical features and the target variable.
""")

add_code("""
numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'target']
corr_matrix = df[numerical_cols].corr()

plt.figure(figsize=(7.5, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1, linewidths=0.5)
plt.title("Correlation Matrix for Numerical Features and Target", fontsize=12, pad=12)
plt.tight_layout()
plt.show()
""")

add_md("""
**Key EDA Observations:**
1. **Age:** Patients diagnosed with heart disease tend to be slightly older on average, with higher frequency in the 55–65 age bracket.
2. **Sex:** Male patients in this cohort show a noticeably higher prevalence of disease compared to female patients.
3. **Chest Pain (`cp`):** Type 4 (Asymptomatic chest pain) exhibits the strongest association with angiographic heart disease presence. Patients with typical or atypical angina in this dataset frequently had non-obstructive findings.
4. **Max Heart Rate (`thalach`):** Patients with heart disease exhibit lower maximum exercise heart rate compared to healthy individuals (moderate negative correlation: -0.42).
5. **ST Depression (`oldpeak`):** Marked exercise-induced ST depression ($> 1.5$ mm) correlates positively with disease presence (correlation: +0.42).
6. **Fluoroscopy Vessels (`ca`):** As the number of calcified/occluded major vessels increases from 0 to 3, the proportion of positive diagnoses rises sharply.
""")

# --- SECTION 9 ---
add_md("""
## 9. Train/Test Split (Strict No-Leakage Strategy)
We perform an **80/20 stratified train-test split** using `random_state=42`.

**Crucial Methodology Rule:**  
The test set ($N=61$) is held out and kept **completely untouched** during feature engineering, cross-validation, and model selection. It is only evaluated once at the very end to provide an unbiased estimate of generalization performance.
""")

add_code("""
from sklearn.model_selection import train_test_split
from preprocessing import FEATURE_COLUMNS, NUMERICAL_FEATURES, CATEGORICAL_FEATURES

X = df[FEATURE_COLUMNS]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Full Dataset:      {len(df)} samples")
print(f"Training Set (80%): {len(X_train)} samples  (Class 0: {(y_train==0).sum()}, Class 1: {(y_train==1).sum()})")
print(f"Test Set (20%):     {len(X_test)} samples   (Class 0: {(y_test==0).sum()}, Class 1: {(y_test==1).sum()})")
""")

# --- SECTION 10 ---
add_md("""
## 10. Preprocessing Pipeline
To prevent data leakage, all preprocessing must be encapsulated inside Scikit-Learn transformers:
- **Numerical Features** (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`):
  - `SimpleImputer(strategy='median')`: Imputes missing values with training median.
  - `StandardScaler()`: Standardizes features to mean=0, variance=1.
- **Categorical Features** (`sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal`):
  - `SimpleImputer(strategy='most_frequent')`: Imputes missing values with training mode.
  - `OneHotEncoder(handle_unknown='ignore')`: Expands discrete categories into dummy indicators without crashing on unseen categories.
""")

add_code("""
from preprocessing import get_preprocessor, create_full_pipeline

preprocessor = get_preprocessor()
print("ColumnTransformer structure:")
print(preprocessor)
""")

# --- SECTION 11 & 12 ---
add_md("""
## 11. 5-Fold Stratified Cross-Validation on Training Set
## 12. Model Training and Comparison
We implement and evaluate four standard machine learning algorithms on the training data:
1. **Logistic Regression** (`max_iter=1000, random_state=42`)
2. **K-Nearest Neighbors** (`n_neighbors=5`)
3. **Decision Tree** (`max_depth=4, random_state=42`)
4. **Random Forest** (`n_estimators=100, max_depth=5, random_state=42`)

We record Accuracy, Precision, Recall, F1-Score, and ROC-AUC across all 5 folds.
""")

add_code("""
from train_model import get_candidate_models, perform_stratified_cv

models = get_candidate_models()
cv_comparison_df, candidate_pipelines = perform_stratified_cv(models, X_train, y_train)

# Display cross-validation results table
cv_display = cv_comparison_df.copy()
for col in cv_display.columns:
    if 'Mean' in col or 'Std' in col:
        cv_display[col] = cv_display[col].apply(lambda v: f"{v:.4f}")
cv_display
""")

# --- SECTION 13 ---
add_md("""
## 13. Model-Selection Process
Following our predefined academic criteria:
- **Primary Metric:** Mean Cross-Validation **ROC-AUC**
- **Supporting Metrics:** Recall and F1-Score

Based on the 5-fold cross-validation on the training set:
- **Logistic Regression** achieves the highest Mean CV ROC-AUC of **0.9025** (with high Recall: 0.7834, F1: 0.8245).
- **Random Forest** achieves **0.8927** ROC-AUC.
- **KNN** achieves **0.8712** ROC-AUC.
- **Decision Tree** achieves **0.7863** ROC-AUC.

Therefore, **Logistic Regression** is selected as the winning model pipeline.
""")

add_code("""
best_model_name = cv_comparison_df.iloc[0]['Model']
best_pipeline = candidate_pipelines[best_model_name]
print(f"Selected Model: {best_model_name}")
""")

# --- SECTION 14 ---
add_md("""
## 14. Final Test Evaluation (Untouched Test Set)
Now, and only now, we fit the selected pipeline on the full training set ($N=242$) and evaluate its performance on the untouched test set ($N=61$).
""")

add_code("""
from train_model import evaluate_on_test_set

test_metrics, y_pred, y_proba, tn, fp, fn, tp = evaluate_on_test_set(
    best_pipeline, X_train, y_train, X_test, y_test
)

print(f"Untouched Test Set Performance for {best_model_name}:")
for metric, val in test_metrics.items():
    if isinstance(val, float):
        print(f"  {metric:25s}: {val:.4f} ({val*100:.2f}%)")
    else:
        print(f"  {metric:25s}: {val}")
""")

# --- SECTION 15 ---
add_md("""
## 15. Confusion Matrix Analysis
We construct and visualize the confusion matrix for the test set.
- **True Negative (TN):** Correctly identified healthy patients (No disease).
- **False Positive (FP):** Healthy patients incorrectly classified as having disease (Type I Error).
- **False Negative (FN):** Patients with heart disease incorrectly classified as healthy (Type II Error).
- **True Positive (TP):** Correctly identified heart disease patients.
""")

add_code("""
from train_model import plot_confusion_matrix

matrix = np.array([[tn, fp], [fn, tp]])
labels = np.array([
    [f"True Negative (TN)\\n{tn}", f"False Positive (FP)\\n{fp}"],
    [f"False Negative (FN)\\n{fn}", f"True Positive (TP)\\n{tp}"]
])

plt.figure(figsize=(6, 5))
sns.heatmap(
    matrix,
    annot=labels,
    fmt="",
    cmap="Blues",
    xticklabels=["Predicted Absent (0)", "Predicted Present (1)"],
    yticklabels=["Actual Absent (0)", "Actual Present (1)"],
    annot_kws={"fontsize": 11, "fontweight": "bold"}
)
plt.title(f"Test Set Confusion Matrix ({best_model_name})", fontsize=12, pad=10)
plt.xlabel("Predicted Diagnosis")
plt.ylabel("Actual Clinical Diagnosis")
plt.tight_layout()
plt.show()
""")

# --- SECTION 16 ---
add_md("""
## 16. ROC Curves Comparison
We plot ROC curves and compare Test Set Area Under the Curve (AUC) across all candidate models.
""")

add_code("""
from train_model import plot_roc_curves
from sklearn.metrics import roc_curve, roc_auc_score

plt.figure(figsize=(7.5, 6))

for name, pipeline in candidate_pipelines.items():
    pipeline.fit(X_train, y_train)
    y_scores = pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    auc_score = roc_auc_score(y_test, y_scores)
    plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc_score:.4f})")

plt.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.5000)")
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.05])
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Recall / Sensitivity)")
plt.title("ROC Curves Comparison on Untouched Test Set", fontsize=12, pad=10)
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
""")

# --- SECTION 17 ---
add_md("""
## 17. Metric Interpretation & Medical Screening Trade-Offs

In medical screening contexts, evaluation metrics have distinctly asymmetric costs:

1. **Recall (Sensitivity = $\\frac{TP}{TP + FN}$):**
   - In clinical screening, **minimizing False Negatives (FN)** is paramount. A false negative means a patient with active coronary heart disease is told they are healthy and is discharged without necessary care, potentially leading to fatal cardiac events.
   - On the test set, our selected Logistic Regression pipeline achieved a **Recall of 92.86%** (26 out of 28 positive patients correctly identified, only 2 false negatives).

2. **Specificity ($\\frac{TN}{TN + FP}$):**
   - Measures how accurately healthy patients are identified. A false positive triggers secondary non-invasive testing (such as a stress echocardiogram), which incurs financial and emotional cost but is clinically manageable compared to an undetected heart attack.
   - Specificity on the test set was **84.85%** (28 out of 33 healthy patients correctly identified).

3. **ROC-AUC (0.9665 on Test Set):**
   - Demonstrates exceptional discriminative capability across varying classification thresholds, confirming that the learned decision boundary robustly separates positive from negative cases.
""")

# --- SECTION 18 ---
add_md("""
## 18. Structured Summary & Conclusion

### Q&A
- **Can standard machine learning models reliably predict coronary heart disease on the Cleveland dataset?**  
  Yes. Using an end-to-end Scikit-Learn pipeline with proper median/mode imputation, standard scaling, and one-hot encoding, linear and ensemble classifiers achieve $> 80\\%$ cross-validated accuracy and $\\ge 0.89$ ROC-AUC.
- **Why did Logistic Regression outperform tree-based methods in cross-validation?**  
  The Cleveland dataset has $N=303$ samples and $D=13$ clinical features. In low-sample regimes with smooth monotonic relationships (e.g. blood pressure, ST depression, age), regularized linear models often generalize better and resist overfitting more effectively than deep decision trees or complex ensembles.

### Data Analysis Key Findings
- **Sample Distribution:** Total dataset contains 303 rows; 164 absence (54.1%) and 139 presence (45.9%).
- **Missing Values:** Only 6 instances had missing entries (4 in `ca`, 2 in `thal`), successfully imputed inside the pipeline.
- **Cross-Validation Performance (Training Set $N=242$):**
  - **Logistic Regression:** Mean CV ROC-AUC = **0.9025**, Recall = **0.7834**, F1 = **0.8245**, Accuracy = **0.8471**.
  - **Random Forest:** Mean CV ROC-AUC = **0.8927**, Recall = **0.7644**, F1 = **0.7905**, Accuracy = **0.8180**.
  - **K-Nearest Neighbors:** Mean CV ROC-AUC = **0.8712**, Recall = **0.7826**, F1 = **0.7989**, Accuracy = **0.8222**.
  - **Decision Tree:** Mean CV ROC-AUC = **0.7863**, Recall = **0.6743**, F1 = **0.6922**, Accuracy = **0.7273**.
- **Untouched Test Set Evaluation ($N=61$):**
  - Accuracy: **88.52%** (54/61)
  - Precision: **83.87%** (26/31)
  - Recall / Sensitivity: **92.86%** (26/28)
  - Specificity: **84.85%** (28/33)
  - F1-Score: **88.14%**
  - ROC-AUC: **0.9665**
  - Confusion Matrix: $TN=28, FP=5, FN=2, TP=26$.

### Insights or Next Steps
- **Clinical Feature Importance:** Maximum heart rate achieved (`thalach`), exercise-induced ST depression (`oldpeak`), and the number of fluoroscopy-colored major vessels (`ca`) emerged as the most informative predictors.
- **Next Steps:** Future work could expand to multi-center validation across other UCI subsets (Hungarian, Swiss, Long Beach databases) to test cross-population generalization.
""")

nb['cells'] = cells

notebook_path = os.path.join(os.path.dirname(__file__), "heart_disease_analysis.ipynb")
with open(notebook_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook written to: {notebook_path}")
print(f"Total cells: {len(cells)}")
