"""
Streamlit Web Application for Heart Disease Prediction.
Educational Machine Learning Demonstration (UCI Cleveland Dataset).
"""

import os
import sys
import pandas as pd
import streamlit as st

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from predict import HeartDiseasePredictor, EXPECTED_COLUMNS

# Page Configuration
st.set_page_config(
    page_title="Heart Disease Prediction | ML Mini-Project",
    page_icon="❤️",
    layout="wide",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #b71c1c;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 1rem;
        margin-bottom: 1rem;
        text-align: center;
    }
    .pred-absent {
        background-color: #e8f5e9;
        border: 2px solid #4caf50;
        color: #1b5e20;
    }
    .pred-present {
        background-color: #ffebee;
        border: 2px solid #e53935;
        color: #b71c1c;
    }
    .disclaimer-box {
        padding: 1rem;
        background-color: #fff8e1;
        border-left: 5px solid #ffb300;
        border-radius: 4px;
        margin-top: 2rem;
        font-size: 0.95rem;
        color: #6d4c41;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Section
st.markdown('<div class="main-title">Heart Disease Prediction Using Machine Learning</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Educational B.Tech ML Mini-Project (TAE-2) — Authentic UCI Cleveland Dataset (303 instances)</div>',
    unsafe_allow_html=True,
)

# Load Model Pipeline
@st.cache_resource
def load_predictor():
    return HeartDiseasePredictor()

try:
    predictor = load_predictor()
except Exception as e:
    st.error(f"Error loading trained model pipeline: {e}")
    st.stop()

# Sidebar: Quick Presets
st.sidebar.header("Quick Sample Presets")
preset = st.sidebar.radio(
    "Load sample patient data:",
    ("Manual Input", "Sample Profile A (Disease Absent)", "Sample Profile B (Disease Present)"),
)

defaults = {
    "age": 54,
    "sex": 1,
    "cp": 1,
    "trestbps": 130,
    "chol": 240,
    "fbs": 0,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.0,
    "slope": 1,
    "ca": 0,
    "thal": 3,
}

if preset == "Sample Profile A (Disease Absent)":
    defaults = {
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
elif preset == "Sample Profile B (Disease Present)":
    defaults = {
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

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Project Methodology Summary:**
    - Model: Scikit-Learn Pipeline
    - Best CV Model: Logistic Regression
    - 5-Fold CV Mean ROC-AUC: 0.9025
    - Test Set Accuracy: 88.52%
    - Test Set Recall: 92.86%
    - Strict Zero Data Leakage
    """
)

# Input Form
with st.form("patient_input_form"):
    st.subheader("Patient Clinical Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=defaults["age"], step=1)
        
        sex_options = {1: "Male (1)", 0: "Female (0)"}
        sex = st.selectbox(
            "Sex",
            options=list(sex_options.keys()),
            format_func=lambda x: sex_options[x],
            index=list(sex_options.keys()).index(defaults["sex"]),
        )
        
        cp_options = {
            1: "1: Typical Angina",
            2: "2: Atypical Angina",
            3: "3: Non-Anginal Pain",
            4: "4: Asymptomatic",
        }
        cp = st.selectbox(
            "Chest Pain Type (cp)",
            options=list(cp_options.keys()),
            format_func=lambda x: cp_options[x],
            index=list(cp_options.keys()).index(defaults["cp"]),
        )
        
        trestbps = st.number_input(
            "Resting Blood Pressure (mm Hg)",
            min_value=80,
            max_value=220,
            value=defaults["trestbps"],
            step=1,
        )
        
        chol = st.number_input(
            "Serum Cholesterol (mg/dl)",
            min_value=100,
            max_value=600,
            value=defaults["chol"],
            step=1,
        )

    with col2:
        fbs_options = {0: "False (<= 120 mg/dl)", 1: "True (> 120 mg/dl)"}
        fbs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl (fbs)",
            options=list(fbs_options.keys()),
            format_func=lambda x: fbs_options[x],
            index=list(fbs_options.keys()).index(defaults["fbs"]),
        )
        
        restecg_options = {
            0: "0: Normal",
            1: "1: ST-T Wave Abnormality",
            2: "2: Left Ventricular Hypertrophy",
        }
        restecg = st.selectbox(
            "Resting ECG Results (restecg)",
            options=list(restecg_options.keys()),
            format_func=lambda x: restecg_options[x],
            index=list(restecg_options.keys()).index(defaults["restecg"]),
        )
        
        thalach = st.number_input(
            "Maximum Heart Rate Achieved (thalach)",
            min_value=60,
            max_value=230,
            value=defaults["thalach"],
            step=1,
        )
        
        exang_options = {0: "No (0)", 1: "Yes (1)"}
        exang = st.selectbox(
            "Exercise-Induced Angina (exang)",
            options=list(exang_options.keys()),
            format_func=lambda x: exang_options[x],
            index=list(exang_options.keys()).index(defaults["exang"]),
        )

    with col3:
        oldpeak = st.number_input(
            "ST Depression by Exercise (oldpeak)",
            min_value=0.0,
            max_value=8.0,
            value=float(defaults["oldpeak"]),
            step=0.1,
            format="%.1f",
        )
        
        slope_options = {
            1: "1: Upsloping",
            2: "2: Flat",
            3: "3: Downsloping",
        }
        slope = st.selectbox(
            "Peak Exercise ST Segment Slope (slope)",
            options=list(slope_options.keys()),
            format_func=lambda x: slope_options[x],
            index=list(slope_options.keys()).index(defaults["slope"]),
        )
        
        ca = st.selectbox(
            "Number of Major Vessels Colored by Fluoroscopy (ca)",
            options=[0, 1, 2, 3],
            index=[0, 1, 2, 3].index(defaults["ca"]),
        )
        
        thal_options = {
            3: "3: Normal",
            6: "6: Fixed Defect",
            7: "7: Reversible Defect",
        }
        thal = st.selectbox(
            "Thalassemia (thal)",
            options=list(thal_options.keys()),
            format_func=lambda x: thal_options[x],
            index=list(thal_options.keys()).index(defaults["thal"]),
        )

    submit_button = st.form_submit_button("Predict Heart Disease Diagnosis", use_container_width=True)

if submit_button:
    input_dict = {
        "age": float(age),
        "sex": float(sex),
        "cp": float(cp),
        "trestbps": float(trestbps),
        "chol": float(chol),
        "fbs": float(fbs),
        "restecg": float(restecg),
        "thalach": float(thalach),
        "exang": float(exang),
        "oldpeak": float(oldpeak),
        "slope": float(slope),
        "ca": float(ca),
        "thal": float(thal),
    }

    result = predictor.predict(input_dict)
    pred_label = result["prediction_label"]
    pred_class = result["predicted_class"]
    pred_prob = result["predicted_probability_percent"]

    st.subheader("Prediction Output")

    if pred_class == 1:
        box_class = "pred-present"
        icon = "⚠️"
    else:
        box_class = "pred-absent"
        icon = "✅"

    st.markdown(
        f"""
        <div class="prediction-box {box_class}">
            <h2 style="margin: 0; font-size: 1.8rem;">{icon} Prediction: {pred_label}</h2>
            <h3 style="margin-top: 0.5rem; font-weight: 500; font-size: 1.3rem;">
                Predicted Probability: <strong>{pred_prob}%</strong>
            </h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Progress bar showing predicted probability
    st.progress(result["predicted_probability"])

    st.caption(
        "Note: The predicted probability reflects the model's posterior output for the positive class (presence of heart disease) generated by the calibrated Scikit-Learn pipeline."
    )

# Mandatory Academic & Medical Disclaimer
st.markdown(
    """
    <div class="disclaimer-box">
        <strong>Academic & Medical Disclaimer:</strong><br>
        This application is an educational machine learning demonstration and is not a medical diagnostic tool.
        Predictions should not be used as a substitute for professional medical advice.
    </div>
    """,
    unsafe_allow_html=True,
)
