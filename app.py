import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
from streamlit_shap import st_shap
from fpdf import FPDF
import base64
import datetime

# --- 1. System Setup & Security ---
st.set_page_config(page_title="CardioPredict UI", layout="wide")

# Strict Zero-Database Architecture Notice
st.sidebar.title("🫀 CardioPredict System")
st.sidebar.markdown("---")
st.sidebar.info("🔒 **Zero-Database Architecture**\n\nAll patient data is processed in volatile memory (RAM) and instantly destroyed after inference. No records are persistently stored.")

app_mode = st.sidebar.radio("Select Clinical Workflow:", ["Single Patient Inference", "Batch CSV Processing"])

# --- 2. Load the AI ---
@st.cache_resource
def load_model():
    model = joblib.load('clinical_heart_model.pkl')
    features = joblib.load('model_features.pkl')
    # Initialize SHAP explainer once to save memory
    explainer = shap.TreeExplainer(model)
    return model, features, explainer

model, expected_features, explainer = load_model()

# --- 3. Helper: PDF Generation ---
def create_pdf(patient_data, probability, risk_level):
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="CardioPredict Clinical Diagnostics Report", ln=True, align='C')
    pdf.set_font("Arial", 'I', 11)
    pdf.cell(200, 8, txt="Attending: Dr. Gawali Krushana Devidas | Akola Cardiology Center", ln=True, align='C')
    pdf.cell(200, 8, txt=f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')
    pdf.ln(10)
    
    # Risk Assessment
    pdf.set_font("Arial", 'B', 14)
    if risk_level == "High Risk":
        pdf.set_text_color(220, 20, 60) # Crimson
        pdf.cell(200, 10, txt=f"DIAGNOSIS: HIGH RISK DETECTED ({probability:.1f}%)", ln=True)
    else:
        pdf.set_text_color(34, 139, 34) # Green
        pdf.cell(200, 10, txt=f"DIAGNOSIS: LOW RISK ({probability:.1f}%)", ln=True)
    
    pdf.set_text_color(0, 0, 0) # Reset to black
    pdf.ln(5)
    
    # Vitals
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Patient Vitals Recorded:", ln=True)
    pdf.set_font("Arial", '', 12)
    
    for key, value in patient_data.items():
        pdf.cell(200, 8, txt=f"- {key}: {value}", ln=True)
        
    pdf.ln(10)
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(200, 10, txt="Note: This is an AI-assisted diagnostic tool. Proceed with clinical screening.", ln=True)
    
    return pdf.output(dest="S").encode("latin-1")


# ==========================================
# WORKFLOW A: SINGLE PATIENT (LIVE SHAP & PDF)
# ==========================================
if app_mode == "Single Patient Inference":
    st.title("Single Patient Diagnostic Tool")
    
    st.header("Patient Vitals")
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
        sex = st.selectbox("Sex", ["Female", "Male"])
        cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"])
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)

    with col2:
        chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
        thalch = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
        oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        ca = st.slider("Number of Major Vessels (CA)", 0, 4, 0)

    if st.button("Run Diagnostics", type="primary"):
        # 1. Prepare Data
        human_readable_data = {
            "Age": age, "Sex": sex, "Chest Pain": cp, "Resting BP": trestbps,
            "Cholesterol": chol, "Max Heart Rate": thalch, "ST Depression": oldpeak, "Major Vessels": ca
        }
        
        patient_data = {
            'age': age, 'trestbps': trestbps, 'chol': chol, 'thalch': thalch,
            'oldpeak': oldpeak, 'ca': ca,
            'sex_Male': 1 if sex == "Male" else 0,
            'cp_atypical angina': 1 if cp == "Atypical Angina" else 0,
            'cp_non-anginal': 1 if cp == "Non-anginal" else 0,
            'cp_typical angina': 1 if cp == "Typical Angina" else 0,
        }
        
        df_patient = pd.DataFrame([patient_data]).reindex(columns=expected_features, fill_value=0)
        
        # 2. Advanced AI Modeling: Optimize for Recall (Threshold = 0.3)
        probability = model.predict_proba(df_patient)[0][1] * 100
        threshold = 30.0 # 30% threshold to aggressively prevent False Negatives
        
        st.divider()
        if probability >= threshold:
            st.error(f"⚠️ **High Risk Detected** (Confidence: {probability:.1f}%)")
            risk_level = "High Risk"
            
            # --- LIVE SHAP INTEGRATION ---
            st.subheader("Explainable AI: Decision Breakdown")
            st.write("Variables pushing the model toward higher risk are in **red**. Variables lowering the risk are in **blue**.")
            
            shap_values = explainer.shap_values(df_patient)
            sv_patient = shap_values[1] if isinstance(shap_values, list) else (shap_values[:, :, 1] if len(np.array(shap_values).shape) == 3 else shap_values)
            
            st_shap(shap.force_plot(explainer.expected_value[1], sv_patient[0, :], df_patient.iloc[0, :]))
            
        else:
            st.success(f"✅ **Low Risk** (Probability of disease: {probability:.1f}%)")
            risk_level = "Low Risk"

        # --- PDF REPORT GENERATION ---
        pdf_bytes = create_pdf(human_readable_data, probability, risk_level)
        st.download_button(
            label="📄 Download Medical Report (PDF)",
            data=pdf_bytes,
            file_name=f"Patient_Report_{datetime.datetime.now().strftime('%Y%m%d%H%M')}.pdf",
            mime="application/pdf"
        )

# ==========================================
# WORKFLOW B: BATCH PREDICTION (CSV UPLOAD)
# ==========================================
elif app_mode == "Batch CSV Processing":
    st.title("Batch Patient Processing")
    st.write("Upload a CSV file containing patient data to process multiple records simultaneously.")
    
    uploaded_file = st.file_uploader("Upload Patient CSV", type=['csv'])
    
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write("Preview of Uploaded Data:")
        st.dataframe(batch_df.head())
        
        if st.button("Process Batch"):
            with st.spinner("Running diagnostics across all patients..."):
                # Preprocess batch to match expected features
                processed_batch = pd.get_dummies(batch_df)
                processed_batch = processed_batch.reindex(columns=expected_features, fill_value=0)
                
                # Predict with Recall optimization (30% threshold)
                probabilities = model.predict_proba(processed_batch)[:, 1] * 100
                predictions = ["High Risk" if prob >= 30.0 else "Low Risk" for prob in probabilities]
                
                # Append results
                results_df = batch_df.copy()
                results_df['Risk Probability (%)'] = probabilities.round(1)
                results_df['Diagnostic Alert'] = predictions
                
                st.success(f"Successfully processed {len(results_df)} patients.")
                
                # Filter and highlight only high-risk patients
                st.subheader("⚠️ High-Risk Patients Requiring Immediate Attention")
                high_risk_df = results_df[results_df['Diagnostic Alert'] == "High Risk"]
                st.dataframe(high_risk_df.style.highlight_max(subset=['Risk Probability (%)'], color='lightcoral'))
                
                # Provide CSV download of results
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Full Batch Results",
                    data=csv,
                    file_name="Processed_Batch_Diagnostics.csv",
                    mime="text/csv",
                )
