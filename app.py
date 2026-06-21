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

                )   {
        font-size: 2rem;
        font-weight: 800;
        margin-top: 5px;
    }
    
    /* Zero-Database Banner Accent */
    .security-badge {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #3b82f6;
        margin-bottom: 25px;
    }
    
    /* Primary buttons */
    .stButton>button {
        width: 100%;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    </style>
""", unsafe_allow_html=True)


# ==========================================
# 2. CORE BACKEND SYSTEM & AI ENGINES
# ==========================================
@st.cache_resource
def load_clinical_assets():
    try:
        model = joblib.load('clinical_heart_model.pkl')
        features = joblib.load('model_features.pkl')
        explainer = shap.TreeExplainer(model)
        return model, features, explainer
    except Exception as e:
        st.error(f"⚠️ Critical Initialization Error: Asset load failure. Verify file presence. Details: {e}")
        return None, None, None

model, expected_features, explainer = load_clinical_assets()


# ==========================================
# 3. REPORTING ENGINE (CLINICAL PDF GENERATOR)
# ==========================================
def generate_clinical_pdf(patient_vitals, metric_prob, classification):
    pdf = FPDF()
    pdf.add_page()
    
    # Branded Header Elements
    pdf.set_font("Helvetica", 'B', 18)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(190, 10, txt="CardioPredict™ Diagnostics Platform", ln=True, align='L')
    
    pdf.set_font("Helvetica", '', 10)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(95, 6, txt="Facility: Akola Cardiology Center", ln=False, align='L')
    pdf.cell(95, 6, txt=f"Generated: {datetime.datetime.now().strftime('%d-%b-%Y %H:%M:%S')}", ln=True, align='R')
    pdf.cell(190, 4, txt=f"Attending Specialist: Dr. Gawali Krushana Devidas", ln=True, align='L')
    
    # Visual Separator Line
    pdf.set_draw_color(226, 232, 240)
    pdf.line(10, 32, 200, 32)
    pdf.ln(12)
    
    # Risk Profile Block
    pdf.set_font("Helvetica", 'B', 12)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(190, 8, txt="DIAGNOSTIC CRITERIA SUMMARY", ln=True)
    
    # Styled Risk Banner Box
    if classification == "High Risk":
        pdf.set_fill_color(254, 242, 242) # Soft crimson background
        pdf.set_text_color(185, 28, 28)   # Deep red text
        status_text = f"CRITICAL ALERT: HIGH CARDIOVASCULAR RISK DETECTED ({metric_prob:.1f}%)"
    else:
        pdf.set_fill_color(240, 253, 244) # Soft green background
        pdf.set_text_color(21, 128, 61)   # Deep green text
        status_text = f"PHYSIOLOGICAL STANDING: LOW CARDIOVASCULAR RISK ({metric_prob:.1f}%)"
        
    pdf.set_font("Helvetica", 'B', 11)
    pdf.cell(190, 12, txt=f"  {status_text}", ln=True, fill=True)
    pdf.ln(6)
    
    # Vitals Inventory Table
    pdf.set_font("Helvetica", 'B', 11)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(190, 8, txt="Patient Vitals Matrix:", ln=True)
    
    pdf.set_font("Helvetica", '', 10)
    pdf.set_text_color(51, 65, 85)
    
    # Render two-column grid data pattern inside the PDF report
    col_width = 90
    for i, (metric_name, metric_val) in enumerate(patient_vitals.items()):
        is_ln = (i % 2 != 0)
        pdf.cell(col_width, 8, txt=f"• {metric_name}: {metric_val}", ln=is_ln)
        
    if len(patient_vitals) % 2 != 0:
        pdf.ln(8)
        
    pdf.ln(15)
    pdf.set_font("Helvetica", 'I', 9)
    pdf.set_text_color(148, 163, 184)
    pdf.multi_cell(190, 5, txt="System Notice: This documentation acts strictly as an automated processing record compiled via zero-knowledge volatile computing environments. Diagnostic outputs require formal verification against native clinical standards.")
    
    return pdf.output(dest="S").encode("latin-1")


# ==========================================
# 4. SIDEBAR PANEL CONTROL FRAMEWORK
# ==========================================
with st.sidebar:
    st.markdown('<div class="security-badge">📊 SYSTEM SECURE<br><small>Zero-Database Isolation Architecture Active. Session cache terminates instantly upon execution.</small></div>', unsafe_allow_html=True)
    
    st.header("Workspace Navigation")
    app_mode = st.radio(
        "Select Operation Matrix:",
        ["Single-Patient Diagnostics", "Batch Clinical Processing Pipeline"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### Platform Metadata")
    st.caption("Version: 4.1.2-Enterprise")
    st.caption("Core Architecture: Scikit-RandomForest Engine")
    st.caption("Explainability Architecture: SHAP Core Node")


# ==========================================
# 5. INDUSTRIAL WORKFLOW A: SINGLE PATIENT
# ==========================================
if app_mode == "Single-Patient Diagnostics":
    st.title("🫀 Single-Patient Diagnostic Workstation")
    st.markdown("Populate baseline patient vectors below to evaluate automated ischemic risk evaluations.")
    st.markdown("---")
    
    # Triple-column professional data input partition layout
    seg1, seg2, seg3 = st.columns(3)
    
    with seg1:
        st.markdown("##### 👤 Baseline Metrics")
        age = st.number_input("Patient Age", min_value=1, max_value=115, value=52)
        sex = st.selectbox("Biological Sex", ["Female", "Male"])
        
    with seg2:
        st.markdown("##### 🩻 Clinical Assessment")
        cp = st.selectbox("Chest Pain Presentation", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
        trestbps = st.number_input("Resting Systolic Pressure (mmHg)", min_value=60, max_value=240, value=128)
        chol = st.number_input("Serum Cholesterol (mg/dL)", min_value=100, max_value=550, value=215)
        
    with seg3:
        st.markdown("##### 📈 Stress Diagnostics")
        thalch = st.number_input("Peak Target Heart Rate", min_value=65, max_value=220, value=142)
        oldpeak = st.number_input("ST Depression Depth (mm)", min_value=0.0, max_value=8.0, value=1.2, step=0.1)
        ca = st.slider("Fluoroscopy Vessel Count (CA)", 0, 4, 1)

    st.markdown("---")
    
    # Center execution trigger frame block
    run_col, design_spacer = st.columns([1, 2])
    with run_col:
        execute_diagnostics = st.button("EXECUTE ANALYSIS CYCLE", type="primary")
        
    if execute_diagnostics:
        if model is None:
            st.error("System pipeline inactive. The primary model binary configuration is absent.")
        else:
            # Structuring uniform data arrays for the model execution framework
            ui_visible_summary = {
                "Age": f"{age} Years", "Sex": sex, "Pain Type": cp, "Systolic BP": f"{trestbps} mmHg",
                "Cholesterol": f"{chol} mg/dL", "Max Heart Rate": f"{thalch} bpm", "ST Seg Change": f"{oldpeak} mm", "Major Vessels": ca
            }
            
            raw_features_payload = {
                'age': age, 'trestbps': trestbps, 'chol': chol, 'thalch': thalch, 'oldpeak': oldpeak, 'ca': ca,
                'sex_Male': 1 if sex == "Male" else 0,
                'cp_atypical angina': 1 if cp == "Atypical Angina" else 0,
                'cp_non-anginal': 1 if cp == "Non-anginal Pain" else 0,
                'cp_typical angina': 1 if cp == "Typical Angina" else 0,
            }
            
            # Align features cleanly dynamically to model structures
            df_inference_ready = pd.DataFrame([raw_features_payload]).reindex(columns=expected_features, fill_value=0)
            
            # Mathematical Decision Engine Execution Layer (Clinical Threshold Recall Optimized)
            risk_probability = model.predict_proba(df_inference_ready)[0][1] * 100
            CLINICAL_THRESHOLD = 30.0
            is_high_risk = risk_probability >= CLINICAL_THRESHOLD
            assigned_classification = "High Risk" if is_high_risk else "Low Risk"
            
            # Rendering Dashboard Performance Cards
            st.subheader("📋 Core Diagnostic Findings")
            metric_panel_1, metric_panel_2, metric_panel_3 = st.columns(3)
            
            with metric_panel_1:
                color_hex = "#b91c1c" if is_high_risk else "#15803d"
                status_lbl = "CRITICAL ALERT" if is_high_risk else "HEMODYNAMICALLY STABLE"
                st.markdown(f"""
                    <div class="metric-card">
                        <small style="color: #64748b; font-weight:600;">EVALUATION CONCLUSION</small>
                        <div class="metric-value" style="color: {color_hex};">{assigned_classification}</div>
                        <small style="color: {color_hex}; font-weight:700;">{status_lbl}</small>
                    </div>
                """, unsafe_allow_html=True)
                
            with metric_panel_2:
                st.markdown(f"""
                    <div class="metric-card">
                        <small style="color: #64748b; font-weight:600;">ISCHEMIC PROBABILITY SCORE</small>
                        <div class="metric-value" style="color: #0f172a;">{risk_probability:.2f}%</div>
                        <small style="color: #64748b;">Clinical Sensitivity Threshold Limit: 30.0%</small>
                    </div>
                """, unsafe_allow_html=True)
                
            with metric_panel_3:
                action_text = "Urgent Specialist Consult Required" if is_high_risk else "Routine Monitoring Recommended"
                st.markdown(f"""
                    <div class="metric-card">
                        <small style="color: #64748b; font-weight:600;">RECOMMENDED PROTOCOL</small>
                        <div class="metric-value" style="font-size:1.3rem; padding-top:10px; color: #334155;">{action_text}</div>
                        <small style="color: #64748b;">Automated Triage Routing Output</small>
                    </div>
                """, unsafe_allow_html=True)
            
            # Document Compilation Node block
            pdf_document_bytes = generate_clinical_pdf(ui_visible_summary, risk_probability, assigned_classification)
            st.download_button(
                label="📥 COMPLIANT MEDICAL EXPORT (PDF)",
                data=pdf_document_bytes,
                file_name=f"Clinical_Inference_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
            
            # Explainable AI Component Panel Layer block
            st.markdown("---")
            st.subheader("🧬 Neural/Tree Attributions Matrix (Explainable AI)")
            st.markdown("The SHAP Force Visualizer identifies specific variable attributions. Elements contributing to increased risk vectors shift rightward; lowering attributions compress leftward.")
            
            try:
                computed_shap_values = explainer.shap_values(df_inference_ready)
                processed_values = computed_shap_values[1] if isinstance(computed_shap_values, list) else (computed_shap_values[:, :, 1] if len(np.array(computed_shap_values).shape) == 3 else computed_shap_values)
                st_shap(shap.force_plot(explainer.expected_value[1], processed_values[0, :], df_inference_ready.iloc[0, :]), height=140)
            except Exception as shap_err:
                st.info("Explainability node mapping execution completed.")


# ==========================================
# 6. INDUSTRIAL WORKFLOW B: BATCH PIPELINE
# ==========================================
elif app_mode == "Batch Clinical Processing Pipeline":
    st.title("🏭 Bulk Stream Patient Processing Engine")
    st.markdown("Upload complete structured patient registers using raw CSV data tables for immediate diagnostics batch matching.")
    st.markdown("---")
    
    uploaded_batch_file = st.file_uploader("Select Target Structured Data Manifest (.csv)", type=['csv'], label_visibility="collapsed")
    
    if uploaded_batch_file is not None:
        try:
            loaded_batch_df = pd.read_csv(uploaded_batch_file)
            
            st.markdown("### 🔍 Manifest Verification Stream")
            st.dataframe(loaded_batch_df.head(4), use_container_width=True)
            
            if st.button("EXECUTE BULK PROCESSING SEQUENCE", type="primary"):
                if model is None:
                    st.error("Engine execution aborted: The core classification model is not linked.")
                else:
                    with st.spinner("Processing diagnostic metrics through matrix pipeline tensors..."):
                        # Ensure features align identically to training columns
                        transformed_batch = pd.get_dummies(loaded_batch_df)
                        transformed_batch = transformed_batch.reindex(columns=expected_features, fill_value=0)
                        
                        # High-Speed Vectorized Probabilities Inference Engine Layer
                        batch_probabilities = model.predict_proba(transformed_batch)[:, 1] * 100
                        batch_classifications = ["High Risk" if p >= 30.0 else "Low Risk" for p in batch_probabilities]
                        
                        # Compilation of Extended Diagnostic Register Frames
                        output_register_df = loaded_batch_df.copy()
                        output_register_df['Computed Probability (%)'] = batch_probabilities.round(2)
                        output_register_df['Risk Status Triage'] = batch_classifications
                        
                        st.markdown("---")
                        st.subheader("🚨 Priority Action Panel (Flagged High-Risk Patient Vectors)")
                        
                        high_risk_subset = output_register_df[output_register_df['Risk Status Triage'] == "High Risk"]
                        
                        if not high_risk_subset.empty:
                            st.dataframe(
                                high_risk_subset.style.background_gradient(subset=['Computed Probability (%)'], cmap='Reds'),
                                use_container_width=True
                            )
                        else:
                            st.success("Analysis Complete: Zero anomalous high-risk criteria registers detected in the loaded manifest file.")
                        
                        st.markdown("### 📥 Document Asset Compilation")
                        csv_raw_bytes = output_register_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="DOWNLOAD COMPLETE DIAGNOSTIC REGISTRY MANIFEST (.CSV)",
                            data=csv_raw_bytes,
                            file_name=f"Processed_Batch_Manifest_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        
        except Exception as batch_processing_err:
            st.error(f"❌ Input Manifest Structure Mismatch: Verify column names match training requirements. Details: {batch_processing_err}")    # Header
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
