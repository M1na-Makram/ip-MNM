import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import io
import contextlib
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import sys

# Add project root to sys.path for cross-directory imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Updated imports based on new structure
from ml_model.train_model import rule_based_predict
from rule_based_system.expert_system import PatientData, HeartDiseaseExpert
from rule_based_system.rules import run_expert_system

st.set_page_config(page_title="Heart Disease Dashboard", layout="wide", page_icon="🫀")

# Custom CSS for styling
st.markdown("""
<style>
:root {
    --primary: #E74C3C;
    --medium: #F39C12;
    --safe: #27AE60;
    --bg: #0E1117;
}
.metric-card {
    background-color: #1E2129;
    padding: 20px;
    border-radius: 10px;
    border-left: 5px solid var(--primary);
}
.risk-badge {
    padding: 10px 20px;
    border-radius: 8px;
    color: white;
    font-weight: bold;
    text-align: center;
    font-size: 1.2rem;
    margin: 10px 0;
}
.high-risk { background-color: var(--primary); }
.medium-risk { background-color: var(--medium); }
.low-risk { background-color: var(--safe); }
</style>
""", unsafe_allow_html=True)

# Paths Configuration
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_CLEAN_PATH = os.path.join(BASE_DIR, 'data', 'cleaned_data.csv')
DATA_RAW_PATH = os.path.join(BASE_DIR, 'data', 'raw_data.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'ml_model', 'heart_model.joblib')
REPORT_PATH = os.path.join(BASE_DIR, 'reports', 'accuracy_comparison.md')

@st.cache_data
def load_data():
    if not os.path.exists(DATA_CLEAN_PATH) or not os.path.exists(DATA_RAW_PATH):
        return None, None
    df_clean = pd.read_csv(DATA_CLEAN_PATH)
    df_raw = pd.read_csv(DATA_RAW_PATH)
    return df_clean, df_raw

@st.cache_resource
def load_models():
    if not os.path.exists(MODEL_PATH):
        return None, None
    model = joblib.load(MODEL_PATH)
    # Fit scaler on base raw data
    df_raw = pd.read_csv(DATA_RAW_PATH).dropna(subset=['target'])
    scaler = MinMaxScaler()
    scaler.fit(df_raw[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']])
    return model, scaler

def run_and_capture_expert(patient_dict):
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        run_expert_system(patient_dict)
    return f.getvalue()


# Retrieve Cache Pointers
df_clean, df_raw = load_data()
model, scaler = load_models()

# Sidebar Navigation with branding
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; padding-bottom: 20px;'>
            <h1 style='color: var(--primary); margin-bottom: 0;'>🫀 CardioGuard</h1>
            <p style='color: #888; font-size: 0.9rem;'>Advanced Clinical Intelligence</p>
        </div>
    """, unsafe_allow_html=True)
    
    page = st.radio("MAIN MENU", 
        ["Patient Risk Assessment", "Data Insights", "Expert System vs ML", "About"],
        label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown("### System Status")
    st.success("ML Model: Loaded") if model else st.error("ML Model: Offline")
    st.success("Rules Engine: Active")
    st.info(f"Database: {len(df_clean) if df_clean is not None else 0} records")

if page == "Patient Risk Assessment":
    st.title("🫀 Patient Risk Assessment")
    st.markdown("---")
    
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.subheader("Patient Profile")
        
        tab1, tab2 = st.tabs(["👤 Demographic & Lifestyle", "🩺 Clinical Measurements"])
        
        with tab1:
            r1c1, r1c2 = st.columns(2)
            with r1c1:
                age = st.slider("Age", 0, 100, 50)
                sex = st.radio("Sex", ["Male", "Female"], horizontal=True)
            with r1c2:
                bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=24.5, step=0.1)
                smoking = st.radio("Smoking Status", ["No", "Yes"], horizontal=True)
            
            r2c1, r2c2 = st.columns(2)
            with r2c1:
                exercise = st.selectbox("Physical Activity", ["None", "Light", "Regular"])
            with r2c2:
                family = st.radio("Family History", ["No", "Yes"], horizontal=True)

        with tab2:
            r3c1, r3c2 = st.columns(2)
            with r3c1:
                chol = st.number_input("Cholesterol (mg/dl)", 50, 600, 200)
                bp = st.number_input("Resting BP (mm Hg)", 60, 250, 120)
                fbs_val = st.number_input("Blood Sugar (mg/dl)", 50, 400, 95)
                thalach = st.number_input("Max Heart Rate", 50, 220, 155)
            with r3c2:
                cp_map = {0: "Asymptomatic", 1: "Typical Angina", 2: "Atypical Angina", 3: "Non-anginal Pain"}
                cp_sel = st.selectbox("Chest Pain Evidence", options=[0, 1, 2, 3], format_func=lambda x: cp_map[x])
                oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0, 0.1)
                ca = st.slider("Vessels Colored (CA)", 0, 4, 0)
                
            r4c1, r4c2 = st.columns(2)
            with r4c1:
                restecg_map = {0: "Normal", 1: "ST-T Wave", 2: "LV Hypertrophy"}
                restecg = st.selectbox("Resting ECG", options=[0,1,2], format_func=lambda x: restecg_map[x])
            with r4c2:
                slope_map = {0: "Upsloping", 1: "Flat", 2: "Downsloping"}
                slope = st.selectbox("ST Slope Type", options=[0,1,2], format_func=lambda x: slope_map[x])
        
        st.markdown("---")
        analyze = st.button("RUN CLINICAL DIAGNOSTICS", use_container_width=True, type="primary")
        
    with col2:
        st.subheader("Diagnostic Results")
        if not analyze:
            st.info("👈 Complete the patient profile and click 'RUN CLINICAL DIAGNOSTICS' to begin analysis.")
            
            # Placeholder for UI balance
            st.markdown("""
                <div style='background-color: #1E2129; padding: 40px; border-radius: 10px; border: 1px dashed #444; text-align: center; color: #666;'>
                    Awaiting Patient Data...
                </div>
            """, unsafe_allow_html=True)
        else:
            with st.spinner("Processing Hybrid Intelligence Engine..."):
                patient_dict = {
                    'age': age, 'cholesterol': chol, 'blood_pressure': bp,
                    'smoking': smoking.lower(), 'exercise': exercise.lower(),
                    'bmi': bmi, 'blood_sugar': fbs_val,
                    'chest_pain': 'typical' if cp_sel == 1 else 'none',
                    'max_heart_rate': thalach, 'family_history': family.lower()
                }
                expert_output = run_and_capture_expert(patient_dict)
                
                # ML Pipeline
                ML_FEATURES = ['age', 'sex', 'trestbps', 'chol', 'fbs', 'thalach', 'exang', 'oldpeak', 'ca', 'cp_1', 'cp_2', 'cp_3', 'restecg_0.529296875', 'restecg_1.0', 'restecg_2.0', 'slope_1', 'slope_2', 'thal_1', 'thal_2', 'thal_3']
                ml_in = pd.DataFrame(columns=ML_FEATURES)
                ml_in.loc[0] = 0
                
                num_vals = np.array([[age, bp, chol, thalach, oldpeak]])
                scaled_nums = scaler.transform(num_vals)
                ml_in['age'] = scaled_nums[0][0]
                ml_in['trestbps'] = scaled_nums[0][1]
                ml_in['chol'] = scaled_nums[0][2]
                ml_in['thalach'] = scaled_nums[0][3]
                ml_in['oldpeak'] = scaled_nums[0][4]
                ml_in['sex'] = 1 if sex == "Male" else 0
                ml_in['ca'] = ca
                
                if fbs_val > 120: ml_in['fbs'] = 1

                if exercise == 'None': ml_in['exang'] = 1
                if cp_sel == 1: ml_in['cp_1'] = 1
                elif cp_sel == 2: ml_in['cp_2'] = 1
                elif cp_sel == 3: ml_in['cp_3'] = 1
                if restecg == 1: ml_in['restecg_1.0'] = 1
                elif restecg == 2: ml_in['restecg_2.0'] = 1
                if slope == 1: ml_in['slope_1'] = 1
                elif slope == 2: ml_in['slope_2'] = 1
                
                prob = model.predict_proba(ml_in)[0][1]
                
                # Visual Gauge
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prob * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "ML Predicted Risk Score", 'font': {'size': 24}},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
                        'bar': {'color': "#E74C3C" if prob > 0.6 else "#F39C12" if prob > 0.3 else "#27AE60"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 35], 'color': 'rgba(39, 174, 96, 0.3)'},
                            {'range': [35, 65], 'color': 'rgba(243, 156, 18, 0.3)'},
                            {'range': [65, 100], 'color': 'rgba(231, 76, 60, 0.3)'}],
                    }
                ))
                fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
                st.plotly_chart(fig, use_container_width=True)

                risk_lvl = "HIGH" if prob >= 0.65 else "MEDIUM" if prob >= 0.35 else "LOW"
                b_color = "var(--primary)" if risk_lvl == "HIGH" else "var(--medium)" if risk_lvl == "MEDIUM" else "var(--safe)"
                
                st.markdown(f"""
                    <div style='background-color: {b_color}; padding: 20px; border-radius: 10px; text-align: center; color: white; margin-bottom: 20px;'>
                        <h2 style='margin: 0;'>{risk_lvl} RISK DETECTED</h2>
                        <small>Classification based on multivariate ML generalized patterns</small>
                    </div>
                """, unsafe_allow_html=True)

                with st.expander("📝 Rule-Based Reasoning Analysis", expanded=True):
                    if expert_output.strip() == "":
                        st.write("✅ No high-risk clinical heuristics triggered.")
                    else:
                        st.markdown(expert_output.replace("\n", "\n\n"))
                
                st.markdown("---")
                report_content = f"--- CARDIO GUARD CLINICAL REPORT ---\n\nPatient Details:\n- Age: {age}\n- Sex: {sex}\n- BMI: {bmi}\n- BP: {bp} mmHg\n- Chol: {chol} mg/dl\n\nRisk Assessment:\n- ML Confidence: {prob*100:.1f}%\n- Classification: {risk_lvl}\n\nLinguistic Reasoning:\n{expert_output}"
                st.download_button("📩 Download Professional Report", data=report_content, file_name=f"Report_{age}_{sex}.txt", use_container_width=True)


elif page == "Data Insights":
    st.title("📊 Data Insights")
    if df_clean is not None and df_raw is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Patient Records", len(df_clean))
        c2.metric("Heart Disease Presence", f"{(df_clean['target'].mean()*100):.1f}%")
        c3.metric("Average Patient Age", f"{df_raw['age'].mean():.1f} yrs")
        c4.metric("Avg Cholesterol", f"{df_raw['chol'].mean():.1f} mg/dl")
        
        st.markdown("---")
        
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            fig1 = px.histogram(df_raw.dropna(subset=['target']), x="age", color="target", title="Age Distribution Colored By Target", barmode='overlay')
            st.plotly_chart(fig1, use_container_width=True)
        with r1c2:
            fig2 = px.scatter(df_raw.dropna(subset=['target']), x="trestbps", y="chol", color="target", title="Cholesterol vs Resting BP")
            st.plotly_chart(fig2, use_container_width=True)
        
        r2c1, r2c2 = st.columns(2)
        with r2c1:
            st.write("**Top 10 Feature Correlations (Heatmap)**")
            corr = df_clean.corr()
            top_10 = corr['target'].abs().sort_values(ascending=False).head(10).index
            fig_corr = px.imshow(df_clean[top_10].corr().round(2), text_auto=True, color_continuous_scale='RdBu_r', aspect='auto')
            st.plotly_chart(fig_corr, use_container_width=True)
            
        with r2c2:
            st.write("**Decision Tree Feature Importance**")
            if model is not None:
                imps = pd.DataFrame({'Feature': model.feature_names_in_, 'Importance': model.feature_importances_})
                top_imps = imps.sort_values(by='Importance', ascending=True).tail(10)
                fig_imp = px.bar(top_imps, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='Reds')
                st.plotly_chart(fig_imp, use_container_width=True)
        
        st.write("### Filterable Database View")
        st.dataframe(df_raw, use_container_width=True)
    else:
        st.warning("Data files are not available.")

elif page == "Expert System vs ML":
    st.title("⚖️ Model Architecture Comparison")
    
    if os.path.exists(REPORT_PATH):
        with open(REPORT_PATH, 'r') as f:
            st.markdown(f.read())
    else:
        st.warning("Comparitive report generated by train_model.py is missing.")

    st.info("💡 **Tradeoff Analysis**: The strict Rule-Based system offers 100% interpretability, assuring clinicians exactly why a decision was reached. Conversely, the Machine Learning system exhibits superior multivariate generalization, utilizing non-linear associations the human eye misses, maximizing predictive power at the cost of strict interpretability.")
    
    if df_clean is not None and model is not None:
        X = df_clean.drop(columns=['target'])
        y = df_clean['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        y_ml = model.predict(X_test)
        
        df_raw_aligned = df_raw.dropna(subset=['target']).reset_index(drop=True)
        raw_test = df_raw_aligned.iloc[X_test.index]
        y_rb = raw_test.apply(rule_based_predict, axis=1)
        
        perf_df = pd.DataFrame({
            'System': ['ML', 'ML', 'ML', 'ML', 'Rule-Based', 'Rule-Based', 'Rule-Based', 'Rule-Based'],
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1', 'Accuracy', 'Precision', 'Recall', 'F1'],
            'Score': [
                accuracy_score(y_test, y_ml), precision_score(y_test, y_ml), recall_score(y_test, y_ml), f1_score(y_test, y_ml),
                accuracy_score(y_test, y_rb), precision_score(y_test, y_rb, zero_division=0), recall_score(y_test, y_rb), f1_score(y_test, y_rb)
            ]
        })
        fig_bar = px.bar(perf_df, x='Metric', y='Score', color='System', barmode='group', title="Testing Holdout Diagnostics (80/20 Split)")
        st.plotly_chart(fig_bar, use_container_width=True)
        
        st.write("### Decision Tree Structure Overview")
        with st.expander("Expand to View Root Nodes"):
            fig, ax = plt.subplots(figsize=(18, 8))
            plot_tree(model, feature_names=list(model.feature_names_in_), class_names=['Healthy', 'Disease'], filled=True, ax=ax, max_depth=3, fontsize=10, rounded=True)
            st.pyplot(fig)
    else:
        st.error("Missing components to render analytical models.")

elif page == "About":
    st.title("ℹ️ About This Project")
    st.markdown("""
    ### Hybrid Intelligence Heart Disease Assessment Ecosystem
    
    Seamlessly bridging diagnostic explainability and multivariate predictive power, this dashboard combines two disparate schools of AI into a modern clinic setting.
    
    **Capabilities:**
    - Live inference over real-time patient streams.
    - Expert rule analysis detailing exact physiological thresholds crossed.
    - Deep feature analysis evaluating the entire cohort pipeline.
    
    **Technology Stack:**
    - **Frontend:** Streamlit, Plotly, HTML/CSS Web Components
    - **Modeling Core:** Scikit-Learn
    - **State Engine:** Experta (Rule Engine)
    - **Data Ops:** Pandas, NumPy

    **Project Team & Responsibilities:**
    - **Mina**: Data Preprocessing & Visualization Orchestration
    - **Nance**: Knowledge Engineering & Experta Logic Implementation
    - **Marsel**: Machine Learning Architecture & Performance Comparison
    """)
