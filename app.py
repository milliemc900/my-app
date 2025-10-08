# diabetes_app.py
# 🌿 Diabetes Prediction App (with Password Access)

import streamlit as st
import pandas as pd
import joblib
import os
import datetime

# ---------- CONFIG ----------
st.set_page_config(page_title="Diabetes Prediction", page_icon="🌿", layout="wide")

# ---------- PASSWORD PROTECTION ----------
PASSWORD = "Millicent123"  # 🔐 Set your password here

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔒 Secure Login")
    password_input = st.text_input("Enter Password to Access the App:", type="password")
    if st.button("Login"):
        if password_input == PASSWORD:
            st.session_state.authenticated = True
            st.success("✅ Access Granted! Welcome.")
            st.rerun()
        else:
            st.error("❌ Incorrect Password. Please try again.")
    st.stop()

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    model_path = "RandomForest_model.pkl"  # Ensure model file is uploaded
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found at: {model_path}. Please upload it to your repository.")
        st.stop()
    return joblib.load(model_path)

model = load_model()

# ---------- HEADER ----------
st.title("🌿 Diabetes Risk Prediction System")
st.markdown("""
This AI-powered system predicts the **risk of diabetes** based on key patient indicators.  
It supports clinicians and individuals in early screening and prevention planning.
""")

# ---------- INPUT FORM ----------
st.subheader("📋 Enter Patient Details")

with st.form("patient_form"):
    col1, col2, col3 = st.columns(3)

    # --- Column 1 ---
    with col1:
        age = st.number_input("Age (years)", min_value=1.0, max_value=120.0, value=40.0)
        gender = st.selectbox("Gender", ["M", "F"])
        weight = st.number_input("Weight (kg)", min_value=20.0, max_value=250.0, value=70.0)

    # --- Column 2 ---
    with col2:
        bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=25.0)
        systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=70, max_value=250, value=120)
        diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=150, value=80)
        blood_sugar = st.number_input("Blood Sugar (mmol/L)", min_value=2.0, max_value=30.0, value=8.0)

    # --- Column 3 ---
    with col3:
        hypertension = st.selectbox("Hypertension (1=Yes, 0=No)", [0, 1])
        both_dm_htn = st.selectbox("Both DM + HTN (1=Yes, 0=No)", [0.0, 1.0])

        # ✅ Treatment combinations available
        treatment = st.selectbox(
            "Treatment Combination",
            [
                'ab', 'abe', 'ae', 'ade', 'e', 'ad',
                'aec', 'ace', 'ce', 'ebe', 'aw', 'ac', 'a'
            ]
        )

    submitted = st.form_submit_button("🔍 Predict Diabetes Risk")

# ---------- PREDICTION ----------
if submitted:
    try:
        bp_combined = f"{systolic_bp}/{diastolic_bp}"

        # Prepare input for prediction
        input_data = pd.DataFrame({
            'AGE': [age],
            'GENDER': [1 if gender == "M" else 0],
            'WEIGHT(kg)': [weight],
            'BMI': [bmi],
            'BP(mmHg)': [bp_combined],
            'BLOOD SUGAR(mmol/L)': [blood_sugar],
            'HTN': [int(hypertension)],
            'BOTH DM+HTN': [float(both_dm_htn)],
            'TREATMENT': [treatment]
        })

        # Match columns to model
        if hasattr(model, "feature_names_in_"):
            model_features = model.feature_names_in_
            for col in model_features:
                if col not in input_data.columns:
                    input_data[col] = 0
            input_data = input_data[model_features]

        # Make prediction
        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1] if hasattr(model, "predict_proba") else 0.5

        # Interpret risk
        if prob < 0.33:
            risk_level = "🟢 Low Risk"
            message = "Your predicted diabetes risk is **low**. Maintain healthy habits and regular exercise."
        elif prob < 0.66:
            risk_level = "🟠 Moderate Risk"
            message = "Your predicted diabetes risk is **moderate**. Consider dietary adjustments and regular screening."
        else:
            risk_level = "🔴 High Risk"
            message = "Your predicted diabetes risk is **high**. Consult a healthcare provider for evaluation."

        # ---------- DISPLAY RESULTS ----------
        st.markdown("## 🧠 Prediction Results")
        st.success(f"**Predicted Status:** {'Diabetic' if pred == 1 else 'Non-Diabetic'}")
        st.write(f"**Probability of Diabetes:** {prob:.2f}")
        st.info(f"**Risk Level:** {risk_level}")
        st.markdown(f"### 💬 Interpretation\n{message}")

        # ---------- SAVE PREDICTION LOG ----------
        record = {
            "Timestamp": [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            "Age": [age],
            "Gender": [gender],
            "Weight": [weight],
            "BMI": [bmi],
            "Systolic_BP": [systolic_bp],
            "Diastolic_BP": [diastolic_bp],
            "Blood_Sugar": [blood_sugar],
            "Hypertension": [hypertension],
            "Treatment": [treatment],
            "Probability": [prob],
            "Risk_Level": [risk_level]
        }

        record_df = pd.DataFrame(record)
        if not os.path.exists("prediction_logs.csv"):
            record_df.to_csv("prediction_logs.csv", index=False)
        else:
            record_df.to_csv("prediction_logs.csv", mode="a", header=False, index=False)

        st.success("✅ Prediction saved to log (prediction_logs.csv)")

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")

# ---------- FOOTER ----------
st.markdown("---")
st.caption("Developed by Millicent Chesang | AI & Data Analytics for Public Health")
