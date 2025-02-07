import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load trained models
def load_models():
    models = {
        "DVT_nc": joblib.load("DVT_nc_gbc_model.joblib"),
        "surgical_site_infection": joblib.load("surgical_site_infection_gbc_model.joblib"),
        "syst_major_complications": joblib.load("syst_major_complications_gbc_model.joblib")
    }
    return models

def predict(models, input_data):
    probabilities = {target: models[target].predict_proba(input_data)[0, 1] for target in models}
    return probabilities

# Preprocessing function
def preprocess_input(age, asa_n, anesthesia_nc, smoking, hypertension, dialysis, steroids, diabetes):
    asa_mapping = {"ASA I": 0, "ASA II": 1, "ASA III": 2, "ASA IV": 3}
    anesthesia_mapping = {"General": 1, "Regional": 2, "Spinal": 3, "MAC": 4}
    
    processed_data = pd.DataFrame({
        "Age": [age],
        "ASA_n": [asa_mapping[asa_n]],
        "anesthesia_nc": [anesthesia_mapping[anesthesia_nc]],
        "Smoke_n": [1 if smoking == "Yes" else 0],
        "Hypertension_n": [1 if hypertension == "Yes" else 0],
        "Dialysis_n": [1 if dialysis == "Yes" else 0],
        "Chornic_Steroid_n": [1 if steroids == "Yes" else 0],
        "Diabetes_nc": [1 if diabetes == "Yes" else 0]
    })
    return processed_data

# Streamlit App
st.set_page_config(page_title="Medical Risk Predictor", page_icon="⚕️", layout="centered")
st.title("🩺 Medical Risk Predictor")
st.write("Predict the probabilities of post-surgical complications based on patient data.")

# User Inputs
st.sidebar.header("🔧 Input Features")
age = st.sidebar.slider("🎂 Age", 18, 100, 50)
asa_n = st.sidebar.selectbox("⚕️ ASA Score", ["ASA I", "ASA II", "ASA III", "ASA IV"], index=1)
anesthesia_nc = st.sidebar.selectbox("💉 Anesthesia Type", ["General", "Regional", "Spinal", "MAC"], index=0)
smoking = st.sidebar.selectbox("🚬 Smoking Status", ["No", "Yes"], index=0)
hypertension = st.sidebar.selectbox("❤️ Hypertension", ["No", "Yes"], index=0)
dialysis = st.sidebar.selectbox("🩸 On Dialysis?", ["No", "Yes"], index=0)
steroids = st.sidebar.selectbox("💊 Chronic Steroid Use?", ["No", "Yes"], index=0)
diabetes = st.sidebar.selectbox("🍬 Diabetes", ["No", "Yes"], index=0)

# Load models
models = load_models()

# Process input data
input_data = preprocess_input(age, asa_n, anesthesia_nc, smoking, hypertension, dialysis, steroids, diabetes)

# Predict probabilities
if st.button("📊 Predict Complication Risks"):
    probabilities = predict(models, input_data)
    st.success("✅ Predictions Generated!")
    
    # Display probabilities
    st.subheader("Predicted Probabilities")
    st.write(f"**DVT Risk:** {probabilities['DVT_nc']:.4f}")
    st.write(f"**Surgical Site Infection Risk:** {probabilities['surgical_site_infection']:.4f}")
    st.write(f"**Major Complication Risk:** {probabilities['syst_major_complications']:.4f}")
    
    # Show last probability component
    last_key = list(probabilities.keys())[-1]
    st.write(f"**Final Probability Component ({last_key}):** {probabilities[last_key]:.4f}")

st.write("\n🔬 Built with Machine Learning to assist medical professionals in risk assessment.")
