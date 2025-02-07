import streamlit as st
from model import load_models, predict
from preprocessing import preprocess_input

# Load trained models
models = load_models()

# Streamlit App UI
st.set_page_config(page_title="Medical Risk Predictor", page_icon="⚕️", layout="centered")
st.title("🩺 Medical Risk Predictor")
st.write("Predict the probabilities of post-surgical complications based on patient data.")

# Sidebar Inputs
st.sidebar.header("🔧 Input Features")
age = st.sidebar.slider("🎂 Age", 18, 100, 50)
asa_n = st.sidebar.selectbox("⚕️ ASA Score", ["ASA I", "ASA II", "ASA III", "ASA IV"], index=1)
anesthesia_nc = st.sidebar.selectbox("💉 Anesthesia Type", ["General", "Regional", "Spinal", "MAC"], index=0)
smoking = st.sidebar.selectbox("🚬 Smoking Status", ["No", "Yes"], index=0)
hypertension = st.sidebar.selectbox("❤️ Hypertension", ["No", "Yes"], index=0)
dialysis = st.sidebar.selectbox("🩸 On Dialysis?", ["No", "Yes"], index=0)
steroids = st.sidebar.selectbox("💊 Chronic Steroid Use?", ["No", "Yes"], index=0)
diabetes = st.sidebar.selectbox("🍬 Diabetes", ["No", "Yes"], index=0)

# Preprocess user input
input_data = preprocess_input(age, asa_n, anesthesia_nc, smoking, hypertension, dialysis, steroids, diabetes)

# Predict and display results
if st.button("📊 Predict Complication Risks"):
    probabilities = predict(models, input_data)
    st.success("✅ Predictions Generated!")
    
    st.subheader("Predicted Probabilities")
    st.write(f"**DVT Risk:** {probabilities['DVT_nc']:.4f}")
    st.write(f"**Surgical Site Infection Risk:** {probabilities['surgical_site_infection']:.4f}")
    st.write(f"**Major Complication Risk:** {probabilities['syst_major_complications']:.4f}")

st.write("🔬 Built with Machine Learning to assist medical professionals in risk assessment.")
