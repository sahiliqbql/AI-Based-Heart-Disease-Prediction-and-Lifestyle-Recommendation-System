import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os

# ===============================
# Page config
# ===============================
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Heart Disease Prediction System")
st.write("Predict heart disease risk and get lifestyle recommendations")

# ===============================
# Load model
# ===============================
@st.cache_resource
def load_model():
    model_file = "heart_model.pkl"
    if os.path.exists(model_file):
        try:
            with open(model_file, "rb") as f:
                data = pickle.load(f)
            # Ensure it's a tuple with model and columns
            if isinstance(data, tuple) and len(data) == 2:
                model, columns = data
                return model, columns
            else:
                st.error("The model file is not in the expected format.")
                return None, None
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None, None
    else:
        st.error("Model file 'heart_model.pkl' not found.")
        return None, None

model, columns = load_model()

if model is None:
    st.stop()  # Stop the app if the model can't be loaded

# ===============================
# Sidebar – Lifestyle Inputs
# ===============================
st.sidebar.header("🧠 Lifestyle Information")

smoking = st.sidebar.selectbox("Smoking", ["No", "Yes"])
physical_activity = st.sidebar.selectbox("Physical Activity", ["Low", "Medium", "High"])
sleep_hours = st.sidebar.slider("Sleep Hours", 3, 10, 7)
stress = st.sidebar.selectbox("Stress Level", ["Low", "Medium", "High"])
diet = st.sidebar.selectbox("Diet Type", ["Healthy", "Mixed", "Junk Food"])

# ===============================
# Main Inputs (Medical)
# ===============================
age = st.number_input("Age", 18, 100, 45)
sex = st.selectbox("Sex", ["Female", "Male"])
cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 400, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Rest ECG (0–2)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope (0–2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0–4)", [0, 1, 2, 3, 4])
thal = st.selectbox("Thal (1–3)", [1, 2, 3])

# ===============================
# Convert inputs to model format
# ===============================
input_data = {
    "age": age,
    "sex": 1 if sex == "Male" else 0,
    "cp": cp,
    "trestbps": trestbps,
    "chol": chol,
    "fbs": fbs,
    "restecg": restecg,
    "thalach": thalach,
    "exang": exang,
    "oldpeak": oldpeak,
    "slope": slope,
    "ca": ca,
    "thal": thal
}

input_df = pd.DataFrame([input_data], columns=columns)

# ===============================
# Prediction Button
# ===============================
if st.button("🔍 Predict"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1] * 100

    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Disease ({probability:.2f}%)")
    else:
        st.success(f"✅ Low Risk of Heart Disease ({probability:.2f}%)")

    # ===============================
    # Lifestyle Recommendations
    # ===============================
    st.subheader("💡 Lifestyle Recommendations")

    recommendations = []

    if smoking == "Yes":
        recommendations.append("🚭 Quit smoking immediately.")

    if physical_activity == "Low":
        recommendations.append("🏃 Exercise at least 30 minutes daily.")

    if sleep_hours < 6:
        recommendations.append("😴 Improve sleep to 7–8 hours.")

    if stress == "High":
        recommendations.append("🧘 Reduce stress with yoga or meditation.")

    if diet == "Junk Food":
        recommendations.append("🥗 Reduce junk food and eat healthy meals.")

    if not recommendations:
        recommendations.append("✅ Your lifestyle habits are good. Keep it up!")

    for rec in recommendations:
        st.write("•", rec)
