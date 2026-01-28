# ===============================
# 1. Import required libraries
# ===============================
import streamlit as st
import pickle
import pandas as pd

# ===============================
# 2. App title
# ===============================
st.title("❤️ Heart Disease Prediction App")
st.write("This app predicts heart disease and suggests lifestyle improvements.")

# ===============================
# 3. Load saved ML model
# ===============================
# The model was saved from HeartDisease.ipynb
with open("heart_model.pkl", "rb") as file:
    model, columns = pickle.load(file)

# ===============================
# 4. Take medical inputs from user
# ===============================
st.header("🩺 Medical Information")

age = st.number_input("Age", 18, 100, 45)
sex = st.selectbox("Sex", ["Female", "Male"])
cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Cholesterol Level", 100, 400, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG Result (0–2)", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("Oldpeak Value", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope (0–2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0–4)", [0, 1, 2, 3, 4])
thal = st.selectbox("Thal (1–3)", [1, 2, 3])

# Convert sex to numeric
sex = 1 if sex == "Male" else 0

# ===============================
# 5. Take lifestyle inputs
# ===============================
st.header("🏃 Lifestyle Information")

smoking = st.selectbox("Do you smoke?", ["No", "Yes"])
activity = st.selectbox("Physical Activity Level", ["Low", "Medium", "High"])
sleep = st.slider("Sleep Hours per Day", 3, 10, 7)
stress = st.selectbox("Stress Level", ["Low", "Medium", "High"])
diet = st.selectbox("Diet Type", ["Healthy", "Mixed", "Junk Food"])

# ===============================
# 6. Predict button
# ===============================
if st.button("🔍 Predict Heart Disease"):

    # Put user input into dictionary
    input_data = {
        "age": age,
        "sex": sex,
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

    # Convert dictionary to DataFrame
    input_df = pd.DataFrame([input_data], columns=columns)

    # ===============================
    # 7. Make prediction
    # ===============================
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1] * 100

    # ===============================
    # 8. Show result
    # ===============================
    if prediction == 1:
        st.error(f"⚠️ High risk of heart disease ({probability:.2f}%)")
    else:
        st.success(f"✅ Low risk of heart disease ({probability:.2f}%)")

    # ===============================
    # 9. Lifestyle recommendations
    # ===============================
    st.subheader("💡 Lifestyle Recommendations")

    if smoking == "Yes":
        st.write("🚭 Quit smoking to reduce heart disease risk.")

    if activity == "Low":
        st.write("🏃 Do at least 30 minutes of exercise daily.")

    if sleep < 6:
        st.write("😴 Improve sleep to 7–8 hours per night.")

    if stress == "High":
        st.write("🧘 Reduce stress with yoga or meditation.")

    if diet == "Junk Food":
        st.write("🥗 Reduce junk food and eat healthy food.")

    if (smoking == "No" and activity != "Low" and sleep >= 6 
        and stress != "High" and diet != "Junk Food"):
        st.write("✅ Your lifestyle habits are good. Keep it up!")
