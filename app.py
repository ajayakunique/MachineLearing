import streamlit as st
import pandas as pd
import pickle

# ---------------------------
# Load trained components
# ---------------------------
LR = pickle.load(open("heart_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
label_encoders = pickle.load(open("label_encoders.pkl", "rb"))

st.set_page_config(page_title="Heart Disease Prediction 💓", page_icon="❤️")
st.title("💓 Heart Disease Prediction App")
st.write("Enter patient details below to predict the likelihood of heart disease.")

# ---------------------------
# Define input fields (matching your dataset columns)
# ---------------------------

age = st.number_input("Age", 1, 120, 50)
sex = st.selectbox("Sex", ["male", "female"])
cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (trestbps)", 50, 250, 120)
chol = st.number_input("Cholesterol (chol)", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])
restecg = st.selectbox("Resting ECG Results (restecg)", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate (thalach)", 60, 250, 150)
exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1])
oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 10.0, 1.0, step=0.1)
slope = st.selectbox("Slope of Peak Exercise ST Segment (slope)", [0, 1, 2])
ca = st.number_input("Number of Major Vessels (ca)", 0, 4, 0)
thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])

# ---------------------------
# Prediction Logic
# ---------------------------
if st.button("🔍 Predict"):

    # ✅ Create DataFrame with all expected columns
    input_data = pd.DataFrame([{
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
    }])

    # ✅ Apply label encoders (if any categorical columns exist)
    for col, le in label_encoders.items():
        if col in input_data.columns:
            input_data[col] = le.transform(input_data[col])

    # ✅ Reorder and fill any missing columns to match training features
    input_data = input_data.reindex(columns=scaler.feature_names_in_, fill_value=0)

    # ✅ Apply scaling
    input_scaled = scaler.transform(input_data)

    # ✅ Predict
    prediction = LR.predict(input_scaled)[0]

    # ✅ Display result
    if prediction == 1:
        st.error("⚠️ The person may have heart disease.")
    else:
        st.success("✅ The person is unlikely to have heart disease.")

st.markdown("---")
st.caption("Developed by Ajay | Powered by Logistic Regression ❤️")
