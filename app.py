import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ---------------------------
# 1️⃣ Function to train and save model (only runs if pkl files not found)
# ---------------------------
def train_and_save_model():
    st.info("Training model... Please wait ⏳")

    # Load dataset
    data_df = pd.read_csv('heart.csv')

    # Remove duplicates
    data_df.drop_duplicates(inplace=True)

    # Handle missing values
    num_cols = data_df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = data_df.select_dtypes(include=['object']).columns

    data_df[num_cols] = data_df[num_cols].fillna(data_df[num_cols].mean())
    for col in cat_cols:
        data_df[col] = data_df[col].fillna(data_df[col].mode()[0])

    # Label encode categorical columns
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        data_df[col] = le.fit_transform(data_df[col])
        label_encoders[col] = le

    # Split features and target
    x = data_df.drop('target', axis=1)
    y = data_df['target']

    # Feature scaling
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(
        x_scaled, y, test_size=0.2, random_state=30
    )

    # Train Logistic Regression model
    LR = LogisticRegression(solver='liblinear')
    LR.fit(x_train, y_train)

    # Evaluate
    y_pred = LR.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Save model components
    pickle.dump(LR, open("heart_model.pkl", "wb"))
    pickle.dump(scaler, open("scaler.pkl", "wb"))
    pickle.dump(label_encoders, open("label_encoders.pkl", "wb"))

    st.success(f"✅ Model trained and saved successfully! Accuracy: {accuracy * 100:.2f}%")
    return LR, scaler, label_encoders


# ---------------------------
# 2️⃣ Load model or train new one
# ---------------------------
if (
    os.path.exists("heart_model.pkl")
    and os.path.exists("scaler.pkl")
    and os.path.exists("label_encoders.pkl")
):
    LR = pickle.load(open("heart_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    label_encoders = pickle.load(open("label_encoders.pkl", "rb"))
else:
    LR, scaler, label_encoders = train_and_save_model()

# ---------------------------
# 3️⃣ Streamlit UI
# ---------------------------
st.set_page_config(page_title="Heart Disease Prediction 💓", page_icon="❤️")
st.title("💓 Heart Disease Prediction App")
st.write("Enter patient details below to predict the likelihood of heart disease.")

# Input fields
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
# 4️⃣ Predict button
# ---------------------------
if st.button("🔍 Predict"):
    # Create input DataFrame
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

    # Apply label encoders if needed
    for col, le in label_encoders.items():
        if col in input_data.columns:
            input_data[col] = le.transform(input_data[col])

    # Reorder columns
    input_data = input_data.reindex(columns=scaler.feature_names_in_, fill_value=0)

    # Scale and predict
    input_scaled = scaler.transform(input_data)
    prediction = LR.predict(input_scaled)[0]

    if prediction == 1:
        st.error("⚠️ The person may have heart disease.")
    else:
        st.success("✅ The person is unlikely to have heart disease.")

st.markdown("---")
st.caption("Developed by Ajay | Powered by Logistic Regression ❤️")
