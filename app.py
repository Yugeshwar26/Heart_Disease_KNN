import streamlit as st
import pickle
import numpy as np

# 1. Page Configuration
st.set_page_config(page_title="Heart Disease Predictor")

st.title("Heart Disease Prediction App")
st.write("This app uses a Machine Learning (KNN) model to predict the likelihood of heart disease.")

# 2. Load the Model
try:
    with open('heart_disease_knn.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'heart_disease_knn.pkl' is in the repository.")
    st.stop()

# 3. Input Fields (13 Features)
# Grouping features for a better layout
st.subheader("Patient Information")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 20, 100, 50)
    sex = st.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], help="0: Typical Angina, 1: Atypical Angina, 2: Non-anginal Pain, 3: Asymptomatic")
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [1, 0], format_func=lambda x: "True" if x == 1 else "False")

with col2:
    restecg = st.selectbox("Resting ECG Results", [0, 1, 2], help="0: Normal, 1: ST-T Wave Abnormality, 2: Left Ventricular Hypertrophy")
    thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 10.0, 1.0, step=0.1)
    slope = st.selectbox("Slope of the Peak Exercise ST Segment", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", [1, 2, 3], format_func=lambda x: {1: "Normal", 2: "Fixed Defect", 3: "Reversable Defect"}.get(x, str(x)))

# 4. Prediction Logic
if st.button("Predict Risk"):
    # Create the feature array in the correct order
    features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    
    # The pipeline will automatically scale the input and predict
    prediction = model.predict(features)
    
    st.divider()
    
    if prediction[0] == 1:
        st.error("⚠️ **Prediction:** Heart Disease Detected")
        st.write("The model predicts a high likelihood of heart disease. Please consult a medical professional.")
    else:
        st.success("✅ **Prediction:** No Heart Disease Detected")
        st.write("The model predicts a low likelihood of heart disease.")
