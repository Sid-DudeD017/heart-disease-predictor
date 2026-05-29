import streamlit as st
import pandas as pd
import joblib

# 1. Load the trained model
model = joblib.load('pipe.pkl')

# 2. Build the User Interface
st.title("Heart Disease Prediction System")
st.write("Enter patient data below to assess risk.")

# Create input fields for the user
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex", ["M", "F"])
    chest_pain = st.selectbox("Chest Pain Type", ["ASY", "NAP", "ATA", "TA"])
    exang = st.selectbox("Exercise Angina (Y/N)", ["Y", "N"])

with col2:
    trestbps = st.number_input("Resting Blood Pressure", min_value=50, max_value=250, value=120)
    thalach = st.number_input("Maximum Heart Rate", min_value=60, max_value=220, value=150)
    restecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])

# 3. The Prediction Logic
if st.button("Calculate Risk"):
    # Pack the user inputs into a DataFrame matching your original training data
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'chest pain': [chest_pain],
        'trestbps': [trestbps],
        'thalach': [thalach],
        'restecg': [restecg],
        'exang': [exang]
    })
    
    # Run the data through your pipeline
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] * 100
    
    # Display the results
    st.markdown("---")
    if prediction == 1:
        st.error(f"High Risk Detected! (Confidence: {probability:.1f}%)")
    else:
        st.success(f"Low Risk. Patient appears healthy. (Risk: {probability:.1f}%)")