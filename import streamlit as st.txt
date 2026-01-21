import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd

model = tf.keras.models.load_model("patient_churn_cnn.h5")

st.title("🏥 Patient Churn Prediction")

age = st.number_input("Age", 0, 100)
gender = st.selectbox("Gender (0=Female,1=Male)", [0,1])
visits = st.number_input("Visits", 0)
charges = st.number_input("Charges")
insurance = st.selectbox("Insurance (0=No,1=Yes)", [0,1])
chronic = st.selectbox("Chronic (0=No,1=Yes)", [0,1])

if st.button("Predict"):
    X = np.array([[age, gender, visits, charges, insurance, chronic]])
    X = (X - X.mean()) / (X.std() + 1e-6)
    X = X[..., np.newaxis]
    pred = model.predict(X)[0][0]
    st.success("Churn" if pred > 0.5 else "No Churn")
