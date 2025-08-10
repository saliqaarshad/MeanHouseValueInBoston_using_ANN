import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
import numpy as np
import json

# Load model
model = load_model('model.keras')

st.title("🏠 Boston Housing Price Predictor")
st.write("Enter the property details to estimate its price.")

# Feature input fields
CRIM = st.number_input("Per capita crime rate (CRIM)", min_value=0.0)
ZN = st.number_input("Proportion of residential land zoned for lots over 25,000 sq.ft. (ZN)", min_value=0.0)
INDUS = st.number_input("Proportion of non-retail business acres per town (INDUS)", min_value=0.0)
CHAS = st.selectbox("Charles River dummy variable (CHAS)", [0, 1])
NOX = st.number_input("Nitric oxides concentration (NOX)", min_value=0.0)
RM = st.number_input("Average number of rooms per dwelling (RM)", min_value=0.0)
AGE = st.number_input("Proportion of owner-occupied units built prior to 1940 (AGE)", min_value=0.0)
DIS = st.number_input("Weighted distances to employment centres (DIS)", min_value=0.0)
RAD = st.number_input("Index of accessibility to radial highways (RAD)", min_value=0.0)
TAX = st.number_input("Full-value property tax rate per $10,000 (TAX)", min_value=0.0)
PTRATIO = st.number_input("Pupil-teacher ratio by town (PTRATIO)", min_value=0.0)
B = st.number_input("1000(Bk - 0.63)^2 where Bk is proportion of blacks by town (B)", min_value=0.0)
LSTAT = st.number_input("% lower status of the population (LSTAT)", min_value=0.0)

# Prepare input data
input_data = np.array([[CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT]])

# Predict
if st.button("Predict Price"):
    prediction = model.predict(input_data)
    st.success(f"🏡 Estimated Price: ${prediction[0][0]:,.2f}k")
