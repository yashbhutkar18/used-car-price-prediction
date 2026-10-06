import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Used Car Price Predictor", page_icon="🚗", layout="wide")
st.title("🚗 Used Car Price Prediction Engine")

MODEL_PATH = os.path.join("model", "car_price_model.pkl")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

model = load_model()
if model is None:
    st.error("Model not found! Run train.py first.")
    st.stop()

st.sidebar.header("Car Details")
year = st.sidebar.slider("Year", 2010, 2024, 2018)
km_driven = st.sidebar.number_input("Kilometers Driven", 1000, 250000, 45000, step=1000)
fuel_type = st.sidebar.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
transmission = st.sidebar.selectbox("Transmission", ["Manual", "Automatic"])
engine_cc = st.sidebar.select_slider("Engine Capacity (CC)", [1000, 1200, 1500, 2000, 2500], value=1200)
mileage_kmpl = st.sidebar.slider("Mileage (kmpl)", 10.0, 28.0, 18.5, 0.1)

input_df = pd.DataFrame([{
    "Year": year,
    "Kilometers_Driven": km_driven,
    "Fuel_Type": fuel_type,
    "Transmission": transmission,
    "Engine_CC": engine_cc,
    "Mileage_kmpl": mileage_kmpl
}])

col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("Selected Features")
    st.dataframe(input_df, use_container_width=True)

with col2:
    st.subheader("Estimated Market Price")
    if st.button("Predict Price", type="primary", use_container_width=True):
        pred = max(0.5, round(model.predict(input_df)[0], 2))
        st.success(f"### Valuation: ₹ {pred:.2f} Lakhs")
