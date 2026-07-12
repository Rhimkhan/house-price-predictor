import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(page_title="House Price Predictor", page_icon="??", layout="wide")
st.title("?? House Price Prediction")
st.markdown("Enter house details to get estimated price")

@st.cache_resource
def load_model():
    try:
        with open('model.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        return None

model = load_model()

if model is None:
    st.error("?? Model file not found!")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    st.subheader("?? Property Details")
    lot_area = st.number_input("Lot Area (sq ft)", 1000, 50000, 10000)
    overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 6)
    year_built = st.number_input("Year Built", 1900, 2025, 2000)
    total_bsmt_sf = st.number_input("Total Basement SF", 0, 5000, 800)
    washroom = st.number_input("Washroom", 0, 10, 2)
    bathroom = st.number_input("Bathroom", 0, 5, 1)

with col2:
    st.subheader("??? Structure Details")
    first_flr_sf = st.number_input("1st Floor SF", 0, 5000, 1000)
    second_flr_sf = st.number_input("2nd Floor SF", 0, 5000, 0)
    garage_area = st.number_input("Garage Area (sq ft)", 0, 2000, 400)
    bedrooms = st.number_input("Bedrooms", 0, 10, 3)
    fireplaces = st.number_input("Fireplaces", 0, 5, 0)

if st.button("?? Predict Price", type="primary"):
    if model:
        try:
            features = [[lot_area, overall_qual, year_built, total_bsmt_sf, first_flr_sf, second_flr_sf, garage_area, bedrooms, washroom, bathroom, fireplaces]]
            price = model.predict(features)[0]
            st.success("? Prediction Complete!")
            st.markdown(f"### ?? Price: ? {price:,.2f}")
            st.subheader("?? View in:")
            c1, c2 = st.columns(2)
            if c1.button("???? INR"):
                st.info(f"? {price:,.2f}")
            if c2.button("?? USD"):
                st.info(f"$ {price * 0.012:,.2f}")
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.error("? Model not loaded!")
