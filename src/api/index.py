import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 House Price Prediction")
st.markdown("Predict house prices based on various features")

@st.cache_resource
def load_model():
    model_path = 'model.pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    return None

model = load_model()

if model is None:
    st.error("⚠️ Model file not found! Please train the model first.")
    st.stop()

st.sidebar.header("📊 Input Features")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📍 Location Features")
    area = st.number_input("Area (sq ft)", min_value=300, max_value=10000, value=1500, step=50)
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3, step=1)
    washroom = st.number_input("Washroom (Full Bathrooms)", min_value=0, max_value=10, value=2, step=1)
    bathroom = st.number_input("Bathroom (Half Bathrooms)", min_value=0, max_value=5, value=1, step=1)

with col2:
    st.subheader("🏗️ Property Features")
    stories = st.number_input("Stories", min_value=1, max_value=5, value=1, step=1)
    parking = st.number_input("Parking Spaces", min_value=0, max_value=10, value=1, step=1)
    year_built = st.number_input("Year Built", min_value=1900, max_value=2025, value=2010, step=1)

st.sidebar.subheader("🏡 Additional Features")

col3, col4 = st.columns(2)

with col3:
    main_road = st.selectbox("Main Road Access", ["No", "Yes"])
    guest_room = st.selectbox("Guest Room", ["No", "Yes"])
    basement = st.selectbox("Basement", ["No", "Yes"])

with col4:
    hot_water_heating = st.selectbox("Hot Water Heating", ["No", "Yes"])
    air_conditioning = st.selectbox("Air Conditioning", ["No", "Yes"])
    pref_area = st.selectbox("Preferred Area", ["No", "Yes"])

furnishing = st.sidebar.selectbox("Furnishing Status", ["Unfurnished", "Semi-Furnished", "Fully Furnished"])

# CURRENCY SELECTOR
st.sidebar.markdown("---")
st.sidebar.subheader("💱 Currency Settings")

currency = st.sidebar.radio(
    "Select Currency",
    ["₹ INR (Indian Rupees)", "$ USD (US Dollars)"],
    index=0
)

if currency == "₹ INR (Indian Rupees)":
    currency_symbol = "₹"
    conversion_rate = 1.0
    currency_code = "INR"
else:
    currency_symbol = "$"
    conversion_rate = 0.012
    currency_code = "USD"

predict_btn = st.sidebar.button("🔮 Predict Price", type="primary", use_container_width=True)

st.markdown("---")

if predict_btn:
    with st.spinner("Calculating price..."):
        try:
            features = [
                area, bedrooms, washroom, bathroom, stories, parking, year_built,
                1 if main_road == "Yes" else 0,
                1 if guest_room == "Yes" else 0,
                1 if basement == "Yes" else 0,
                1 if hot_water_heating == "Yes" else 0,
                1 if air_conditioning == "Yes" else 0,
                1 if pref_area == "Yes" else 0,
                0 if furnishing == "Unfurnished" else (1 if furnishing == "Semi-Furnished" else 2)
            ]
            features_array = np.array(features).reshape(1, -1)
            prediction_inr = model.predict(features_array)[0]
            predicted_price = prediction_inr * conversion_rate
            
            st.success("✅ Prediction Complete!")
            
            col_result1, col_result2, col_result3 = st.columns([1, 2, 1])
            with col_result2:
                st.markdown(f"""
                <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; color: white;">
                    <h2 style="margin: 0; font-size: 1.2rem;">Predicted Price</h2>
                    <h1 style="margin: 10px 0; font-size: 3.5rem;">{currency_symbol} {predicted_price:,.2f}</h1>
                    <p style="margin: 0; opacity: 0.8;">Based on your selected features</p>
                    <p style="margin: 5px 0 0 0; opacity: 0.6; font-size: 0.9rem;">Currency: {currency_code}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with st.expander("📋 Input Summary"):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.write("**📍 Location Features**")
                    st.write(f"- Area: {area} sq ft")
                    st.write(f"- Bedrooms: {bedrooms}")
                    st.write(f"- Washroom (Full): {washroom}")
                    st.write(f"- Bathroom (Half): {bathroom}")
                    st.write(f"- Stories: {stories}")
                    st.write(f"- Year Built: {year_built}")
                with col_b:
                    st.write("**🏗️ Property Features**")
                    st.write(f"- Main Road: {main_road}")
                    st.write(f"- Guest Room: {guest_room}")
                    st.write(f"- Basement: {basement}")
                    st.write(f"- Hot Water Heating: {hot_water_heating}")
                    st.write(f"- Air Conditioning: {air_conditioning}")
                    st.write(f"- Preferred Area: {pref_area}")
                    st.write(f"- Furnishing: {furnishing}")
                    st.write(f"- Currency: {currency_code}")
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")

st.markdown("---")
st.caption("🏠 House Price Prediction App | Built with Streamlit")

st.sidebar.markdown("---")
st.sidebar.caption("💡 Adjust all parameters and click Predict Price")
st.sidebar.caption(f"💱 Current Currency: {currency_code}")
