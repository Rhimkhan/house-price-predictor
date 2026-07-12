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
    # ✅ CHANGED: Full Bathrooms → Washroom
    washroom = st.number_input("Washroom", min_value=0, max_value=10, value=2, step=1, help="Full bathrooms with shower/bathtub")
    # ✅ CHANGED: Half Bathrooms → Bathroom
    bathroom = st.number_input("Bathroom", min_value=0, max_value=5, value=1, step=1, help="Half bathrooms (toilet + sink)")

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
            
            st.success("✅ Prediction Complete!")
            
            # ✅ PRICE SHOW KAREIN PEHLE INR MEIN
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background: #f0f2f6; border-radius: 10px; margin: 20px 0;">
                <h3 style="margin: 0; color: #333;">💰 Predicted Price</h3>
                <h1 style="margin: 10px 0; font-size: 3rem; color: #667eea;">₹ {prediction_inr:,.2f}</h1>
                <p style="margin: 0; color: #666;">Indian Rupees (INR)</p>
            </div>
            """, unsafe_allow_html=True)
            
            # ✅ YEHI HAI WOH OPTION - INR YA USD MEIN DEKHNA HAI
            st.subheader("💱 View price in:")
            
            col_btn1, col_btn2 = st.columns(2)
            
            with col_btn1:
                if st.button("🇮🇳 Indian Rupees (INR)", use_container_width=True):
                    st.balloons()
                    st.markdown(f"""
                    <div style="text-align: center; padding: 25px; background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%); border-radius: 15px; color: white; margin: 10px 0;">
                        <h2 style="margin: 0; font-size: 1.2rem;">🏠 House Price</h2>
                        <h1 style="margin: 10px 0; font-size: 3.5rem;">₹ {prediction_inr:,.2f}</h1>
                        <p style="margin: 0; opacity: 0.9;">Indian Rupees (INR)</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col_btn2:
                if st.button("💵 US Dollars (USD)", use_container_width=True):
                    usd_price = prediction_inr * 0.012
                    st.balloons()
                    st.markdown(f"""
                    <div style="text-align: center; padding: 25px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white; margin: 10px 0;">
                        <h2 style="margin: 0; font-size: 1.2rem;">🏠 House Price</h2>
                        <h1 style="margin: 10px 0; font-size: 3.5rem;">$ {usd_price:,.2f}</h1>
                        <p style="margin: 0; opacity: 0.9;">US Dollars (USD)</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Input summary
            with st.expander("📋 Input Summary"):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.write("**📍 Location Features**")
                    st.write(f"- Area: {area} sq ft")
                    st.write(f"- Bedrooms: {bedrooms}")
                    st.write(f"- Washroom: {washroom}")
                    st.write(f"- Bathroom: {bathroom}")
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
                    
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

st.markdown("---")
st.caption("🏠 House Price Prediction App | Built with Streamlit")
