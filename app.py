import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. Page Configuration
st.set_page_config(page_title="AI Crop Recommender", page_icon="🌱", layout="centered")

st.title("🌱 AI-Powered Crop Recommendation System")
st.write("Enter the soil and weather characteristics below to find the best crop to cultivate.")

# 2. Load the Dataset
@st.cache_data # This keeps the app running fast
def load_data():
    # Make sure 'Crop_recommendation.csv' is in the same folder as this script!
    df = pd.read_csv("https://raw.githubusercontent.com/nanditha-07/AI-Crop-Recommendation-System/main/Crop_recommendation.csv")
    return df

try:
    df = load_data()

    # 3. Separate Features (X) and Target Label (y)
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = df['label']

    # 4. Train the Machine Learning Model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # 5. User Input UI (Sidebar or Main Form)
    st.subheader("📊 Input Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        n = st.number_input("Nitrogen (N) value:", min_value=0, max_value=150, value=50)
        p = st.number_input("Phosphorus (P) value:", min_value=0, max_value=150, value=50)
        k = st.number_input("Potassium (K) value:", min_value=0, max_value=250, value=50)
        ph = st.number_input("Soil pH value (0-14):", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
    
    with col2:
        temp = st.number_input("Average Temperature (°C):", min_value=0.0, max_value=50.0, value=25.0, step=0.5)
        hum = st.number_input("Relative Humidity (%):", min_value=0.0, max_value=100.0, value=60.0, step=0.5)
        rain = st.number_input("Average Rainfall (mm):", min_value=0.0, max_value=300.0, value=100.0, step=1.0)

    # 6. Prediction Button
    if st.button("Recommend Best Crop", type="primary"):
        user_features = [[n, p, k, temp, hum, ph, rain]]
        prediction = model.predict(user_features)
        
        st.success(f"🎉 The best recommended crop for your farm is: **{prediction[0].upper()}**")

except FileNotFoundError:
    st.error("⚠️ Error: 'Crop_recommendation.csv' file not found. Please place it in the same directory as this code file.")
