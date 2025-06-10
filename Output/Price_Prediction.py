import pickle
import numpy as np
import pandas as pd
import streamlit as st
import requests
from io import BytesIO
from sklearn.preprocessing import LabelEncoder

# URLs for the car data and model artifacts
CAR_DATA_URL = 'https://raw.githubusercontent.com/Avijit-Jana/Used-Car-Price-Prediction/main/Data%20Preprocessing%20%26%20Cleaning/car_data.xlsx'
MODEL_URL = 'https://raw.githubusercontent.com/Avijit-Jana/Used-Car-Price-Prediction/main/Model/model.pkl'
LABEL_ENCODER_URL = 'https://raw.githubusercontent.com/Avijit-Jana/Used-Car-Price-Prediction/main/Data%20Preprocessing%20%26%20Cleaning/label_encoder.pkl'
SCALER_URL = 'https://raw.githubusercontent.com/Avijit-Jana/Used-Car-Price-Prediction/main/Model/scaler.pkl'
CAR_IMAGE_URL = 'https://raw.githubusercontent.com/Avijit-Jana/Used-Car-Price-Prediction/main/Output/car.png'

# Utility to load binary file from URL
@st.cache_data
def load_binary_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    return BytesIO(response.content)

# Load the saved Random Forest model from the URL
@st.cache_resource
def load_model():
    bio = load_binary_from_url(MODEL_URL)
    return pickle.load(bio)

# Load label encoder and scaler from URLs
@st.cache_resource
def load_preprocessors():
    enc_bio = load_binary_from_url(LABEL_ENCODER_URL)
    scl_bio = load_binary_from_url(SCALER_URL)
    label_encoder = pickle.load(enc_bio)
    scaler = pickle.load(scl_bio)
    return label_encoder, scaler

# Function to predict using the Random Forest model
def predict_model(features, car_data):
    model_rf = load_model()
    label_encoder, scaler = load_preprocessors()
    features_df = pd.DataFrame([features])

    # Columns to encode
    columns_to_encode = ['Fuel type', 'Body type', 'transmission',
                         'model', 'variantName', 'Insurance Validity', 'City']

    for column in columns_to_encode:
        label_encoder.fit(car_data[column])
        features_df[column] = label_encoder.transform(features_df[column])

    # Scale the features
    features_scaled = scaler.transform(features_df)
    # Predict the price
    return model_rf.predict(features_scaled)[0]

# Main function
def main():
    # Load the dataset from URL
    try:
        df = pd.read_excel(load_binary_from_url(CAR_DATA_URL))
    except Exception as e:
        st.error(f"Error loading data from URL: {e}")
        return

    # Sidebar
    with st.sidebar:
        st.header('About:')
        st.write('This project leverages a machine learning model to predict the resale value of a car, offering users insights based on various vehicle features. By building a Streamlit app, we created an intuitive and interactive interface, allowing users to input details like model year, mileage, and other car specifics to receive an estimated resale price.')
        st.markdown('## -Developed by Avijit Jana')
        st.image(CAR_IMAGE_URL, width=250)

    # Set the title of the web app
    st.header(':car: :orange[_CarDekho_] Resale Car Price Prediction :car:', divider="red")
    st.write('---')

    # Layout for user inputs
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        owner = int(st.selectbox('Owner number', df['ownerNo'].unique()))
        reg_year = int(st.selectbox('Registration Year', df['Registration Year'].unique()))
    with col2:
        fuel = st.selectbox('Fuel Type', df['Fuel type'].unique())
        mileage = int(st.selectbox('Mileage(kmpl)', df['Mileage(kmpl)'].unique()))
    with col3:
        model_year = int(st.selectbox('Model Year', df['modelYear'].unique()))
        engine = int(st.selectbox('Engine(CC)', df['Engine(CC)'].unique()))
    with col4:
        km_driven = int(st.selectbox('Kilometers Driven', df['Kilometers driven'].unique()))
        torque = int(st.selectbox('Torque(Nm)', df['Torque(Nm)'].unique()))
    with col5:
        city = st.selectbox('City', df['City'].unique())
        transmission = st.selectbox('Transmission', df['transmission'].unique())

    col1, col2, col3 = st.columns(3)
    with col1:
        body = st.selectbox('Body Type', df['Body type'].unique())
    with col2:
        insurance = st.selectbox('Insurance Validity', df['Insurance Validity'].unique())
    with col3:
        max_power = int(st.selectbox('Max Power(bhp)', df['Max Power(bhp)'].unique()))

    col1, col2 = st.columns(2)
    with col1:
        model_name = st.selectbox('Model Name', df['model'].unique())
    with col2:
        variant = st.selectbox('Variant Name', df['variantName'].unique())

    features = {
        'Fuel type': fuel,
        'Body type': body,
        'Kilometers driven': km_driven,
        'transmission': transmission,
        'ownerNo': owner,
        'model': model_name,
        'modelYear': model_year,
        'variantName': variant,
        'Registration Year': reg_year,
        'Insurance Validity': insurance,
        'Mileage(kmpl)': mileage,
        'Engine(CC)': engine,
        'Max Power(bhp)': max_power,
        'Torque(Nm)': torque,
        'City': city
    }

    st.markdown("""
        <style>
        .stButton > button {
            font-size: 25px;
            height: 50px;
            width: 200px;
            color: green;
            display: block;
            margin: auto;
        }
        </style>
        """, unsafe_allow_html=True)

    if st.button("# Predict Price"):
        prediction = predict_model(features, df)
        st.markdown(f'## Resale Price of the Car: ₹ `{prediction:.2f}`')

if __name__ == '__main__':
    main()
