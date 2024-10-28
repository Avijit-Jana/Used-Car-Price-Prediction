import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder



# Load the saved Random Forest model from the pickle file
def load_model():
    file = '../Model/model.pkl'
    with open(file, 'rb') as f:
        model1 = pickle.load(f) 
    return model1

# Function to predict using the Random Forest model
def predict_model(features,car_data):
    model_rf = load_model()                 # Load the model
    features_df = pd.DataFrame([features])  # Create a DataFrame from the features

    # Load pre-fitted label encoder and scaler object from the pickle files
    label_file = '../Data Preprocessing & Cleaning/label_encoder.pkl'
    scaler_file = '../Model/scaler.pkl'
    with open(label_file, 'rb') as f:
        label_encoder = pickle.load(f)
    with open(scaler_file, 'rb') as f1:
        scaler = pickle.load(f1)
    
    # columns to encode
    columns_to_encode = ['Fuel type','Body type','transmission','model','variantName','Insurance Validity','City'] 
    
    for column in columns_to_encode:
        label_encoder.fit(car_data[column])  # Fit the label encoder on the entire column data
        features_df[column] = label_encoder.transform(features_df[column])  
    
    # Scale the features
    features_df = scaler.transform(features_df)
    # Predict the price
    prediction = model_rf.predict(features_df)  
    # return the prediction
    return prediction[0]


# main function
def main():
    # Load the dataset
    df = pd.read_excel('../Data Preprocessing & Cleaning/car_data.xlsx')

    # Sidebar
    with st.sidebar:
        st.header('About:')
        st.write('This project leverages a machine learning model to predict the resale value of a car, offering users insights based on various vehicle features. By building a Streamlit app, we created an intuitive and interactive interface, allowing users to input details like model year, mileage, and other car specifics to receive an estimated resale price.')
        st.markdown('## -Developed by Avijit Jana')
        st.image('../Output/car.png', width=250)

    # Set the title of the web app
    st.header(':car: :orange[_CarDekho_] Resale Car Price Prediction :car:', divider="red")

    # Create a form to take input from the user
    col1, col2, col3, col4, col5 = st.columns(5)

    # Create columns for getting user input
    # Section 1
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
        transmission = st.selectbox('transmission', df['transmission'].unique())
    
    # Section 2
    col1, col2, col3 = st.columns(3)
    with col1:
        body = st.selectbox('Body Type', df['Body type'].unique())
    with col2:
        insurance = st.selectbox('Insurance Validity', df['Insurance Validity'].unique())
    with col3:
        max_power = int(st.selectbox('Max Power(bhp)', df['Max Power(bhp)'].unique()))

    # Section 3
    col1, col2 = st.columns(2)
    with col1:
        model_name = st.selectbox('Model Name', df['model'].unique())
    with col2:
        variant = st.selectbox('Variant Name', df['variantName'].unique())

    # Dictionary for user inputs
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
    
    # Inject CSS styles globally for the button
    st.markdown("""
        <style>
        .stButton > button {
            font-size: 25px;
            height: 50px;
            width: 200px;
            color: green;
            # background-color: cyan;
            display: block;
            margin: auto;
        }
        </style>
        """, unsafe_allow_html=True)

    # Button without inline HTML
    if st.button("# Predict Price"):
        prediction = predict_model(features,df)
        st.markdown(f'## Resale Price of the Car: ₹ `{prediction:.2f}`')

if __name__ == '__main__':
    main()