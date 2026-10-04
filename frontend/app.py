import streamlit as st
import pandas as pd
import requests
from pathlib import Path
import os

# Fallback backend URL for testing purposes
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:7860")

# Streamlit UI for SuperKart Product Store Sales Prediction
st.title("SuperKart Product Store Sales Prediction App")
st.write("This app helps in predicting the future sales for Superkart.")
st.write("Move the sliders below to adjust values and get a prediction.")

# Collect user input using sliders
Product_Weight = st.number_input("Enter the weight of the product", min_value=0.0, value=12.66)
Product_Sugar_Content = st.selectbox("Select Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
Product_Allocated_Area = st.number_input("Ratio of the allocated display area", min_value=0.0, value=0.027)
Product_MRP = st.number_input("Maximum retail price of each product", min_value=0.0, value=117.08)
Store_Size = st.selectbox("Select the Store Size", ["Small", "Medium", "High"])
Store_Location_City_Type = st.selectbox("Select the Store Location", ["Tier 1", "Tier 2", "Tier 3"])
Store_Type = st.selectbox("Select the Store Type", ["Departmental Store", "Food Mart", "Supermarket Type1", "Supermarket Type2"])
Product_Id = st.text_input("Product ID", value="FD6114")
Store_Establishment_Year = st.number_input("Enter the Year Established", min_value=1900, max_value=2026, value=2009)
Product_Type = st.selectbox("Select the Product Type", ["Dairy", "Meat", "Frozen Foods", "Fruits and Vegetables", "Seafood", "Snack Foods", "Household", "Health and Hygiene", "Canned", "Baking Goods", "Others"])

# Create input DataFrame representing the raw schema expected by backend's preprocess_features
input_data = {
  "Product_Weight": Product_Weight,
  "Product_Sugar_Content": Product_Sugar_Content,
  "Product_Allocated_Area": Product_Allocated_Area,
  "Product_MRP": Product_MRP,
  "Store_Size": Store_Size,
  "Store_Location_City_Type": Store_Location_City_Type,
  "Store_Type": Store_Type,
  "Product_Id": Product_Id,
  "Store_Establishment_Year": int(Store_Establishment_Year),
  "Product_Type": Product_Type
}

# Making prediction when the "Predict" Button is clicked
if st.button("Predict"):
    response = requests.post(f"{BACKEND_URL}/v1/superkart_salesrevenue_predictor", json=input_data)    # Send Data to Flask API
    if response.status_code == 200:
        result = response.json()["Predicted_Sales_Total"]
        st.success(f"Predicted Sales Total (in dollar): {result:.2f}")
    else:
        st.error("Unable to connect to the prediction API")

# Batch Prediction
st.subheader("Batch Prediction")

# Allow users to upload a CSV file for batch prediction
upload_file = st.file_uploader("Upload CSV file for batch prediction", type=["csv"])

# Make batch prediction when the "Predict Batch" button is clicked
if upload_file is not None:
  if st.button("Predict Batch", type="primary"):
    response = requests.post(f"{BACKEND_URL}/v1/salesbatch", files={"file": upload_file.getvalue()})    # Send file to Flask API
    if response.status_code == 200:
      result_df = pd.DataFrame(response.json())
      st.write(result_df)
      st.success("Batch Prediction completed successfully!")
    else:
        st.error("Unable to connect to the prediction API")
