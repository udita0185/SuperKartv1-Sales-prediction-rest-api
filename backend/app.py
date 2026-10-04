import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from pathlib import Path
import socket
import sys

# Initialize Flask app
ProductStore_sales_revenue_api = Flask("SuperKart Product Store Sales Revenue Predictor")

# Load the trained Superkart Product Store Sales model
MODEL_PATH = Path(__file__).with_name("SuperKart_model_v1_0.joblib")
model = joblib.load(MODEL_PATH)

# Feature Preprocessing function
def preprocess_features(input_df):
    df = input_df.copy()

    # 1. Clean categorical text variations
    if 'Product_Sugar_Content' in df.columns:
        df['Product_Sugar_Content'] = df['Product_Sugar_Content'].replace({'reg': 'Regular'})

    # 2. Extract structural high-level product character prefixes
    if 'Product_Id' in df.columns:
        df['Product_Id_char'] = df['Product_Id'].str[:2]
    else:
        df['Product_Id_char'] = 'FD'

    # 3. Calculate store maturity in years since establishment
    if 'Store_Establishment_Year' in df.columns:
        df['Store_Age_Year'] = 2026 - df['Store_Establishment_Year'].astype(int)
    elif 'Store_Age_Years' in df.columns:
        df['Store_Age_Year'] = df['Store_Age_Years']
    else:
        df['Store_Age_Year'] = 17

    # 4. Group detailed items into Perishables/Non-Perishables
    perishable_types = ["Dairy", "Meat", "Frozen Foods", "Fruits and Vegetables", "Seafood"]
    if 'Product_Type' in df.columns:
        df['Product_Type_Category'] = df['Product_Type'].apply(lambda x: "Perishable" if x in perishable_types else "Non Perishable")
    elif 'Product_Type_Category' in df.columns:
        df['Product_Type_Category'] = df['Product_Type_Category']
    else:
        df['Product_Type_Category'] = "Non Perishable"

    # 5. Apply ordinal label-encoding structures for Store_Size
    if 'Store_Size' in df.columns:
        df['Store_Size'] = df['Store_Size'].astype(str).replace(r'^\d\((.*?)\)$', r'\1', regex=True)
        df['Store_Size'] = df['Store_Size'].replace({"Small": 1, "Medium": 2, "High": 3})
    else:
        df['Store_Size'] = 2

    # 6. Reconstruct precise model feature columns
    expected_features = [
        'Product_Weight', 'Product_Allocated_Area', 'Product_MRP', 'Store_Size', 'Store_Age_Year',
        'Product_Sugar_Content_No Sugar', 'Product_Sugar_Content_Regular',
        'Store_Location_City_Type_Tier 2', 'Store_Location_City_Type_Tier 3',
        'Store_Type_Food Mart', 'Store_Type_Supermarket Type1', 'Store_Type_Supermarket Type2',
        'Product_Id_char_FD', 'Product_Id_char_NC', 'Product_Type_Category_Perishable'
    ]

    final_df = pd.DataFrame(0, index=df.index, columns=expected_features)

    for col in ['Product_Weight', 'Product_Allocated_Area', 'Product_MRP', 'Store_Size', 'Store_Age_Year']:
        if col in df.columns:
            final_df[col] = df[col]

    if 'Product_Sugar_Content' in df.columns:
        final_df['Product_Sugar_Content_No Sugar'] = (df['Product_Sugar_Content'] == 'No Sugar').astype(int)
        final_df['Product_Sugar_Content_Regular'] = (df['Product_Sugar_Content'] == 'Regular').astype(int)

    if 'Store_Location_City_Type' in df.columns:
        final_df['Store_Location_City_Type_Tier 2'] = (df['Store_Location_City_Type'] == 'Tier 2').astype(int)
        final_df['Store_Location_City_Type_Tier 3'] = (df['Store_Location_City_Type'] == 'Tier 3').astype(int)

    if 'Store_Type' in df.columns:
        final_df['Store_Type_Food Mart'] = (df['Store_Type'] == 'Food Mart').astype(int)
        final_df['Store_Type_Supermarket Type1'] = (df['Store_Type'] == 'Supermarket Type1').astype(int)
        final_df['Store_Type_Supermarket Type2'] = (df['Store_Type'] == 'Supermarket Type2').astype(int)

    if 'Product_Id_char' in df.columns:
        df['Product_Id_char_clean'] = df['Product_Id_char'].astype(str).replace(r'\(.*\)', '', regex=True).str.strip()
        final_df['Product_Id_char_FD'] = (df['Product_Id_char_clean'] == 'FD').astype(int)
        final_df['Product_Id_char_NC'] = (df['Product_Id_char_clean'] == 'NC').astype(int)

    if 'Product_Type_Category' in df.columns:
        df['Product_Type_Category_clean'] = df['Product_Type_Category'].astype(str).replace(r's$', '', regex=True)
        final_df['Product_Type_Category_Perishable'] = (df['Product_Type_Category_clean'] == 'Perishable').astype(int)

    return final_df[expected_features]

# Define routes
@ProductStore_sales_revenue_api.get('/')
def home():
  return "Welcome to the SuperKart Product Store Sales Prediction API!"

@ProductStore_sales_revenue_api.post('/v1/superkart_salesrevenue_predictor')
def predict_sales_price():
    sales_data = request.get_json()
    input_data = pd.DataFrame([sales_data])
    processed_input = preprocess_features(input_data)
    prediction = model.predict(processed_input).tolist()[0]
    return jsonify({'Predicted_Sales_Total': prediction})

@ProductStore_sales_revenue_api.post('/v1/salesbatch')
def predict_sales_batch():
    file = request.files['file']
    input_data = pd.read_csv(file)
    processed_input = preprocess_features(input_data)
    predictions = model.predict(processed_input).tolist()
    input_data['Predicted_Sales_Total'] = predictions
    result = input_data.to_dict(orient="records")
    return jsonify(result)

def find_available_port(start_port=7860, max_attempts=100):
    port = start_port
    for _ in range(max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('0.0.0.0', port))
                return port
            except OSError:
                port += 1
    raise RuntimeError("No available ports found")

if __name__ == '__main__':
    port = find_available_port(7860)
    print(f"Starting SuperKart API on port {port}...")
    ProductStore_sales_revenue_api.run(debug=True, host='0.0.0.0', port=port)
