import streamlit as st
import pandas as pd
import tensorflow as tf
import pickle

modal = tf.keras.models.load_model('model.h5')

with open("label_encoder.pkl","rb") as file:
    label_encoder = pickle.load(file)

with open("one_Customer_Segment.pkl","rb") as file:
    one_Customer_Segment = pickle.load(file)

with open("one_hot_encoder_brand.pkl","rb") as file:
    one_hot_encoder_brand = pickle.load(file)

with open("one_hot_encoder.pkl","rb") as file:
    one_hot_encoder = pickle.load(file)

with open("one_model.pkl","rb") as file:
    one_model = pickle.load(file)

with open("one_Vehical.pkl","rb") as file:
    one_Vehical = pickle.load(file)

with open("Scaler_X.pkl","rb") as file:
    Scaler_X = pickle.load(file)

with open("Scaler_y.pkl","rb") as file:
    Scaler_y = pickle.load(file)

st.title("Car Revenue predictor")

region = st.selectbox("Region",one_hot_encoder.categories_[0])

brand = st.selectbox("Brand", one_hot_encoder_brand.categories_[0])

model = st.selectbox("Model",one_model.categories_[0])

vehical = st.selectbox("Vehicle Type",one_Vehical.categories_[0])

Battery_Capacity_kWh = st.number_input("Battery Capacity kWh")

Discount_Percentage = st.number_input("Discount Percentage")

custmer_segment = st.selectbox("Customer Segment",one_Customer_Segment.categories_[0])

def to_binary(value):
    if value == "Yes":
        return 1
    else:
        return 0

Fast_Charging = st.selectbox("Fast Charging Option",["Yes","No"])
Fast_Charging_Option = to_binary(Fast_Charging)

Units_Sold	= st.number_input("Unit Sold")


input_data = pd.DataFrame({
    "Battery_Capacity_kWh":[Battery_Capacity_kWh],
    "Discount_Percentage":[Discount_Percentage],
    "Fast_Charging_Option":[Fast_Charging_Option],
    "Units_Sold":[Units_Sold]
})

one_encode_region = one_hot_encoder.transform([[region]]).toarray()
encoder = pd.DataFrame(one_encode_region,columns=one_hot_encoder.get_feature_names_out(["Region"]))
input_data = pd.concat([input_data.reset_index(drop=True),encoder],axis=1)


hot_encode_brand = one_hot_encoder_brand.transform([[brand]]).toarray()
encode = pd.DataFrame(hot_encode_brand,columns=one_hot_encoder_brand.get_feature_names_out(["Brand"]))
input_data = pd.concat([input_data.reset_index(drop=True),encode],axis=1)

hot_encode_model = one_model.transform([[model]]).toarray()
encode_model = pd.DataFrame(hot_encode_model,columns=one_model.get_feature_names_out(["Model"]))
input_data = pd.concat([input_data.reset_index(drop=True),encode_model],axis=1)

hot_encode_Vehicle = one_Vehical.transform([[vehical]]).toarray()
encode_Vehicle = pd.DataFrame(hot_encode_Vehicle,columns=one_Vehical.get_feature_names_out(["Vehicle_Type"]))
input_data = pd.concat([input_data.reset_index(drop=True),encode_Vehicle],axis=1)

hot_encode_Customer_segment = one_Customer_Segment.transform([[custmer_segment]]).toarray()
encode_customer = pd.DataFrame(hot_encode_Customer_segment,columns=one_Customer_Segment.get_feature_names_out(["Customer_Segment"]))
input_data = pd.concat([input_data.reset_index(drop=True),encode_customer],axis=1)

input_data_scaled = Scaler_X.transform(input_data)

if st.button("Estimated Revenue"):
    Estimated_revenue = modal.predict(input_data_scaled)
    prediction = Scaler_y.inverse_transform(Estimated_revenue.reshape(-1,1))
    prediction_prob = prediction[0][0]
    st.success(f"Predicted Revenue is {prediction_prob:.2f}")