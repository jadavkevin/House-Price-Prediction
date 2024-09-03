import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sklearn.linear_model as lm
import streamlit as st
# import streamlit.components.v1 as components

# Load the dataset
data = pd.read_csv("house_price_prediction_dataset.csv")
data
data.head()
data.tail()
data.describe()

# No of rows and columns
data.shape
data.shape[0]  #rows
data.shape[1]  #columns

# Null values
data.isnull().sum()
data[["num_bedrooms","num_bathrooms","square_footage","age_of_house"]]

# Train the model
reg = lm.LinearRegression()
reg.fit(data[["num_bedrooms","num_bathrooms","square_footage","age_of_house"]],data.house_price)
      # y_predict = reg.predict([[3,1,3000,25]])
      # print(y_predict)
      # print(reg.predict([[10,5,5000,10]]))
      # y_predict_inr= y_predict*80
      # print(y_predict_inr)

#Streamlit UI
st.title('Home Price Prediction')

# Input fields for the user
bedrooms = st.number_input('No. of Bedrooms', min_value=1, step=1)
bathrooms = st.number_input('No. of Bathrooms : ', min_value=1, step=1)
age = st.number_input('How old is the house (in years)?', min_value=1, step=1)
area = st.number_input('Area of the house : ',  min_value=500, step=250)

# Predict house price
if st.button('Predict House Price : '):
    cost = reg.predict(np.array([[bedrooms, bathrooms, area, age]]))
    result_html = f"""
    <div style="background-color: #f2f2f2; padding: 20px; border: 1px solid #ddd; border-radius: 10px;">
        <h2>Predicted House Price</h2>
        <p>The predicted house price is : 
            <span style="font-size: 24px; font-weight: bold; font-style: italic;"> 
                Rs {cost[0] * 80:.2f}
            </span>
        </p>
    </div>
    <button style="
        background-color: #337ab7; 
        color: white; 
        padding: 10px 20px; 
        border: none; 
        border-radius: 5px; 
        font-size: 18px; 
        cursor: pointer;
        margin-top: 20px; ">
        <a href="tel:+91-799-069-4362" 
            style="
                color: white; 
                text-decoration: none; " 
            class="phone-link">
            Contact us for more information
        </a>
        <i class="margin" style="margin-left: 10px;"></i>
    </button> 
    """
    st.markdown(result_html, unsafe_allow_html=True)



    
