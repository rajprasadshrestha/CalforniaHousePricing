import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.datasets import fetch_california_housing

#Loading the dataset
ca = fetch_california_housing()

df = pd.DataFrame(data=ca.data, columns=ca.feature_names)
df['Price'] = ca.target

#title of the web app
st.title('California Housing Price Prediction') 

#Data Overview
st.write('### Data Overview')
st.write(df.head(10))

#Split the dataset
X = df.drop('Price', axis=1)
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)


#Model Selection
st.write('### Model Selection')
model_name = st.selectbox('Choose the Model',( 'Linear Regression', 'Ridge Regression', 'Lasso Regression'))


#Initialize the models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(alpha=0.01)
}


#Train the model
models[model_name].fit(X_train_sc, y_train)
y_pred = models[model_name].predict(X_test_sc)


#Evaluation Metrics
test_mse = mean_squared_error(y_test, y_pred)
test_r2 = r2_score(y_test, y_pred)

st.write('### Evaluation Metrics')
st.write(f'Test MSE: {test_mse}')
st.write(f'Test R2: {test_r2}')


#Features input by the user
st.write('### Select the Features')
user_input = {}
for feature in X.columns:
    user_input[feature] = st.number_input(feature,value=float(X[feature].mean()))

#Convert the user input into dataframe
user_input_df = pd.DataFrame([user_input])

#Scale the user input
user_input_sc = scaler.transform(user_input_df)

#Predict the price
model_name=models[model_name]
prediction = model_name.predict(user_input_sc)
st.write('### Predict House Price')
st.write(f'Predicted Price for the inputs: {prediction[0]}')






