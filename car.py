import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
def load_data():
    df = pd.read_csv("car_prices.csv")  # Ensure you have a dataset
    return df

# Streamlit UI
st.title("Car Price Prediction")

df = load_data()
st.write("### Dataset Preview", df.head())

# Selecting relevant features
features = ['year', 'mileage', 'brand', 'fuel_type']  # Modify based on dataset
if all(col in df.columns for col in features + ['price']):
    df = df[features + ['price']]
    df = pd.get_dummies(df, drop_first=True)  # Convert categorical variables
else:
    st.error("Dataset doesn't have the required columns!")
    st.stop()

# Split data
X = df.drop(columns=['price'])
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Display model performance
st.write("### Model Performance")
st.write(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f}")
st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
st.write(f"R² Score: {r2_score(y_test, y_pred):.2f}")

# User input for prediction
st.write("### Predict Car Price")
year = st.number_input("Year", min_value=1990, max_value=2025, value=2015)
mileage = st.number_input("Mileage", min_value=0, max_value=300000, value=50000)
brand = st.selectbox("Brand", df.columns[df.columns.str.startswith('brand_')])
fuel_type = st.selectbox("Fuel Type", df.columns[df.columns.str.startswith('fuel_type_')])

if st.button("Predict Price"):
    user_data = pd.DataFrame([[year, mileage]], columns=['year', 'mileage'])
    for col in df.columns:
        if col.startswith('brand_') or col.startswith('fuel_type_'):
            user_data[col] = 1 if col == brand or col == fuel_type else 0
    price_pred = model.predict(user_data)[0]
    st.write(f"### Estimated Price: ${price_pred:.2f}")