import pathlib
import sys

# This adds the path of the …/src folder
# to the PYTHONPATH variable
sys.path.append(str(pathlib.Path().absolute()).split("/src")[0] + "/src")

import streamlit as st
from mymodule.loader import get_name

# Import necessary libraries
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

# Load the data and create the model
@st.cache
def load_data():
    df = pd.read_csv("emissions_data.csv")
    X = df[["Engine Size(L)", "Fuel Consumption City (L/100 km)"]]
    y = df["Emissions"]
    model = LinearRegression().fit(X, y)
    return df, model

df, model = load_data()

# Create the main app
def main():
    # Create the input widgets
    engine_size = st.number_input("Engine Size (L):")
    fuel_consumption = st.number_input("Fuel Consumption City (L/100 km):")

    # Make the prediction
    prediction = model.predict([[engine_size, fuel_consumption]])[0]
    st.write(f"Emissions: {prediction:.2f}")

if __name__ == "__main__":
    main()
