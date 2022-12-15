import streamlit as st
from utils import PrepProcesor, columns
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

model = joblib.load('xgbpipe.joblib')
st.title('Wordt je gepromoveerd?')
sales = st.text_input("Wat zijn uw sales?", 0,3)
performance = st.slider("Wat is uw performance score?",0,5)
customer_rate = st.slider("Wat is uw customer rate?",0,5)


def predict():
    row = np.array([sales, customer_rate, performance])
    X = pd.DataFrame([row], columns=columns)
    prediction = model.predict(X)
    if prediction[0] == 1:
        st.success('Je wordt gepromoveerd! :thumbsup:')
    else:
        st.error('Je wordt niet gepromoveerd! :thumbsdown:')

trigger = st.button('Voorspel', on_click=predict)