import streamlit as st
import pickle
import pandas as pd

st.title("Diamond Price Prediction")

with open("C:/Users/raval/Downloads/models.pkl", "rb") as file:
    models = pickle.load(file)

c1, c2, c3 = st.columns(3)

carat1 = c1.number_input("Enter carat", 0.20, 5.00, 0.35, 0.01)
depth1 = c2.number_input("Enter depth", 40.0, 80.0, 61.0)
table1 = c3.number_input("Enter table", 40.0, 80.0, 57.0)

c4, c5, c6 = st.columns(3)

x1 = c4.number_input("Enter x", 0.0, 10.0, 4.0)
y1 = c5.number_input("Enter y", 0.0, 10.0, 4.0)
z1 = c6.number_input("Enter z", 0.0, 10.0, 2.5)

c7, c8, c9 = st.columns(3)

cut1 = c7.selectbox("Enter cut", ['Fair','Good','Very Good','Premium','Ideal'])

color1 = c8.selectbox("Enter color", ['D','E','F','G','H','I','J'])

clarity1 = c9.selectbox("Enter clarity", ['I1','SI1','SI2','VS1','VS2','VVS1','VVS2','IF'])

b1 = st.button("Predict Price")

if b1:

    data = pd.DataFrame({
        'carat': [carat1],
        'depth': [depth1],
        'table': [table1],
        'x': [x1],
        'y': [y1],
        'z': [z1]
    })

    cat_data = pd.DataFrame({
        'cut': [cut1],
        'color': [color1],
        'clarity': [clarity1]
    })

    data_s = models['Scaler'].transform(data)

    data_s = pd.DataFrame(data_s, columns=data.columns)

    cat_data_e = models['Encoder'].transform(cat_data)

    cat_data_e = pd.DataFrame(cat_data_e, columns=cat_data.columns)

    final_data = data_s.join(cat_data_e)

    prediction = models['Pred_model'].predict(final_data)

    st.success(f"Predicted Diamond Price: {prediction[0]:,.2f}")
