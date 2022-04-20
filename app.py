import streamlit as st
import pandas as pd
import numpy as np
import keras
import joblib
from sklearn.ensemble import RandomForestRegressor
from prediction import get_prediction
from keras.models import load_model

model = load_model(r'model.hdf5')

st.set_page_config(page_title="DMSP Particle Precipitate Prediction",
                   page_icon="🚧", layout="wide")


st.markdown("<h1 style='text-align: center;'>DMSP Particle Precipitate Prediction 🚧</h1>", unsafe_allow_html=True)
def main():
    with st.form('prediction_form'):

        st.subheader("Enter the input for following features:")
        Bx_6hr = st.text_input("Bx_6hr", value="", max_chars=6)
        AL_6hr = st.text_input("AL_6hr", value="", max_chars=6)
        vsw_6hr = st.text_input("vsw_6hr", value="", max_chars=6)
        psw_6hr = st.text_input("psw_6hr", value="", max_chars=6)
        Bz_3hr = st.text_input("Bz_3hr", value="", max_chars=6)
        By_3hr = st.text_input("By_3hr", value="", max_chars=6)
        psw_1hr = st.text_input("psw_1hr", value="", max_chars=6)
        By_45min = st.text_input("By_45min", value="", max_chars=6)
        AL_45min = st.text_input("AL_45min", value="", max_chars=6)
        Bz_10min = st.text_input("Bz_10min", value="", max_chars=6)
        submit = st.form_submit_button("Predict")


    if submit:
        data = np.array([float(Bx_6hr),float(AL_6hr),float(vsw_6hr),float(psw_6hr),
        float(Bz_3hr),float(By_3hr),float(psw_1hr),float(By_45min),float(AL_45min),float(Bz_10min)]).reshape(1,-1)

        pred = get_prediction(data=data, model=model)

        st.write(f"The predicted survival of patient is:  {pred}")

if __name__ == '__main__':
    main()
