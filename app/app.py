import os
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model


import warnings
warnings.filterwarnings('ignore')

st.title("Прогноз свойств композита")
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

scaler_X_mod = joblib.load(os.path.join(MODEL_DIR,'scaler_X_mod.pkl'))
scaler_y_mod = joblib.load(os.path.join(MODEL_DIR,'scaler_y_mod.pkl'))
ridge_model = joblib.load(os.path.join(MODEL_DIR,'Ridge.pkl'))

scaler_X_n = joblib.load(os.path.join(MODEL_DIR,'scaler_X.pkl'))
scaler_y_n = joblib.load(os.path.join(MODEL_DIR,'scaler_y.pkl'))
nn_model = load_model(os.path.join(MODEL_DIR,'neural_network_model.keras'))

st.set_page_config(page_title="Прогнозирование свойств композитов", layout="centered")

st.header("1. Прогнозирование модуля упругости и прочности")
st.caption("Модель: Ридж-регрессия")

with st.form("ridge_form"):
    st.subheader("Параметры рецептуры")
    
    col1, col2 = st.columns(2)
    with col1:
        ratio = st.number_input("Соотношение матрица-наполнитель", value=1.86, format="%.4f")
        density = st.number_input("Плотность, кг/м³", value=2030.0, format="%.1f")
        modulus = st.number_input("Модуль упругости, ГПа", value=738.7, format="%.2f")
        hardener = st.number_input("Количество отвердителя, м.%", value=30.0, format="%.2f")
        epoxy = st.number_input("Содержание эпоксидных групп, %", value=22.27, format="%.2f")
        temp = st.number_input("Температура вспышки, °С", value=100.0, format="%.1f")
    with col2:
        surface_density = st.number_input("Поверхностная плотность, г/м²", value=210.0, format="%.1f")
        resin = st.number_input("Потребление смолы, г/м²", value=220.0, format="%.1f")
        angle = st.number_input("Угол нашивки, град", value=0.0, format="%.1f")
        step = st.number_input("Шаг нашивки", value=4.0, format="%.4f")
        stitch_density = st.number_input("Плотность нашивки", value=57.0, format="%.1f")
    
    submitted_ridge = st.form_submit_button("Рассчитать")
    
    if submitted_ridge:
        input_data = pd.DataFrame([[ratio, density, modulus, hardener, epoxy, temp, 
                                     surface_density, resin, angle, step, stitch_density]],
                                   columns=['Соотношение матрица-наполнитель', 'Плотность, кг/м3',
                                            'модуль упругости, ГПа', 'Количество отвердителя, м.%',
                                            'Содержание эпоксидных групп,%_2', 'Температура вспышки, С_2',
                                            'Поверхностная плотность, г/м2', 'Потребление смолы, г/м2',
                                            'Угол нашивки, град', 'Шаг нашивки', 'Плотность нашивки'])
        
        X_scaled = scaler_X_mod.transform(input_data)
        y_pred_scaled = ridge_model.predict(X_scaled)
        y_pred = scaler_y_mod.inverse_transform(y_pred_scaled)
        
        st.success(f"Модуль упругости при растяжении: **{y_pred[0][0]:.2f} ГПа**")
        st.success(f"Прочность при растяжении: **{y_pred[0][1]:.2f} МПа**")


st.divider()

# Second part

st.header("2. Прогнозирование соотношения матрица-наполнитель")
st.caption("Модель: нейронная сеть")

with st.form("nn_form"):
    st.subheader("Параметры рецептуры и свойств")
    
    col3, col4 = st.columns(2)
    with col3:

        density2 = st.number_input("Плотность, кг/м³", value=2030.0, format="%.1f")
        modulus2 = st.number_input("Модуль упругости, ГПа", value=738.7, format="%.2f")
        hardener2 = st.number_input("Количество отвердителя, м.%", value=30.0, format="%.2f")
        epoxy2 = st.number_input("Содержание эпоксидных групп, %", value=22.27, format="%.2f")
        temp2 = st.number_input("Температура вспышки, °С", value=100.0, format="%.1f")
        surface_density2 = st.number_input("Поверхностная плотность, г/м²", value=210.0, format="%.1f")
    with col4:
        elastic = st.number_input("Модуль упругости при растяжении, ГПа", value=70.0, format="%.2f")
        strength = st.number_input("Прочность при растяжении, МПа", value=3000.0, format="%.1f")
        resin2 = st.number_input("Потребление смолы, г/м²", value=220.0, format="%.1f")
        angle2 = st.number_input("Угол нашивки, град", value=0.0, format="%.1f")
        step2 = st.number_input("Шаг нашивки", value=4.0, format="%.4f")
        stitch_density2 = st.number_input("Плотность нашивки", value=57.0, format="%.1f")
    
    submitted_nn = st.form_submit_button("Рассчитать")
    
    if submitted_nn:
        input_data2 = pd.DataFrame([[elastic, strength, density2, modulus2, hardener2, epoxy2,
                                      temp2, surface_density2, resin2, angle2, step2, stitch_density2]],
                                    columns=['Модуль упругости при растяжении, ГПа',
                                             'Прочность при растяжении, МПа',
                                             'Плотность, кг/м3', 'модуль упругости, ГПа',
                                             'Количество отвердителя, м.%',
                                             'Содержание эпоксидных групп,%_2',
                                             'Температура вспышки, С_2',
                                             'Поверхностная плотность, г/м2',
                                             'Потребление смолы, г/м2',
                                             'Угол нашивки, град', 'Шаг нашивки', 'Плотность нашивки'])
        
        X_scaled2 = scaler_X_n.transform(input_data2)
        y_pred_scaled2 = nn_model.predict(X_scaled2, verbose=0).flatten()
        y_pred2 = scaler_y_n.inverse_transform(y_pred_scaled2.reshape(-1, 1)).flatten()
        
        st.info(f"Соотношение матрица-наполнитель: **{y_pred2[0]:.4f}**")


st.divider()