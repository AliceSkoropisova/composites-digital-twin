import streamlit as st
import numpy as np
import joblib

st.title("Прогноз свойств композита")

# Поля для ввода 10 признаков
col1, col2 = st.columns(2)
with col1:
    density = st.number_input("Плотность, кг/м3", value=2000)
    modulus = st.number_input("Модуль упругости, ГПа", value=700)
    hardener = st.number_input("Количество отвердителя, м.%", value=30)
    epoxy = st.number_input("Содержание эпоксидных групп,%", value=22)
    temp = st.number_input("Температура вспышки, С", value=100)
with col2:
    surface_density = st.number_input("Поверхностная плотность, г/м2", value=210)
    resin = st.number_input("Потребление смолы, г/м2", value=70)
    angle = st.number_input("Угол нашивки, град", value=0)
    step = st.number_input("Шаг нашивки", value=5)
    pack_density = st.number_input("Плотность нашивки", value=60)

if st.button("Предсказать"):
    user_input = np.array([[density, modulus, hardener, epoxy, temp,
                            surface_density, resin, angle, step, pack_density]])
    
    # model_elastic = joblib.load("model_elastic.pkl")
    # model_strength = joblib.load("model_strength.pkl")
    # scaler = joblib.load("scaler.pkl")
    
    X_scaled = scaler.transform(user_input)
    elastic = model_elastic.predict(X_scaled)[0]
    strength = model_strength.predict(X_scaled)[0]
    
    st.success(f"Модуль упругости при растяжении: {elastic:.2f} ГПа")
    st.success(f"Прочность при растяжении: {strength:.2f} МПа")

