import os
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from predictor import predecir_bloque, VARIABLES_MULTIVAR, VAR_ACUMULATIVA

# Rutas relativas al propio archivo. Asi funciona igual en local y en
# Streamlit Cloud, sin depender de Google Drive.
BASE = os.path.dirname(os.path.abspath(__file__))
ARCHIVO_DATOS = os.path.join(BASE, "datos", "datos_preprocesados_10min.csv")
CARPETA_MODELOS = os.path.join(BASE, "modelos_predictivos")

st.set_page_config(page_title="E-Visor - Prediccion por bloque", layout="wide")
st.title("E-Visor: prediccion de consumo 24h por bloque")
st.caption("Comparacion entre el consumo real y el pronostico del modelo en la ventana de prueba")


@st.cache_data
def cargar_datos():
    # Se lee una sola vez y queda en cache
    return pd.read_csv(ARCHIVO_DATOS, parse_dates=['timestamp'])


@st.cache_data
def bloques_disponibles():
    # Un bloque por carpeta en modelos_predictivos
    return sorted([
        d for d in os.listdir(CARPETA_MODELOS)
        if os.path.isdir(os.path.join(CARPETA_MODELOS, d))
    ])


df = cargar_datos()
bloques = bloques_disponibles()

bloque = st.selectbox("Selecciona un bloque (edificio):", bloques)

if st.button("Generar prediccion"):
    with st.spinner(f"Calculando prediccion de {bloque}..."):
        r = predecir_bloque(df, bloque, CARPETA_MODELOS)

    if r['usa_covariables']:
        st.info("Este bloque usa covariables de calendario (hora y dia de la semana).")

    # Una grafica por cada variable instantanea
    for var in VARIABLES_MULTIVAR:
        fig, ax = plt.subplots(figsize=(11, 3))
        r['real'][var].plot(ax=ax, label='Real', color='black')
        r['pred'][var].plot(ax=ax, label='Prediccion', color='orange', linestyle='--')
        ax.set_title(f"{var} - prediccion 24h")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    # Grafica de la energia acumulada (deriva)
    fig, ax = plt.subplots(figsize=(11, 3))
    r['real_energia'][VAR_ACUMULATIVA].plot(ax=ax, label='Real', color='black')
    r['pred_energia'][VAR_ACUMULATIVA].plot(ax=ax, label='Prediccion (deriva)',
                                            color='orange', linestyle='--')
    ax.set_title(f"{VAR_ACUMULATIVA} - prediccion con deriva")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

    st.success("Prediccion generada. Selecciona otro bloque para comparar.")
