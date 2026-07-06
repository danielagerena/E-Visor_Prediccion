import os
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import matplotlib as mpl

from predictor import predecir_bloque, VARIABLES_MULTIVAR, VAR_ACUMULATIVA

# Rutas relativas al propio archivo (funciona en local y en Streamlit Cloud)
BASE = os.path.dirname(os.path.abspath(__file__))
ARCHIVO_DATOS = os.path.join(BASE, "datos", "datos_preprocesados_10min.csv")
CARPETA_MODELOS = os.path.join(BASE, "modelos_predictivos")

# Colores del tema oscuro
FONDO = "#0e1117"
TEXTO = "#e0e0e0"
REJILLA = "#333333"

# Forzar que TODO el texto de las graficas sea claro, incluidas las fechas.
# Se hace en la configuracion global para que ningun elemento quede oscuro.
mpl.rcParams['text.color'] = TEXTO
mpl.rcParams['axes.labelcolor'] = TEXTO
mpl.rcParams['xtick.color'] = TEXTO
mpl.rcParams['ytick.color'] = TEXTO
mpl.rcParams['xtick.labelcolor'] = TEXTO
mpl.rcParams['ytick.labelcolor'] = TEXTO

# Notas para casos especiales, para que los lideres no se confundan
NOTAS = {
    'B8_CPA': "Medidor con lectura anomala. La prediccion de este bloque no es representativa.",
    'B15_BIBL': "El modelo es fiable en general; el dia usado como prueba fue atipicamente bajo.",
    'B4_PRIM': "El modelo es fiable en general; el dia usado como prueba fue atipico.",
    'B9_SFA1': "El modelo es fiable en general; el dia usado como prueba fue atipico.",
}

# Color de la etiqueta de confianza
COLOR_CONFIANZA = {
    'Alta': '#2ecc71',
    'Media': '#f1c40f',
    'Limitada': '#e74c3c',
    'Sin dato': '#888888',
}

st.set_page_config(page_title="E-Visor - Prediccion por bloque", layout="wide")
st.title("E-Visor: prediccion de consumo 24h por bloque")
st.caption("Comparacion entre el consumo real y el pronostico del modelo, con la historia de la semana previa")

# Nota fija con el contexto del analisis y como leer el error
st.info(
    "Periodo analizado: 10 de febrero al 31 de mayo de 2026 "
    "(con un apagon de mantenimiento de ~12 dias en Semana Santa). "
    "Prediccion a 24 horas con modelos LSTM, uno por edificio."
)
st.warning(
    "Como leer el error: el MAPE y el WAPE son porcentajes de EQUIVOCACION, "
    "no de acierto. Mas bajo es mejor. Por ejemplo, un MAPE de 20% significa "
    "que el modelo se equivoca en promedio un 20%, es decir acierta cerca del 80%."
)

@st.cache_data
def cargar_datos():
    return pd.read_csv(ARCHIVO_DATOS, parse_dates=['timestamp'])


@st.cache_data
def bloques_disponibles():
    return sorted([
        d for d in os.listdir(CARPETA_MODELOS)
        if os.path.isdir(os.path.join(CARPETA_MODELOS, d))
    ])


def graficar(contexto, real, pred, var, titulo, mape):
    # Dibuja historia previa + real + prediccion sobre fondo oscuro
    fig, ax = plt.subplots(figsize=(11, 3.2))
    fig.patch.set_facecolor(FONDO)
    ax.set_facecolor(FONDO)

    # Linea real continua: contexto (semana previa) + dia de prueba
    historico = contexto.append(real)
    historico[var].plot(ax=ax, label='Medicion real', color='#c9d600', linewidth=1.3)

    # Prediccion punteada solo sobre el dia pronosticado
    pred[var].plot(ax=ax, label='Prediccion 24h', color='orange',
                   linestyle='--', linewidth=1.6)

    # Linea vertical que marca el inicio de la prediccion
    inicio = real.start_time()
    ax.axvline(inicio, color='#888888', linestyle=':', linewidth=1)

    # Titulo por fuera de la grafica, con el MAPE al lado
    ax.set_title(f"{titulo}   |   MAPE {mape:.1f}%", color=TEXTO,
                 fontsize=12, loc='left', pad=10)

    # Estilo oscuro
    ax.tick_params(colors=TEXTO, labelsize=8)

    # Forzar color claro tambien en las etiquetas de fecha del eje x
    for etiqueta in ax.get_xticklabels():
        etiqueta.set_color(TEXTO)
    for etiqueta in ax.get_yticklabels():
        etiqueta.set_color(TEXTO)

    ax.set_xlabel("")
    for spine in ax.spines.values():
        spine.set_color(REJILLA)
    ax.grid(alpha=0.2, color=REJILLA)
    leg = ax.legend(facecolor=FONDO, edgecolor=REJILLA, labelcolor=TEXTO, fontsize=8)
    fig.tight_layout()
    return fig


df = cargar_datos()
bloques = bloques_disponibles()

bloque = st.selectbox("Selecciona un bloque (edificio):", bloques)

if st.button("Generar prediccion"):
    with st.spinner(f"Calculando prediccion de {bloque}..."):
        r = predecir_bloque(df, bloque, CARPETA_MODELOS)

    # Badge de confianza para el bloque
    color = COLOR_CONFIANZA.get(r['confianza'], '#888888')
    st.markdown(
        f"**Confianza de la prediccion:** "
        f"<span style='background-color:{color};color:#000;"
        f"padding:3px 10px;border-radius:6px;font-weight:600'>"
        f"{r['confianza']}</span> "
        f"<span style='color:#888'>(WAPE {r['wape_ap']:.1f}% en potencia activa)</span>",
        unsafe_allow_html=True
    )

    if bloque in NOTAS:
        st.caption("Nota: " + NOTAS[bloque])
    if r['usa_covariables']:
        st.caption("Este bloque usa covariables de calendario (hora y dia de la semana).")

    # Una grafica por variable instantanea
    for var in VARIABLES_MULTIVAR:
        fig = graficar(r['contexto'], r['real'], r['pred'], var, var, r['metricas'][var])
        st.pyplot(fig)
        plt.close(fig)

    # Grafica de energia
    fig = graficar(r['contexto_energia'], r['real_energia'], r['pred_energia'],
                   VAR_ACUMULATIVA, VAR_ACUMULATIVA, r['metricas'][VAR_ACUMULATIVA])
    st.pyplot(fig)
    plt.close(fig)