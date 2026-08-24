"""Tablero E-Visor.

Cambio central frente a la version anterior: la app NO calcula nada. Ya no
carga modelos ni corre inferencia; solo consulta tres tablas. Eso elimina de un
golpe el riesgo de quedarse sin memoria, el arranque lento y la dependencia de
que alguien tenga la pagina abierta para que existan predicciones.
"""
import datetime as dt
import io
import os
import sys

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config, db                      # noqa: E402
from app import tema                            # noqa: E402

st.set_page_config(page_title="E-Visor UPB", layout="wide", page_icon="⚡")

LOGO = os.path.join(os.path.dirname(__file__), "assets", "logo.svg")
if os.path.exists(LOGO):
    st.logo(LOGO, size="large")


# --------------------------------------------------------------- consultas
@st.cache_data(ttl=300, show_spinner=False)
def cargar_predicciones(bloque):
    return db.consultar(
        "SELECT p.* FROM predicciones p WHERE p.entity_id = :b "
        "AND p.ts_origen = (SELECT MAX(ts_origen) FROM predicciones WHERE entity_id = :b) "
        "ORDER BY ts_objetivo",
        {"b": bloque},
    )


@st.cache_data(ttl=300, show_spinner=False)
def cargar_historia(bloque, dias=7):
    corte = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=dias)
    return db.consultar(
        "SELECT * FROM mediciones WHERE entity_id = :b AND ts >= :d ORDER BY ts",
        {"b": bloque, "d": corte},
    )


@st.cache_data(ttl=300, show_spinner=False)
def cargar_alertas():
    return db.consultar(
        "SELECT e.*, r.nombre, r.severidad, r.variable, r.umbral "
        "FROM eventos_alerta e JOIN reglas_alerta r ON r.id = e.regla_id "
        "WHERE e.estado = 'activa' ORDER BY e.iniciado_en DESC"
    )


@st.cache_data(ttl=300, show_spinner=False)
def frescura():
    r = db.consultar("SELECT MAX(ts) AS t FROM mediciones")
    if r.empty or not r["t"].iloc[0]:
        return None
    ultimo = pd.Timestamp(r["t"].iloc[0])
    return (pd.Timestamp.utcnow().tz_localize(None) - ultimo.tz_localize(None)).total_seconds() / 60


@st.cache_data(ttl=300, show_spinner=False)
def bloques_disponibles():
    r = db.consultar("SELECT DISTINCT entity_id FROM mediciones ORDER BY entity_id")
    return r["entity_id"].tolist()


# --------------------------------------------------------------- graficas
def grafico(historia, prediccion, variable):
    """Historia reciente y pronostico en un solo eje, con la frontera marcada."""
    h = historia[["ts", variable]].rename(columns={"ts": "momento"})
    h["serie"] = "Medicion real"
    p = prediccion[["ts_objetivo", variable]].rename(columns={"ts_objetivo": "momento"})
    p["serie"] = "Pronostico 24 h"
    datos = pd.concat([h, p], ignore_index=True).dropna(subset=[variable])

    lineas = alt.Chart(datos).mark_line(strokeWidth=1.6).encode(
        x=alt.X("momento:T", title=None),
        y=alt.Y(f"{variable}:Q", title=tema.NOMBRES_VARIABLES.get(variable, variable)),
        color=alt.Color("serie:N", title=None,
                        scale=alt.Scale(range=[tema.VERDE_UPB, tema.AMBAR])),
        strokeDash=alt.StrokeDash("serie:N", legend=None),
        tooltip=["momento:T", alt.Tooltip(f"{variable}:Q", format=".2f"), "serie:N"],
    )
    if not p.empty:
        frontera = alt.Chart(pd.DataFrame({"x": [p["momento"].min()]})).mark_rule(
            color=tema.GRIS, strokeDash=[4, 4]
        ).encode(x="x:T")
        return (lineas + frontera).properties(height=240)
    return lineas.properties(height=240)


# --------------------------------------------------------------- interfaz
st.title("E-Visor: consumo previsto por bloque")

minutos = frescura()
estado, texto = tema.etiqueta_frescura(minutos)
st.markdown(
    f"<span style='background:{tema.FRESCURA[estado]};color:white;padding:3px 10px;"
    f"border-radius:6px;font-weight:600'>Datos actualizados {texto}</span>",
    unsafe_allow_html=True,
)
if estado != "fresco":
    st.warning(
        "Los datos no estan al dia. El pronostico que ves puede no reflejar la "
        "situacion actual del campus."
    )

alertas = cargar_alertas()
if not alertas.empty:
    st.subheader(f"Alertas activas ({len(alertas)})")
    vista = alertas[["nombre", "entity_id", "severidad", "ts_prevista", "valor_extremo"]]
    vista.columns = ["Alerta", "Bloque", "Severidad", "Prevista para", "Valor"]
    st.dataframe(vista, use_container_width=True, hide_index=True)

st.divider()

col_a, col_b = st.columns([2, 1])
bloques = bloques_disponibles()
bloque = col_a.selectbox("Bloque", bloques, format_func=lambda b: config.nombre_corto(b))
dias = col_b.slider("Dias de historia", 1, 14, 7)

prediccion = cargar_predicciones(bloque)
historia = cargar_historia(bloque, dias)

if prediccion.empty:
    st.info("Todavia no hay pronostico para este bloque.")
    st.stop()

historia["ts"] = pd.to_datetime(historia["ts"])
prediccion["ts_objetivo"] = pd.to_datetime(prediccion["ts_objetivo"])

st.caption(
    f"Pronostico generado a partir de las {prediccion['ts_origen'].iloc[0]} "
    f"con el modelo {prediccion['modelo_version'].iloc[0]}"
)

for variable in config.VARIABLES_MULTIVAR + [config.VAR_ACUMULATIVA]:
    if variable in prediccion.columns:
        st.altair_chart(grafico(historia, prediccion, variable), use_container_width=True)

# --------------------------------------------------------------- descarga
st.divider()
st.subheader("Descargar el pronostico")

tabla = prediccion.rename(columns=tema.NOMBRES_VARIABLES)
marca = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
nombre = f"prediccion_{config.nombre_corto(bloque)}_{marca}"

c1, c2 = st.columns(2)
c1.download_button(
    "Descargar CSV",
    tabla.to_csv(index=False).encode("utf-8-sig"),   # utf-8-sig: tildes correctas en Excel
    file_name=f"{nombre}.csv",
    mime="text/csv",
    use_container_width=True,
)

buffer = io.BytesIO()
with pd.ExcelWriter(buffer, engine="xlsxwriter") as w:
    tabla.to_excel(w, sheet_name="Prediccion", index=False)
c2.download_button(
    "Descargar Excel",
    buffer.getvalue(),
    file_name=f"{nombre}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    use_container_width=True,
)

st.caption(
    "La tabla incluye la version del modelo y la hora de generacion de cada "
    "pronostico, para que cualquier resultado se pueda rastrear meses despues."
)
