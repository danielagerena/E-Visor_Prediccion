"""E-Visor - tablero de predicciones.

Esta app NO carga el modelo. Lee archivos que produjeron src/inferir_v4.py y
scripts/evaluar_por_bloque.py. Por eso no necesita TensorFlow, arranca en
segundos y no se acerca al limite de memoria de Streamlit Community Cloud.

Se corre con:
    streamlit run app/app_v4.py
"""
import json
import os

import altair as alt
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
RUTA_PRED = "datos/predicciones.csv"
RUTA_HIST = "datos/historia_reciente.csv"
RUTA_FICHA = "modelos/v4/modelo_produccion_ficha.json"
RUTA_ERROR_HORIZONTE = "modelos/v4/error_por_horizonte_real.csv"
RUTA_ERROR_BLOQUE = "modelos/v4/error_por_bloque.csv"
RUTA_LOGO = "app/logo_upb.png"

# ---------------------------------------------------------------------------
# Colores
# ---------------------------------------------------------------------------
# Institucionales. Reemplazar por los del manual de identidad de la UPB.
AZUL_UPB = "#12284c"
GRIS_UPB = "#6b7280"

# Series. Validados para daltonismo y contraste; si se cambian, hay que volver
# a validarlos, no elegirlos a ojo.
COLOR_MEDIDO = "#2a78d6"
COLOR_PREDICHO = "#eb6834"
TINTA = "#52514e"
TINTA_TENUE = "#898781"
REJILLA = "#e1e0d9"

# Estados. Nunca van solos: siempre con icono y texto.
BUENO, AVISO, CRITICO = "#0ca30c", "#fab219", "#d03b3b"

# ---------------------------------------------------------------------------
# Variables y reglas
# ---------------------------------------------------------------------------
POT = "activepower"
FP = "totalpowerfactor"
VOL = "voltaje_promedio"
INC = "activeenergyimport_incremento"
ACU = "activeenergyimport_absoluto"

NOMBRES = {
    POT: "Potencia activa (W)",
    FP: "Factor de potencia",
    VOL: "Voltaje promedio (V)",
    INC: "Energia cada 10 min (kWh)",
    ACU: "Energia acumulada (kWh)",
}
CORTOS = {POT: "Potencia activa", FP: "Factor de potencia",
          VOL: "Voltaje", INC: "Energia importada"}

VENTANA = pd.Timedelta(hours=24)       # cuanto se muestra hacia atras

FACTOR_POTENCIA_MINIMO = 0.90          # CREG 101-035 de 2024
VOLTAJE_NOMINAL = 120.0
TOLERANCIA_VOLTAJE = 0.10

# Umbrales de confiabilidad por variable, en WAPE %: (moderada desde, baja desde).
# Ademas, un bloque es de baja confiabilidad en una variable si el modelo no le
# gana a copiar la semana anterior, sin importar el valor del error.
UMBRALES = {
    POT: (25, 40),
    INC: (25, 40),
    FP: (5, 10),
    VOL: (1, 2),
}

MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
         "agosto", "septiembre", "octubre", "noviembre", "diciembre"]


def fecha_larga(t):
    # %B depende del idioma del sistema y en Streamlit Cloud sale en ingles
    return f"{t.day} de {MESES[t.month - 1]} de {t.year}, {t:%H:%M}"


def confianza(var, wape, wape_trivial):
    """Devuelve 'alta', 'moderada' o 'baja' y el motivo."""
    moderada, baja = UMBRALES[var]
    if wape >= wape_trivial:
        return "baja", "no le gana a copiar la semana anterior"
    if wape > baja:
        return "baja", f"error mayor a {baja} %"
    if wape > moderada:
        return "moderada", f"error entre {moderada} y {baja} %"
    return "alta", f"error de {moderada} % o menos"


ESTILO_NIVEL = {"alta": (BUENO, "Alta"), "moderada": (AVISO, "Moderada"),
                "baja": (CRITICO, "Baja")}

st.set_page_config(page_title="E-Visor", layout="wide")


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------
@st.cache_data(ttl=600)
def cargar(ruta, columnas_fecha=("ts",)):
    if not os.path.isfile(ruta):
        return None
    return pd.read_csv(ruta, parse_dates=list(columnas_fecha))


@st.cache_data(ttl=600)
def cargar_json(ruta):
    if not os.path.isfile(ruta):
        return {}
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


pred = cargar(RUTA_PRED)
hist = cargar(RUTA_HIST)
ficha = cargar_json(RUTA_FICHA)
curva = cargar(RUTA_ERROR_HORIZONTE, columnas_fecha=())
errores = cargar(RUTA_ERROR_BLOQUE, columnas_fecha=())

if pred is None:
    st.error(
        f"No se encontro **{RUTA_PRED}**.\n\n"
        "Las predicciones las genera `src/inferir_v4.py`. Corre:\n\n"
        "```\npython -m src.inferir_v4 --artefactos modelos/v4 "
        "--csv datos/datos_preprocesados_10min.csv "
        "--salida datos/predicciones.csv "
        "--salida-historia datos/historia_reciente.csv\n```")
    st.stop()


# ---------------------------------------------------------------------------
# Encabezado
# ---------------------------------------------------------------------------
cab1, cab2 = st.columns([1, 6])
with cab1:
    if os.path.isfile(RUTA_LOGO):
        st.image(RUTA_LOGO, width=110)
with cab2:
    st.markdown(
        f"<h1 style='color:{AZUL_UPB};margin-bottom:0'>E-Visor</h1>"
        f"<p style='color:{GRIS_UPB};margin-top:0'>Prediccion de consumo "
        f"electrico a 24 horas &middot; Ecocampus Central UPB</p>",
        unsafe_allow_html=True)

bloques = sorted(pred.bloque.unique())
sel1, sel2 = st.columns([2, 5])
with sel1:
    bloque = st.selectbox("Bloque", bloques)

p = pred[pred.bloque == bloque]
h = hist[hist.bloque == bloque] if hist is not None else None
origen = pd.to_datetime(p.origen.iloc[0])

with sel2:
    st.caption(f"Prediccion generada desde el ultimo dato disponible: "
               f"**{fecha_larga(origen)}**")
    calidad = p.calidad.iloc[0]
    if not calidad.startswith("buena"):
        st.warning(f"Calidad degradada en este bloque ({calidad}). "
                   "Falto parte de la historia y se uso una referencia alterna.")


# ---------------------------------------------------------------------------
# Confiabilidad del bloque seleccionado
# ---------------------------------------------------------------------------
def tarjeta(titulo, valor, nivel, detalle):
    color, texto = ESTILO_NIVEL[nivel]
    icono = {"alta": "&#10003;", "moderada": "!", "baja": "&#10007;"}[nivel]
    return (f"<div style='border-left:4px solid {color};padding:.45rem .8rem;"
            f"background:rgba(0,0,0,.02);height:100%'>"
            f"<div style='color:{TINTA};font-size:.85rem'>{titulo}</div>"
            f"<div style='font-size:1.4rem;font-weight:600'>{valor}</div>"
            f"<div style='font-size:.85rem'><strong>{icono} Confiabilidad "
            f"{texto.lower()}</strong><br><span style='color:{TINTA_TENUE}'>"
            f"{detalle}</span></div></div>")


if errores is not None:
    eb = errores[errores.bloque == bloque].set_index("variable")
    st.markdown("**Error de este bloque medido sobre agosto** "
                f"<span style='color:{TINTA_TENUE}'>(WAPE, mas bajo es mejor)</span>",
                unsafe_allow_html=True)
    columnas = st.columns(4)
    bajas = []
    for col, var in zip(columnas, [POT, FP, VOL, INC]):
        if var not in eb.index:
            continue
        r = eb.loc[var]
        nivel, motivo = confianza(var, r.wape_modelo, r.wape_ingenuo)
        if nivel == "baja":
            bajas.append((CORTOS[var], motivo))
        col.markdown(tarjeta(CORTOS[var], f"{r.wape_modelo:.1f} %", nivel,
                             f"semana anterior: {r.wape_ingenuo:.1f} %"),
                     unsafe_allow_html=True)
    if bajas:
        lista = "; ".join(f"**{v}** ({m})" for v, m in bajas)
        st.error(f"**Predicciones de baja confiabilidad en {bloque}:** {lista}. "
                 "Usar estas predicciones solo como referencia orientativa.")
else:
    # Respaldo mientras no exista el archivo de errores por variable
    marcados = ficha.get("bloques_que_pierden_contra_trivial", [])
    if bloque in marcados:
        st.error(f"**{bloque} requiere revision.** En agosto el modelo no le "
                 "gano a copiar la semana anterior en este bloque.")
    st.caption("Falta modelos/v4/error_por_bloque.csv. Generalo con "
               "scripts/evaluar_por_bloque.py para ver el error por variable.")


# ---------------------------------------------------------------------------
# Cifras del dia
# ---------------------------------------------------------------------------
def serie(datos, var):
    return datos[datos.variable == var].sort_values("ts")


pot, fp, vol = serie(p, POT), serie(p, FP), serie(p, VOL)
inc, acu = serie(p, INC), serie(p, ACU)

pico = pot.loc[pot.valor.idxmax()]
energia = (inc.valor.sum() if len(inc) else pot.valor.sum() / 6) / 1000
fp_min = fp.valor.min()
vol_fuera = int(((vol.valor - VOLTAJE_NOMINAL).abs()
                 > VOLTAJE_NOMINAL * TOLERANCIA_VOLTAJE).sum())

st.write("")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Energia prevista 24 h", f"{energia:,.0f} kWh")
c2.metric("Pico de potencia previsto", f"{pico.valor:,.0f} W",
          f"a las {pico.ts:%H:%M}", delta_color="off")
c3.metric("Factor de potencia minimo", f"{fp_min:.3f}",
          "bajo el minimo" if fp_min < FACTOR_POTENCIA_MINIMO else "dentro de norma",
          delta_color="inverse" if fp_min < FACTOR_POTENCIA_MINIMO else "off")
c4.metric("Puntos con voltaje fuera de rango", f"{vol_fuera}",
          "de 144" if vol_fuera else "ninguno", delta_color="off")


# ---------------------------------------------------------------------------
# Alertas de operacion
# ---------------------------------------------------------------------------
alertas = []
if fp_min < FACTOR_POTENCIA_MINIMO:
    horas = fp[fp.valor < FACTOR_POTENCIA_MINIMO]
    alertas.append((CRITICO, "Factor de potencia bajo",
                    f"Se preven {len(horas)} intervalos con factor de potencia "
                    f"menor a {FACTOR_POTENCIA_MINIMO:.2f}, el minimo de la "
                    f"CREG 101-035 de 2024. Por debajo de ese valor el operador "
                    f"cobra el transporte de la energia reactiva en exceso. "
                    f"Primer intervalo: {horas.ts.min():%H:%M}."))
if vol_fuera:
    alertas.append((AVISO, "Voltaje fuera de rango",
                    f"{vol_fuera} de 144 intervalos se salen de "
                    f"{VOLTAJE_NOMINAL:.0f} V mas o menos {TOLERANCIA_VOLTAJE:.0%}."))

if alertas:
    st.subheader("Alertas")
    for color, titulo, texto in alertas:
        icono = "!" if color == CRITICO else "i"
        st.markdown(
            f"<div style='border-left:4px solid {color};padding:.5rem .9rem;"
            f"margin-bottom:.5rem;background:rgba(0,0,0,.02)'>"
            f"<strong>{icono} {titulo}</strong><br>"
            f"<span style='color:{GRIS_UPB}'>{texto}</span></div>",
            unsafe_allow_html=True)
else:
    st.success("Sin alertas previstas para las proximas 24 horas.")


# ---------------------------------------------------------------------------
# Graficas: 24 horas medidas y 24 predichas, una por variable
# ---------------------------------------------------------------------------
st.subheader("24 horas anteriores y 24 horas siguientes")
st.caption(f"Prediccion de 144 puntos cada 10 minutos, de {p.ts.min():%d/%m %H:%M} "
           f"a {p.ts.max():%d/%m %H:%M}. La linea punteada marca donde termina "
           f"lo medido.")

mostrar_margen = st.checkbox(
    "Mostrar el margen de error tipico en la potencia activa", value=True,
    help="Es el error medido sobre agosto para cada distancia de pronostico. "
         "No es un intervalo de confianza estadistico: es cuanto se equivoco "
         "el modelo en promedio a esa misma distancia.")


def datos_grafica(var, escala=1.0):
    """Une lo medido (24 h atras) con lo predicho (24 h adelante)."""
    predicho = serie(p, var)[["ts", "valor"]].assign(serie="Predicho")
    partes = [predicho]
    if h is not None:
        medido = serie(h, var)
        medido = medido[(medido.ts > origen - VENTANA) & (medido.ts <= origen)]
        medido = medido[["ts", "valor"]].assign(serie="Medido")
        if len(medido):
            partes.append(medido)
            # El ultimo punto medido tambien inicia la linea predicha, para que
            # las dos queden unidas en el corte y no aparezca un hueco
            partes.append(medido.tail(1).assign(serie="Predicho"))
    d = pd.concat(partes, ignore_index=True)
    d["valor"] = d.valor * escala
    return d


def grafica(var, escala=1.0, titulo=None, formato=",.2f", banda=False):
    d = datos_grafica(var, escala)
    titulo = titulo or NOMBRES[var]
    base = alt.Chart(d).encode(
        x=alt.X("ts:T", title=None,
                axis=alt.Axis(grid=False, domainColor="#c3c2b7",
                              tickColor="#c3c2b7", labelColor=TINTA_TENUE,
                              format="%d/%m %H:%M")))
    linea = base.mark_line(strokeWidth=2).encode(
        y=alt.Y("valor:Q", title=titulo, scale=alt.Scale(zero=False),
                axis=alt.Axis(gridColor=REJILLA, domain=False, ticks=False,
                              labelColor=TINTA_TENUE, titleColor=TINTA)),
        color=alt.Color("serie:N", title=None,
                        scale=alt.Scale(domain=["Medido", "Predicho"],
                                        range=[COLOR_MEDIDO, COLOR_PREDICHO]),
                        legend=alt.Legend(orient="top", direction="horizontal")),
        tooltip=[alt.Tooltip("ts:T", title="Instante", format="%d/%m %H:%M"),
                 alt.Tooltip("valor:Q", title=titulo, format=formato),
                 alt.Tooltip("serie:N", title="")])
    capas = [linea]

    if banda and curva is not None:
        m = serie(p, var).reset_index(drop=True)[["ts", "valor"]].copy()
        m["paso"] = m.index + 1
        m = m.merge(curva[["paso", "wape_modelo"]], on="paso", how="left")
        m["bajo"] = m.valor * escala * (1 - m.wape_modelo / 100)
        m["alto"] = m.valor * escala * (1 + m.wape_modelo / 100)
        capas.insert(0, alt.Chart(m).mark_area(opacity=0.15, color=COLOR_PREDICHO)
                     .encode(x="ts:T", y=alt.Y("bajo:Q", title=titulo), y2="alto:Q"))

    capas.append(alt.Chart(pd.DataFrame({"ts": [origen]}))
                 .mark_rule(color=TINTA_TENUE, strokeDash=[4, 4]).encode(x="ts:T"))
    return (alt.layer(*capas).properties(height=230)
            .configure_view(strokeWidth=0).configure_legend(labelColor=TINTA))


st.altair_chart(grafica(POT, formato=",.0f", banda=mostrar_margen), width="stretch")
g1, g2 = st.columns(2)
with g1:
    st.altair_chart(grafica(FP, formato=".3f"), width="stretch")
with g2:
    st.altair_chart(grafica(VOL, formato=".1f"), width="stretch")


# ---------------------------------------------------------------------------
# Energia activa importada
# ---------------------------------------------------------------------------
st.subheader("Energia activa importada")
if len(acu) and len(inc):
    st.caption("No tiene modelo propio: se obtiene acumulando la potencia activa "
               "predicha. En 10 minutos, la energia en Wh es la potencia en W "
               "dividida entre 6.")

    lectura_ahora = None
    if h is not None and len(serie(h, ACU)):
        m = serie(h, ACU)
        m = m[m.ts <= origen]
        if len(m):
            lectura_ahora = m.valor.iloc[-1] / 1000

    e1, e2, e3 = st.columns(3)
    if lectura_ahora is not None:
        e1.metric("Lectura actual del contador", f"{lectura_ahora:,.1f} kWh")
    e2.metric("Lectura prevista en 24 h", f"{acu.valor.iloc[-1] / 1000:,.1f} kWh")
    e3.metric("Energia prevista 24 h", f"{energia:,.1f} kWh")

    vista = st.radio("Ver", ["Acumulado", "Incrementos cada 10 minutos"],
                     horizontal=True, label_visibility="collapsed")
    if vista == "Acumulado":
        st.altair_chart(grafica(ACU, escala=1 / 1000, formato=",.1f"),
                        width="stretch")
    else:
        st.altair_chart(grafica(INC, escala=1 / 1000, formato=",.3f"),
                        width="stretch")
else:
    st.info("Estas predicciones no traen energia acumulada. Vuelve a correr "
            "src/inferir_v4.py con la version actual.")


# ---------------------------------------------------------------------------
# Confiabilidad de todos los bloques
# ---------------------------------------------------------------------------
if errores is not None:
    st.subheader("Confiabilidad de cada bloque")
    st.caption("Error medido sobre agosto de 2026 con un modelo que no habia visto "
               "ese mes, comparado con copiar la semana anterior. Mas bajo es mejor.")

    var_sel = st.radio("Variable", [POT, FP, VOL, INC], horizontal=True,
                       format_func=lambda v: CORTOS[v], key="var_conf")
    d = errores[errores.variable == var_sel].copy()
    d[["nivel", "motivo"]] = d.apply(
        lambda r: pd.Series(confianza(var_sel, r.wape_modelo, r.wape_ingenuo)), axis=1)

    # Un bloque con error enorme aplasta la escala de los demas: se nombra aparte
    limite = d.wape_modelo.median() * 4
    fuera = d[d.wape_modelo > limite]
    dg = d[d.wape_modelo <= limite]

    largo = dg.melt(id_vars="bloque", value_vars=["wape_modelo", "wape_ingenuo"],
                    var_name="fuente", value_name="wape")
    largo["fuente"] = largo.fuente.map({"wape_modelo": "Modelo",
                                        "wape_ingenuo": "Semana anterior"})
    orden = dg.sort_values("wape_modelo").bloque.tolist()
    moderada, baja = UMBRALES[var_sel]

    barras = alt.Chart(largo).mark_bar(
        cornerRadiusEnd=4, stroke="#fcfcfb", strokeWidth=2
    ).encode(
        y=alt.Y("bloque:N", sort=orden, title=None,
                axis=alt.Axis(labelColor=TINTA, domain=False, ticks=False)),
        x=alt.X("wape:Q", title="Error (WAPE %)",
                axis=alt.Axis(gridColor=REJILLA, domain=False, ticks=False,
                              labelColor=TINTA_TENUE, titleColor=TINTA)),
        yOffset=alt.YOffset("fuente:N"),
        color=alt.Color("fuente:N", title=None,
                        scale=alt.Scale(domain=["Modelo", "Semana anterior"],
                                        range=[COLOR_MEDIDO, COLOR_PREDICHO]),
                        legend=alt.Legend(orient="top", direction="horizontal")),
        tooltip=[alt.Tooltip("bloque:N", title="Bloque"),
                 alt.Tooltip("fuente:N", title=""),
                 alt.Tooltip("wape:Q", title="WAPE %", format=".2f")])
    umbral = alt.Chart(pd.DataFrame({"x": [baja]})).mark_rule(
        color=CRITICO, strokeDash=[4, 4]).encode(x="x:Q")
    grafico = (alt.layer(barras, umbral).properties(height=30 * len(orden))
               .configure_view(strokeWidth=0))
    st.altair_chart(grafico, width="stretch")
    st.caption(f"La linea roja punteada es el umbral de baja confiabilidad para "
               f"esta variable: {baja} %.")

    for _, r in fuera.iterrows():
        st.caption(f"**{r.bloque}** queda fuera de la grafica: "
                   f"{r.wape_modelo:.1f} % de error.")

    tabla = d.sort_values("wape_modelo")[["bloque", "wape_modelo", "wape_ingenuo",
                                           "nivel", "motivo"]]
    tabla["nivel"] = tabla.nivel.map(lambda n: ESTILO_NIVEL[n][1])
    tabla.columns = ["Bloque", "Error del modelo (%)", "Error semana anterior (%)",
                     "Confiabilidad", "Motivo"]
    st.dataframe(tabla.round(2), width="stretch", hide_index=True)

    with st.expander("Como se decide la confiabilidad"):
        filas = "\n".join(f"| {CORTOS[v]} | hasta {a} % | {a} a {b} % | mas de {b} % |"
                          for v, (a, b) in UMBRALES.items())
        st.markdown(f"""
| Variable | Alta | Moderada | Baja |
|---|---|---|---|
{filas}

Ademas, cualquier bloque cuyo error sea igual o mayor que el de copiar la semana
anterior se marca como **baja**, sin importar el valor: en ese bloque el modelo
no aporta sobre la regla mas simple.

Los umbrales son distintos por variable porque sus errores tienen escalas muy
distintas: la potencia se equivoca alrededor del 25 %, el voltaje alrededor del
0,4 %. Un umbral unico no serviria para las dos.
""")


# ---------------------------------------------------------------------------
# Tabla descargable
# ---------------------------------------------------------------------------
st.subheader("Tabla de predicciones")
todo = st.checkbox("Incluir los 16 bloques en la descarga", value=False)
datos_tabla = pred if todo else p

ancha = (datos_tabla.pivot_table(index=["bloque", "ts"], columns="variable",
                                 values="valor")
         .reset_index().sort_values(["bloque", "ts"]))
for v in (INC, ACU):
    if v in ancha.columns:
        ancha[v] = ancha[v] / 1000
ancha = ancha.rename(columns=NOMBRES)

st.dataframe(ancha, width="stretch", hide_index=True, height=300)
st.download_button(
    "Descargar en CSV",
    ancha.to_csv(index=False).encode("utf-8-sig"),
    file_name=f"prediccion_{'campus' if todo else bloque}_{origen:%Y%m%d_%H%M}.csv",
    mime="text/csv")


# ---------------------------------------------------------------------------
# Pie: de donde salen estos numeros
# ---------------------------------------------------------------------------
with st.expander("Como se generan estas predicciones"):
    medido = ficha.get("wape_agosto_medido", {})
    trivial = ficha.get("regla_trivial_wape", {})
    st.markdown(f"""
**Modelo:** version `{ficha.get('version', '?')}`.

Una sola red LSTM para los 16 bloques. Para cada prediccion recibe las ultimas
48 horas del bloque, las mismas 24 horas que debe predecir pero de la semana
anterior, y la hora y el dia de la semana. Necesita
{ficha.get('dias_de_historia_requeridos', 7)} dias de historia disponibles.

La energia activa importada no tiene modelo propio: se calcula acumulando la
potencia activa predicha y sumandola a la ultima lectura del contador.

**Desempeno del campus medido sobre agosto de 2026**, con un modelo que no habia
visto ese mes:

| Variable | Modelo | Copiar la semana anterior |
|---|---|---|
| Potencia activa | {medido.get(POT, '?')} % | {trivial.get(POT, '?')} % |
| Factor de potencia | {medido.get(FP, '?')} % | {trivial.get(FP, '?')} % |
| Voltaje | {medido.get(VOL, '?')} % | {trivial.get(VOL, '?')} % |

El error es WAPE: error total dividido entre el consumo total.
""")