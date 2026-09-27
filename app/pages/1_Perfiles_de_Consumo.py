"""E-Visor - perfiles de consumo del campus.

Segunda página del tablero. Igual que la principal, NO calcula nada: lee lo que
dejó scripts/clustering.py en resultados/clustering/.

Va en app/pages/1_Perfiles_de_consumo.py
"""
import os

import altair as alt
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
CARPETA = "resultados/clustering"
RUTA_PERFILES = f"{CARPETA}/perfiles_semana.csv"
RUTA_RESUMEN = f"{CARPETA}/resumen.csv"
RUTA_LOGO = "app/logo_upb.png"

# ---------------------------------------------------------------------------
# Colores
# ---------------------------------------------------------------------------
AZUL_UPB = "#12284c"
GRIS_UPB = "#6b7280"
TINTA = "#52514e"
TINTA_TENUE = "#898781"
REJILLA = "#e1e0d9"

# Paleta categórica en orden fijo. Verificada para daltonismo contra el fondo
# claro: la peor pareja adyacente queda en delta E 9,1 en protanopia. Algunos
# colores tienen contraste bajo contra el fondo, así que las series llevan
# siempre etiqueta además del color.
COLORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
MAX_COMPARAR = len(COLORES)

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------
POT, FP, VOL = "activepower", "totalpowerfactor", "voltaje_promedio"

NOMBRES = {POT: "Potencia activa", FP: "Factor de potencia",
           VOL: "Voltaje promedio"}
FORMATO = {POT: ".2f", FP: ".2f", VOL: ".2f"}

PREGUNTA = {
    POT: "Qué edificios consumen con el mismo ritmo a lo largo de la semana.",
    FP: "Qué edificios comparten el mismo comportamiento de calidad de energía.",
    VOL: "Qué edificios comparten las mismas condiciones de voltaje.",
}

DIAS = ["Lun", "Mar", "Mie", "Jue", "Vie", "Sab", "Dom"]

# Nombre de negocio de cada grupo. Se identifica por los bloques que contiene y
# no por su número, porque el número puede cambiar si se vuelve a correr el
# clustering. Un grupo toma el nombre si contiene todos los bloques de la firma.
FIRMAS = {
    POT: [
        ({"B4_PRIM", "B5_BACH"}, "Colegio: de día y sin fines de semana"),
        ({"B17_POLI", "B18_PARQ"}, "Uso en las tardes y los fines de semana"),
        ({"B15_BIBL", "B12_DERE", "B10_ARQ"},
         "Académico de jornada larga: de 6 a. m. a 10 p. m., de lunes a sábado"),
    ],
    FP: [
        ({"B7_CTIC"}, "Carga constante tipo centro de datos"),
        ({"B8_AA", "B8_CPA", "ECOVILLA"},
         "Carga baja: el factor de potencia se vuelve inestable"),
    ],
    VOL: [
        ({"ECOVILLA"}, "Caso aislado"),
        ({"B17_POLI", "B18_PARQ"}, "Uso en las tardes y los fines de semana"),
    ],
}
GENERICO = {POT: "Sin patrón semanal marcado", FP: "Comportamiento común",
            VOL: "Perfil común del campus"}


def nombrar(var, miembros):
    for firma, nombre in FIRMAS.get(var, []):
        if firma <= set(miembros):
            return nombre
    return GENERICO[var]


st.set_page_config(page_title="E-Visor - Perfiles", layout="wide")


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------
@st.cache_data(ttl=600)
def cargar(ruta):
    if not os.path.isfile(ruta):
        return None
    return pd.read_csv(ruta)


perfiles = cargar(RUTA_PERFILES)
resumen = cargar(RUTA_RESUMEN)

cab1, cab2 = st.columns([1, 6])
with cab1:
    if os.path.isfile(RUTA_LOGO):
        st.image(RUTA_LOGO, width=110)
with cab2:
    st.markdown(
        f"<h1 style='color:{AZUL_UPB};margin-bottom:0'>Perfiles de consumo</h1>"
        f"<p style='color:{GRIS_UPB};margin-top:0'>Cómo se agrupan los 16 bloques "
        f"del Ecocampus Central según su semana típica</p>",
        unsafe_allow_html=True)

if perfiles is None:
    st.error(
        f"No se encontró **{RUTA_PERFILES}**.\n\n"
        "Lo genera el análisis de agrupamiento. Corre desde la raíz "
        "del repositorio:\n\n```\npython scripts/clustering.py\n```")
    st.stop()


# ---------------------------------------------------------------------------
# Idea principal
# ---------------------------------------------------------------------------
st.info(
    "**No hay un solo mapa del campus, hay tres.** El edificio que más consume "
    "no es el que peor calidad de energía tiene, ni el que vive condiciones de "
    "voltaje distintas. Por eso el análisis se hizo variable por variable.")

var = st.radio("Mapa", [POT, FP, VOL], horizontal=True,
               format_func=lambda v: NOMBRES[v])
st.caption(PREGUNTA[var])

d = perfiles[perfiles.variable == var].copy()

# Cuánto pesa cada bloque en el consumo del campus. Solo tiene sentido con la
# potencia: es la única de las tres variables que se puede sumar.
consumo = perfiles[perfiles.variable == POT].groupby("bloque").valor.mean()
peso = consumo / consumo.sum() * 100

miembros = {g: sorted(b) for g, b in d.groupby("grupo").bloque.unique().items()}
nombres = {g: nombrar(var, b) for g, b in miembros.items()}
# En la gráfica el nombre lleva el número delante: dos grupos pueden compartir
# el nombre genérico y la leyenda no admite etiquetas repetidas.
etiquetas_grupo = {g: f"{g}. {n}" for g, n in nombres.items()}
d["etiqueta"] = d.grupo.map(etiquetas_grupo)


# ---------------------------------------------------------------------------
# Cifras del mapa
# ---------------------------------------------------------------------------
if resumen is not None and var in set(resumen.analisis):
    r = resumen[resumen.analisis == var].iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Grupos", f"{int(r.k_usado)}")
    c2.metric("Separación entre grupos", f"{r.silueta:.2f}",
              help="Silueta, de 0 a 1. Compara qué tan parecidos son los "
                   "edificios de un mismo grupo frente a qué tan distintos son "
                   "de los demás grupos. Con 16 edificios los grupos son "
                   "vecindarios, no islas separadas.")
    c3.metric("Estabilidad", f"{r.estabilidad_media:.2f}",
              f"peor caso {r.estabilidad_peor:.2f}", delta_color="off",
              help="Se saca un edificio, se vuelve a agrupar y se mide qué "
                   "tanto se parece el resultado nuevo al original. Se repite "
                   "sacando cada uno de los 16 y se reporta el promedio y el "
                   "peor caso. Cerca de 1 quiere decir que los grupos no "
                   "dependen de un solo edificio.")


# ---------------------------------------------------------------------------
# Eje y separadores, compartidos por las dos gráficas
# ---------------------------------------------------------------------------
TITULO_Y = "Consumo respecto a lo normal del edificio"

eje_x = alt.X("hora_semana:Q", title=None,
              scale=alt.Scale(domain=[0, 167], nice=False),
              axis=alt.Axis(values=[12 + 24 * i for i in range(7)],
                            labelExpr=f"{DIAS}[floor(datum.value/24)]",
                            grid=False, domainColor="#c3c2b7",
                            tickColor="#c3c2b7", labelColor=TINTA_TENUE))


def eje_y(titulo=TITULO_Y):
    return alt.Y("forma:Q", title=titulo, scale=alt.Scale(zero=False),
                 axis=alt.Axis(gridColor=REJILLA, domain=False, ticks=False,
                               labelColor=TINTA_TENUE, titleColor=TINTA))


separadores = (alt.Chart(pd.DataFrame({"x": [24 * i for i in range(1, 7)]}))
               .mark_rule(color=REJILLA, strokeWidth=1).encode(x="x:Q"))


# ---------------------------------------------------------------------------
# Semana típica de cada grupo
# ---------------------------------------------------------------------------
st.subheader("Semana típica de cada grupo")
st.caption(
    "Cada edificio está comparado consigo mismo: por encima de cero está "
    "consumiendo más de lo que es normal en él, y por debajo, menos. Se "
    "grafica así porque los grupos se formaron por el ritmo del consumo y no "
    "por su tamaño. Si se comparara el valor en kilovatios, lo único que "
    "separaría sería los edificios grandes de los pequeños, que es algo que ya "
    "se sabe sin necesidad de un modelo.")

curvas = d.groupby(["grupo", "etiqueta", "hora_semana", "dia", "hora"],
                   as_index=False).forma.mean()
orden = [etiquetas_grupo[g] for g in sorted(etiquetas_grupo)]
color_grupo = alt.Color(
    "etiqueta:N", title=None,
    scale=alt.Scale(domain=orden, range=COLORES[:len(orden)]),
    legend=alt.Legend(orient="top", direction="vertical", columns=1,
                      labelColor=TINTA, labelLimit=0))

lineas = alt.Chart(curvas).mark_line(strokeWidth=2).encode(
    x=eje_x, y=eje_y(), color=color_grupo,
    tooltip=[alt.Tooltip("etiqueta:N", title="Grupo"),
             alt.Tooltip("dia:N", title="Día"),
             alt.Tooltip("hora:Q", title="Hora"),
             alt.Tooltip("forma:Q", title=TITULO_Y, format=FORMATO[var])])

# Etiqueta directa al final de cada curva: el color no puede ser la única forma
# de distinguir las series.
numeros = (alt.Chart(curvas[curvas.hora_semana == 167])
           .mark_text(align="left", dx=6, fontSize=11, fontWeight=600)
           .encode(x=eje_x, y=eje_y(), color=color_grupo,
                   text=alt.Text("grupo:N")))

st.altair_chart(
    alt.layer(separadores, lineas, numeros)
    .properties(height=340, padding={"right": 40})
    .configure_view(strokeWidth=0),
    width="stretch")


# ---------------------------------------------------------------------------
# Quién está en cada grupo
# ---------------------------------------------------------------------------
st.subheader("Qué edificios hay en cada grupo")

tabla = pd.DataFrame([
    {"Grupo": g, "Perfil": nombres[g], "Edificios": len(miembros[g]),
     "% del consumo del campus": round(peso.reindex(miembros[g]).sum(), 1),
     "Bloques": ", ".join(miembros[g])}
    for g in sorted(miembros)])
st.dataframe(tabla, hide_index=True, width="stretch")

if var == POT:
    st.caption("El grupo sin patrón semanal marcado no es un segmento de "
               "edificios parecidos: es donde cae lo que no encaja en ninguno "
               "de los patrones claros, cada uno por su propia razón. Lo que "
               "dice es que a esos edificios no se les puede aplicar una "
               "política basada en horarios.")


# ---------------------------------------------------------------------------
# Comparar edificios de un mismo grupo
# ---------------------------------------------------------------------------
st.subheader("Comparar edificios de un mismo grupo")
st.caption("Sirve para ver qué tan parecidos son de verdad los edificios que "
           "quedaron juntos, y cuál de ellos se aleja del promedio.")

cg, cb = st.columns([2, 5])
with cg:
    grupo = st.selectbox("Grupo", sorted(miembros),
                         format_func=lambda g: f"{g}. {nombres[g]}")
disponibles = miembros[grupo]
with cb:
    elegidos = st.multiselect(
        "Edificios", disponibles, default=disponibles[:MAX_COMPARAR],
        max_selections=MAX_COMPARAR,
        help=f"Hasta {MAX_COMPARAR} a la vez, para que cada uno conserve su "
             "propio color.")

if not elegidos:
    st.info("Elige al menos un edificio para comparar.")
else:
    promedio = (d[d.grupo == grupo].groupby("hora_semana", as_index=False)
                .forma.mean().assign(serie="Promedio del grupo"))
    unos = (d[d.bloque.isin(elegidos)][["bloque", "hora_semana", "forma"]]
            .rename(columns={"bloque": "serie"}))
    junto = pd.concat([promedio, unos], ignore_index=True)

    series = ["Promedio del grupo"] + elegidos
    color_bloque = alt.Color(
        "serie:N", title=None,
        scale=alt.Scale(domain=series, range=[TINTA_TENUE] + COLORES[:len(elegidos)]),
        legend=alt.Legend(orient="top", direction="horizontal",
                          labelColor=TINTA, labelLimit=0))

    comparacion = alt.Chart(junto).mark_line().encode(
        x=eje_x, y=eje_y(), color=color_bloque,
        strokeWidth=alt.condition(alt.datum.serie == "Promedio del grupo",
                                  alt.value(3), alt.value(1.6)),
        strokeDash=alt.condition(alt.datum.serie == "Promedio del grupo",
                                 alt.value([6, 4]), alt.value([1, 0])),
        tooltip=[alt.Tooltip("serie:N", title=""),
                 alt.Tooltip("forma:Q", title=TITULO_Y, format=FORMATO[var])])

    st.altair_chart(
        alt.layer(separadores, comparacion).properties(height=300)
        .configure_view(strokeWidth=0),
        width="stretch")
    st.caption("La línea gris punteada es el promedio del grupo.")


# ---------------------------------------------------------------------------
# Cómo leer estos grupos
# ---------------------------------------------------------------------------
with st.expander("Cómo leer estos grupos"):
    st.markdown(
        "- **Los grupos son vecindarios, no islas.** La separación entre "
        "grupos es moderada, así que un edificio en el borde podría pasarse al "
        "grupo vecino. Sirven para orientar decisiones, no para clasificar de "
        "forma definitiva.\n"
        "- **Son 16 edificios.** Es una muestra pequeña y conviene no "
        "sobreinterpretar un grupo de uno o dos.\n"
        "- **Se agrupó por ritmo, no por tamaño.** Cada bloque se compara "
        "consigo mismo antes de compararlo con los demás.\n"
        "- **Se usó la semana típica de los períodos de clase**, promediando "
        "cada hora de cada día de la semana y dejando fuera las vacaciones.\n"
        "- **La energía acumulada quedó fuera a propósito**: su incremento es "
        "la potencia activa dividida entre 6, así que incluirla sería contar "
        "el consumo dos veces.")

st.caption(f"Fuente: `{RUTA_PERFILES}`, generado por `scripts/clustering.py`.")