"""E-Visor - funciones de entrenamiento y evaluacion (LSTM multivariable multistep).

Todas las funciones son puras: no leen ni escriben variables globales. Eso
permite pegarlas tal cual en una celda de Colab hoy y copiarlas sin cambios a
src/modelo.py el dia que la configuracion pase a produccion.
"""
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Constantes del proyecto
# ----------------------------------------------------------------------------
VARIABLES = ["activepower", "totalpowerfactor", "voltaje_promedio"]
PREFIJO = "SmartMeter_SM_"
FRECUENCIA = "10min"

PASOS_HORA = 6
PASOS_DIA = 144
PASOS_SEMANA = 1008

LARGO_ENTRADA = 288      # 48 horas que mira el modelo hacia atras
HORIZONTE = 144          # 24 horas que predice
PASO_MUESTREO = 6        # una ventana por hora al armar el entrenamiento
LIMITE_INTERPOLACION = 6  # huecos de hasta 1 hora se rellenan; los demas se dejan

PERIODOS_CLASE = [("2026-02-10", "2026-05-30"), ("2026-07-14", "2026-09-01")]


# ----------------------------------------------------------------------------
# 1. Carga y organizacion de los datos
# ----------------------------------------------------------------------------
def cargar_datos(ruta_csv):
    """Lee el CSV preprocesado y agrega la columna 'bloque' sin el prefijo."""
    df = pd.read_csv(ruta_csv, parse_dates=["timestamp"])
    df["bloque"] = df["entity_id"].str.replace(PREFIJO, "", regex=False)
    return df.sort_values(["bloque", "timestamp"])


def tabla_bloque(df, bloque, limite_interp=LIMITE_INTERPOLACION):
    """Serie de un bloque sobre una rejilla continua de 10 minutos.

    Los huecos cortos se interpolan. Los largos se dejan como nulos: mas
    adelante se descartan las ventanas que los contengan, en vez de inventar
    datos.
    """
    d = df[df.bloque == bloque].set_index("timestamp").sort_index()
    idx = pd.date_range(d.index.min(), d.index.max(), freq=FRECUENCIA)
    d = d.reindex(idx)[VARIABLES]
    return d.interpolate(limit=limite_interp, limit_area="inside")


def mascara_clase(indice, periodos=PERIODOS_CLASE):
    """Marca que instantes caen dentro de un periodo de clase."""
    m = np.zeros(len(indice), dtype=bool)
    for a, b in periodos:
        m |= (indice >= pd.Timestamp(a)) & (indice < pd.Timestamp(b))
    return m


# ----------------------------------------------------------------------------
# 2. Construccion del dataset supervisado
# ----------------------------------------------------------------------------
def contexto_temporal(marcas):
    """Hora del dia y dia de la semana en seno y coseno.

    Se usan seno y coseno para que las 23:50 y las 00:00 queden cerca. Con un
    numero plano (0 a 23) el modelo veria un salto enorme a medianoche.
    """
    h = marcas.hour + marcas.minute / 60.0
    d = marcas.dayofweek
    return np.column_stack([
        np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
        np.sin(2 * np.pi * d / 7), np.cos(2 * np.pi * d / 7),
    ]).astype("float32")


def ventanas(tabla, origenes, largo_entrada=None, horizonte=None):
    """Arma las tres entradas y la salida para una lista de origenes.

    Para un origen t el modelo recibe:
      - pasado:  valores de t-largo_entrada+1 a t
      - semana:  los mismos instantes que va a predecir, pero 7 dias antes
      - contexto: hora y dia de la semana del origen
    y debe producir los valores de t+1 a t+horizonte.

    Todo lo que entra ya existia en el instante t. No hay fuga del futuro.
    """
    largo_entrada = LARGO_ENTRADA if largo_entrada is None else largo_entrada
    horizonte = HORIZONTE if horizonte is None else horizonte
    vals = tabla[VARIABLES].to_numpy("float32")
    origenes = np.asarray(origenes, dtype=int)

    i_pas = origenes[:, None] + np.arange(-largo_entrada + 1, 1)
    i_fut = origenes[:, None] + np.arange(1, horizonte + 1)
    i_sem = i_fut - PASOS_SEMANA

    x_pas, x_sem, y = vals[i_pas], vals[i_sem], vals[i_fut]
    x_ctx = contexto_temporal(tabla.index[origenes])

    completo = ~(np.isnan(x_pas).any((1, 2)) |
                 np.isnan(x_sem).any((1, 2)) |
                 np.isnan(y).any((1, 2)))
    return x_pas[completo], x_sem[completo], x_ctx[completo], y[completo], origenes[completo]


def origenes_validos(tabla, desde, hasta, paso, largo_entrada=None,
                     horizonte=None, solo_clase=True):
    """Posiciones de origen cuyo tramo a predecir cae en el rango pedido.

    Un origen es valido si hay suficiente historia atras (una semana) y si el
    tramo completo a predecir existe.
    """
    largo_entrada = LARGO_ENTRADA if largo_entrada is None else largo_entrada
    horizonte = HORIZONTE if horizonte is None else horizonte
    n = len(tabla)
    minimo = max(largo_entrada - 1, PASOS_SEMANA - 1)
    pos = np.arange(minimo, n - horizonte)

    inicio_objetivo = tabla.index[pos + 1]
    fin_objetivo = tabla.index[pos + horizonte]
    ok = (inicio_objetivo >= pd.Timestamp(desde)) & (fin_objetivo < pd.Timestamp(hasta))
    if solo_clase:
        ok &= mascara_clase(inicio_objetivo) & mascara_clase(fin_objetivo)
    return pos[ok][::paso]


def origenes_diarios(tabla, desde, hasta):
    """Un origen por dia, de forma que cada prediccion cubra de 00:00 a 23:50."""
    dias = pd.date_range(pd.Timestamp(desde), pd.Timestamp(hasta), freq="D")
    posiciones = []
    for d in dias:
        if d not in tabla.index:
            continue
        p = tabla.index.get_loc(d) - 1
        if p < max(LARGO_ENTRADA - 1, PASOS_SEMANA - 1):
            continue
        if p + HORIZONTE >= len(tabla):
            continue
        posiciones.append(p)
    return np.array(posiciones, dtype=int)


# ----------------------------------------------------------------------------
# 3. Escalamiento por bloque
# ----------------------------------------------------------------------------
def limites(tabla, hasta_entrenamiento):
    """Minimo y maximo de cada variable, calculados solo con datos de entrenamiento."""
    d = tabla.loc[tabla.index < pd.Timestamp(hasta_entrenamiento), VARIABLES]
    mn = d.min().to_numpy("float32")
    mx = d.max().to_numpy("float32")
    rango = np.where(mx - mn < 1e-9, 1.0, mx - mn).astype("float32")
    return mn, rango


def escalar(a, mn, rango):
    """Lleva los valores al rango -1 a 1 usando los limites del bloque."""
    return 2 * (a - mn) / rango - 1


def desescalar(a, mn, rango):
    return (a + 1) / 2 * rango + mn


# ----------------------------------------------------------------------------
# 4. El modelo
# ----------------------------------------------------------------------------
def construir_modelo(n_contexto, unidades=128, densa=256, dropout=0.2,
                     largo_entrada=None, horizonte=None,
                     n_vars=len(VARIABLES), tasa=1e-3):
    """LSTM multivariable multistep con tres entradas.

    Una rama lee las ultimas 48 horas, otra lee las mismas horas de la semana
    pasada, y el contexto dice que hora y que dia se esta prediciendo (y, en
    modo global, de que bloque se trata).
    """
    from tensorflow.keras import Input, Model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Concatenate, Reshape
    from tensorflow.keras.optimizers import Adam

    largo_entrada = LARGO_ENTRADA if largo_entrada is None else largo_entrada
    horizonte = HORIZONTE if horizonte is None else horizonte

    ent_pas = Input(shape=(largo_entrada, n_vars), name="pasado")
    ent_sem = Input(shape=(horizonte, n_vars), name="semana")
    ent_ctx = Input(shape=(n_contexto,), name="contexto")

    h1 = LSTM(unidades)(ent_pas)
    h2 = LSTM(unidades // 2)(ent_sem)
    z = Concatenate()([h1, h2, ent_ctx])
    z = Dense(densa, activation="relu")(z)
    z = Dropout(dropout)(z)
    salida = Reshape((horizonte, n_vars))(Dense(horizonte * n_vars)(z))

    modelo = Model([ent_pas, ent_sem, ent_ctx], salida)
    # Error absoluto: es la misma familia del WAPE con el que se mide.
    # Entrenar con error cuadratico seria apuntarle a un blanco distinto.
    modelo.compile(optimizer=Adam(learning_rate=tasa), loss="mae")
    return modelo


def fijar_semilla(semilla):
    import random, os
    import tensorflow as tf
    os.environ["PYTHONHASHSEED"] = str(semilla)
    random.seed(semilla)
    np.random.seed(semilla)
    tf.random.set_seed(semilla)


# ----------------------------------------------------------------------------
# 5. Metricas
# ----------------------------------------------------------------------------
def wape(real, pred):
    """Error total dividido entre el consumo total. No se dispara cerca de cero."""
    real, pred = np.asarray(real, float), np.asarray(pred, float)
    m = ~(np.isnan(real) | np.isnan(pred))
    total = np.sum(np.abs(real[m]))
    return 100 * np.sum(np.abs(real[m] - pred[m])) / total if total > 0 else np.nan


def rmse(real, pred):
    real, pred = np.asarray(real, float), np.asarray(pred, float)
    m = ~(np.isnan(real) | np.isnan(pred))
    return float(np.sqrt(np.mean((real[m] - pred[m]) ** 2)))


def tabla_metricas(detalle):
    """WAPE y RMSE por bloque y variable a partir del detalle punto a punto."""
    filas = []
    for (b, v), g in detalle.groupby(["bloque", "variable"]):
        filas.append({"bloque": b, "variable": v,
                      "wape_modelo": wape(g.real, g.pred),
                      "wape_ingenuo": wape(g.real, g.ingenuo),
                      "rmse_modelo": rmse(g.real, g.pred)})
    return pd.DataFrame(filas)


def error_por_horizonte(detalle, variable="activepower"):
    """Como crece el error a medida que la prediccion se aleja."""
    d = detalle[detalle.variable == variable]
    return d.groupby("paso").apply(
        lambda g: pd.Series({"wape_modelo": wape(g.real, g.pred),
                             "wape_ingenuo": wape(g.real, g.ingenuo)}))


# ----------------------------------------------------------------------------
# 6. Armado de los conjuntos y entrenamiento
# ----------------------------------------------------------------------------
RANGOS_TREN = [("2026-02-10", "2026-05-16"), ("2026-07-14", "2026-08-01")]
RANGO_VAL = ("2026-05-16", "2026-05-30")
RANGO_PRUEBA = ("2026-08-01", "2026-09-01")


def preparar_tablas(df, bloques, hasta_entrenamiento=RANGO_PRUEBA[0]):
    """Una tabla y unos limites de escalado por bloque."""
    tablas, lims = {}, {}
    for b in bloques:
        t = tabla_bloque(df, b)
        tablas[b] = t
        lims[b] = limites(t, hasta_entrenamiento)
    return tablas, lims


def armar_conjunto(tablas, lims, bloques, rangos, paso, indice_bloque=None):
    """Junta las ventanas de los bloques pedidos, ya escaladas.

    Si indice_bloque es None se entrena por bloque (el contexto solo lleva
    calendario). Si trae el diccionario de posiciones, se entrena un modelo
    global y el contexto lleva ademas la identidad del bloque.
    """
    P, S, C, Y, ORIG, BLQ = [], [], [], [], [], []
    for b in bloques:
        t, (mn, rango) = tablas[b], lims[b]
        for desde, hasta in rangos:
            pos = origenes_validos(t, desde, hasta, paso)
            if len(pos) == 0:
                continue
            p, s, c, y, orig = ventanas(t, pos)
            if len(y) == 0:
                continue
            P.append(escalar(p, mn, rango))
            S.append(escalar(s, mn, rango))
            Y.append(escalar(y, mn, rango))
            if indice_bloque is not None:
                ident = np.zeros((len(c), len(indice_bloque)), "float32")
                ident[:, indice_bloque[b]] = 1.0
                c = np.hstack([c, ident])
            C.append(c)
            ORIG.append(t.index[orig])
            BLQ.append(np.repeat(b, len(y)))
    if not Y:
        return None
    return {"pasado": np.concatenate(P), "semana": np.concatenate(S),
            "contexto": np.concatenate(C), "y": np.concatenate(Y),
            "origen": np.concatenate(ORIG), "bloque": np.concatenate(BLQ)}


def entrenar(conj_tren, conj_val, semilla, epocas=60, lote=256, paciencia=8,
             unidades=128, verbose=0):
    """Entrena una red y devuelve el modelo y su historia."""
    from tensorflow.keras.callbacks import EarlyStopping
    fijar_semilla(semilla)
    modelo = construir_modelo(n_contexto=conj_tren["contexto"].shape[1],
                              unidades=unidades)
    parada = EarlyStopping(monitor="val_loss", patience=paciencia,
                           restore_best_weights=True)
    historia = modelo.fit(
        x={"pasado": conj_tren["pasado"], "semana": conj_tren["semana"],
           "contexto": conj_tren["contexto"]},
        y=conj_tren["y"],
        validation_data=({"pasado": conj_val["pasado"], "semana": conj_val["semana"],
                          "contexto": conj_val["contexto"]}, conj_val["y"]),
        epochs=epocas, batch_size=lote, callbacks=[parada], verbose=verbose)
    return modelo, historia


def evaluar(modelo, tablas, lims, bloques, rango=RANGO_PRUEBA, indice_bloque=None):
    """Una prediccion por dia. Devuelve el detalle punto a punto.

    La columna 'ingenuo' es la regla trivial: el valor de hace una semana. Sale
    de la misma ventana que recibe el modelo, asi que los dos se miden sobre
    exactamente los mismos instantes.
    """
    filas = []
    for b in bloques:
        t, (mn, rango_esc) = tablas[b], lims[b]
        pos = origenes_diarios(t, rango[0], pd.Timestamp(rango[1]) - pd.Timedelta(days=1))
        if len(pos) == 0:
            continue
        p, s, c, y, orig = ventanas(t, pos)
        if len(y) == 0:
            continue
        if indice_bloque is not None:
            ident = np.zeros((len(c), len(indice_bloque)), "float32")
            ident[:, indice_bloque[b]] = 1.0
            c = np.hstack([c, ident])
        pred = modelo.predict({"pasado": escalar(p, mn, rango_esc),
                               "semana": escalar(s, mn, rango_esc),
                               "contexto": c}, verbose=0)
        pred = desescalar(pred, mn, rango_esc)
        n, h, v = y.shape
        pasos = np.tile(np.arange(1, h + 1), n)
        for j, var in enumerate(VARIABLES):
            filas.append(pd.DataFrame({
                "bloque": b, "variable": var, "paso": pasos,
                "origen": np.repeat(t.index[orig], h),
                "real": y[:, :, j].ravel(),
                "pred": pred[:, :, j].ravel(),
                "ingenuo": s[:, :, j].ravel()}))
    return pd.concat(filas, ignore_index=True)


def comparar_pareado(tabla_a, tabla_b, variable="activepower",
                     nombre_a="A", nombre_b="B"):
    """Prueba de Wilcoxon sobre el WAPE por bloque.

    Cada bloque aporta una observacion, no cada punto de la serie. Los errores
    de instantes seguidos estan correlacionados y usarlos como si fueran
    independientes infla la significancia.
    """
    from scipy.stats import wilcoxon
    a = tabla_a[tabla_a.variable == variable].set_index("bloque").wape_modelo
    b = tabla_b[tabla_b.variable == variable].set_index("bloque").wape_modelo
    comun = a.index.intersection(b.index)
    a, b = a.loc[comun], b.loc[comun]
    est, p = wilcoxon(a, b)
    return pd.DataFrame({nombre_a: a, nombre_b: b, "diferencia": a - b}), est, p


# ----------------------------------------------------------------------------
# 7. Guardado de artefactos
# ----------------------------------------------------------------------------
def guardar_artefactos(carpeta, nombre, modelo, lims, indice_bloque, ficha):
    """Guarda el modelo, los limites de escalado y la ficha de la version.

    Sin los limites el modelo no sirve: predice en el rango -1 a 1 y hay que
    devolverlo a vatios. Sin la ficha, dentro de seis meses nadie va a poder
    explicar de donde salio un numero.
    """
    import json, os
    os.makedirs(carpeta, exist_ok=True)
    modelo.save(f"{carpeta}/{nombre}.keras")
    np.savez(f"{carpeta}/{nombre}_limites.npz",
             **{f"{b}_mn": v[0] for b, v in lims.items()},
             **{f"{b}_rango": v[1] for b, v in lims.items()})
    with open(f"{carpeta}/{nombre}_ficha.json", "w", encoding="utf-8") as f:
        json.dump({**ficha, "indice_bloque": indice_bloque}, f,
                  indent=2, ensure_ascii=False)


def cargar_artefactos(carpeta, nombre):
    import json
    from tensorflow.keras.models import load_model
    modelo = load_model(f"{carpeta}/{nombre}.keras")
    z = np.load(f"{carpeta}/{nombre}_limites.npz")
    bloques = sorted({k.rsplit("_", 1)[0] for k in z.files})
    lims = {b: (z[f"{b}_mn"], z[f"{b}_rango"]) for b in bloques}
    with open(f"{carpeta}/{nombre}_ficha.json", encoding="utf-8") as f:
        ficha = json.load(f)
    return modelo, lims, ficha
