"""Limpieza y remuestreo de las mediciones crudas.

Es el contenido de los notebooks 1 y 3 convertido en funciones puras: reciben
un DataFrame y devuelven otro, sin leer ni escribir archivos. Asi la misma
transformacion la usan el cargue historico, la ingesta automatica y las pruebas.
"""
import numpy as np
import pandas as pd

from . import config

COL_TIEMPO = config.COL_TIEMPO
COL_EDIFICIO = config.COL_EDIFICIO


def normalizar_columnas(df, col_fecha="Fecha y Hora", col_entidad="ID Entidad"):
    """Renombra las columnas del export del proveedor a los nombres internos."""
    df = df.rename(columns={col_fecha: COL_TIEMPO, col_entidad: COL_EDIFICIO})
    df[COL_TIEMPO] = pd.to_datetime(df[COL_TIEMPO], errors="coerce")
    return df.dropna(subset=[COL_TIEMPO, COL_EDIFICIO])


def deduplicar(df):
    """Quita filas repetidas por solapamiento entre exports."""
    df = df.drop_duplicates()
    return df.drop_duplicates(subset=[COL_TIEMPO, COL_EDIFICIO], keep="first")


def construir_voltaje_promedio(df):
    """Promedia las tres fases en una sola variable de voltaje."""
    fases = [c for c in ("v1", "v2", "v3") if c in df.columns]
    if fases:
        df["voltaje_promedio"] = df[fases].mean(axis=1)
    return df


def aplicar_reglas_calidad(df, reglas=None):
    """Marca como nulo todo valor fisicamente imposible.

    Devuelve el DataFrame corregido y un reporte para auditoria.
    """
    reglas = reglas or config.REGLAS_CALIDAD
    reporte = []
    for variable, (vmin, vmax) in reglas.items():
        if variable not in df.columns:
            continue
        fuera = ((df[variable] < vmin) | (df[variable] > vmax)) & df[variable].notna()
        n = int(fuera.sum())
        if n:
            malos = df.loc[fuera, variable]
            reporte.append({
                "variable": variable,
                "rango_valido": f"[{vmin}, {vmax}]",
                "n_fuera_rango": n,
                "min_detectado": float(malos.min()),
                "max_detectado": float(malos.max()),
            })
            df.loc[fuera, variable] = np.nan
    return df, pd.DataFrame(reporte)


def remuestrear_10min(df):
    """Lleva las mediciones (llegan cada ~30 s) a la rejilla regular de 10 min.

    Cada variable usa el criterio que le corresponde:
    - potencia: mediana, para que un pico espurio no arrastre el intervalo
    - factor de potencia y voltaje: promedio
    - energia: ultimo valor, porque es un contador acumulado
    """
    salidas = []
    for edificio, grupo in df.groupby(COL_EDIFICIO):
        g = grupo.sort_values(COL_TIEMPO).set_index(COL_TIEMPO)
        res = pd.DataFrame()
        if "activepower" in g:
            res["activepower"] = g["activepower"].resample("10min").median()
        if "totalpowerfactor" in g:
            res["totalpowerfactor"] = g["totalpowerfactor"].resample("10min").mean()
        if "voltaje_promedio" in g:
            res["voltaje_promedio"] = g["voltaje_promedio"].resample("10min").mean()
        if "activeenergyimport" in g:
            res[config.VAR_ACUMULATIVA] = g["activeenergyimport"].resample("10min").last()
        res[COL_EDIFICIO] = edificio
        salidas.append(res.reset_index().rename(columns={"index": COL_TIEMPO}))
    if not salidas:
        return pd.DataFrame()
    out = pd.concat(salidas, ignore_index=True)
    return out.sort_values([COL_EDIFICIO, COL_TIEMPO]).reset_index(drop=True)


def calcular_incremento_energia(df, techo=None, umbral_rollover=None):
    """Convierte el contador acumulado en energia consumida por intervalo.

    El medidor da la vuelta al llegar a su techo (~100 millones). Cuando eso
    pasa, la resta simple da un salto enorme negativo. La correccion fisica es
    sumarle el techo, no reemplazar el dato por una mediana: el consumo real de
    ese intervalo se conoce y no hay por que inventarlo.
    """
    techo = techo or config.TECHO_CONTADOR
    umbral = umbral_rollover or config.UMBRAL_ROLLOVER

    df = df.sort_values([COL_EDIFICIO, COL_TIEMPO]).reset_index(drop=True)
    inc = df.groupby(COL_EDIFICIO)[config.VAR_ACUMULATIVA].diff()

    rollover = inc < umbral
    inc = inc.where(~rollover, inc + techo)

    # Un incremento negativo pequeno es ruido de medicion, no consumo negativo
    inc = inc.where(inc >= 0, np.nan)

    df[config.VAR_INCREMENTO] = inc
    df.attrs["n_rollovers"] = int(rollover.sum())
    return df


def interpolar_huecos_cortos(df, limite=None, variables=None):
    """Rellena huecos cortos e ignora los largos.

    Un hueco de una hora se puede interpolar sin mentir. El apagon de ocho dias
    no: ahi se deja el vacio para que el modelo no aprenda una linea recta que
    nunca ocurrio.
    """
    limite = limite or config.LIMITE_INTERPOLACION_PASOS
    variables = variables or [v for v in config.VARIABLES_TODAS if v in df.columns]

    partes = []
    for _, grupo in df.groupby(COL_EDIFICIO):
        g = grupo.sort_values(COL_TIEMPO).copy()
        for var in variables:
            g[var] = g[var].interpolate(method="linear", limit=limite)
        partes.append(g)
    return pd.concat(partes, ignore_index=True)


def preprocesar(df_crudo, ya_normalizado=False):
    """Cadena completa: crudo del proveedor -> rejilla de 10 min lista para el modelo."""
    df = df_crudo if ya_normalizado else normalizar_columnas(df_crudo)
    df = deduplicar(df)
    df = construir_voltaje_promedio(df)
    df, reporte = aplicar_reglas_calidad(df)
    df10 = remuestrear_10min(df)
    df10 = calcular_incremento_energia(df10)
    df10 = interpolar_huecos_cortos(df10)
    df10.attrs["reporte_calidad"] = reporte
    return df10
