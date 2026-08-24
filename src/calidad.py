"""Compuerta de datos.

Antes de entrenar o de predecir se revisa que el lote tenga sentido. Si no lo
tiene, el proceso se detiene aqui y no gasta GPU ni publica un modelo malo.
"""
import numpy as np
import pandas as pd

from . import config


def validar_lote(df, minimo_filas=100, max_pct_nulos=40.0, variables=None):
    """Revisa un DataFrame de mediciones y devuelve (ok, problemas)."""
    variables = variables or [v for v in config.VARIABLES_MULTIVAR if v in df.columns]
    problemas = []

    if len(df) < minimo_filas:
        problemas.append(f"Muy pocas filas: {len(df)} < {minimo_filas}")

    if df.duplicated(subset=[config.COL_EDIFICIO, config.COL_TIEMPO]).any():
        problemas.append("Hay filas duplicadas en la clave (entity_id, ts)")

    for var in variables:
        pct = 100 * df[var].isna().mean()
        if pct > max_pct_nulos:
            problemas.append(f"{var}: {pct:.1f} % de nulos supera el limite de {max_pct_nulos} %")

    for var, (vmin, vmax) in config.REGLAS_CALIDAD.items():
        if var in df.columns:
            fuera = ((df[var] < vmin) | (df[var] > vmax)) & df[var].notna()
            if fuera.any():
                problemas.append(f"{var}: {int(fuera.sum())} valores fuera del rango fisico")

    if config.COL_TIEMPO in df.columns and len(df) > 1:
        ts = pd.to_datetime(df[config.COL_TIEMPO])
        if ts.max() - ts.min() < pd.Timedelta(hours=24):
            problemas.append("El lote cubre menos de 24 horas")

    return len(problemas) == 0, problemas


def validar_prediccion(valores, variable):
    """Prueba de humo de una prediccion recien generada."""
    arr = np.asarray(valores, dtype=float)
    problemas = []

    if arr.size == 0:
        problemas.append("La prediccion vino vacia")
        return False, problemas
    if np.isnan(arr).any():
        problemas.append(f"{int(np.isnan(arr).sum())} valores nulos en la prediccion")
    if np.isinf(arr).any():
        problemas.append("La prediccion contiene infinitos")

    rango = config.REGLAS_CALIDAD.get(variable)
    if rango is not None:
        vmin, vmax = rango
        fuera = int(((arr < vmin) | (arr > vmax)).sum())
        if fuera:
            problemas.append(f"{fuera} valores predichos fuera del rango fisico de {variable}")

    if np.nanstd(arr) == 0:
        problemas.append("La prediccion es una linea plana: el modelo no esta respondiendo")

    return len(problemas) == 0, problemas
