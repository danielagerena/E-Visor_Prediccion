"""Construccion de las series que entran al modelo.

Este archivo es el que garantiza que entrenamiento e inferencia vean
exactamente las mismas variables. Cuando cambie una transformacion, cambia en
los dos lados a la vez, que es el error mas comun al pasar de notebook a
produccion.

Darts se importa dentro de las funciones para que el resto del paquete (y las
pruebas) funcione sin tener PyTorch instalado.
"""
import numpy as np
import pandas as pd

from . import config


def cargar_mediciones(bloque=None, desde=None, hasta=None):
    """Lee mediciones de la base y las devuelve ordenadas."""
    from . import db

    sql = "SELECT * FROM mediciones WHERE 1=1"
    params = {}
    if bloque:
        sql += " AND entity_id = :b"
        params["b"] = config.nombre_completo(bloque)
    if desde is not None:
        sql += " AND ts >= :d"
        params["d"] = desde
    if hasta is not None:
        sql += " AND ts <= :h"
        params["h"] = hasta
    sql += " ORDER BY entity_id, ts"

    df = db.consultar(sql, params)
    if not df.empty:
        df[config.COL_TIEMPO] = pd.to_datetime(df[config.COL_TIEMPO])
    return df


def series_bloque(df, bloque):
    """Convierte el DataFrame de un bloque en las dos series de Darts.

    Devuelve (serie_multivariable, serie_energia_acumulada). Los huecos quedan
    como nulos: es Darts quien decide como tratarlos, no este archivo.
    """
    from darts import TimeSeries

    completo = config.nombre_completo(bloque)
    dfb = df[df[config.COL_EDIFICIO] == completo].sort_values(config.COL_TIEMPO)
    if dfb.empty:
        return None, None
    dfb = dfb.set_index(config.COL_TIEMPO)

    serie_mv = TimeSeries.from_dataframe(
        dfb[config.VARIABLES_MULTIVAR], freq=config.FRECUENCIA,
        fill_missing_dates=True, fillna_value=None,
    )
    serie_en = TimeSeries.from_dataframe(
        dfb[[config.VAR_ACUMULATIVA]], freq=config.FRECUENCIA,
        fill_missing_dates=True, fillna_value=None,
    )
    return serie_mv, serie_en


def covariables_calendario(serie, scaler_cov=None, ajustar=False):
    """Hora del dia y dia de la semana, escaladas a [0, 1].

    Son deterministas: se conocen para cualquier fecha futura, asi que
    escalarlas sobre todo el rango no filtra informacion del futuro.
    """
    from darts.dataprocessing.transformers import Scaler
    from darts.utils.timeseries_generation import datetime_attribute_timeseries
    from sklearn.preprocessing import MinMaxScaler

    cov = datetime_attribute_timeseries(serie, attribute="hour").stack(
        datetime_attribute_timeseries(serie, attribute="dayofweek")
    )
    if ajustar or scaler_cov is None:
        scaler_cov = Scaler(MinMaxScaler(feature_range=(0, 1)))
        return scaler_cov.fit_transform(cov), scaler_cov
    return scaler_cov.transform(cov), scaler_cov


def extender_covariables_al_futuro(serie_mv, scaler_cov, pasos=None):
    """Agrega el calendario del dia que viene, que ya se conoce de antemano."""
    from darts import TimeSeries

    pasos = pasos or config.HORIZONTE
    cov_pasado, _ = covariables_calendario(serie_mv, scaler_cov)
    indice_futuro = pd.date_range(
        serie_mv.end_time() + serie_mv.freq, periods=pasos, freq=config.FRECUENCIA
    )
    serie_futura = TimeSeries.from_times_and_values(
        indice_futuro, np.zeros((pasos, serie_mv.n_components))
    )
    cov_futuro, _ = covariables_calendario(serie_futura, scaler_cov)
    return cov_pasado.append(cov_futuro)


def pendiente_energia(df_bloque, dias=None):
    """Ritmo de consumo por intervalo, medido sobre las ultimas semanas.

    Se calcula sobre el INCREMENTO, no sobre el contador acumulado: es la
    cantidad que realmente se consume y la unica sobre la que un error
    porcentual significa algo.
    """
    dias = dias or config.DIAS_VENTANA_DERIVA
    pasos = dias * config.PASOS_DIA
    inc = df_bloque[config.VAR_INCREMENTO].tail(pasos).dropna()
    if inc.empty:
        return 0.0
    return float(inc.median())


def ultimo_valor_valido(serie_valores):
    arr = np.asarray(serie_valores, dtype=float)
    validos = arr[~np.isnan(arr)]
    return float(validos[-1]) if validos.size else np.nan


def wape(real, pred):
    """Error total dividido entre el consumo total.

    Se prefiere al MAPE porque no se dispara cuando el valor real se acerca a
    cero, que es justo lo que pasa con la potencia activa de madrugada.
    """
    real = np.asarray(real, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mascara = ~(np.isnan(real) | np.isnan(pred))
    if mascara.sum() == 0:
        return np.nan
    total = np.sum(np.abs(real[mascara]))
    if total == 0:
        return np.nan
    return 100 * np.sum(np.abs(real[mascara] - pred[mascara])) / total
