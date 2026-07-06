import os
import numpy as np
import pandas as pd
import joblib
from darts import TimeSeries
from darts.models import BlockRNNModel
from darts.dataprocessing.transformers import MissingValuesFiller
from darts.utils.timeseries_generation import datetime_attribute_timeseries

# Constantes del pipeline (las mismas del notebook 4)
VARIABLES_MULTIVAR = ['activepower', 'totalpowerfactor', 'voltaje_promedio']
VAR_ACUMULATIVA = 'activeenergyimport_absoluto'
COL_TIEMPO = 'timestamp'
COL_EDIFICIO = 'entity_id'
HORIZONTE = 144            # 144 pasos de 10 min = 24 horas
PORCENTAJE_TEST = 0.15
PASOS_SEMANA = 7 * 144     # 7 dias de historia previa para el contexto


def cargar_serie_bloque(df, nombre_corto):
    # Reconstruye las series del bloque igual que en el entrenamiento
    full = 'SmartMeter_SM_' + nombre_corto
    dfb = df[df[COL_EDIFICIO] == full].sort_values(COL_TIEMPO).set_index(COL_TIEMPO)

    serie_mv = TimeSeries.from_dataframe(
        dfb[VARIABLES_MULTIVAR], freq='10min',
        fill_missing_dates=True, fillna_value=None
    )
    serie_en = TimeSeries.from_dataframe(
        dfb[[VAR_ACUMULATIVA]], freq='10min',
        fill_missing_dates=True, fillna_value=None
    )
    return serie_mv, serie_en


def _mape(real_vals, pred_vals):
    # Error porcentual promedio, ignorando valores reales cercanos a cero
    mask = np.abs(real_vals) > 1e-9
    if mask.sum() == 0:
        return np.nan
    return 100 * np.mean(np.abs(real_vals[mask] - pred_vals[mask]) / np.abs(real_vals[mask]))


def _wape(real_vals, pred_vals):
    # Error total dividido entre el consumo total. No se dispara cerca de cero.
    s = np.sum(np.abs(real_vals))
    return 100 * np.sum(np.abs(real_vals - pred_vals)) / s if s > 0 else np.nan


def _etiqueta_confianza(wape):
    # Traduce el WAPE a una palabra que un lider entiende de inmediato
    if np.isnan(wape):
        return "Sin dato"
    if wape <= 25:
        return "Alta"
    if wape <= 45:
        return "Media"
    return "Limitada"


def _cargar_artefactos(carpeta):
    # Carga el modelo y sus transformadores. Comun a las dos vistas.
    scaler = joblib.load(os.path.join(carpeta, 'scaler.joblib'))
    modelo = BlockRNNModel.load(os.path.join(carpeta, 'modelo_blockrnn.pt'))
    try:
        modelo.to_cpu()   # forzar CPU en servidores sin GPU
    except Exception:
        pass
    return scaler, modelo


def _covariables(serie_mv, scaler_cov):
    # Construye y escala las covariables de calendario (hora y dia de semana)
    cov_hora = datetime_attribute_timeseries(serie_mv, attribute='hour')
    cov_dia = datetime_attribute_timeseries(serie_mv, attribute='dayofweek')
    cov = cov_hora.stack(cov_dia)
    return scaler_cov.transform(cov)


def predecir_bloque(df, nombre_corto, carpeta_modelos):
    # VISTA VALIDACION: se para al final del train y predice el dia siguiente,
    # que si tiene valores reales para comparar.
    carpeta = os.path.join(carpeta_modelos, nombre_corto)
    serie_mv, serie_en = cargar_serie_bloque(df, nombre_corto)

    # Misma particion train/test del entrenamiento
    N = len(serie_mv)
    n_test = int(N * PORCENTAJE_TEST)
    n_train_val = N - n_test

    train_val = serie_mv[:n_train_val]
    test = serie_mv[n_train_val:]
    energia_train_val = serie_en[:n_train_val]
    energia_test = serie_en[n_train_val:]

    scaler, modelo = _cargar_artefactos(carpeta)

    filler = MissingValuesFiller()
    train_val_s = scaler.transform(filler.transform(train_val))

    # Covariables de calendario si el bloque las uso
    ruta_cov = os.path.join(carpeta, 'scaler_cov.joblib')
    cov_s = None
    if os.path.exists(ruta_cov):
        scaler_cov = joblib.load(ruta_cov)
        cov_s = _covariables(serie_mv, scaler_cov)

    # Prediccion del horizonte desde el fin del train
    pred_s = modelo.predict(n=HORIZONTE, series=train_val_s, past_covariates=cov_s)
    pred = scaler.inverse_transform(pred_s)

    real = filler.transform(test)[:HORIZONTE]

    # Contexto: ultimos 7 dias reales antes del pronostico
    contexto = train_val[-PASOS_SEMANA:]
    contexto_en = energia_train_val[-PASOS_SEMANA:]

    # Energia con la deriva (diccionario con pendiente y punto de partida)
    d = joblib.load(os.path.join(carpeta, 'modelo_deriva.joblib'))
    pasos = np.arange(1, HORIZONTE + 1)
    valores_en = d['ultimo_valor'] + d['pendiente'] * pasos
    pred_en = TimeSeries.from_times_and_values(
        energia_test[:HORIZONTE].time_index,
        valores_en.reshape(-1, 1),
        columns=[VAR_ACUMULATIVA]
    )
    real_en = energia_test[:HORIZONTE]

    # Metricas en vivo (MAPE por variable + WAPE de activepower para la etiqueta)
    metricas = {}
    for var in VARIABLES_MULTIVAR:
        metricas[var] = _mape(real[var].values().flatten(),
                              pred[var].values().flatten())
    metricas[VAR_ACUMULATIVA] = _mape(real_en.values().flatten(),
                                      pred_en.values().flatten())

    wape_ap = _wape(real['activepower'].values().flatten(),
                    pred['activepower'].values().flatten())

    return {
        'contexto': contexto,
        'real': real,
        'pred': pred,
        'contexto_energia': contexto_en,
        'real_energia': real_en,
        'pred_energia': pred_en,
        'metricas': metricas,
        'wape_ap': wape_ap,
        'confianza': _etiqueta_confianza(wape_ap),
        'usa_covariables': cov_s is not None,
    }


def predecir_futuro(df, nombre_corto, carpeta_modelos):
    # VISTA FUTURO: usa TODA la historia disponible y predice las proximas
    # 24 horas. No hay valores reales para comparar.
    carpeta = os.path.join(carpeta_modelos, nombre_corto)
    serie_mv, serie_en = cargar_serie_bloque(df, nombre_corto)

    scaler, modelo = _cargar_artefactos(carpeta)

    filler = MissingValuesFiller()
    serie_s = scaler.transform(filler.transform(serie_mv))

    # Covariables de calendario si el bloque las uso
    ruta_cov = os.path.join(carpeta, 'scaler_cov.joblib')
    cov_s = None
    if os.path.exists(ruta_cov):
        scaler_cov = joblib.load(ruta_cov)
        cov_s = _covariables(serie_mv, scaler_cov)

        # El calendario debe cubrir tambien el dia que viene. Como es
        # deterministico (se sabe que dia y hora sera), se construye solo.
        futuro_idx = pd.date_range(
            serie_mv.end_time() + serie_mv.freq,
            periods=HORIZONTE, freq='10min'
        )
        serie_futura = TimeSeries.from_times_and_values(
            futuro_idx, np.zeros((HORIZONTE, serie_mv.n_components))
        )
        cov_fut_s = _covariables(serie_futura, scaler_cov)
        cov_s = cov_s.append(cov_fut_s)

    # Prediccion de las proximas 24 horas
    pred_s = modelo.predict(n=HORIZONTE, series=serie_s, past_covariates=cov_s)
    pred = scaler.inverse_transform(pred_s)

    # Contexto: ultimos 7 dias reales antes del pronostico
    contexto = serie_mv[-PASOS_SEMANA:]
    contexto_en = serie_en[-PASOS_SEMANA:]

    # Energia con la deriva, partiendo del ultimo valor real conocido
    d = joblib.load(os.path.join(carpeta, 'modelo_deriva.joblib'))
    pasos = np.arange(1, HORIZONTE + 1)
    valores_reales = serie_en.values().flatten()
    ultimo_real = valores_reales[~np.isnan(valores_reales)][-1]
    valores_en = ultimo_real + d['pendiente'] * pasos
    fut_idx = pd.date_range(
        serie_en.end_time() + serie_en.freq, periods=HORIZONTE, freq='10min'
    )
    pred_en = TimeSeries.from_times_and_values(
        fut_idx, valores_en.reshape(-1, 1), columns=[VAR_ACUMULATIVA]
    )

    return {
        'contexto': contexto,
        'real': None,
        'pred': pred,
        'contexto_energia': contexto_en,
        'real_energia': None,
        'pred_energia': pred_en,
        'metricas': None,
        'wape_ap': None,
        'confianza': None,
        'usa_covariables': cov_s is not None,
    }