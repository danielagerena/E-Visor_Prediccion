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


def predecir_bloque(df, nombre_corto, carpeta_modelos):
    # Genera la prediccion de 24h del bloque sobre la ventana de test,
    # usando los artefactos ya guardados. No reentrena nada.
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

    # Cargar artefactos del bloque
    scaler = joblib.load(os.path.join(carpeta, 'scaler.joblib'))
    modelo = BlockRNNModel.load(os.path.join(carpeta, 'modelo_blockrnn.pt'))

    # En servidores sin GPU (como Streamlit Cloud) forzar prediccion en CPU
    try:
        modelo.to_cpu()
    except Exception:
        pass

    # Preparar el train escalado con el MISMO scaler guardado (transform, no fit)
    filler = MissingValuesFiller()
    train_val_filled = filler.transform(train_val)
    train_val_s = scaler.transform(train_val_filled)

    # Covariables de calendario solo si el bloque las uso
    ruta_cov = os.path.join(carpeta, 'scaler_cov.joblib')
    cov_s = None
    if os.path.exists(ruta_cov):
        scaler_cov = joblib.load(ruta_cov)
        cov_hora = datetime_attribute_timeseries(serie_mv, attribute='hour')
        cov_dia = datetime_attribute_timeseries(serie_mv, attribute='dayofweek')
        cov = cov_hora.stack(cov_dia)
        cov_s = scaler_cov.transform(cov)

    # Prediccion del horizonte desde el fin del train
    pred_s = modelo.predict(n=HORIZONTE, series=train_val_s, past_covariates=cov_s)
    pred = scaler.inverse_transform(pred_s)

    # Valores reales del test (rellenando huecos igual que en el entrenamiento)
    real = filler.transform(test)[:HORIZONTE]

    # Contexto: ultimos 7 dias del train para mostrar la historia previa
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
        rv = real[var].values().flatten()
        pv = pred[var].values().flatten()
        metricas[var] = _mape(rv, pv)
    metricas[VAR_ACUMULATIVA] = _mape(
        real_en.values().flatten(), pred_en.values().flatten()
    )

    ap_real = real['activepower'].values().flatten()
    ap_pred = pred['activepower'].values().flatten()
    wape_ap = _wape(ap_real, ap_pred)
    confianza = _etiqueta_confianza(wape_ap)

    return {
        'contexto': contexto,
        'real': real,
        'pred': pred,
        'contexto_energia': contexto_en,
        'real_energia': real_en,
        'pred_energia': pred_en,
        'metricas': metricas,
        'wape_ap': wape_ap,
        'confianza': confianza,
        'usa_covariables': cov_s is not None,
    }

def predecir_futuro(df, nombre_corto, carpeta_modelos):
    # Pronostico real a futuro: usa TODA la historia disponible y predice
    # las siguientes 24 horas. No hay valores reales para comparar.
    carpeta = os.path.join(carpeta_modelos, nombre_corto)
    serie_mv, serie_en = cargar_serie_bloque(df, nombre_corto)

    # Cargar artefactos
    scaler = joblib.load(os.path.join(carpeta, 'scaler.joblib'))
    modelo = BlockRNNModel.load(os.path.join(carpeta, 'modelo_blockrnn.pt'))
    try:
        modelo.to_cpu()
    except Exception:
        pass

    # Usar toda la serie como historia (no se reserva test)
    filler = MissingValuesFiller()
    serie_filled = filler.transform(serie_mv)
    serie_s = scaler.transform(serie_filled)

    # Covariables de calendario si el bloque las uso
    ruta_cov = os.path.join(carpeta, 'scaler_cov.joblib')
    cov_s = None
    if os.path.exists(ruta_cov):
        scaler_cov = joblib.load(ruta_cov)
        cov_hora = datetime_attribute_timeseries(serie_mv, attribute='hour')
        cov_dia = datetime_attribute_timeseries(serie_mv, attribute='dayofweek')
        cov = cov_hora.stack(cov_dia)
        cov_s = scaler_cov.transform(cov)

    # Para predecir a futuro, las covariables deben cubrir tambien el dia
    # que viene. Como el calendario es deterministico, se extiende solo.
    if cov_s is not None:
        futuro_idx = pd.date_range(
            serie_mv.end_time() + serie_mv.freq,
            periods=HORIZONTE, freq='10min'
        )
        serie_futura = TimeSeries.from_times_and_values(
            futuro_idx,
            np.zeros((HORIZONTE, serie_mv.n_components))
        )
        cov_h = datetime_attribute_timeseries(serie_futura, attribute='hour')
        cov_d = datetime_attribute_timeseries(serie_futura, attribute='dayofweek')
        cov_fut = cov_h.stack(cov_d)
        cov_fut_s = scaler_cov.transform(cov_fut)
        cov_s = cov_s.append(cov_fut_s)

    # Prediccion de las proximas 24 horas
    pred_s = modelo.predict(n=HORIZONTE, series=serie_s, past_covariates=cov_s)
    pred = scaler.inverse_transform(pred_s)

    # Contexto: ultimos 7 dias reales antes del pronostico
    contexto = serie_mv[-PASOS_SEMANA:]
    contexto_en = serie_en[-PASOS_SEMANA:]

    # Energia con la deriva
    d = joblib.load(os.path.join(carpeta, 'modelo_deriva.joblib'))
    pasos = np.arange(1, HORIZONTE + 1)
    ultimo_real = serie_en.values().flatten()
    ultimo_real = ultimo_real[~np.isnan(ultimo_real)][-1]
    valores_en = ultimo_real + d['pendiente'] * pasos
    fut_idx = pd.date_range(
        serie_en.end_time() + serie_en.freq, periods=HORIZONTE, freq='10min'
    )
    pred_en = TimeSeries.from_times_and_values(
        fut_idx, valores_en.reshape(-1, 1), columns=[VAR_ACUMULATIVA]
    )

    return {
        'contexto': contexto,
        'pred': pred,
        'contexto_energia': contexto_en,
        'pred_energia': pred_en,
        'usa_covariables': cov_s is not None,
    }