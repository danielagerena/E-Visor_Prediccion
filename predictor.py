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
HORIZONTE = 144           # 144 pasos de 10 min = 24 horas
PORCENTAJE_TEST = 0.15


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

    # Covariables de calendario solo si el bloque las uso.
    # Se detecta por la existencia de su scaler propio.
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

    # Energia: la deriva es un diccionario con pendiente y punto de partida
    d = joblib.load(os.path.join(carpeta, 'modelo_deriva.joblib'))
    pasos = np.arange(1, HORIZONTE + 1)
    valores_en = d['ultimo_valor'] + d['pendiente'] * pasos
    pred_en = TimeSeries.from_times_and_values(
        energia_test[:HORIZONTE].time_index,
        valores_en.reshape(-1, 1),
        columns=[VAR_ACUMULATIVA]
    )
    real_en = energia_test[:HORIZONTE]

    return {
        'real': real,
        'pred': pred,
        'real_energia': real_en,
        'pred_energia': pred_en,
        'usa_covariables': cov_s is not None,
    }
