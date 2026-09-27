"""Pruebas de las compuertas de calidad."""
import numpy as np
import pandas as pd

from src import calidad, config


def _mediciones(n=200, nulos=0):
    valores = np.linspace(1000, 2000, n)
    if nulos:
        valores[:nulos] = np.nan
    return pd.DataFrame({
        config.COL_EDIFICIO: "SmartMeter_SM_B10_ARQ",
        config.COL_TIEMPO: pd.date_range("2026-06-01", periods=n, freq="10min"),
        "activepower": valores,
        "totalpowerfactor": np.full(n, 0.95),
        "voltaje_promedio": np.full(n, 121.0),
    })


def test_lote_sano_pasa():
    ok, problemas = calidad.validar_lote(_mediciones())
    assert ok, problemas


def test_lote_corto_se_rechaza():
    ok, problemas = calidad.validar_lote(_mediciones(20))
    assert not ok
    assert any("pocas filas" in p for p in problemas)


def test_exceso_de_nulos_se_rechaza():
    ok, problemas = calidad.validar_lote(_mediciones(200, nulos=120))
    assert not ok
    assert any("nulos" in p for p in problemas)


def test_duplicados_en_la_clave_se_rechazan():
    df = pd.concat([_mediciones(), _mediciones()], ignore_index=True)
    ok, problemas = calidad.validar_lote(df)
    assert not ok
    assert any("duplicadas" in p for p in problemas)


def test_prediccion_con_nulos_no_pasa_la_prueba_de_humo():
    ok, problemas = calidad.validar_prediccion([1.0, np.nan, 3.0], "activepower")
    assert not ok
    assert any("nulos" in p for p in problemas)


def test_prediccion_plana_no_pasa():
    """Un modelo que devuelve siempre lo mismo esta roto, aunque no falle."""
    ok, problemas = calidad.validar_prediccion([500.0] * 144, "activepower")
    assert not ok
    assert any("plana" in p for p in problemas)


def test_prediccion_fuera_de_rango_fisico_no_pasa():
    ok, problemas = calidad.validar_prediccion([0.5, 0.9, 5.0], "totalpowerfactor")
    assert not ok
