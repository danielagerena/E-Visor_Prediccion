"""Pruebas del preprocesamiento.

Son pocas y cortas a proposito: cubren las tres cosas que, si se rompen en
silencio, ensucian todo lo que viene despues.
"""
import numpy as np
import pandas as pd
import pytest

from src import config, preprocesamiento as pre


def _crudo(n=400, edificios=("SmartMeter_SM_B10_ARQ",)):
    filas = []
    for e in edificios:
        ts = pd.date_range("2026-06-01", periods=n, freq="30s")
        filas.append(pd.DataFrame({
            config.COL_TIEMPO: ts,
            config.COL_EDIFICIO: e,
            "activepower": np.linspace(1000, 2000, n),
            "totalpowerfactor": np.full(n, 0.95),
            "v1": np.full(n, 120.0),
            "v2": np.full(n, 121.0),
            "v3": np.full(n, 122.0),
            "activeenergyimport": np.arange(n) * 100.0 + 1_000_000,
        }))
    return pd.concat(filas, ignore_index=True)


def test_voltaje_promedio_es_la_media_de_las_tres_fases():
    df = pre.construir_voltaje_promedio(_crudo(10))
    assert df["voltaje_promedio"].iloc[0] == pytest.approx(121.0)


def test_reglas_de_calidad_anulan_valores_imposibles():
    df = _crudo(10)
    df.loc[3, "totalpowerfactor"] = 2.0        # imposible: el maximo fisico es 1
    df, reporte = pre.aplicar_reglas_calidad(df)
    assert np.isnan(df.loc[3, "totalpowerfactor"])
    assert reporte.loc[0, "n_fuera_rango"] == 1


def test_remuestreo_produce_rejilla_de_10_minutos():
    df = pre.construir_voltaje_promedio(_crudo(400))
    df10 = pre.remuestrear_10min(df)
    diferencias = df10[config.COL_TIEMPO].diff().dropna().unique()
    assert list(diferencias) == [pd.Timedelta(minutes=10)]


def test_rollover_del_contador_no_inventa_consumo():
    """Cuando el medidor da la vuelta, el consumo del intervalo se conoce.

    La version anterior reemplazaba ese dato por la mediana del bloque, es
    decir, se inventaba un valor. Sumar el techo del contador recupera el
    consumo real.
    """
    ts = pd.date_range("2026-06-01", periods=4, freq="10min")
    df = pd.DataFrame({
        config.COL_EDIFICIO: "SmartMeter_SM_B10_ARQ",
        config.COL_TIEMPO: ts,
        config.VAR_ACUMULATIVA: [99_999_000, 99_999_500, 300, 800],
    })
    out = pre.calcular_incremento_energia(df)
    incrementos = out[config.VAR_INCREMENTO].tolist()
    assert incrementos[1] == pytest.approx(500)
    assert incrementos[2] == pytest.approx(800)    # 300 - 99_999_500 + 100_000_000
    assert incrementos[3] == pytest.approx(500)


def test_huecos_largos_no_se_interpolan():
    """Un apagon de dias debe quedar como vacio, no como una linea recta."""
    ts = pd.date_range("2026-06-01", periods=200, freq="10min")
    valores = np.arange(200, dtype=float)
    valores[20:150] = np.nan                      # hueco de mas de 12 horas
    df = pd.DataFrame({
        config.COL_EDIFICIO: "SmartMeter_SM_B10_ARQ",
        config.COL_TIEMPO: ts,
        "activepower": valores,
    })
    out = pre.interpolar_huecos_cortos(df, variables=["activepower"])
    assert out["activepower"].isna().sum() > 0


def test_huecos_cortos_si_se_interpolan():
    ts = pd.date_range("2026-06-01", periods=50, freq="10min")
    valores = np.arange(50, dtype=float)
    valores[10:13] = np.nan                       # media hora
    df = pd.DataFrame({
        config.COL_EDIFICIO: "SmartMeter_SM_B10_ARQ",
        config.COL_TIEMPO: ts,
        "activepower": valores,
    })
    out = pre.interpolar_huecos_cortos(df, variables=["activepower"])
    assert out["activepower"].isna().sum() == 0
