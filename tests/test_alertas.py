"""Pruebas del motor de alertas.

`evaluar_reglas` es una funcion pura, asi que se puede probar con datos
inventados sin base de datos ni modelos.
"""
import pandas as pd

from src import alertas, config


def _reglas(**cambios):
    base = {
        "id": 1,
        "nombre": "Factor de potencia bajo",
        "entity_id": None,
        "variable": "totalpowerfactor",
        "operador": "<",
        "umbral": 0.90,
        "umbral_salida": 0.93,
        "pasos_consecutivos": 3,
        "severidad": "alta",
        "destinatarios": "",
        "activa": 1,
        "modo_sombra": 1,
    }
    base.update(cambios)
    return pd.DataFrame([base])


def _predicciones(valores):
    return pd.DataFrame({
        config.COL_EDIFICIO: "SmartMeter_SM_B10_ARQ",
        "ts_objetivo": pd.date_range("2026-06-01", periods=len(valores), freq="10min"),
        "totalpowerfactor": valores,
    })


def test_un_punto_aislado_no_genera_evento():
    """Persistencia: un solo valor bajo es ruido, no un problema."""
    df = _predicciones([0.95, 0.95, 0.80, 0.95, 0.95])
    assert alertas.evaluar_reglas(df, _reglas()) == []


def test_tres_pasos_seguidos_si_generan_evento():
    df = _predicciones([0.95, 0.80, 0.82, 0.85, 0.95])
    eventos = alertas.evaluar_reglas(df, _reglas())
    assert len(eventos) == 1
    assert eventos[0]["valor_extremo"] == 0.80
    assert eventos[0]["duracion_pasos"] == 3


def test_regla_desactivada_no_evalua():
    df = _predicciones([0.5] * 10)
    assert alertas.evaluar_reglas(df, _reglas(activa=0)) == []


def test_operador_mayor_que_detecta_sobretension():
    df = pd.DataFrame({
        config.COL_EDIFICIO: "SmartMeter_SM_B10_ARQ",
        "ts_objetivo": pd.date_range("2026-06-01", periods=6, freq="10min"),
        "voltaje_promedio": [120, 120, 135, 136, 137, 120],
    })
    reglas = _reglas(variable="voltaje_promedio", operador=">", umbral=132,
                     umbral_salida=128, pasos_consecutivos=3)
    eventos = alertas.evaluar_reglas(df, reglas)
    assert len(eventos) == 1
    assert eventos[0]["valor_extremo"] == 137


def test_la_regla_solo_aplica_al_bloque_indicado():
    df = _predicciones([0.5] * 10)
    reglas = _reglas(entity_id="SmartMeter_SM_B3_RECT")
    assert alertas.evaluar_reglas(df, reglas) == []


def test_dos_tramos_separados_generan_dos_eventos():
    df = _predicciones([0.80, 0.80, 0.80, 0.99, 0.99, 0.80, 0.80, 0.80])
    eventos = alertas.evaluar_reglas(df, _reglas())
    assert len(eventos) == 2


def test_bloque_casi_apagado_no_dispara_factor_de_potencia():
    """Con 270 W de carga el factor de potencia bajo es fisica, no un problema.

    Sin esta guarda la regla se dispara permanentemente en los medidores de
    consumo minimo y la gente deja de leer las alertas.
    """
    df = _predicciones([0.5] * 10)
    df["activepower"] = 270.0
    reglas = _reglas(potencia_minima=2000)
    assert alertas.evaluar_reglas(df, reglas) == []


def test_bloque_en_operacion_si_dispara_factor_de_potencia():
    df = _predicciones([0.5] * 10)
    df["activepower"] = 25000.0
    reglas = _reglas(potencia_minima=2000)
    assert len(alertas.evaluar_reglas(df, reglas)) == 1
