"""Prueba de que la escritura es idempotente.

Es la propiedad que permite volver a correr un job sin miedo: si el proveedor
reenvia datos o el cron se dispara dos veces, las filas se sobrescriben en vez
de duplicarse.
"""
import datetime as dt

import pandas as pd
import pytest

from src import config, db


@pytest.fixture
def base_temporal(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_URL", f"sqlite:///{tmp_path / 'prueba.db'}")
    db._motor = None
    db.crear_esquema()
    yield
    db._motor = None


def _fila(valor):
    return pd.DataFrame([{
        config.COL_EDIFICIO: "SmartMeter_SM_B10_ARQ",
        config.COL_TIEMPO: dt.datetime(2026, 6, 1, 10, 0),
        "activepower": valor,
    }])


def test_upsert_no_duplica_y_actualiza(base_temporal):
    db.upsert(_fila(100.0), "mediciones", [config.COL_EDIFICIO, config.COL_TIEMPO])
    db.upsert(_fila(200.0), "mediciones", [config.COL_EDIFICIO, config.COL_TIEMPO])

    r = db.consultar("SELECT * FROM mediciones")
    assert len(r) == 1
    assert r["activepower"].iloc[0] == 200.0


def test_registro_de_ejecuciones(base_temporal):
    db.registrar_ejecucion("ingesta", dt.datetime.now(dt.timezone.utc), "ok", 5, "prueba")
    r = db.consultar("SELECT * FROM ejecuciones")
    assert len(r) == 1
    assert r["estado"].iloc[0] == "ok"


def test_puntero_de_modelo_en_produccion(base_temporal):
    filas = pd.DataFrame([
        {"version": "v1.0.0", "bloque": "SmartMeter_SM_B10_ARQ", "estado": "archivada",
         "metricas": "{}", "creado_en": dt.datetime(2026, 1, 1)},
        {"version": "v1.1.0", "bloque": "SmartMeter_SM_B10_ARQ", "estado": "produccion",
         "metricas": "{}", "creado_en": dt.datetime(2026, 6, 1)},
    ])
    db.upsert(filas, "modelos", ["version", "bloque"])
    assert db.modelo_en_produccion("SmartMeter_SM_B10_ARQ")["version"] == "v1.1.0"
