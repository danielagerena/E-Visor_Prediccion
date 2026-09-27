"""Acceso a la base de datos.

Una sola puerta de entrada. El resto del codigo nunca escribe SQL de conexion
ni sabe si detras hay PostgreSQL o SQLite.
"""
import datetime as dt
import json

import pandas as pd
from sqlalchemy import create_engine, text

from . import config

_motor = None


def motor():
    """Devuelve el motor de conexion, creandolo la primera vez."""
    global _motor
    if _motor is None:
        _motor = create_engine(config.DB_URL, future=True)
    return _motor


def crear_esquema():
    """Ejecuta esquema.sql. Es idempotente: se puede correr muchas veces."""
    ruta = config.RAIZ / "src" / "esquema.sql"
    # Se quitan los comentarios antes de partir por ';' para que ninguna
    # sentencia quede escondida detras de una linea de comentario.
    limpio = "\n".join(
        linea for linea in ruta.read_text(encoding="utf-8").splitlines()
        if not linea.strip().startswith("--")
    )
    with motor().begin() as con:
        for sentencia in limpio.split(";"):
            if sentencia.strip():
                con.execute(text(sentencia))


def consultar(sql, params=None):
    """Devuelve un DataFrame a partir de una consulta."""
    return pd.read_sql(text(sql), motor(), params=params or {})


def upsert(df, tabla, claves):
    """Inserta o actualiza filas segun la clave primaria.

    Es lo que hace que un job se pueda repetir sin duplicar nada: si la fila
    ya existe con esa clave, se sobrescribe en vez de agregarse.
    """
    if df.empty:
        return 0

    columnas = list(df.columns)
    no_clave = [c for c in columnas if c not in claves]
    lista_cols = ", ".join(columnas)
    lista_vals = ", ".join(f":{c}" for c in columnas)
    conflicto = ", ".join(claves)

    if no_clave:
        actualizacion = ", ".join(f"{c} = excluded.{c}" for c in no_clave)
        sufijo = f"DO UPDATE SET {actualizacion}"
    else:
        sufijo = "DO NOTHING"

    sql = text(
        f"INSERT INTO {tabla} ({lista_cols}) VALUES ({lista_vals}) "
        f"ON CONFLICT ({conflicto}) {sufijo}"
    )

    # SQLite no acepta objetos Timestamp de pandas; se pasan a datetime nativo
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.to_pydatetime()

    registros = df.astype(object).where(pd.notna(df), None).to_dict("records")
    for r in registros:
        for k, v in r.items():
            if isinstance(v, pd.Timestamp):
                r[k] = v.to_pydatetime()
    with motor().begin() as con:
        for i in range(0, len(registros), 1000):
            con.execute(sql, registros[i:i + 1000])
    return len(registros)


def registrar_ejecucion(job, inicio, estado, filas=0, detalle=""):
    """Deja constancia de una corrida. El vigilante lee esta tabla."""
    with motor().begin() as con:
        con.execute(
            text(
                "INSERT INTO ejecuciones (job, inicio, fin, estado, filas, detalle) "
                "VALUES (:job, :inicio, :fin, :estado, :filas, :detalle)"
            ),
            {
                "job": job,
                "inicio": inicio,
                "fin": dt.datetime.now(dt.timezone.utc),
                "estado": estado,
                "filas": int(filas),
                "detalle": detalle[:2000],
            },
        )


def ultima_medicion(entity_id=None):
    """Marca de tiempo mas reciente en mediciones. Base de la ingesta incremental."""
    sql = "SELECT MAX(ts) AS ts FROM mediciones"
    params = {}
    if entity_id:
        sql += " WHERE entity_id = :e"
        params["e"] = entity_id
    r = consultar(sql, params)
    return pd.to_datetime(r["ts"].iloc[0]) if not r.empty and r["ts"].iloc[0] else None


def modelo_en_produccion(bloque):
    """Version del modelo marcada como produccion para un bloque."""
    r = consultar(
        "SELECT version, url, metricas FROM modelos "
        "WHERE bloque = :b AND estado = 'produccion' "
        "ORDER BY creado_en DESC LIMIT 1",
        {"b": bloque},
    )
    if r.empty:
        return None
    fila = r.iloc[0].to_dict()
    if fila.get("metricas"):
        fila["metricas"] = json.loads(fila["metricas"])
    return fila
