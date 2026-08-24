"""Carga en la base el historico que ya existe.

Sirve para tener la base poblada y el tablero funcionando ANTES de que
Sistemas entregue el acceso a la plataforma. Es lo que convierte "estoy
esperando permisos" en "el sistema esta listo y solo falta conectar la fuente".

Uso:
    python -m src.cargar_historico --csv datos/datos_preprocesados_10min.csv
"""
import argparse
import datetime as dt

import pandas as pd

from . import config, db, preprocesamiento


def cargar(ruta_csv, ya_preprocesado=True):
    df = pd.read_csv(ruta_csv)

    # El CSV del notebook usa 'timestamp'; internamente la columna se llama 'ts'
    if "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": config.COL_TIEMPO})
    df[config.COL_TIEMPO] = pd.to_datetime(df[config.COL_TIEMPO])

    if not ya_preprocesado:
        df = preprocesamiento.preprocesar(df, ya_normalizado=True)
    else:
        # El preprocesado viejo no traia la columna de incremento corregido
        if config.VAR_INCREMENTO not in df.columns:
            df = preprocesamiento.calcular_incremento_energia(df)

    df["ingerido_en"] = dt.datetime.now(dt.timezone.utc)
    columnas = ([config.COL_EDIFICIO, config.COL_TIEMPO]
                + [c for c in config.VARIABLES_TODAS if c in df.columns]
                + ["ingerido_en"])
    return db.upsert(df[columnas], "mediciones", [config.COL_EDIFICIO, config.COL_TIEMPO])


def main():
    parser = argparse.ArgumentParser(description="Cargue del historico de E-Visor")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--crudo", action="store_true",
                        help="el CSV viene sin preprocesar")
    args = parser.parse_args()

    db.crear_esquema()
    n = cargar(args.csv, ya_preprocesado=not args.crudo)
    print(f"Filas cargadas en mediciones: {n:,}")


if __name__ == "__main__":
    main()
