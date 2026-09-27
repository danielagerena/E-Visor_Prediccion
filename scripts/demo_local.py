"""Demostracion local del pipeline, sin credenciales y sin modelos.

Sirve para mostrar el sistema funcionando antes de que Sistemas entregue el
acceso a la plataforma. Levanta una base SQLite, carga el historico que ya
existe, siembra las reglas y evalua el motor de alertas.

Las "predicciones" de esta demo son las ultimas 24 horas medidas, marcadas
como version DEMO. Sirven para probar el recorrido completo; no son un
pronostico y no deben presentarse como tal.

Uso:
    python -m scripts.demo_local --csv datos/datos_preprocesados_10min.csv
"""
import argparse
import datetime as dt

import pandas as pd

from src import alertas, cargar_historico, config, db


def fabricar_predicciones_demo():
    """Copia las ultimas 24 horas medidas a la tabla de predicciones."""
    df = db.consultar(
        "SELECT * FROM mediciones WHERE ts >= "
        "(SELECT datetime(MAX(ts), '-1 day') FROM mediciones)"
    )
    if df.empty:
        return 0
    df["ts"] = pd.to_datetime(df["ts"])
    origen = df["ts"].min()

    salida = pd.DataFrame({
        config.COL_EDIFICIO: df[config.COL_EDIFICIO],
        "ts_origen": origen,
        "ts_objetivo": df["ts"],
        "activepower": df["activepower"],
        "totalpowerfactor": df["totalpowerfactor"],
        "voltaje_promedio": df["voltaje_promedio"],
        config.VAR_ACUMULATIVA: df[config.VAR_ACUMULATIVA],
        "modelo_version": "DEMO-no-es-pronostico",
        "generado_en": dt.datetime.now(dt.timezone.utc),
    })
    return db.upsert(salida, "predicciones",
                     [config.COL_EDIFICIO, "ts_origen", "ts_objetivo"])


def main():
    parser = argparse.ArgumentParser(description="Demo local de E-Visor")
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()

    print("1. Creando el esquema...")
    db.crear_esquema()

    print("2. Cargando el historico...")
    n = cargar_historico.cargar(args.csv)
    print(f"   {n:,} filas en mediciones")

    print("3. Sembrando las reglas de alerta...")
    print(f"   reglas creadas: {alertas.sembrar_reglas()}")

    print("4. Generando predicciones de demostracion...")
    print(f"   {fabricar_predicciones_demo():,} filas")

    print("5. Evaluando el motor de alertas...")
    eventos = alertas.ejecutar()
    if eventos:
        resumen = pd.DataFrame(eventos).groupby(["nombre", "severidad"]).size()
        print(resumen.to_string())
    else:
        print("   sin eventos")

    print("\nListo. Levanta el tablero con:  streamlit run app/app.py")


if __name__ == "__main__":
    main()
