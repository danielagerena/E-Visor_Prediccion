"""Calcula el error de agosto por bloque y por variable.

Usa el modelo del EXPERIMENTO (modelo_global), el que se entreno sin ver
agosto. No usa el modelo de produccion: ese ya vio agosto y su error sobre ese
mes no significa nada.

Va en scripts/. Se corre desde la raiz del repositorio:

    python scripts/evaluar_por_bloque.py --artefactos modelos/v4 --csv datos/datos_preprocesados_10min.csv

Escribe modelos/v4/error_por_bloque.csv, que es lo que lee la app.
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
from src import evisor_lstm as E  # noqa: E402

VAR_INC = "activeenergyimport_incremento"
PRUEBA = ("2026-08-01", "2026-09-01")


def error_incrementos(det, df):
    """Error de la energia cada 10 minutos.

    La energia no tiene modelo propio: se deriva de la potencia predicha
    dividiendo entre 6. Aqui se compara esa derivacion contra el incremento que
    registro el contador de verdad, y la regla trivial contra el incremento de
    la semana anterior.
    """
    ap = det[det.variable == "activepower"].copy()
    ap["ts"] = ap.origen + pd.to_timedelta(ap.paso * 10, unit="min")
    medido = (df.dropna(subset=[VAR_INC])
                .set_index(["bloque", "timestamp"])[VAR_INC])

    real = medido.reindex(pd.MultiIndex.from_arrays([ap.bloque, ap.ts])).values
    antes = medido.reindex(pd.MultiIndex.from_arrays(
        [ap.bloque, ap.ts - pd.Timedelta(days=7)])).values
    return pd.DataFrame({"bloque": ap.bloque.values, "variable": VAR_INC,
                         "paso": ap.paso.values, "real": real,
                         "pred": ap.pred.values / 6.0, "ingenuo": antes})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--artefactos", default="modelos/v4")
    p.add_argument("--nombre", default="modelo_global")
    p.add_argument("--csv", default="datos/datos_preprocesados_10min.csv")
    p.add_argument("--salida", default=None)
    args = p.parse_args()
    warnings.filterwarnings("ignore")

    modelo, lims, ficha = E.cargar_artefactos(args.artefactos, args.nombre)
    if "produccion" in str(ficha.get("version", "")):
        print("AVISO: este es el modelo de produccion, que ya vio agosto. "
              "El error que salga no es una medicion honesta. Usa modelo_global.")
    E.LARGO_ENTRADA = ficha.get("largo_entrada", E.LARGO_ENTRADA)
    E.HORIZONTE = ficha.get("horizonte", E.HORIZONTE)
    print(f"Modelo {args.nombre} | entrada {E.LARGO_ENTRADA} pasos")

    df = E.cargar_datos(args.csv)
    bloques = [b for b in sorted(df.bloque.unique()) if b in lims]
    tablas = {b: E.tabla_bloque(df, b) for b in bloques}

    print("Evaluando agosto...")
    det = E.evaluar(modelo, tablas, lims, bloques, PRUEBA, ficha.get("indice_bloque"))
    det_inc = error_incrementos(det, df)

    tabla = E.tabla_metricas(pd.concat([det, det_inc], ignore_index=True))
    tabla["gana"] = np.where(tabla.wape_modelo < tabla.wape_ingenuo, "si", "no")
    tabla = tabla[["bloque", "variable", "wape_modelo", "wape_ingenuo", "gana"]].round(3)

    salida = args.salida or os.path.join(args.artefactos, "error_por_bloque.csv")
    tabla.to_csv(salida, index=False)
    print(f"Guardado en {salida}\n")

    ancha = tabla.pivot(index="bloque", columns="variable", values="wape_modelo")
    print("WAPE % sobre agosto, por bloque y variable")
    print(ancha.round(2).to_string())


if __name__ == "__main__":
    main()
