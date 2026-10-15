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

# Tramos de agosto donde el dato no es medido sino interpolado: el relleno de
# huecos cortos rellena tambien las primeras 12 horas de los huecos largos.
# Los detecto diagnostico_interpolacion.py. Afectan a los 16 bloques.
VENTANAS_INVENTADAS = [("2026-08-13 01:10", "2026-08-13 13:00"),
                       ("2026-08-20 01:10", "2026-08-20 13:00")]


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
                         "origen": ap.origen.values, "paso": ap.paso.values,
                         "real": real, "pred": ap.pred.values / 6.0,
                         "ingenuo": antes})


def quitar_inventados(todo):
    """Deja fuera las filas cuyo valor real no fue medido sino interpolado.

    Se descartan dos casos: cuando el instante que se evalua cae en un tramo
    inventado, y cuando el que cae ahi es el de la semana anterior, porque ese
    es el valor con el que se compara la regla trivial.
    """
    ts = todo.origen + pd.to_timedelta(todo.paso * 10, unit="min")
    malo = pd.Series(False, index=todo.index)
    for a, b in VENTANAS_INVENTADAS:
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        malo |= ts.between(a, b)
        malo |= (ts + pd.Timedelta(days=7)).between(a, b)
    return todo[~malo], int(malo.sum())


def wape_agrupado(todo):
    """WAPE del conjunto: error total sobre consumo total, por variable."""
    g = todo.groupby("variable")
    return pd.DataFrame({
        "wape_modelo": g.apply(
            lambda x: (x.real - x.pred).abs().sum() / x.real.abs().sum() * 100,
            include_groups=False),
        "wape_ingenuo": g.apply(
            lambda x: (x.real - x.ingenuo).abs().sum() / x.real.abs().sum()
            * 100, include_groups=False),
        "filas": g.size(),
    })


def curva_por_variable(detalle):
    """Como crece el error a medida que la prediccion se aleja, por variable.

    Es lo que la app usa para dibujar el margen de error alrededor de cada
    curva. Antes solo existia para la potencia activa, y por eso el margen solo
    se podia mostrar ahi.
    """
    partes = []
    for v in sorted(detalle.variable.unique()):
        c = E.error_por_horizonte(detalle, v).reset_index()
        c.insert(0, "variable", v)
        partes.append(c)
    return pd.concat(partes, ignore_index=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--artefactos", default="modelos/v4")
    p.add_argument("--nombre", default="modelo_global")
    p.add_argument("--csv", default="datos/datos_preprocesados_10min.csv")
    p.add_argument("--salida", default=None)
    p.add_argument("--salida-limpia", default=None,
                   help="CSV con el error calculado sin los tramos interpolados")
    p.add_argument("--sin-bloques", default="B8_CPA",
                   help="Bloques que se dejan fuera del WAPE del conjunto")
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

    todo = pd.concat([det, det_inc], ignore_index=True)
    todo = todo.dropna(subset=["real", "pred", "ingenuo"])

    tabla = E.tabla_metricas(todo)
    tabla["gana"] = np.where(tabla.wape_modelo < tabla.wape_ingenuo, "si", "no")
    tabla = tabla[["bloque", "variable", "wape_modelo", "wape_ingenuo", "gana"]].round(3)

    salida = args.salida or os.path.join(args.artefactos, "error_por_bloque.csv")
    tabla.to_csv(salida, index=False)
    print(f"Guardado en {salida}")

    curva = curva_por_variable(todo).round(3)
    ruta_curva = os.path.join(args.artefactos, "error_por_horizonte_variable.csv")
    curva.to_csv(ruta_curva, index=False)
    print(f"Guardado en {ruta_curva}\n")

    ancha = tabla.pivot(index="bloque", columns="variable", values="wape_modelo")
    print("WAPE % sobre agosto, por bloque y variable")
    print(ancha.round(2).to_string())

    # Comparacion de escenarios
    limpio, quitadas = quitar_inventados(todo)
    print(f"\nFilas descartadas por caer en tramos interpolados: {quitadas:,}"
          f" de {len(todo):,} ({quitadas / len(todo) * 100:.2f} %)")

    fuera = [b.strip() for b in args.sin_bloques.split(",") if b.strip()]
    sin = todo[~todo.bloque.isin(fuera)]
    sin_limpio = limpio[~limpio.bloque.isin(fuera)]

    escenarios = [("todo el mes, los 16 bloques", todo),
                  ("sin los tramos interpolados", limpio),
                  (f"sin {', '.join(fuera)}", sin),
                  (f"sin tramos interpolados y sin {', '.join(fuera)}",
                   sin_limpio)]
    for nombre, marco in escenarios:
        if len(marco):
            print(f"\nWAPE % del conjunto, {nombre}")
            print(wape_agrupado(marco).round(3).to_string())

    if args.salida_limpia:
        t = E.tabla_metricas(limpio)
        t["gana"] = np.where(t.wape_modelo < t.wape_ingenuo, "si", "no")
        t[["bloque", "variable", "wape_modelo", "wape_ingenuo", "gana"]].round(
            3).to_csv(args.salida_limpia, index=False)
        print(f"\nGuardado en {args.salida_limpia}")


if __name__ == "__main__":
    main()