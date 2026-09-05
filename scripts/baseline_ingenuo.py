"""Compara el modelo contra dos predictores triviales.

Un error del 37 % no significa nada por si solo. La pregunta que importa es:
que tan bien lo haria alguien que no usa ningun modelo?

Se comparan dos reglas sin inteligencia, sobre EXACTAMENTE los mismos instantes
que evaluo el modelo, para que la comparacion sea justa:

- ingenuo_dia:    "manana a esta hora va a consumir lo mismo que hoy a esta hora"
- ingenuo_semana: "el proximo martes a las 3 pm va a consumir lo mismo que el
                   martes pasado a las 3 pm"

La segunda suele ser dificil de superar en edificios, porque el consumo sigue el
horario academico y las semanas se parecen mucho entre si.

Si el modelo no le gana a estas reglas, el modelo no esta aportando nada y el
problema no se arregla reentrenando.

Uso:
    python -m scripts.baseline_ingenuo --detalle resultados/desempeno_v1.csv
"""
import argparse

import numpy as np
import pandas as pd

from src import config, features


def _wape(real, pred):
    real = np.asarray(real, dtype=float)
    pred = np.asarray(pred, dtype=float)
    m = ~(np.isnan(real) | np.isnan(pred))
    if m.sum() == 0:
        return np.nan
    total = np.sum(np.abs(real[m]))
    if total == 0:
        return np.nan
    return 100 * np.sum(np.abs(real[m] - pred[m])) / total


def main():
    parser = argparse.ArgumentParser(description="Comparacion contra predictores ingenuos")
    parser.add_argument("--detalle", required=True,
                        help="CSV punto a punto que produjo evaluar_modelos")
    parser.add_argument("--salida", default=None)
    args = parser.parse_args()

    print("Leyendo la evaluacion del modelo...")
    det = pd.read_csv(args.detalle, parse_dates=["origen"])
    # horas_adelante viene como decimal (10 minutos son 0,166666...), y al
    # convertirlo a duracion quedan errores de nanosegundos que hacen fallar el
    # cruce. Se redondea a la rejilla de 10 minutos, que es la real.
    det["objetivo"] = (det["origen"]
                       + pd.to_timedelta(det["horas_adelante"], unit="h")).dt.round("10min")
    print(f"  {len(det):,} puntos evaluados, {det.bloque.nunique()} bloques")

    print("Leyendo las mediciones para construir los predictores ingenuos...")
    med = features.cargar_mediciones()
    med["bloque"] = med[config.COL_EDIFICIO].str.replace(config.PREFIJO_MEDIDOR, "", regex=False)

    # Se pasa a formato largo para poder cruzar por (bloque, variable, instante)
    largo = med.melt(
        id_vars=["bloque", config.COL_TIEMPO],
        value_vars=config.VARIABLES_MULTIVAR,
        var_name="variable", value_name="valor",
    ).rename(columns={config.COL_TIEMPO: "objetivo"})

    for nombre, horas in [("ingenuo_dia", 24), ("ingenuo_semana", 168)]:
        desfasado = largo.copy()
        desfasado["objetivo"] = desfasado["objetivo"] + pd.Timedelta(hours=horas)
        desfasado = desfasado.rename(columns={"valor": nombre})
        det = det.merge(desfasado, on=["bloque", "variable", "objetivo"], how="left")

    # Para que la comparacion sea justa se evaluan los tres predictores sobre
    # exactamente los mismos puntos: se descarta cualquier instante donde a
    # alguno le falte el dato. Si no, el ingenuo se veria mejor solo por haber
    # sido evaluado en menos puntos.
    antes = len(det)
    det = det.dropna(subset=["real", "pred", "ingenuo_dia", "ingenuo_semana"])
    print(f"  {len(det):,} puntos comparables "
          f"({antes - len(det):,} descartados por falta de dato en algun predictor)")

    # ---------------- comparacion ----------------
    filas = []
    for (bloque, var), g in det.groupby(["bloque", "variable"]):
        filas.append({
            "bloque": bloque,
            "variable": var,
            "modelo": _wape(g.real, g.pred),
            "ingenuo_dia": _wape(g.real, g.ingenuo_dia),
            "ingenuo_semana": _wape(g.real, g.ingenuo_semana),
        })
    comp = pd.DataFrame(filas)
    comp["mejor_ingenuo"] = comp[["ingenuo_dia", "ingenuo_semana"]].min(axis=1)
    comp["ventaja_%"] = 100 * (comp["mejor_ingenuo"] - comp["modelo"]) / comp["mejor_ingenuo"]

    print("\n" + "=" * 74)
    print("EL MODELO CONTRA LAS REGLAS TRIVIALES   (WAPE, mas bajo es mejor)")
    print("=" * 74)

    for var in config.VARIABLES_MULTIVAR:
        sub = comp[comp.variable == var].sort_values("ventaja_%", ascending=False)
        print(f"\n{var}")
        print(sub[["bloque", "modelo", "ingenuo_dia", "ingenuo_semana", "ventaja_%"]]
              .round(1).to_string(index=False))

        g = det[det.variable == var]
        m = _wape(g.real, g.pred)
        d = _wape(g.real, g.ingenuo_dia)
        s = _wape(g.real, g.ingenuo_semana)
        mejor = min(d, s)
        print(f"  Global: modelo {m:.1f} %  |  ingenuo dia {d:.1f} %  |  "
              f"ingenuo semana {s:.1f} %")
        if m < mejor:
            print(f"  El modelo GANA por {100 * (mejor - m) / mejor:.1f} %")
        else:
            print(f"  El modelo PIERDE contra la regla trivial "
                  f"por {100 * (m - mejor) / mejor:.1f} %")

        gana = (sub.modelo < sub.mejor_ingenuo).sum()
        print(f"  Bloques donde el modelo gana: {gana} de {len(sub)}")

    if args.salida:
        comp.round(2).to_csv(args.salida, index=False)
        print(f"\nComparacion guardada en: {args.salida}")


if __name__ == "__main__":
    main()
