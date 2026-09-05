"""Mide el desempeño real de un conjunto de modelos sobre un periodo que no vieron.

Es solo inferencia: no entrena nada, corre en CPU y no toca los artefactos.

Para cada bloque se para en un instante, predice las 24 horas siguientes, compara
contra lo que realmente ocurrio, y repite avanzando dia a dia. Eso da cientos de
pronosticos en vez de uno solo, que es la diferencia entre saber como se comporta
el modelo y haber tenido suerte una vez.

La metrica principal es WAPE: el error total dividido entre el consumo total del
periodo. Se prefiere al MAPE porque el MAPE se dispara de madrugada, cuando el
consumo se acerca a cero, y produce numeros como 139 % que no describen nada.

Uso:
    python -m scripts.evaluar_modelos --desde 2026-06-01 --hasta 2026-08-31 --etiqueta v1.0.0
"""
import argparse
import logging
import os
import warnings

import numpy as np
import pandas as pd

from src import config, features

# PyTorch Lightning imprime mucho ruido por cada prediccion. Aqui estorba.
# Hay que silenciarlo DESPUES de importar darts: al importarse, Lightning
# configura sus propios niveles y pisa cualquier ajuste anterior.
warnings.filterwarnings("ignore")
from darts.models import BlockRNNModel  # noqa: E402

for nombre in ("pytorch_lightning", "lightning.pytorch", "lightning",
               "pytorch_lightning.utilities.rank_zero",
               "pytorch_lightning.accelerators.cuda"):
    logging.getLogger(nombre).setLevel(logging.ERROR)

# La energia acumulada no entra en esta evaluacion a proposito: no la produce un
# modelo entrenado sino una pendiente que se recalcula con datos recientes, asi
# que no cambia al reentrenar y no aporta a la comparacion entre versiones.


def _cargar(carpeta):
    import joblib

    scaler = joblib.load(os.path.join(carpeta, "scaler.joblib"))
    modelo = BlockRNNModel.load(os.path.join(carpeta, "modelo_blockrnn.pt"))
    try:
        modelo.to_cpu()
    except Exception:
        pass
    ruta_cov = os.path.join(carpeta, "scaler_cov.joblib")
    scaler_cov = joblib.load(ruta_cov) if os.path.exists(ruta_cov) else None
    return scaler, modelo, scaler_cov


def _metricas(real, pred):
    real = np.asarray(real, dtype=float)
    pred = np.asarray(pred, dtype=float)
    m = ~(np.isnan(real) | np.isnan(pred))
    if m.sum() == 0:
        return {"WAPE_%": np.nan, "MAE": np.nan, "RMSE": np.nan, "n": 0}
    err = real[m] - pred[m]
    total = np.sum(np.abs(real[m]))
    return {
        "WAPE_%": 100 * np.sum(np.abs(err)) / total if total > 0 else np.nan,
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "n": int(m.sum()),
    }


def evaluar_bloque(df, bloque, desde, hasta, paso_horas=24, carpeta_modelos=None):
    """Pronosticos encadenados de un bloque. Devuelve un DataFrame largo."""
    from darts.dataprocessing.transformers import MissingValuesFiller

    carpeta_modelos = carpeta_modelos or config.CARPETA_MODELOS
    carpeta = os.path.join(str(carpeta_modelos), bloque)

    serie_mv, _ = features.series_bloque(df, bloque)
    if serie_mv is None:
        return pd.DataFrame()

    scaler, modelo, scaler_cov = _cargar(carpeta)
    filler = MissingValuesFiller()
    serie_s = scaler.transform(filler.transform(serie_mv))

    cov = None
    if scaler_cov is not None:
        cov, _ = features.covariables_calendario(serie_mv, scaler_cov)

    inicio = pd.Timestamp(desde)
    fin = pd.Timestamp(hasta)
    paso = int(paso_horas * 60 / 10)

    pronosticos = modelo.historical_forecasts(
        series=serie_s,
        past_covariates=cov,
        start=inicio,
        forecast_horizon=config.HORIZONTE,
        stride=paso,
        retrain=False,
        last_points_only=False,
        verbose=False,
    )

    real = filler.transform(serie_mv)
    filas = []
    for pronostico in pronosticos:
        p = scaler.inverse_transform(pronostico)
        if p.end_time() > fin:
            continue
        r = real.slice_intersect(p)
        if len(r) == 0:
            continue
        origen = p.start_time() - serie_mv.freq
        for i, var in enumerate(config.VARIABLES_MULTIVAR):
            vr = r[var].values().flatten()
            vp = p[var].values().flatten()[:len(vr)]
            for k in range(len(vr)):
                filas.append({
                    "bloque": bloque,
                    "variable": var,
                    "origen": origen,
                    "horas_adelante": (k + 1) * 10 / 60,
                    "real": vr[k],
                    "pred": vp[k],
                })
    return pd.DataFrame(filas)


def main():
    parser = argparse.ArgumentParser(description="Evaluacion de modelos de E-Visor")
    parser.add_argument("--desde", required=True, help="inicio del periodo, AAAA-MM-DD")
    parser.add_argument("--hasta", required=True, help="fin del periodo, AAAA-MM-DD")
    parser.add_argument("--etiqueta", default="v1.0.0", help="nombre de la version evaluada")
    parser.add_argument("--paso-horas", type=int, default=24,
                        help="cada cuantas horas se lanza un pronostico nuevo")
    parser.add_argument("--bloques", nargs="*", default=None)
    parser.add_argument("--modelos", default=None, help="carpeta de artefactos")
    parser.add_argument("--salida", default=None, help="CSV con el detalle punto a punto")
    args = parser.parse_args()

    bloques = args.bloques or config.BLOQUES
    print(f"Evaluando {args.etiqueta} del {args.desde} al {args.hasta}")
    print(f"Un pronostico de 24 h cada {args.paso_horas} horas\n")

    df = features.cargar_mediciones()
    if df.empty:
        raise SystemExit("No hay mediciones en la base. Corre src.cargar_historico primero.")

    partes = []
    for bloque in bloques:
        try:
            parte = evaluar_bloque(df, bloque, args.desde, args.hasta,
                                   args.paso_horas, args.modelos)
            if parte.empty:
                print(f"  {bloque:<10} sin datos en el periodo")
                continue
            partes.append(parte)
            n = parte.origen.nunique()
            print(f"  {bloque:<10} {n} pronosticos")
        except Exception as e:
            print(f"  {bloque:<10} FALLO: {e}")

    if not partes:
        raise SystemExit("No se pudo evaluar ningun bloque.")

    detalle = pd.concat(partes, ignore_index=True)

    # ---------------- resumen por bloque y variable ----------------
    filas = []
    for (bloque, var), g in detalle.groupby(["bloque", "variable"]):
        m = _metricas(g.real, g.pred)
        m.update({"bloque": bloque, "variable": var,
                  "pronosticos": g.origen.nunique()})
        filas.append(m)
    resumen = pd.DataFrame(filas)[
        ["bloque", "variable", "pronosticos", "WAPE_%", "MAE", "RMSE", "n"]]

    print("\n" + "=" * 70)
    print(f"DESEMPENO DE {args.etiqueta}   ({args.desde} a {args.hasta})")
    print("=" * 70)

    for var in config.VARIABLES_MULTIVAR:
        sub = resumen[resumen.variable == var].sort_values("WAPE_%")
        print(f"\n{var}")
        print(sub[["bloque", "pronosticos", "WAPE_%", "MAE"]].round(2).to_string(index=False))
        print(f"  Promedio ponderado: {_metricas(detalle[detalle.variable == var].real, detalle[detalle.variable == var].pred)['WAPE_%']:.1f} %")

    # ---------------- como se degrada con el horizonte ----------------
    print("\n" + "-" * 70)
    print("Como empeora el error a medida que se aleja el pronostico")
    tramos = [(0, 1, "1 h"), (1, 6, "6 h"), (6, 12, "12 h"), (12, 24, "24 h")]
    filas_h = []
    for var in config.VARIABLES_MULTIVAR:
        d = detalle[detalle.variable == var]
        fila = {"variable": var}
        for lo, hi, nombre in tramos:
            t = d[(d.horas_adelante > lo) & (d.horas_adelante <= hi)]
            fila[nombre] = round(_metricas(t.real, t.pred)["WAPE_%"], 1)
        filas_h.append(fila)
    print(pd.DataFrame(filas_h).to_string(index=False))

    if args.salida:
        detalle.to_csv(args.salida, index=False)
        resumen.to_csv(args.salida.replace(".csv", "_resumen.csv"), index=False)
        print(f"\nDetalle punto a punto: {args.salida}")
        print(f"Resumen por bloque:    {args.salida.replace('.csv', '_resumen.csv')}")


if __name__ == "__main__":
    main()
