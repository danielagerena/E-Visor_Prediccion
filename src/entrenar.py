"""Entrenamiento de un bloque, ejecutable sin intervencion humana.

Es el notebook 4 convertido en script. Diferencias importantes frente al
notebook:

1. La validacion es de origen deslizante con varios cortes, no un solo fold.
   Un fold no es validacion cruzada: es una medicion suelta.
2. El conjunto de prueba temporal es siempre el mismo (los ultimos 30 dias) y
   se guarda en la metadata, para que campeon y retador se comparen sobre lo
   mismo. Compararlos sobre particiones distintas no significa nada.
3. La metrica principal es WAPE, no MAPE. El MAPE de la potencia activa se
   dispara de madrugada, cuando el consumo se acerca a cero.
4. Cada corrida deja un paquete completo: modelo, escaladores y metadata.json
   con rango de datos, metricas, versiones y commit. Sin eso, una prediccion
   del pasado no se puede explicar.
"""
import argparse
import datetime as dt
import json
import os
import subprocess

import numpy as np
import pandas as pd

from . import calidad, config, features


def _commit_actual():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "desconocido"


def cortes_origen_deslizante(n_total, n_cortes=4, pasos_prueba=None):
    """Origen deslizante: cada corte entrena con mas historia y prueba adelante.

    Se prefiere a "mas folds" porque con pocos meses de datos, mas folds
    significa bloques de prueba mas cortos y estimaciones mas ruidosas.
    """
    pasos_prueba = pasos_prueba or config.PASOS_DIA * 7
    primero = int(n_total * 0.5)
    ultimo = n_total - pasos_prueba
    if ultimo <= primero:
        return []
    return [int(c) for c in np.linspace(primero, ultimo, n_cortes)]


def construir_modelo(cfg):
    from darts.models import BlockRNNModel

    return BlockRNNModel(
        model="LSTM",
        input_chunk_length=config.INPUT_CHUNK,
        output_chunk_length=config.HORIZONTE,
        hidden_dim=cfg["hidden_dim"],
        n_rnn_layers=cfg["n_rnn_layers"],
        dropout=cfg["dropout"],
        n_epochs=cfg["n_epochs"],
        batch_size=cfg["batch_size"],
        random_state=config.SEED,
        force_reset=True,
        save_checkpoints=False,
    )


def entrenar_bloque(bloque, salida, n_cortes=4, warm_start_desde=None):
    """Entrena un bloque y deja el paquete de version en `salida`."""
    import joblib
    from darts.dataprocessing.transformers import MissingValuesFiller, Scaler
    from sklearn.preprocessing import MinMaxScaler

    df = features.cargar_mediciones(bloque=bloque)
    ok, problemas = calidad.validar_lote(df)
    if not ok:
        raise ValueError(f"Compuerta de datos: {'; '.join(problemas)}")

    serie_mv, serie_en = features.series_bloque(df, bloque)
    cfg = config.config_bloque(bloque)
    filler = MissingValuesFiller()

    # Prueba temporal fija: los ultimos 30 dias, iguales para todos los modelos
    pasos_prueba = config.PASOS_DIA * 30
    n = len(serie_mv)
    corte_prueba = n - pasos_prueba
    entrenamiento = serie_mv[:corte_prueba]
    prueba = serie_mv[corte_prueba:]

    scaler = Scaler(MinMaxScaler(feature_range=(0, 1)))
    entrenamiento_s = scaler.fit_transform(filler.transform(entrenamiento))

    cov, scaler_cov = (None, None)
    if cfg["usar_covariables_calendario"]:
        cov, scaler_cov = features.covariables_calendario(serie_mv, ajustar=True)

    # ---- validacion de origen deslizante ----
    resultados = []
    for i, corte in enumerate(cortes_origen_deslizante(len(entrenamiento), n_cortes), 1):
        tr = entrenamiento[:corte]
        va = entrenamiento[corte:corte + config.HORIZONTE]
        if len(va) < config.HORIZONTE:
            continue
        sc = Scaler(MinMaxScaler(feature_range=(0, 1)))
        tr_s = sc.fit_transform(filler.transform(tr))
        modelo = construir_modelo(cfg)
        modelo.fit(series=tr_s, past_covariates=cov, verbose=False)
        pred = sc.inverse_transform(
            modelo.predict(n=config.HORIZONTE, series=tr_s, past_covariates=cov)
        )
        real = filler.transform(va)
        for var in config.VARIABLES_MULTIVAR:
            resultados.append({
                "corte": i, "variable": var,
                "WAPE_%": features.wape(real[var].values().flatten(),
                                        pred[var].values().flatten()),
            })
        print(f"  corte {i} terminado")

    df_cv = pd.DataFrame(resultados)

    # ---- modelo final con todo lo anterior a la prueba ----
    modelo_final = construir_modelo(cfg)
    if warm_start_desde:
        from darts.models import BlockRNNModel
        modelo_final = BlockRNNModel.load(warm_start_desde)
        modelo_final.n_epochs = max(5, cfg["n_epochs"] // 10)
    modelo_final.fit(series=entrenamiento_s, past_covariates=cov, verbose=False)

    pred = scaler.inverse_transform(
        modelo_final.predict(n=config.HORIZONTE, series=entrenamiento_s, past_covariates=cov)
    )
    real = filler.transform(prueba)[:config.HORIZONTE]

    metricas = {
        var: float(features.wape(real[var].values().flatten(),
                                 pred[var].values().flatten()))
        for var in config.VARIABLES_MULTIVAR
    }

    # Energia: se evalua sobre el consumo por intervalo, no sobre el contador
    dfb = df.sort_values(config.COL_TIEMPO)
    ritmo = features.pendiente_energia(dfb.iloc[:corte_prueba])
    real_inc = dfb[config.VAR_INCREMENTO].iloc[corte_prueba:corte_prueba + config.HORIZONTE]
    metricas[config.VAR_INCREMENTO] = float(
        features.wape(real_inc.values, np.repeat(ritmo, len(real_inc)))
    )

    # ---- paquete de version ----
    os.makedirs(salida, exist_ok=True)
    modelo_final.save(os.path.join(salida, "modelo_blockrnn.pt"))
    joblib.dump(scaler, os.path.join(salida, "scaler.joblib"))
    if scaler_cov is not None:
        joblib.dump(scaler_cov, os.path.join(salida, "scaler_cov.joblib"))
    joblib.dump({"pendiente": ritmo, "dias_ventana": config.DIAS_VENTANA_DERIVA},
                os.path.join(salida, "modelo_deriva.joblib"))
    df_cv.to_csv(os.path.join(salida, "validacion_origen_deslizante.csv"), index=False)

    metadata = {
        "bloque": bloque,
        "entrenado_en": dt.datetime.now(dt.timezone.utc).isoformat(),
        "datos_desde": str(df[config.COL_TIEMPO].min()),
        "datos_hasta": str(df[config.COL_TIEMPO].max()),
        "prueba_desde": str(prueba.start_time()),
        "prueba_hasta": str(prueba.end_time()),
        "hiperparametros": cfg,
        "variables": config.VARIABLES_MULTIVAR,
        "metricas_wape": metricas,
        "commit": _commit_actual(),
        "warm_start": bool(warm_start_desde),
    }
    with open(os.path.join(salida, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(json.dumps(metricas, indent=2))
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento de E-Visor")
    parser.add_argument("--bloque", required=True)
    parser.add_argument("--salida", required=True)
    parser.add_argument("--cortes", type=int, default=4)
    parser.add_argument("--warm-start", default=None,
                        help="ruta al modelo actual para continuar entrenando")
    args = parser.parse_args()
    entrenar_bloque(args.bloque, args.salida, args.cortes, args.warm_start)


if __name__ == "__main__":
    main()
