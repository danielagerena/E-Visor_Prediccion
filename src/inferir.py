"""Genera el pronostico de 24 horas de cada bloque y lo guarda en la base.

Esta es la pieza que le quita el trabajo pesado a Streamlit. La app pasa a leer
una tabla; los modelos solo se cargan aqui, en un proceso que corre solo.
"""
import argparse
import datetime as dt
import json
import os

import numpy as np
import pandas as pd

from . import calidad, config, db, features


def _cargar_artefactos(carpeta):
    """Carga modelo y escaladores de un bloque, forzando CPU."""
    import joblib
    from darts.models import BlockRNNModel

    scaler = joblib.load(os.path.join(carpeta, "scaler.joblib"))
    modelo = BlockRNNModel.load(os.path.join(carpeta, "modelo_blockrnn.pt"))
    try:
        modelo.to_cpu()
    except Exception:
        pass

    ruta_cov = os.path.join(carpeta, "scaler_cov.joblib")
    scaler_cov = joblib.load(ruta_cov) if os.path.exists(ruta_cov) else None
    return scaler, modelo, scaler_cov


def predecir_bloque(df, bloque, carpeta_modelos=None):
    """Pronostica las proximas 24 horas de un bloque.

    Devuelve un DataFrame con una fila por instante predicho, listo para
    escribir en la tabla `predicciones`.
    """
    from darts.dataprocessing.transformers import MissingValuesFiller

    carpeta_modelos = carpeta_modelos or config.CARPETA_MODELOS
    carpeta = os.path.join(str(carpeta_modelos), bloque)

    serie_mv, serie_en = features.series_bloque(df, bloque)
    if serie_mv is None:
        return pd.DataFrame()

    scaler, modelo, scaler_cov = _cargar_artefactos(carpeta)
    filler = MissingValuesFiller()
    serie_s = scaler.transform(filler.transform(serie_mv))

    cov = None
    if scaler_cov is not None:
        cov = features.extender_covariables_al_futuro(serie_mv, scaler_cov)

    pred_s = modelo.predict(n=config.HORIZONTE, series=serie_s, past_covariates=cov)
    pred = scaler.inverse_transform(pred_s)

    # Energia acumulada: se reconstruye sumando el consumo previsto al ultimo
    # valor real del contador. Predecir el contador directamente da errores
    # porcentuales ridiculamente bajos que no significan nada.
    dfb = df[df[config.COL_EDIFICIO] == config.nombre_completo(bloque)]
    ritmo = features.pendiente_energia(dfb)
    ultimo = features.ultimo_valor_valido(serie_en.values().flatten())
    acumulada = ultimo + np.cumsum(np.repeat(ritmo, config.HORIZONTE))

    indice = pd.date_range(
        serie_mv.end_time() + serie_mv.freq,
        periods=config.HORIZONTE, freq=config.FRECUENCIA,
    )

    salida = pd.DataFrame({
        config.COL_EDIFICIO: config.nombre_completo(bloque),
        "ts_origen": serie_mv.end_time(),
        "ts_objetivo": indice,
    })
    for var in config.VARIABLES_MULTIVAR:
        salida[var] = pred[var].values().flatten()
    salida[config.VAR_ACUMULATIVA] = acumulada

    # Prueba de humo: si el modelo devuelve nulos o una linea plana, no se guarda
    for var in config.VARIABLES_MULTIVAR:
        ok, problemas = calidad.validar_prediccion(salida[var], var)
        if not ok:
            raise ValueError(f"{bloque}/{var}: " + "; ".join(problemas))

    return salida


def ejecutar(bloques=None, carpeta_modelos=None):
    """Recorre todos los bloques. Un bloque que falla no tumba a los demas."""
    inicio = dt.datetime.now(dt.timezone.utc)
    bloques = bloques or config.BLOQUES
    df = features.cargar_mediciones()
    if df.empty:
        db.registrar_ejecucion("inferencia", inicio, "error", 0, "no hay mediciones")
        raise RuntimeError("No hay mediciones en la base")

    total, fallos = 0, []
    for bloque in bloques:
        try:
            salida = predecir_bloque(df, bloque, carpeta_modelos)
            if salida.empty:
                fallos.append(f"{bloque}: sin datos")
                continue
            info = db.modelo_en_produccion(config.nombre_completo(bloque))
            salida["modelo_version"] = info["version"] if info else "v1.0.0"
            salida["generado_en"] = dt.datetime.now(dt.timezone.utc)
            total += db.upsert(
                salida, "predicciones",
                [config.COL_EDIFICIO, "ts_origen", "ts_objetivo"],
            )
            print(f"OK {bloque}")
        except Exception as e:
            fallos.append(f"{bloque}: {e}")
            print(f"FALLO {bloque}: {e}")

    estado = "ok" if not fallos else "error"
    db.registrar_ejecucion("inferencia", inicio, estado, total, "; ".join(fallos))
    if len(fallos) == len(bloques):
        raise RuntimeError("Ningun bloque pudo predecir: " + "; ".join(fallos))
    return total


def main():
    parser = argparse.ArgumentParser(description="Inferencia de E-Visor")
    parser.add_argument("--bloques", nargs="*", default=None)
    parser.add_argument("--modelos", default=None)
    args = parser.parse_args()

    db.crear_esquema()
    n = ejecutar(args.bloques, args.modelos)
    print(f"Filas de prediccion escritas: {n}")


if __name__ == "__main__":
    main()
