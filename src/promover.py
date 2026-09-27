"""Compuertas de promocion de modelos.

La regla de oro: el proceso nunca sobrescribe un modelo. Publica una version
nueva e inmutable y, solo si pasa todas las compuertas, mueve el puntero de
produccion. Revertir es volver a mover el puntero, un UPDATE de una linea.
"""
import argparse
import datetime as dt
import json
import os

import pandas as pd

from . import calidad, config, db, inferir_v4

TOLERANCIA = 0.02   # el retador puede ser hasta 2 % peor y aun asi entrar


def evaluar_compuertas(metadata, metricas_campeon=None, tolerancia=TOLERANCIA):
    """Devuelve (aprobado, motivos). Cada motivo explica por que se rechaza."""
    motivos = []

    metricas = metadata.get("metricas_wape") or {}
    if not metricas:
        motivos.append("El entrenamiento no reporto metricas")

    for var, valor in metricas.items():
        if valor is None or pd.isna(valor):
            motivos.append(f"{var}: metrica nula")
        elif valor > 100:
            motivos.append(f"{var}: WAPE de {valor:.1f} % es peor que no predecir nada")

    if metricas_campeon:
        for var, valor in metricas.items():
            ref = metricas_campeon.get(var)
            if ref is None or pd.isna(ref) or valor is None or pd.isna(valor):
                continue
            if valor > ref * (1 + tolerancia):
                motivos.append(
                    f"{var}: el retador empeora ({valor:.2f} % frente a {ref:.2f} %)"
                )

    if metadata.get("prueba_desde") is None:
        motivos.append("Falta el rango del conjunto de prueba en la metadata")

    return len(motivos) == 0, motivos


def prueba_de_humo(carpeta, bloque):
    """Carga el artefacto en limpio y genera 24 horas.

    Atrapa el fallo clasico: un modelo que entrena bien pero no se puede volver
    a cargar por un cambio de version de libreria.
    """
    from . import features

    df = features.cargar_mediciones(bloque=bloque)
    salida = inferir_v4.predecir_bloque(df, bloque, carpeta_modelos=os.path.dirname(carpeta))
    if salida.empty:
        return False, ["La prediccion de prueba vino vacia"]
    for var in config.VARIABLES_MULTIVAR:
        ok, problemas = calidad.validar_prediccion(salida[var], var)
        if not ok:
            return False, problemas
    return True, []


def promover(bloque, version, carpeta, url="", con_humo=True):
    """Registra la version y, si pasa todo, la marca como produccion."""
    completo = config.nombre_completo(bloque)
    with open(os.path.join(carpeta, "metadata.json"), encoding="utf-8") as f:
        metadata = json.load(f)

    campeon = db.modelo_en_produccion(completo)
    metricas_campeon = (campeon or {}).get("metricas", {}).get("metricas_wape") \
        if campeon else None

    aprobado, motivos = evaluar_compuertas(metadata, metricas_campeon)

    if aprobado and con_humo:
        ok_humo, problemas = prueba_de_humo(carpeta, bloque)
        if not ok_humo:
            aprobado = False
            motivos.extend(problemas)

    fila = {
        "version": version,
        "bloque": completo,
        "estado": "produccion" if aprobado else "rechazada",
        "url": url,
        "metricas": json.dumps(metadata),
        "datos_desde": metadata.get("datos_desde"),
        "datos_hasta": metadata.get("datos_hasta"),
        "commit_sha": metadata.get("commit"),
        "creado_en": dt.datetime.now(dt.timezone.utc),
    }
    db.upsert(pd.DataFrame([fila]), "modelos", ["version", "bloque"])

    if aprobado and campeon:
        from sqlalchemy import text
        with db.motor().begin() as con:
            con.execute(
                text("UPDATE modelos SET estado='archivada' "
                     "WHERE bloque=:b AND version<>:v AND estado='produccion'"),
                {"b": completo, "v": version},
            )

    return aprobado, motivos


def revertir(bloque, version):
    """Vuelve a poner en produccion una version anterior."""
    from sqlalchemy import text

    completo = config.nombre_completo(bloque)
    with db.motor().begin() as con:
        con.execute(
            text("UPDATE modelos SET estado='archivada' "
                 "WHERE bloque=:b AND estado='produccion'"), {"b": completo})
        con.execute(
            text("UPDATE modelos SET estado='produccion' "
                 "WHERE bloque=:b AND version=:v"), {"b": completo, "v": version})
    return True


def main():
    parser = argparse.ArgumentParser(description="Promocion de modelos de E-Visor")
    parser.add_argument("--bloque", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--carpeta", required=True)
    parser.add_argument("--url", default="")
    parser.add_argument("--revertir", action="store_true")
    parser.add_argument("--sin-humo", action="store_true")
    args = parser.parse_args()

    db.crear_esquema()
    if args.revertir:
        revertir(args.bloque, args.version)
        print(f"{args.bloque} revertido a {args.version}")
        return

    aprobado, motivos = promover(args.bloque, args.version, args.carpeta,
                                 args.url, con_humo=not args.sin_humo)
    if aprobado:
        print(f"{args.bloque}: {args.version} promovida a produccion")
    else:
        print(f"{args.bloque}: {args.version} RECHAZADA")
        for m in motivos:
            print(f"  - {m}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
