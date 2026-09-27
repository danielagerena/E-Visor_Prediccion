"""Reemplaza al notebook 3: convierte el consolidado crudo en la rejilla de 10 minutos.

Hace lo mismo que el notebook, con dos diferencias que importan:

1. Corrige los reinicios del contador sumando el techo del medidor, en vez de
   reemplazar ese dato por la mediana del bloque. El notebook inventaba un
   valor que en realidad se conoce.
2. Imprime un reporte de validacion para que puedas revisar los datos nuevos
   antes de entrenar sobre ellos.

Entrada:  el CSV que produce el notebook 1 (datos_crudos_consolidados.csv)
Salida:   el CSV de 10 minutos listo para subir al repositorio

Uso:
    python -m scripts.preprocesar_crudo \
        --entrada datos/datos_crudos_consolidados.csv \
        --salida  datos/datos_preprocesados_10min.csv
"""
import argparse

import numpy as np
import pandas as pd

from src import config, preprocesamiento as pre

# La salida conserva el nombre 'timestamp' y la columna de energia absoluta
# para que la app que hoy esta publicada siga funcionando sin cambios.
COLUMNAS_SALIDA = [
    config.COL_EDIFICIO,
    "timestamp",
    "activepower",
    "totalpowerfactor",
    "voltaje_promedio",
    config.VAR_ACUMULATIVA,
    config.VAR_INCREMENTO,
]


def reporte_huecos(df, horas=1, variable="activepower"):
    """Tramos sin dato mayores a N horas, por bloque.

    Tras el remuestreo la rejilla es continua: los huecos no son filas que
    falten, son filas con valor nulo. Por eso se buscan rachas de nulos
    consecutivos y no saltos en la marca de tiempo.
    """
    pasos_minimos = int(horas * 60 / 10)
    filas = []
    for edificio, g in df.groupby(config.COL_EDIFICIO):
        g = g.sort_values(config.COL_TIEMPO).reset_index(drop=True)
        nulo = g[variable].isna().to_numpy()
        inicio = None
        for i in range(len(nulo) + 1):
            hay_nulo = nulo[i] if i < len(nulo) else False
            if hay_nulo and inicio is None:
                inicio = i
            elif not hay_nulo and inicio is not None:
                if i - inicio >= pasos_minimos:
                    filas.append({
                        "bloque": config.nombre_corto(edificio),
                        "desde": g[config.COL_TIEMPO].iloc[inicio],
                        "hasta": g[config.COL_TIEMPO].iloc[i - 1],
                        "horas": round((i - inicio) * 10 / 60, 1),
                    })
                inicio = None
    return pd.DataFrame(filas)


def reporte_por_mes(df):
    """Filas y porcentaje de nulos por mes. Sirve para ver que los meses
    nuevos llegaron completos."""
    t = df.copy()
    t["mes"] = t[config.COL_TIEMPO].dt.to_period("M").astype(str)
    tabla = t.groupby("mes").agg(
        filas=("activepower", "size"),
        pct_nulos=("activepower", lambda s: s.isna().mean() * 100),
    )
    tabla.columns = ["filas", "% nulos"]
    return tabla.round(1)


def main():
    parser = argparse.ArgumentParser(description="Preprocesamiento de E-Visor")
    parser.add_argument("--entrada", required=True, help="CSV consolidado del notebook 1")
    parser.add_argument("--salida", required=True, help="CSV de 10 minutos a generar")
    args = parser.parse_args()

    print("Leyendo el consolidado crudo...")
    df = pd.read_csv(args.entrada, low_memory=False)

    # El notebook 1 llama 'timestamp' a la columna de tiempo; el paquete la
    # llama 'ts'. Se renombra al entrar y se devuelve al nombre viejo al salir.
    if "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": config.COL_TIEMPO})
    df[config.COL_TIEMPO] = pd.to_datetime(df[config.COL_TIEMPO], errors="coerce")
    df = df.dropna(subset=[config.COL_TIEMPO, config.COL_EDIFICIO])

    print(f"  {len(df):,} filas crudas")
    print(f"  {df[config.COL_EDIFICIO].nunique()} bloques")
    print(f"  del {df[config.COL_TIEMPO].min()} al {df[config.COL_TIEMPO].max()}")

    print("\nProcesando...")
    salida = pre.preprocesar(df, ya_normalizado=True)

    # ---------------- reporte de validacion ----------------
    print("\n" + "=" * 62)
    print("REPORTE DE VALIDACION")
    print("=" * 62)

    calidad = salida.attrs.get("reporte_calidad")
    if calidad is not None and not calidad.empty:
        print("\nValores fisicamente imposibles convertidos a nulo:")
        print(calidad.to_string(index=False))
    else:
        print("\nSin valores fuera de los rangos fisicos")

    print(f"\nReinicios del contador corregidos: {salida.attrs.get('n_rollovers', 0)}")

    print(f"\nRejilla final: {len(salida):,} filas, "
          f"{salida[config.COL_EDIFICIO].nunique()} bloques")
    print(f"Rango: {salida[config.COL_TIEMPO].min()} a {salida[config.COL_TIEMPO].max()}")

    print("\nFilas y nulos por mes (revisar que los meses nuevos esten completos):")
    print(reporte_por_mes(salida).to_string())

    print("\nNulos restantes por variable (son los huecos largos, no se imputan):")
    presentes = [v for v in config.VARIABLES_TODAS if v in salida.columns]
    nulos = salida[presentes].isna().sum()
    for var, n in nulos.items():
        print(f"  {var:<32} {n:>8,}  ({100 * n / len(salida):.1f} %)")

    huecos = reporte_huecos(salida)
    if not huecos.empty:
        print(f"\nHuecos mayores a 1 hora: {len(huecos)}. Los 10 mas largos:")
        print(huecos.sort_values("horas", ascending=False).head(10).to_string(index=False))
    else:
        print("\nSin huecos mayores a 1 hora")

    print("\nEstadisticas de las variables finales:")
    print(salida[presentes].describe().round(2).to_string())

    # ---------------- exportacion ----------------
    salida = salida.rename(columns={config.COL_TIEMPO: "timestamp"})
    columnas = [c for c in COLUMNAS_SALIDA if c in salida.columns]
    salida[columnas].to_csv(args.salida, index=False)

    print("\n" + "=" * 62)
    print(f"Exportado: {args.salida}")
    print(f"Columnas: {', '.join(columnas)}")


if __name__ == "__main__":
    main()
