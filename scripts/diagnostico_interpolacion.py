"""Mide cuántos valores inventa el relleno de huecos y dónde están los vacíos.

Dos partes:

  --demo   Demuestra con una serie de juguete qué hace realmente
           interpolate(limit=N) frente a lo que se quería que hiciera.

  (normal) Reconstruye la rejilla de 10 minutos a partir del CSV crudo,
           mide todos los huecos, lista los más largos con sus fechas
           exactas y cuenta cuántos valores rellena de más la versión
           actual de interpolar_huecos_cortos.

Uso:
    python diagnostico_interpolacion.py --demo
    python diagnostico_interpolacion.py --entrada datos/datos_crudos_consolidados.csv
"""
import argparse

import numpy as np
import pandas as pd

LIMITE = 72                      # config.LIMITE_INTERPOLACION_PASOS, 12 horas

RANGOS = {
    "activepower": (0, 2_000_000),
    "totalpowerfactor": (-1.0, 1.0),
    "activeenergyimport": (0, 500_000_000),
    "v1": (50, 500), "v2": (50, 500), "v3": (50, 500),
}
INSTANTANEAS = ["activepower", "totalpowerfactor", "voltaje_promedio"]


# ---------------------------------------------------------------------------
# Parte 1: demostración de la semántica
# ---------------------------------------------------------------------------
def demo():
    print("=" * 72)
    print("DEMOSTRACION: que hace interpolate(limit=N)")
    print("=" * 72)

    # Serie con dos huecos: uno de 3 pasos (corto) y otro de 8 (largo)
    s = pd.Series([10.0] + [np.nan] * 3 + [20.0] + [np.nan] * 8 + [100.0])
    limite = 4

    actual = s.interpolate(method="linear", limit=limite)
    deseado = s.copy()
    vacio = s.isna()
    racha = (vacio != vacio.shift()).cumsum()
    largo = vacio.groupby(racha).transform("sum").where(vacio, 0)
    relleno = s.interpolate(method="linear", limit_area="inside")
    deseado = s.where(~(vacio & (largo <= limite)), relleno)

    tabla = pd.DataFrame({"original": s, "actual": actual, "deseado": deseado})
    print(f"\nlimite = {limite} pasos\n")
    print(tabla.to_string(float_format=lambda x: f"{x:8.2f}"))
    print("\nHueco corto (filas 1-3, largo 3 <= 4): ambos lo rellenan. Bien.")
    print("Hueco largo (filas 5-12, largo 8 > 4): la version actual rellena")
    print("los primeros 4 valores; la deseada no toca ninguno.")

    # Segundo efecto: la cola
    s2 = pd.Series([10.0, 20.0, 30.0] + [np.nan] * 6)
    print("\n" + "-" * 72)
    print("Segundo efecto: valores al final de la serie")
    print("-" * 72)
    tabla2 = pd.DataFrame({
        "original": s2,
        "actual": s2.interpolate(method="linear", limit=limite),
        "deseado": s2.interpolate(method="linear", limit_area="inside"),
    })
    print()
    print(tabla2.to_string(float_format=lambda x: f"{x:8.2f}"))
    print("\nLa version actual copia el ultimo valor medido hacia adelante")
    print(f"{limite} pasos. La deseada deja la cola vacia.")
    print(f"\npandas {pd.__version__}")


# ---------------------------------------------------------------------------
# Parte 2: medición sobre los datos reales
# ---------------------------------------------------------------------------
def rejilla(ruta):
    """Reconstruye la rejilla de 10 minutos desde el CSV crudo."""
    df = pd.read_csv(ruta)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "entity_id"])
    df["bloque"] = df["entity_id"].str.replace("SmartMeter_SM_", "", regex=False)

    for var, (bajo, alto) in RANGOS.items():
        if var in df.columns:
            fuera = (df[var] < bajo) | (df[var] > alto)
            df.loc[fuera, var] = np.nan

    df["voltaje_promedio"] = df[["v1", "v2", "v3"]].mean(axis=1)

    salidas = []
    for bloque, g in df.groupby("bloque"):
        g = g.set_index("timestamp").sort_index()
        r = g[INSTANTANEAS].resample("10min").mean()
        r["activeenergyimport"] = g["activeenergyimport"].resample("10min").last()
        r["bloque"] = bloque
        salidas.append(r)
    return pd.concat(salidas)


def rachas_vacias(serie):
    """Devuelve una tabla con inicio, fin y largo de cada racha de nulos."""
    vacio = serie.isna().to_numpy()
    if not vacio.any():
        return pd.DataFrame(columns=["inicio", "fin", "pasos"])
    cambio = np.flatnonzero(np.diff(vacio.astype(int)) != 0) + 1
    bordes = np.concatenate(([0], cambio, [len(vacio)]))
    filas = []
    for a, b in zip(bordes[:-1], bordes[1:]):
        if vacio[a]:
            filas.append({"inicio": serie.index[a],
                          "fin": serie.index[b - 1],
                          "pasos": b - a,
                          "al_final": b == len(vacio)})
    return pd.DataFrame(filas)


def medir(ruta, limite):
    g = rejilla(ruta)
    print("=" * 72)
    print("MEDICION SOBRE LOS DATOS REALES")
    print("=" * 72)
    print(f"\nRejilla: {len(g):,} filas, {g['bloque'].nunique()} bloques")
    print(f"Periodo: {g.index.min()} a {g.index.max()}")
    print(f"Limite de interpolacion: {limite} pasos "
          f"({limite * 10 / 60:.0f} horas)\n")

    todas = []
    for bloque, gb in g.groupby("bloque"):
        gb = gb.sort_index()
        for var in INSTANTANEAS + ["activeenergyimport"]:
            r = rachas_vacias(gb[var])
            if len(r):
                r["bloque"] = bloque
                r["variable"] = var
                todas.append(r)
    r = pd.concat(todas, ignore_index=True) if todas else pd.DataFrame()

    if r.empty:
        print("No hay nulos en la rejilla.")
        return

    r["horas"] = r["pasos"] * 10 / 60
    r["largo"] = r["pasos"] > limite
    # Valores que la version actual rellena dentro de huecos largos
    r["inventados"] = np.where(r["largo"], np.minimum(r["pasos"], limite), 0)
    # En la cola no hay valor posterior: el relleno es copia del ultimo
    r.loc[r["al_final"], "inventados"] = np.minimum(
        r.loc[r["al_final"], "pasos"], limite)

    print("-" * 72)
    print("Resumen por variable")
    print("-" * 72)
    res = r.groupby("variable").apply(lambda x: pd.Series({
        "huecos": len(x),
        "huecos_largos": int(x["largo"].sum()),
        "pasos_vacios": int(x["pasos"].sum()),
        "valores_inventados": int(x["inventados"].sum()),
    }), include_groups=False)
    res["pct_rejilla"] = (res["valores_inventados"] / len(g) * 100).round(3)
    print()
    print(res.to_string())

    print("\n" + "-" * 72)
    print("Los 15 huecos mas largos, con sus fechas exactas")
    print("-" * 72)
    print()
    cols = ["bloque", "variable", "inicio", "fin", "pasos", "horas",
            "inventados"]
    print(r.nlargest(15, "pasos")[cols].to_string(
        index=False, float_format=lambda x: f"{x:.1f}"))

    print("\n" + "-" * 72)
    print("Huecos de mas de 24 horas comunes a todas las variables")
    print("-" * 72)
    comun = (r[r["horas"] > 24]
             .groupby(["bloque", "inicio", "fin"])
             .size().reset_index(name="variables"))
    print()
    print(comun.to_string(index=False) if len(comun)
          else "Ninguno supera las 24 horas.")

    # Los valores inventados ocupan los primeros 'limite' pasos del hueco
    r["fin_inventado"] = r["inicio"] + pd.to_timedelta(
        (r["inventados"] - 1).clip(lower=0) * 10, unit="m")

    print("\n" + "-" * 72)
    print("En que mes caen los valores inventados")
    print("-" * 72)
    inv = r[r["inventados"] > 0].copy()
    inv["mes"] = inv["inicio"].dt.to_period("M")
    print()
    print(inv.groupby("mes")["inventados"].agg(
        huecos="count", valores="sum").to_string())

    print("\n" + "-" * 72)
    print("Huecos largos dentro de la ventana de evaluacion (agosto 2026)")
    print("-" * 72)
    ago = inv[(inv["fin_inventado"] >= "2026-08-01") &
              (inv["inicio"] < "2026-09-01")]
    print()
    if len(ago):
        print(ago[["bloque", "variable", "inicio", "fin", "pasos", "horas",
                   "inventados"]].to_string(
            index=False, float_format=lambda x: f"{x:.1f}"))
        print(f"\nValores inventados dentro de agosto: "
              f"{int(ago['inventados'].sum()):,}")
    else:
        print("Ninguno. La evaluacion de agosto no toca valores inventados.")

    total = int(r["inventados"].sum())
    print("\n" + "=" * 72)
    print(f"TOTAL de valores inventados por el defecto: {total:,} "
          f"({total / len(g) / 4 * 100:.3f} % de las celdas de la rejilla)")
    print("=" * 72)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--entrada", default="datos/datos_crudos_consolidados.csv")
    ap.add_argument("--limite", type=int, default=LIMITE)
    args = ap.parse_args()
    if args.demo:
        demo()
    else:
        medir(args.entrada, args.limite)