"""Genera las cuatro figuras del análisis exploratorio que faltan en el informe.

Lee la rejilla de 10 minutos ya preprocesada y escribe cuatro PNG:

  eda_patron_semanal.png    Mapa de calor hora x dia de la semana, por variable
  eda_correlacion.png       Matriz de correlación de Spearman entre variables
  eda_autocorrelacion.png   Autocorrelación de la potencia activa
  eda_mapa_vacios.png       Mapa de datos faltantes por bloque y día

Uso:
    python figuras_eda.py
    python figuras_eda.py --bloque B7_CTIC
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

VARIABLES = ["activepower", "totalpowerfactor", "voltaje_promedio"]
ETIQUETAS = {"activepower": "Potencia activa (W)",
             "totalpowerfactor": "Factor de potencia",
             "voltaje_promedio": "Voltaje promedio (V)",
             "activeenergyimport": "Energía importada (Wh)"}
DIAS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado",
        "Domingo"]
PASOS_DIA = 144


CANDIDATOS_BLOQUE = ["bloque", "edificio", "entity_id", "medidor", "id_bloque"]
CANDIDATOS_TIEMPO = ["timestamp", "fecha_hora", "fecha", "tiempo"]


def cargar(ruta):
    """Lee la rejilla y normaliza los nombres de las dos columnas clave."""
    df = pd.read_csv(ruta)
    col_t = next((c for c in CANDIDATOS_TIEMPO if c in df.columns), None)
    col_b = next((c for c in CANDIDATOS_BLOQUE if c in df.columns), None)
    if col_t is None or col_b is None:
        raise SystemExit("No encuentro las columnas de tiempo o bloque.\n"
                         f"Columnas del archivo: {list(df.columns)}")
    print(f"Columna de tiempo: {col_t}   Columna de bloque: {col_b}")
    df = df.rename(columns={col_t: "timestamp", col_b: "bloque"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["bloque"] = df["bloque"].astype(str).str.replace(
        "SmartMeter_SM_", "", regex=False)
    return df.sort_values(["bloque", "timestamp"])


# ---------------------------------------------------------------------------
def patron_semanal(df, bloque, salida):
    g = df[df["bloque"] == bloque].copy()
    g["dia"] = g["timestamp"].dt.dayofweek
    g["hora"] = g["timestamp"].dt.hour

    fig, ejes = plt.subplots(3, 1, figsize=(10, 9))
    for ax, var in zip(ejes, VARIABLES):
        m = g.pivot_table(index="dia", columns="hora", values=var,
                          aggfunc="mean").reindex(range(7))
        im = ax.imshow(m.to_numpy(), aspect="auto", cmap="YlOrRd",
                       interpolation="nearest")
        ax.set_yticks(range(7), DIAS, fontsize=8)
        ax.set_xticks(range(0, 24, 2), [f"{h}h" for h in range(0, 24, 2)],
                      fontsize=8)
        ax.set_title(ETIQUETAS[var], fontsize=10)
        plt.colorbar(im, ax=ax, pad=0.01)
    fig.suptitle(f"Patrón semanal promedio · {bloque}", fontsize=12)
    fig.tight_layout()
    fig.savefig(salida, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {salida}")


# ---------------------------------------------------------------------------
def correlacion(df, bloque, salida):
    cols = [c for c in VARIABLES + ["activeenergyimport"] if c in df.columns]
    g = df[df["bloque"] == bloque][cols].dropna()
    c = g.corr(method="spearman")

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(c.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    nombres = [{"activepower": "Potencia\nactiva",
                "totalpowerfactor": "Factor de\npotencia",
                "voltaje_promedio": "Voltaje\npromedio",
                "activeenergyimport": "Energía\nimportada"}[c] for c in cols]
    ax.set_xticks(range(len(cols)), nombres, fontsize=8)
    ax.set_yticks(range(len(cols)), nombres, fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = c.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(v) > 0.6 else "black")
    ax.set_title(f"Correlación de Spearman · {bloque}\n"
                 f"({len(g):,} observaciones)", fontsize=11)
    plt.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(salida, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {salida}")
    return c


# ---------------------------------------------------------------------------
def autocorrelacion(df, bloque, salida, max_dias=15):
    s = df[df["bloque"] == bloque].set_index("timestamp")["activepower"]
    s = s.asfreq("10min").interpolate(limit_area="inside")
    x = s.dropna().to_numpy()
    x = x - x.mean()
    n = len(x)
    max_lag = max_dias * PASOS_DIA
    denom = np.dot(x, x)
    acf = np.array([np.dot(x[:n - k], x[k:]) / denom
                    for k in range(max_lag + 1)])

    fig, ax = plt.subplots(figsize=(10, 4))
    horas = np.arange(max_lag + 1) / 6
    ax.plot(horas, acf, color="#38784B", lw=0.9)
    ax.axhline(0, color="black", lw=0.6)
    for d in range(1, max_dias + 1):
        ax.axvline(d * 24, color="#E97132", lw=0.6, ls=":", alpha=0.7)
    ax.axvline(168, color="#0E2841", lw=1.4, ls="--",
               label="168 h (una semana)")
    ax.set_xlabel("Rezago (horas)")
    ax.set_ylabel("Autocorrelación")
    ax.set_title(f"Autocorrelación de la potencia activa · {bloque}\n"
                 "Las líneas punteadas marcan múltiplos de 24 horas",
                 fontsize=11)
    ax.legend(fontsize=8)
    ax.margins(x=0.01)
    fig.tight_layout()
    fig.savefig(salida, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {salida}")

    picos = {f"{d} d": acf[d * PASOS_DIA] for d in (1, 2, 7, 14)
             if d * PASOS_DIA <= max_lag}
    return picos


# ---------------------------------------------------------------------------
def mapa_vacios(df, salida):
    d = df.copy()
    d["fecha"] = d["timestamp"].dt.date
    m = (d.assign(vacio=d["activepower"].isna())
           .pivot_table(index="bloque", columns="fecha", values="vacio",
                        aggfunc="mean"))
    m = m.reindex(sorted(m.index))

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(m.to_numpy() * 100, aspect="auto", cmap="Reds",
                   vmin=0, vmax=100, interpolation="nearest")
    ax.set_yticks(range(len(m.index)), m.index, fontsize=8)
    fechas = list(m.columns)
    paso = max(1, len(fechas) // 20)
    pos = range(0, len(fechas), paso)
    ax.set_xticks(list(pos), [str(fechas[i]) for i in pos], fontsize=7,
                  rotation=90)
    ax.set_title("Porcentaje de la jornada sin dato de potencia activa, "
                 "por bloque y día", fontsize=11)
    plt.colorbar(im, ax=ax, label="% del día faltante", shrink=0.8)
    fig.tight_layout()
    fig.savefig(salida, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {salida}")

    # Días en que todos los bloques tienen más de la mitad del día vacía
    por_dia = m.mean(axis=0)
    return por_dia[por_dia > 0.5]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="datos/datos_preprocesados_10min.csv")
    ap.add_argument("--bloque", default="B10_ARQ")
    args = ap.parse_args()

    df = cargar(args.csv)
    if args.bloque not in set(df["bloque"]):
        raise SystemExit(f"El bloque {args.bloque} no esta en el archivo.\n"
                         f"Disponibles: {sorted(set(df['bloque']))}")
    print(f"Filas: {len(df):,}   Bloques: {df['bloque'].nunique()}")
    print(f"Periodo: {df['timestamp'].min()} a {df['timestamp'].max()}")
    print("\nFiguras generadas:")

    patron_semanal(df, args.bloque, "eda_patron_semanal.png")
    c = correlacion(df, args.bloque, "eda_correlacion.png")
    picos = autocorrelacion(df, args.bloque, "eda_autocorrelacion.png")
    vacios = mapa_vacios(df, "eda_mapa_vacios.png")

    print(f"\nCorrelación de Spearman en {args.bloque}:")
    print(c.round(3).to_string())
    print(f"\nAutocorrelación de la potencia en {args.bloque}:")
    for k, v in picos.items():
        print(f"  rezago {k}: {v:.3f}")
    print("\nDías con más de la mitad de la jornada vacía en promedio:")
    print(vacios.round(3).to_string() if len(vacios) else "  ninguno")