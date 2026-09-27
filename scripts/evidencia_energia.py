"""Evidencia numerica sobre la energia activa importada.

Responde dos preguntas con datos calculados, no con afirmaciones:

  1. Que relacion tiene el contador de energia con la potencia activa, y por
     que el contador no entra al clustering.
  2. Como se comporta ese contador en B7_CTIC frente a los demas bloques.

Va en scripts/. Se corre desde la raiz del repositorio:

    python scripts/evidencia_energia.py

Escribe tablas y figuras en resultados/energia/.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SALIDA = "resultados/energia"
RUTA_CSV = "datos/datos_preprocesados_10min.csv"
PREFIJO = "SmartMeter_SM_"

COL_POT = "activepower"
COL_NIVEL = "activeenergyimport_absoluto"
COL_INC = "activeenergyimport_incremento"

# Un paso de 10 minutos es un sexto de hora: una potencia de 6 W sostenida
# durante ese paso deja 1 Wh en el contador.
PASOS_POR_HORA = 6

BLOQUE_EJEMPLO = "B7_CTIC"
BLOQUE_CONTROL = "B10_ARQ"   # se reemplaza por el de mejor concordancia si no existe

COLOR_A, COLOR_B = "#2a78d6", "#eb6834"
GRIS = "#6b7280"


# ---------------------------------------------------------------------------
# 1. Carga
# ---------------------------------------------------------------------------
def cargar(ruta):
    df = pd.read_csv(ruta, parse_dates=["timestamp"])
    df["bloque"] = df["entity_id"].str.replace(PREFIJO, "", regex=False)
    df = df.sort_values(["bloque", "timestamp"])

    if COL_INC not in df.columns:
        # Si el CSV solo trae el contador, el incremento se calcula aqui.
        # Los saltos negativos son reinicios del contador y se descartan.
        inc = df.groupby("bloque")[COL_NIVEL].diff()
        df[COL_INC] = inc.where(inc >= 0)
    return df


# ---------------------------------------------------------------------------
# 2. Tabla por bloque
# ---------------------------------------------------------------------------
def tabla_por_bloque(df):
    """Compara, bloque por bloque, el contador contra la potencia medida.

    Tres angulos distintos de la misma pregunta:

      corr_nivel      el contador tal cual (numero que solo sube) contra la
                      potencia. Deberia ser cercana a cero: son cosas distintas.
      corr_incremento lo que subio el contador en cada paso contra la potencia.
                      Deberia ser cercana a uno: es la misma cosa.
      desacuerdo_%    cuanta energia separa al contador de la potencia, en
                      porcentaje de la energia total. Es la prueba mas dura:
                      una correlacion baja puede venir de un consumo plano,
                      pero un desacuerdo alto solo puede venir de un problema
                      de registro.
    """
    filas = []
    for bloque, d in df.groupby("bloque"):
        d = d.dropna(subset=[COL_POT, COL_INC])
        if len(d) < 1000:
            continue
        pot, inc = d[COL_POT].values, d[COL_INC].values
        esperado = pot / PASOS_POR_HORA          # Wh que deberia marcar cada paso

        total_esperado = esperado.sum()
        desacuerdo = np.abs(inc - esperado).sum() / total_esperado * 100

        con_nivel = d.dropna(subset=[COL_NIVEL])
        corr_nivel = (con_nivel[COL_NIVEL].corr(con_nivel[COL_POT])
                      if len(con_nivel) > 1000 else np.nan)

        filas.append({
            "bloque": bloque,
            "n": len(d),
            "potencia_media_W": pot.mean(),
            "variabilidad_cv": pot.std() / pot.mean(),
            "corr_nivel": corr_nivel,
            "corr_incremento": np.corrcoef(inc, pot)[0, 1],
            "razon_pot_inc": pot.sum() / inc.sum(),
            "desacuerdo_%": desacuerdo,
            "incrementos_cero_%": (inc == 0).mean() * 100,
            "cero_con_potencia_%": ((inc == 0) & (pot > 500)).mean() * 100,
            "valores_distintos": len(np.unique(inc)),
        })
    return pd.DataFrame(filas).sort_values("desacuerdo_%").reset_index(drop=True)


def tabla_mensual(df, bloque):
    """El mismo chequeo mes a mes para un solo bloque."""
    d = df[df.bloque == bloque].dropna(subset=[COL_POT, COL_INC]).copy()
    d["mes"] = d.timestamp.dt.to_period("M").astype(str)
    filas = []
    for mes, g in d.groupby("mes"):
        pot, inc = g[COL_POT].values, g[COL_INC].values
        esperado = pot / PASOS_POR_HORA
        filas.append({
            "mes": mes,
            "n": len(g),
            "potencia_media_W": pot.mean(),
            "corr_incremento": np.corrcoef(inc, pot)[0, 1],
            "razon_pot_inc": pot.sum() / inc.sum(),
            "desacuerdo_%": np.abs(inc - esperado).sum() / esperado.sum() * 100,
        })
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# 3. Figuras
# ---------------------------------------------------------------------------
def figura_nivel_vs_incremento(df, bloque, ruta):
    """Por que el contador no entra al clustering, en dos dibujos."""
    d = df[df.bloque == bloque].dropna(subset=[COL_POT, COL_INC, COL_NIVEL])
    d = d.iloc[::3]   # se ralea la nube para que el grafico no pese

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))

    r1 = d[COL_NIVEL].corr(d[COL_POT])
    ax[0].scatter(d[COL_NIVEL] / 1e6, d[COL_POT] / 1000, s=3, alpha=.25, color=GRIS)
    ax[0].set_xlabel("Contador acumulado (MWh)")
    ax[0].set_ylabel("Potencia activa (kW)")
    ax[0].set_title(f"Contador tal cual\ncorrelacion = {r1:.3f}")

    r2 = d[COL_INC].corr(d[COL_POT])
    ax[1].scatter(d[COL_INC], d[COL_POT] / 1000, s=3, alpha=.25, color=COLOR_A)
    lim = np.array([0, d[COL_INC].quantile(.999)])
    ax[1].plot(lim, lim * PASOS_POR_HORA / 1000, color=COLOR_B, lw=1.5,
               label="potencia = 6 x incremento")
    ax[1].set_xlabel("Lo que subio el contador en 10 min (Wh)")
    ax[1].set_ylabel("Potencia activa (kW)")
    ax[1].set_title(f"Incremento del contador\ncorrelacion = {r2:.3f}")
    ax[1].legend(fontsize=8)

    fig.suptitle(f"{bloque}: el contador acumulado no aporta informacion nueva",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(ruta, dpi=140)
    plt.close(fig)


def figura_desacuerdo(tabla, ruta):
    """Ranking de bloques por desacuerdo entre contador y potencia."""
    t = tabla.sort_values("desacuerdo_%")
    colores = [COLOR_B if v > 15 else COLOR_A for v in t["desacuerdo_%"]]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(t.bloque, t["desacuerdo_%"], color=colores)
    ax.axvline(15, color=GRIS, ls="--", lw=1)
    ax.text(15.3, -0.6, "umbral de revision", color=GRIS, fontsize=8)
    ax.set_xlabel("Diferencia entre contador y potencia (% de la energia total)")
    ax.set_title("Que tan de acuerdo estan el contador y el medidor de potencia")
    fig.tight_layout()
    fig.savefig(ruta, dpi=140)
    plt.close(fig)


def figura_ctic(df, bloque, control, ruta):
    """CTIC contra un bloque sano: consumo plano no es consumo mal medido."""
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))
    semana = ("2026-08-04", "2026-08-11")

    for j, b in enumerate([bloque, control]):
        d = df[df.bloque == b].dropna(subset=[COL_POT, COL_INC])
        s = d[(d.timestamp >= semana[0]) & (d.timestamp < semana[1])]

        ax[0, j].plot(s.timestamp, s[COL_POT] / 1000, lw=.8, color=COLOR_A)
        ax[0, j].set_title(f"{b}: potencia de una semana de agosto")
        ax[0, j].set_ylabel("kW")
        ax[0, j].set_ylim(bottom=0)
        ax[0, j].tick_params(axis="x", rotation=30, labelsize=7)

        pot, inc = d[COL_POT].values, d[COL_INC].values
        r = np.corrcoef(inc, pot)[0, 1]
        des = np.abs(inc - pot / PASOS_POR_HORA).sum() / (pot / PASOS_POR_HORA).sum() * 100
        cv = pot.std() / pot.mean()
        ax[1, j].scatter(inc[::3], pot[::3] / 1000, s=3, alpha=.25, color=COLOR_A)
        lim = np.array([0, np.quantile(inc, .999)])
        ax[1, j].plot(lim, lim * PASOS_POR_HORA / 1000, color=COLOR_B, lw=1.5)
        ax[1, j].set_xlabel("Incremento del contador (Wh en 10 min)")
        ax[1, j].set_ylabel("Potencia activa (kW)")
        ax[1, j].set_title(f"correlacion {r:.3f} | desacuerdo {des:.1f} % | "
                           f"variabilidad {cv:.2f}", fontsize=10)

    fig.suptitle("Correlacion baja con desacuerdo normal = carga plana, "
                 "no medidor danado", fontsize=12)
    fig.tight_layout()
    fig.savefig(ruta, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Programa
# ---------------------------------------------------------------------------
def main():
    ruta = sys.argv[1] if len(sys.argv) > 1 else RUTA_CSV
    os.makedirs(SALIDA, exist_ok=True)

    print(f"Leyendo {ruta} ...")
    df = cargar(ruta)
    print(f"{df.bloque.nunique()} bloques | {df.timestamp.min().date()} a "
          f"{df.timestamp.max().date()}")

    tabla = tabla_por_bloque(df)
    tabla.round(4).to_csv(f"{SALIDA}/contador_vs_potencia.csv", index=False)

    vista = tabla[["bloque", "potencia_media_W", "variabilidad_cv", "corr_nivel",
                   "corr_incremento", "razon_pot_inc", "desacuerdo_%",
                   "incrementos_cero_%"]]
    print("\nCONTADOR CONTRA POTENCIA, POR BLOQUE")
    print("(razon teorica = 6.000; desacuerdo bajo = el contador confirma la potencia)")
    print(vista.round(3).to_string(index=False))

    print("\nRESUMEN GLOBAL")
    print(f"  correlacion media contador acumulado vs potencia : "
          f"{tabla.corr_nivel.mean():.3f}")
    print(f"  correlacion media incremento vs potencia         : "
          f"{tabla.corr_incremento.mean():.3f}")
    print(f"  razon mediana potencia / incremento              : "
          f"{tabla.razon_pot_inc.median():.3f}  (teoria: 6.000)")
    print(f"  desacuerdo mediano                               : "
          f"{tabla['desacuerdo_%'].median():.2f} %")

    sospechosos = tabla[tabla["desacuerdo_%"] > 15]
    print("\nBLOQUES CON DESACUERDO ALTO (posible problema de registro)")
    print(sospechosos.round(3).to_string(index=False) if len(sospechosos)
          else "  ninguno")
    sospechosos.round(4).to_csv(f"{SALIDA}/contadores_sospechosos.csv", index=False)

    # Relacion entre variabilidad del consumo y correlacion, sobre los 16 bloques.
    rho = tabla["variabilidad_cv"].corr(tabla["corr_incremento"], method="spearman")
    print(f"\nRelacion variabilidad vs correlacion entre bloques (Spearman): {rho:.3f}")
    print("  Sirve para no exagerar: la carga plana explica el caso de "
          f"{BLOQUE_EJEMPLO}, no es una regla general.")

    if BLOQUE_EJEMPLO in set(tabla.bloque):
        mensual = tabla_mensual(df, BLOQUE_EJEMPLO)
        mensual.round(3).to_csv(f"{SALIDA}/ctic_por_mes.csv", index=False)
        print(f"\n{BLOQUE_EJEMPLO} MES A MES")
        print(mensual.round(3).to_string(index=False))

    control = (BLOQUE_CONTROL if BLOQUE_CONTROL in set(tabla.bloque)
               else tabla.bloque.iloc[0])

    print("\nGenerando figuras...")
    figura_nivel_vs_incremento(df, BLOQUE_EJEMPLO, f"{SALIDA}/nivel_vs_incremento.png")
    figura_desacuerdo(tabla, f"{SALIDA}/desacuerdo_por_bloque.png")
    figura_ctic(df, BLOQUE_EJEMPLO, control, f"{SALIDA}/ctic_vs_control.png")
    print(f"Listo. Resultados en {SALIDA}/")


if __name__ == "__main__":
    main()
