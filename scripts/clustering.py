"""Clustering de bloques del campus a partir de la semana tipica.

Responde tres preguntas separadas, una por variable, y las compara contra el
clustering conjunto:

  potencia activa    -> que edificios consumen con el mismo ritmo
  factor de potencia -> que edificios comparten el mismo problema de calidad
  voltaje promedio   -> que edificios comparten condiciones electricas

Va en scripts/. Se corre desde la raiz del repositorio:

    python scripts/clustering.py

Escribe tablas y figuras en resultados/clustering/.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

sys.path.insert(0, os.getcwd())
from src import evisor_lstm as E  # noqa: E402

SALIDA = "resultados/clustering"
RUTA_CSV = "datos/datos_preprocesados_10min.csv"

# Se usan las mismas tres variables del modelo predictivo. La energia acumulada
# queda fuera a proposito: su incremento es la potencia dividida entre 6, asi
# que incluirla seria contar el consumo dos veces.
VARIABLES = ["activepower", "totalpowerfactor", "voltaje_promedio"]
NOMBRES = {"activepower": "Potencia activa",
           "totalpowerfactor": "Factor de potencia",
           "voltaje_promedio": "Voltaje promedio",
           "conjunto": "Las tres juntas"}

HORAS_SEMANA = 168
RANGO_K = range(2, 8)
SEED = 42

# Numero de grupos de cada analisis.
#
# Es una decision de negocio, no estadistica: con 2 grupos la silueta es mas
# alta pero lo que separa es un edificio atipico, no un segmento util. Se fija
# en 4 por recomendacion del area de ciencia de datos. El script imprime
# igualmente cual seria el mejor numero segun la silueta, para que la decision
# quede documentada y no parezca arbitraria.
#
# Poner None en una variable para que la silueta decida.
K_POR_ANALISIS = {
    "activepower": 4,
    "totalpowerfactor": 4,
    "voltaje_promedio": 4,
    "conjunto": 4,
}

# Paleta validada para daltonismo y contraste
COLORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
TINTA, TENUE, REJILLA = "#52514e", "#898781", "#e1e0d9"
DIAS = ["Lun", "Mar", "Mie", "Jue", "Vie", "Sab", "Dom"]


# ---------------------------------------------------------------------------
# 1. Semana tipica
# ---------------------------------------------------------------------------
def semana_tipica(df, bloques):
    """Perfil de 168 horas por bloque y variable.

    Se promedia cada hora de cada dia de la semana sobre todo el periodo de
    clase. Promediar quita el ruido y deja la forma: como consume ese edificio
    un martes cualquiera.
    """
    perfiles = {}
    for b in bloques:
        d = df[df.bloque == b].set_index("timestamp").sort_index()
        d = d[E.mascara_clase(d.index)]          # fuera vacaciones y Semana Santa
        d = d[VARIABLES].resample("1h").mean()
        d = d.assign(dia=d.index.dayofweek, hora=d.index.hour)
        for var in VARIABLES:
            p = d.pivot_table(index="dia", columns="hora", values=var, aggfunc="mean")
            p = p.reindex(index=range(7), columns=range(24))
            perfiles[(b, var)] = p.to_numpy().ravel()
    matriz = {}
    for var in VARIABLES:
        matriz[var] = pd.DataFrame(
            [perfiles[(b, var)] for b in bloques], index=bloques,
            columns=[f"D{d}_H{h:02d}" for d in range(7) for h in range(24)])
    return matriz


def normalizar_forma(m):
    """Cada bloque se centra en su propia media y se divide por su desviacion.

    Asi dos edificios quedan cerca si tienen la MISMA FORMA de consumo, aunque
    uno consuma diez veces mas que el otro. Agrupar por magnitud solo separaria
    grandes de pequenos, que es algo que ya se sabe sin modelo.
    """
    centrado = m.sub(m.mean(axis=1), axis=0)
    desv = m.std(axis=1).replace(0, 1)
    return centrado.div(desv, axis=0)


# ---------------------------------------------------------------------------
# 2. Elegir k y agrupar
# ---------------------------------------------------------------------------
def evaluar_k(X, rango_k=RANGO_K):
    filas = []
    for k in rango_k:
        if k >= len(X):
            break
        km = KMeans(n_clusters=k, random_state=SEED, n_init=50).fit(X)
        ward = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(X)
        filas.append({"k": k,
                      "silueta_kmeans": silhouette_score(X, km.labels_),
                      "silueta_ward": silhouette_score(X, ward.labels_),
                      "inercia": km.inertia_})
    return pd.DataFrame(filas)


def agrupar(X, k):
    """Se usa Ward: es jerarquico, produce el dendrograma y no depende del azar."""
    return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)


def estabilidad_dejando_uno_fuera(X, k, etiquetas):
    """Quita un bloque, reagrupa y mide cuanto se parece al agrupamiento original.

    Si al quitar un solo edificio los grupos se reordenan, los grupos no son
    robustos y hay que decirlo.
    """
    puntajes = []
    for i in range(len(X)):
        resto = np.delete(np.arange(len(X)), i)
        nuevas = agrupar(X.iloc[resto], k)
        puntajes.append(adjusted_rand_score(etiquetas[resto], nuevas))
    return float(np.mean(puntajes)), float(np.min(puntajes))


# ---------------------------------------------------------------------------
# 3. Figuras
# ---------------------------------------------------------------------------
def figura_dendrogramas(matrices, ks, ruta):
    fig, axes = plt.subplots(1, len(matrices), figsize=(5.2 * len(matrices), 4.6))
    for ax, (var, X) in zip(np.atleast_1d(axes), matrices.items()):
        Z = linkage(X.values, method="ward")
        dendrogram(Z, labels=list(X.index), ax=ax, leaf_font_size=8,
                   color_threshold=Z[-(ks[var] - 1), 2], above_threshold_color=TENUE)
        ax.set_title(NOMBRES[var], color=TINTA)
        ax.tick_params(axis="x", rotation=90, labelsize=7)
        ax.set_ylabel("Distancia", color=TINTA)
        for lado in ("top", "right", "left"):
            ax.spines[lado].set_visible(False)
    fig.suptitle("Que edificios se parecen entre si, variable por variable",
                 fontsize=13, color=TINTA)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)


def figura_pertenencia(tabla, ruta):
    """Matriz de 16 bloques por cada agrupacion. Un color por grupo."""
    cols = list(tabla.columns)
    datos = tabla.to_numpy()
    fig, ax = plt.subplots(figsize=(1.9 * len(cols) + 2.5, 0.42 * len(tabla) + 1.8))
    ax.imshow(datos, cmap=matplotlib.colors.ListedColormap(COLORES[:datos.max() + 1]),
              aspect="auto", vmin=0, vmax=len(COLORES) - 1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([NOMBRES[c] for c in cols], color=TINTA)
    ax.set_yticks(range(len(tabla)))
    ax.set_yticklabels(tabla.index, color=TINTA, fontsize=9)
    for i in range(len(tabla)):
        for j in range(len(cols)):
            ax.text(j, i, f"G{datos[i, j] + 1}", ha="center", va="center",
                    color="white", fontsize=9, fontweight="bold")
    ax.set_title("A que grupo pertenece cada bloque segun la variable",
                 color=TINTA, pad=12)
    ax.set_xticks(np.arange(-.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(tabla), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)


def figura_siluetas(metricas, ruta):
    """Que tan separados quedan los grupos segun cuantos se pidan."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (var, ev) in enumerate(metricas.items()):
        ax.plot(ev.k, ev.silueta_ward, marker="o", color=COLORES[i],
                linewidth=2, label=NOMBRES[var])
    ax.axhline(0.25, color=TENUE, linestyle="--", linewidth=1)
    ax.text(ev.k.max(), 0.26, "por debajo de 0,25 los grupos son debiles",
            ha="right", fontsize=8, color=TENUE)
    ax.set_xlabel("Numero de grupos", color=TINTA)
    ax.set_ylabel("Silueta (mas alto, grupos mas separados)", color=TINTA)
    ax.set_title("Cuantos grupos tiene realmente el campus", color=TINTA)
    ax.legend(frameon=False)
    ax.grid(axis="y", color=REJILLA)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)


def figura_perfiles(X, etiquetas, var, ruta):
    """Semana tipica promedio de cada grupo, en forma normalizada."""
    d = X.copy()
    d["grupo"] = etiquetas
    perfil = d.groupby("grupo").mean()
    fig, ax = plt.subplots(figsize=(12, 4))
    for i, (g, fila) in enumerate(perfil.iterrows()):
        ax.plot(range(HORAS_SEMANA), fila.values, color=COLORES[i], linewidth=2,
                label=f"Grupo {g + 1} ({(etiquetas == g).sum()} bloques)")
    for d_ in range(1, 7):
        ax.axvline(d_ * 24, color=REJILLA, linewidth=1)
    ax.set_xticks([d_ * 24 + 12 for d_ in range(7)])
    ax.set_xticklabels(DIAS, color=TINTA)
    ax.set_ylabel("Perfil normalizado", color=TINTA)
    ax.set_title(f"Semana tipica de cada grupo - {NOMBRES[var]}", color=TINTA)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    ax.grid(axis="y", color=REJILLA)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Programa
# ---------------------------------------------------------------------------
def main():
    os.makedirs(SALIDA, exist_ok=True)
    print("Cargando datos...")
    df = E.cargar_datos(RUTA_CSV)
    bloques = sorted(df.bloque.unique())
    print(f"  {len(bloques)} bloques | {df.timestamp.min()} a {df.timestamp.max()}")

    print("Construyendo la semana tipica de cada bloque...")
    crudas = semana_tipica(df, bloques)
    matrices = {var: normalizar_forma(m).fillna(0) for var, m in crudas.items()}
    matrices["conjunto"] = pd.concat(
        [matrices[v].add_prefix(f"{v}_") for v in VARIABLES], axis=1)
    print(f"  matriz por variable: {matrices[VARIABLES[0]].shape} | "
          f"conjunto: {matrices['conjunto'].shape}")

    resumen, etiquetas, ks, metricas = [], {}, {}, {}
    for var, X in matrices.items():
        ev = evaluar_k(X)
        metricas[var] = ev
        ev.insert(0, "analisis", var)
        k_silueta = int(ev.loc[ev.silueta_ward.idxmax(), "k"])
        k = K_POR_ANALISIS.get(var) or k_silueta
        lab = agrupar(X, k)
        media, peor = estabilidad_dejando_uno_fuera(X, k, lab)
        ks[var], etiquetas[var] = k, lab
        resumen.append({"analisis": var, "k_usado": k,
                        "k_mejor_silueta": k_silueta,
                        "silueta": float(ev.loc[ev.k == k, "silueta_ward"].iloc[0]),
                        "silueta_maxima": ev.silueta_ward.max(),
                        "estabilidad_media": media, "estabilidad_peor": peor,
                        "tamanos": np.bincount(lab).tolist()})
        ev.to_csv(f"{SALIDA}/metricas_k_{var}.csv", index=False)

    resumen = pd.DataFrame(resumen).round(3)
    resumen.to_csv(f"{SALIDA}/resumen.csv", index=False)
    print("\nRESUMEN")
    print(resumen.to_string(index=False))

    tabla = pd.DataFrame(etiquetas, index=bloques)
    tabla.index.name = "bloque"
    tabla.to_csv(f"{SALIDA}/pertenencia.csv")
    print("\nPERTENENCIA (grupo de cada bloque, contando desde 0)")
    print(tabla.to_string())

    print("\nCUANTO COINCIDEN LAS AGRUPACIONES (indice Rand ajustado)")
    claves = list(matrices)
    ari = pd.DataFrame(index=claves, columns=claves, dtype=float)
    for a in claves:
        for b in claves:
            ari.loc[a, b] = adjusted_rand_score(etiquetas[a], etiquetas[b])
    ari.round(3).to_csv(f"{SALIDA}/coincidencia_ari.csv")
    print(ari.round(3).to_string())

    # Grupos muy pequenos no son un segmento: son bloques que se comportan
    # distinto a todos los demas. Se reportan aparte porque eso, por si solo,
    # ya es un resultado util para mantenimiento.
    filas = []
    for var, lab in etiquetas.items():
        s_ = pd.Series(lab, index=bloques)
        for g in np.unique(lab):
            miembros = list(s_[s_ == g].index)
            if len(miembros) <= 3:
                filas.append({"analisis": var, "grupo": int(g) + 1,
                              "n": len(miembros), "bloques": ", ".join(miembros)})
    atipicos = pd.DataFrame(filas)
    atipicos.to_csv(f"{SALIDA}/bloques_atipicos.csv", index=False)
    print("\nBLOQUES QUE SE SEPARAN DEL RESTO")
    print(atipicos.to_string(index=False) if len(atipicos) else "  ninguno")

    print("\nGenerando figuras...")
    figura_siluetas(metricas, f"{SALIDA}/siluetas.png")
    figura_dendrogramas({v: matrices[v] for v in VARIABLES}, ks,
                        f"{SALIDA}/dendrogramas.png")
    figura_pertenencia(tabla, f"{SALIDA}/pertenencia.png")
    for var in VARIABLES:
        figura_perfiles(matrices[var], etiquetas[var], var,
                        f"{SALIDA}/perfiles_{var}.png")
    print(f"Listo. Resultados en {SALIDA}/")


if __name__ == "__main__":
    main()
