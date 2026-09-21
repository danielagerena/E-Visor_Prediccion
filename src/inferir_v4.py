"""Genera las predicciones de 24 horas con el modelo global v4.

Este archivo es el que corre en produccion. Usa las mismas funciones de
evisor_lstm que se usaron para entrenar, asi que no puede desincronizarse:
si cambia la forma de construir las entradas, cambia en los dos lados a la vez.

Uso:
    python -m src.inferir_v4 --artefactos modelos/v4 --csv datos/datos_preprocesados_10min.csv
    python -m src.inferir_v4 --artefactos modelos/v4 --csv datos/... --salida predicciones.csv
"""
import argparse
import warnings

import numpy as np
import pandas as pd

from . import evisor_lstm as E

# Cuantas semanas hacia atras se buscan si falta la referencia de la semana
# anterior. Los cortes de mantenimiento del campus duran hasta ocho dias, asi
# que mirar solo una semana atras no alcanza.
SEMANAS_ALTERNATIVAS = 4

# La energia acumulada no se predice con un modelo aparte: es la integral de la
# potencia activa. En 10 minutos la energia en Wh es la potencia en W dividida
# entre 6. Se verifico en los datos: la razon mediana es 5,974 en los 16 bloques
# y la correlacion entre el incremento del contador y la potencia es 0,973.
COL_CONTADOR = "activeenergyimport_absoluto"
VAR_INCREMENTO = "activeenergyimport_incremento"
PASOS_POR_HORA = 6
TECHO_CONTADOR = 100_000_000   # el contador vuelve a cero al llegar aqui


def _ventana_semana(tabla, i_ini, i_fin):
    """Devuelve la referencia de la semana anterior y como se obtuvo.

    Primero intenta la semana inmediatamente anterior. Si tiene datos
    faltantes, retrocede semana a semana. Si ninguna sirve, cae al perfil del
    dia anterior, que es peor referencia pero siempre esta mas cerca.
    """
    vals = tabla[E.VARIABLES].to_numpy("float32")
    for k in range(1, SEMANAS_ALTERNATIVAS + 1):
        a, b = i_ini - k * E.PASOS_SEMANA, i_fin - k * E.PASOS_SEMANA
        if a < 0:
            break
        v = vals[a:b]
        if not np.isnan(v).any():
            return v, ("semana_anterior" if k == 1 else f"semana_-{k}")

    a, b = i_ini - E.PASOS_DIA, i_fin - E.PASOS_DIA
    if a >= 0:
        v = vals[a:b]
        if not np.isnan(v).any():
            return v, "dia_anterior"
    return None, "sin_referencia"


def _ventana_pasado(tabla, i_ini, i_fin):
    """Las ultimas 48 horas. Si faltan datos se rellenan con la semana previa."""
    vals = tabla[E.VARIABLES].to_numpy("float32")
    v = vals[i_ini:i_fin].copy()
    if not np.isnan(v).any():
        return v, "completo"

    a = i_ini - E.PASOS_SEMANA
    if a >= 0:
        respaldo = vals[a:i_fin - E.PASOS_SEMANA]
        faltan = np.isnan(v)
        v[faltan] = respaldo[faltan]
    if np.isnan(v).any():
        return None, "pasado_incompleto"
    return v, "pasado_rellenado"


def preparar_entradas(tabla, origen_pos, bloque, lims, indice_bloque,
                      largo_entrada, horizonte):
    """Arma las tres entradas del modelo para un instante de origen."""
    mn, rango = lims[bloque]
    i_pas_ini = origen_pos - largo_entrada + 1
    i_pas_fin = origen_pos + 1
    if i_pas_ini < 0:
        return None, "historia_insuficiente"

    pasado, calidad_pas = _ventana_pasado(tabla, i_pas_ini, i_pas_fin)
    if pasado is None:
        return None, calidad_pas

    i_fut_ini, i_fut_fin = origen_pos + 1, origen_pos + 1 + horizonte
    semana, calidad_sem = _ventana_semana(tabla, i_fut_ini, i_fut_fin)
    if semana is None:
        return None, calidad_sem

    ctx = E.contexto_temporal(tabla.index[[origen_pos]])
    if indice_bloque is not None:
        ident = np.zeros((1, len(indice_bloque)), "float32")
        ident[0, indice_bloque[bloque]] = 1.0
        ctx = np.hstack([ctx, ident])

    entradas = {
        "pasado": E.escalar(pasado[None, :, :], mn, rango),
        "semana": E.escalar(semana[None, :, :], mn, rango),
        "contexto": ctx,
    }
    calidad = "buena" if (calidad_pas == "completo"
                          and calidad_sem == "semana_anterior") else "degradada"
    return entradas, f"{calidad}:{calidad_pas}/{calidad_sem}"


def energia_desde_potencia(potencia_w, ultimo_nivel, techo=TECHO_CONTADOR):
    """Convierte la potencia predicha en energia acumulada.

    No hace falta un modelo de deriva: la energia es la integral de la potencia.
    Se acumula el incremento de cada paso y se suma a la ultima lectura real del
    contador. El modulo maneja el caso en que el contador se desborde.
    """
    incremento = potencia_w / PASOS_POR_HORA          # W durante 10 min -> Wh
    nivel = (ultimo_nivel + np.cumsum(incremento)) % techo
    return incremento, nivel


def predecir(modelo, lims, indice_bloque, tablas, ficha, origen=None,
             contadores=None):
    """Predice 24 horas para todos los bloques. Devuelve formato largo."""
    largo_entrada = ficha.get("largo_entrada", E.LARGO_ENTRADA)
    horizonte = ficha.get("horizonte", E.HORIZONTE)
    version = ficha.get("version", "desconocida")

    filas, avisos = [], []
    for bloque, tabla in tablas.items():
        if origen is None:
            # El ultimo instante con dato real, no el ultimo de la rejilla
            validos = tabla[E.VARIABLES].dropna().index
            if len(validos) == 0:
                avisos.append((bloque, "sin_datos"))
                continue
            pos = tabla.index.get_loc(validos[-1])
        else:
            marca = pd.Timestamp(origen)
            if marca not in tabla.index:
                avisos.append((bloque, "origen_fuera_de_rejilla"))
                continue
            pos = tabla.index.get_loc(marca)

        entradas, calidad = preparar_entradas(
            tabla, pos, bloque, lims, indice_bloque, largo_entrada, horizonte)
        if entradas is None:
            avisos.append((bloque, calidad))
            continue

        mn, rango = lims[bloque]
        crudo = modelo.predict(entradas, verbose=0)[0]
        # El modelo trabaja en el rango -1 a 1. Dejarlo salir de ahi equivale a
        # predecir un valor que ese medidor nunca ha registrado: en los datos
        # ningun bloque tiene potencia negativa, pero el modelo si la produce si
        # no se acota. Recortar es la forma mas simple de impedirlo.
        pred = E.desescalar(np.clip(crudo, -1.0, 1.0), mn, rango)
        marcas = tabla.index[pos + 1:pos + 1 + horizonte]
        if len(marcas) < horizonte:
            # El futuro se sale de la rejilla cargada: se extiende
            marcas = pd.date_range(tabla.index[pos] + pd.Timedelta(minutes=10),
                                   periods=horizonte, freq=E.FRECUENCIA)

        for j, var in enumerate(E.VARIABLES):
            filas.append(pd.DataFrame({
                "bloque": bloque, "ts": marcas, "variable": var,
                "valor": pred[:, j], "origen": tabla.index[pos],
                "version": version, "calidad": calidad}))

        # Energia acumulada: se deriva de la potencia ya predicha
        if contadores is not None and bloque in contadores:
            nivel_actual = contadores[bloque].iloc[:pos + 1].dropna()
            if len(nivel_actual):
                inc, nivel = energia_desde_potencia(
                    pred[:, E.VARIABLES.index("activepower")],
                    float(nivel_actual.iloc[-1]))
                for var, valores in [(VAR_INCREMENTO, inc), (COL_CONTADOR, nivel)]:
                    filas.append(pd.DataFrame({
                        "bloque": bloque, "ts": marcas, "variable": var,
                        "valor": valores, "origen": tabla.index[pos],
                        "version": version, "calidad": calidad}))

    salida = pd.concat(filas, ignore_index=True) if filas else pd.DataFrame()
    return salida, avisos


def main():
    p = argparse.ArgumentParser(description="Predicciones de 24 horas, modelo global v4")
    p.add_argument("--artefactos", required=True, help="carpeta con el modelo y los limites")
    p.add_argument("--nombre", default="modelo_produccion")
    p.add_argument("--csv", required=True, help="CSV preprocesado de 10 minutos")
    p.add_argument("--origen", default=None,
                   help="instante de origen; por defecto el ultimo dato disponible")
    p.add_argument("--salida", default=None, help="CSV donde escribir las predicciones")
    p.add_argument("--salida-historia", default=None,
                   help="CSV con los ultimos dias medidos, para que la app no "
                        "tenga que leer el archivo completo de 44 MB")
    p.add_argument("--dias-historia", type=int, default=2)
    p.add_argument("--archivo", default=None,
                   help="carpeta donde guardar una copia comprimida de cada "
                        "prediccion, para compararla despues con lo medido")
    args = p.parse_args()

    warnings.filterwarnings("ignore")

    print("Cargando artefactos...")
    modelo, lims, ficha = E.cargar_artefactos(args.artefactos, args.nombre)
    indice_bloque = ficha.get("indice_bloque")
    print(f"  version {ficha.get('version')} | {len(lims)} bloques | "
          f"entrada {ficha.get('largo_entrada')} pasos")

    print("Cargando historia...")
    df = E.cargar_datos(args.csv)
    # Solo hace falta el ultimo mes: la ventana mas larga que mira el modelo son
    # siete dias. Leer mas es tiempo perdido en produccion.
    corte = df.timestamp.max() - pd.Timedelta(days=30)
    df = df[df.timestamp >= corte]
    bloques = [b for b in sorted(df.bloque.unique()) if b in lims]
    tablas = {b: E.tabla_bloque(df, b) for b in bloques}

    # Ultimas lecturas del contador, en la misma rejilla de 10 minutos
    contadores = {}
    if COL_CONTADOR in df.columns:
        for b in bloques:
            d = df[df.bloque == b].set_index("timestamp").sort_index()
            contadores[b] = d[COL_CONTADOR].reindex(tablas[b].index)
    print(f"  {len(df):,} filas | {len(bloques)} bloques | hasta {df.timestamp.max()}")

    print("Prediciendo...")
    pred, avisos = predecir(modelo, lims, indice_bloque, tablas, ficha,
                            origen=args.origen, contadores=contadores)

    if pred.empty:
        print("No se pudo predecir ningun bloque.")
    else:
        print(f"  {len(pred):,} filas | "
              f"{pred.bloque.nunique()} bloques | "
              f"de {pred.ts.min()} a {pred.ts.max()}")
        degradados = sorted(pred[pred.calidad.str.startswith("degradada")].bloque.unique())
        if degradados:
            print(f"  calidad degradada en: {', '.join(degradados)}")
        marcados = ficha.get("bloques_que_pierden_contra_trivial", [])
        if marcados:
            print(f"  marcar en el tablero: {', '.join(marcados)}")

    for bloque, motivo in avisos:
        print(f"  SIN PREDICCION {bloque}: {motivo}")

    if args.salida and not pred.empty:
        pred.to_csv(args.salida, index=False)
        print(f"Guardado en {args.salida}")

    if args.archivo and not pred.empty:
        import os
        os.makedirs(args.archivo, exist_ok=True)
        marca = pd.Timestamp(pred.origen.max())
        ruta = os.path.join(args.archivo, f"pred_{marca:%Y%m%d_%H%M}.csv.gz")
        pred.to_csv(ruta, index=False, compression="gzip")
        print(f"Copia archivada en {ruta}")

    if args.salida_historia:
        corte_h = df.timestamp.max() - pd.Timedelta(days=args.dias_historia)
        vars_hist = E.VARIABLES + [v for v in (VAR_INCREMENTO, COL_CONTADOR)
                                   if v in df.columns]
        hist = (df[df.timestamp >= corte_h]
                .melt(id_vars=["bloque", "timestamp"], value_vars=vars_hist,
                      var_name="variable", value_name="valor")
                .rename(columns={"timestamp": "ts"})
                .dropna(subset=["valor"]))
        hist.to_csv(args.salida_historia, index=False)
        print(f"Historia de {args.dias_historia} dias guardada en "
              f"{args.salida_historia} ({len(hist):,} filas)")


if __name__ == "__main__":
    main()