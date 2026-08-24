"""Motor de alertas.

Se evalua dentro del job de inferencia, no dentro de Streamlit: una alerta que
solo existe cuando alguien mira la pantalla no es una alerta.

El motor tiene tres defensas contra el ruido, que es lo que hace que la gente
termine silenciando estos sistemas:
- persistencia: hacen falta N pasos seguidos fuera de rango, no uno solo
- histeresis: se enciende en un umbral y se apaga en otro mas holgado
- deduplicacion: mientras el evento siga activo no se crea otro
"""
import argparse
import datetime as dt

import pandas as pd
from sqlalchemy import text

from . import config, db

# Reglas iniciales sugeridas. Se insertan una vez y despues se editan en la
# base de datos, sin tocar el codigo ni volver a desplegar.
REGLAS_INICIALES = [
    {
        "nombre": "Factor de potencia bajo",
        "entity_id": None,
        "variable": "totalpowerfactor",
        "operador": "<",
        "umbral": 0.90,
        "umbral_salida": 0.93,
        "pasos_consecutivos": 6,          # una hora seguida
        "potencia_minima": 2000,          # solo cuando el bloque esta operando
        "severidad": "alta",
        "referencia": "Factor de potencia inductivo minimo 0.90 exigido por la "
                      "regulacion colombiana; por debajo se cobra transporte de "
                      "exceso de energia reactiva. Verificar la resolucion vigente.",
    },
    {
        "nombre": "Sobretension",
        "entity_id": None,
        "variable": "voltaje_promedio",
        "operador": ">",
        "umbral": config.VOLTAJE_NOMINAL * (1 + config.TOLERANCIA_VOLTAJE),
        "umbral_salida": config.VOLTAJE_NOMINAL * 1.07,
        "pasos_consecutivos": 3,
        "severidad": "alta",
        "referencia": "Nominal 120 V con tolerancia de +-10 %. Confirmar el "
                      "nivel de tension y la norma que aplica al campus.",
    },
    {
        "nombre": "Subtension",
        "entity_id": None,
        "variable": "voltaje_promedio",
        "operador": "<",
        "umbral": config.VOLTAJE_NOMINAL * (1 - config.TOLERANCIA_VOLTAJE),
        "umbral_salida": config.VOLTAJE_NOMINAL * 0.93,
        "pasos_consecutivos": 3,
        "severidad": "media",
        "referencia": "Nominal 120 V con tolerancia de +-10 %.",
    },
]


def sembrar_reglas(destinatarios=""):
    """Inserta las reglas iniciales si la tabla esta vacia."""
    existentes = db.consultar("SELECT COUNT(*) AS n FROM reglas_alerta")["n"].iloc[0]
    if existentes:
        return 0
    filas = []
    for i, r in enumerate(REGLAS_INICIALES, start=1):
        fila = dict(r)
        fila["id"] = i
        fila["destinatarios"] = destinatarios
        fila["activa"] = 1
        fila["modo_sombra"] = 1        # primero se calibra, despues se notifica
        filas.append(fila)
    return db.upsert(pd.DataFrame(filas), "reglas_alerta", ["id"])


def _tramos_consecutivos(mascara, minimo):
    """Devuelve los indices donde hay al menos `minimo` valores True seguidos."""
    tramos = []
    inicio = None
    for i, v in enumerate(list(mascara) + [False]):
        if v and inicio is None:
            inicio = i
        elif not v and inicio is not None:
            if i - inicio >= minimo:
                tramos.append((inicio, i - 1))
            inicio = None
    return tramos


def evaluar_reglas(df_pred, reglas):
    """Funcion pura: predicciones + reglas -> eventos candidatos.

    No toca la base de datos, asi que se puede probar con datos inventados.
    """
    eventos = []
    for _, regla in reglas.iterrows():
        if not int(regla.get("activa", 1)):
            continue
        var = regla["variable"]
        if var not in df_pred.columns:
            continue

        sub = df_pred
        if regla.get("entity_id"):
            sub = sub[sub[config.COL_EDIFICIO] == regla["entity_id"]]

        for edificio, grupo in sub.groupby(config.COL_EDIFICIO):
            g = grupo.sort_values("ts_objetivo").reset_index(drop=True)

            # Un bloque casi apagado tiene factor de potencia bajo por fisica,
            # no por un problema: con 270 W de carga no hay penalizacion que
            # cobrar. Sin esta guarda, la regla se dispara todo el tiempo en los
            # medidores de consumo minimo y nadie vuelve a leer las alertas.
            minimo = float(regla.get("potencia_minima") or 0)
            if minimo > 0 and "activepower" in g.columns:
                if (g["activepower"].dropna() < minimo).all():
                    continue

            serie = g[var]
            es_menor = regla["operador"] == "<"
            fuera = serie < regla["umbral"] if es_menor else serie > regla["umbral"]

            for ini, fin in _tramos_consecutivos(fuera.fillna(False), int(regla["pasos_consecutivos"])):
                valores = serie.iloc[ini:fin + 1].dropna()
                if valores.empty:
                    continue
                extremo = valores.min() if es_menor else valores.max()
                eventos.append({
                    "regla_id": int(regla["id"]),
                    "nombre": regla["nombre"],
                    "entity_id": edificio,
                    "ts_prevista": g["ts_objetivo"].iloc[ini],
                    "valor_extremo": float(extremo),
                    "severidad": regla.get("severidad", "media"),
                    "duracion_pasos": fin - ini + 1,
                    "modo_sombra": int(regla.get("modo_sombra", 1)),
                    "destinatarios": regla.get("destinatarios") or "",
                })
    return eventos


def _resolver_eventos(df_pred, reglas):
    """Cierra los eventos activos cuya condicion ya volvio al lado seguro.

    La histeresis vive aqui: para apagar se exige cruzar `umbral_salida`, que es
    mas holgado que el de encendido. Sin esto, un valor oscilando alrededor del
    limite genera decenas de correos.
    """
    activos = db.consultar("SELECT * FROM eventos_alerta WHERE estado = 'activa'")
    if activos.empty:
        return 0

    cerrados = 0
    ahora = dt.datetime.now(dt.timezone.utc)
    for _, ev in activos.iterrows():
        regla = reglas[reglas["id"] == ev["regla_id"]]
        if regla.empty:
            continue
        regla = regla.iloc[0]
        var = regla["variable"]
        salida = regla["umbral_salida"] if pd.notna(regla["umbral_salida"]) else regla["umbral"]

        g = df_pred[df_pred[config.COL_EDIFICIO] == ev["entity_id"]]
        if g.empty or var not in g.columns:
            continue
        serie = g[var].dropna()
        if serie.empty:
            continue

        seguro = (serie > salida).all() if regla["operador"] == "<" else (serie < salida).all()
        if seguro:
            with db.motor().begin() as con:
                con.execute(
                    text("UPDATE eventos_alerta SET estado='resuelta', resuelto_en=:t WHERE id=:i"),
                    {"t": ahora, "i": int(ev["id"])},
                )
            cerrados += 1
    return cerrados


def registrar_eventos(eventos):
    """Guarda eventos nuevos y evita duplicar los que ya estan activos."""
    if not eventos:
        return 0
    activos = db.consultar(
        "SELECT regla_id, entity_id FROM eventos_alerta WHERE estado = 'activa'"
    )
    ya = set(zip(activos["regla_id"], activos["entity_id"])) if not activos.empty else set()

    ahora = dt.datetime.now(dt.timezone.utc)
    nuevos = []
    for ev in eventos:
        clave = (ev["regla_id"], ev["entity_id"])
        if clave in ya:
            continue
        ya.add(clave)
        nuevos.append({
            "regla_id": ev["regla_id"],
            "entity_id": ev["entity_id"],
            "iniciado_en": ahora,
            "ts_prevista": ev["ts_prevista"],
            "valor_extremo": ev["valor_extremo"],
            "estado": "activa",
            "ultimo_aviso": ahora if not ev["modo_sombra"] else None,
        })

    if not nuevos:
        return 0
    df = pd.DataFrame(nuevos)
    with db.motor().begin() as con:
        df.to_sql("eventos_alerta", con, if_exists="append", index=False)
    return len(nuevos)


def ejecutar():
    """Evalua las reglas sobre el pronostico mas reciente de cada bloque."""
    inicio = dt.datetime.now(dt.timezone.utc)
    reglas = db.consultar("SELECT * FROM reglas_alerta WHERE activa = 1")
    if reglas.empty:
        db.registrar_ejecucion("alertas", inicio, "ok", 0, "sin reglas configuradas")
        return []

    df_pred = db.consultar(
        "SELECT p.* FROM predicciones p JOIN ("
        "  SELECT entity_id, MAX(ts_origen) AS ts_origen FROM predicciones GROUP BY entity_id"
        ") u ON p.entity_id = u.entity_id AND p.ts_origen = u.ts_origen"
    )
    if df_pred.empty:
        db.registrar_ejecucion("alertas", inicio, "ok", 0, "sin predicciones")
        return []

    df_pred["ts_objetivo"] = pd.to_datetime(df_pred["ts_objetivo"])
    eventos = evaluar_reglas(df_pred, reglas)
    n_nuevos = registrar_eventos(eventos)
    n_cerrados = _resolver_eventos(df_pred, reglas)

    db.registrar_ejecucion(
        "alertas", inicio, "ok", n_nuevos,
        f"nuevos={n_nuevos} resueltos={n_cerrados} evaluados={len(eventos)}"
    )
    return eventos


def main():
    parser = argparse.ArgumentParser(description="Motor de alertas de E-Visor")
    parser.add_argument("--sembrar", action="store_true", help="crea las reglas iniciales")
    args = parser.parse_args()

    db.crear_esquema()
    if args.sembrar:
        print(f"Reglas iniciales creadas: {sembrar_reglas()}")
    eventos = ejecutar()
    print(f"Eventos detectados en el pronostico vigente: {len(eventos)}")


if __name__ == "__main__":
    main()
