"""Ingesta de mediciones nuevas.

Hoy la unica fuente disponible es la exportacion manual en CSV de la plataforma
del proveedor. Cuando Sistemas entregue el acceso programatico, solo hay que
escribir una clase nueva con el metodo `leer` y cambiar una linea en `main`.
El resto del pipeline no se entera del cambio.
"""
import argparse
import datetime as dt
import glob
import os

import pandas as pd

from . import config, db, preprocesamiento

# Cuanta historia se vuelve a leer en cada corrida. No es solo un margen por
# datos que lleguen tarde: el preprocesamiento necesita contexto anterior para
# calcular los incrementos de energia y detectar los reinicios del contador.
SOLAPAMIENTO_HORAS = 72


class FuenteCSV:
    """Lee los archivos que hoy se descargan a mano de la plataforma."""

    def __init__(self, carpeta, patron="etsmartmeter*.csv"):
        self.carpeta = carpeta
        self.patron = patron

    def leer(self, desde=None):
        archivos = sorted(glob.glob(os.path.join(self.carpeta, self.patron)))
        if not archivos:
            return pd.DataFrame()
        partes = [pd.read_csv(a, low_memory=False) for a in archivos]
        df = pd.concat(partes, ignore_index=True)
        df = preprocesamiento.normalizar_columnas(df)
        if desde is not None:
            df = df[df[config.COL_TIEMPO] > desde]
        return df


class FuenteSQL:
    """Lee directamente de la base del proveedor, por ejemplo CrateDB.

    CrateDB habla el protocolo de PostgreSQL, asi que sirve el mismo driver
    que ya usa el proyecto. Solo hace falta la cadena de conexion de un
    usuario de solo lectura en EVISOR_FUENTE_URL y, si el nombre de la tabla
    difiere, EVISOR_FUENTE_TABLA.

    Es una conexion aparte de la del proyecto a proposito: la base de origen
    es de lectura y las tablas propias viven en otra.
    """

    COLUMNAS = ["timestamp", "entity_id", "activepower", "totalpowerfactor",
                "activeenergyimport", "v1", "v2", "v3"]

    def __init__(self, url=None, tabla=None, prefijo=None):
        self.url = url or os.getenv("EVISOR_FUENTE_URL", "")
        self.tabla = tabla or os.getenv("EVISOR_FUENTE_TABLA", "mediciones")
        self.prefijo = prefijo or config.PREFIJO_MEDIDOR

    def leer(self, desde=None):
        if not self.url:
            raise RuntimeError(
                "Falta la conexion a la base de origen. Configurar "
                "EVISOR_FUENTE_URL con un usuario de solo lectura."
            )
        from sqlalchemy import create_engine, text

        columnas = ", ".join(self.COLUMNAS)
        sql = f"SELECT {columnas} FROM {self.tabla} WHERE entity_id LIKE :pre"
        params = {"pre": self.prefijo + "%"}
        if desde is not None:
            sql += " AND timestamp >= :desde"
            params["desde"] = desde.to_pydatetime()
        sql += " ORDER BY entity_id, timestamp"

        motor = create_engine(self.url, future=True)
        df = pd.read_sql(text(sql), motor, params=params)
        return df.rename(columns={"timestamp": config.COL_TIEMPO})


class FuenteAPI:
    """Plantilla para cuando exista el API del proveedor.

    Deja escrito lo que hay que pedirle a Sistemas: url, token, y un parametro
    de fecha para traer solo lo nuevo.
    """

    def __init__(self, url=None, token=None):
        self.url = url or config.PROVEEDOR_URL
        self.token = token or config.PROVEEDOR_TOKEN

    def leer(self, desde=None):
        if not self.url or not self.token:
            raise RuntimeError(
                "Falta el acceso al API del proveedor. Configurar "
                "EVISOR_PROVEEDOR_URL y EVISOR_PROVEEDOR_TOKEN."
            )
        import requests  # importado aqui para no exigirlo mientras no se use

        params = {}
        if desde is not None:
            params["desde"] = desde.isoformat()
        r = requests.get(
            self.url,
            headers={"Authorization": f"Bearer {self.token}"},
            params=params,
            timeout=60,
        )
        r.raise_for_status()
        return preprocesamiento.normalizar_columnas(pd.DataFrame(r.json()))


def ejecutar(fuente, solapamiento_horas=None):
    """Trae lo nuevo, lo preprocesa y lo guarda.

    Se pide bastante mas atras de la ultima marca guardada porque el
    preprocesamiento no es independiente de la ventana: el incremento de
    energia se calcula con una diferencia contra la fila anterior, de modo que
    la primera fila de cada lote se queda sin incremento, y un reinicio del
    contador que caiga justo en el borde se interpreta mal. Reprocesar varios
    dias cada vez elimina ese problema, y como la escritura es un upsert no
    duplica nada: sobrescribe.
    """
    solapamiento_horas = solapamiento_horas or SOLAPAMIENTO_HORAS
    inicio = dt.datetime.now(dt.timezone.utc)
    ultima = db.ultima_medicion()
    desde = ultima - pd.Timedelta(hours=solapamiento_horas) if ultima is not None else None

    crudo = fuente.leer(desde=desde)
    if crudo.empty:
        db.registrar_ejecucion("ingesta", inicio, "ok", 0, "sin datos nuevos")
        return 0

    df10 = preprocesamiento.preprocesar(crudo, ya_normalizado=True)
    df10["ingerido_en"] = dt.datetime.now(dt.timezone.utc)

    columnas = [config.COL_EDIFICIO, config.COL_TIEMPO] + \
               [c for c in config.VARIABLES_TODAS if c in df10.columns] + ["ingerido_en"]
    n = db.upsert(df10[columnas], "mediciones", [config.COL_EDIFICIO, config.COL_TIEMPO])
    db.registrar_ejecucion("ingesta", inicio, "ok", n, f"desde={desde}")
    return n


def main():
    parser = argparse.ArgumentParser(description="Ingesta de mediciones de E-Visor")
    parser.add_argument("--fuente", choices=["csv", "sql", "api"], default="csv")
    parser.add_argument("--carpeta", default=str(config.CARPETA_DATOS))
    args = parser.parse_args()

    db.crear_esquema()
    fuentes = {"csv": lambda: FuenteCSV(args.carpeta),
               "sql": FuenteSQL,
               "api": FuenteAPI}
    fuente = fuentes[args.fuente]()
    inicio = dt.datetime.now(dt.timezone.utc)
    try:
        n = ejecutar(fuente)
        print(f"Ingesta terminada: {n} filas escritas")
    except Exception as e:
        db.registrar_ejecucion("ingesta", inicio, "error", 0, str(e))
        raise


if __name__ == "__main__":
    main()
