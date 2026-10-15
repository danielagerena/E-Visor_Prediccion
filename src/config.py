"""Configuracion central del proyecto E-Visor.

Todo lo que antes estaba repetido en cada notebook vive aqui.
Los valores sensibles (URL de la base, tokens) se leen de variables de
entorno para que nunca queden escritos en el repositorio.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------- rutas
RAIZ = Path(__file__).resolve().parent.parent
CARPETA_DATOS = Path(os.getenv("EVISOR_DATOS", RAIZ / "datos"))
CARPETA_MODELOS = Path(os.getenv("EVISOR_MODELOS", RAIZ / "modelos" / "v4"))

# ---------------------------------------------------------------- conexion
# Ejemplo Postgres: postgresql+psycopg://usuario:clave@host:5432/evisor
# Ejemplo local:    sqlite:///evisor.db
DB_URL = os.getenv("EVISOR_DB_URL", f"sqlite:///{RAIZ / 'evisor.db'}")

# Credenciales de la plataforma del proveedor (aun no disponibles)
PROVEEDOR_URL = os.getenv("EVISOR_PROVEEDOR_URL", "")
PROVEEDOR_TOKEN = os.getenv("EVISOR_PROVEEDOR_TOKEN", "")

# ---------------------------------------------------------------- columnas
COL_TIEMPO = "ts"
COL_EDIFICIO = "entity_id"
PREFIJO_MEDIDOR = "SmartMeter_SM_"

VARIABLES_MULTIVAR = ["activepower", "totalpowerfactor", "voltaje_promedio"]
VAR_ACUMULATIVA = "activeenergyimport_absoluto"
VAR_INCREMENTO = "activeenergyimport_incremento"
VARIABLES_TODAS = VARIABLES_MULTIVAR + [VAR_ACUMULATIVA, VAR_INCREMENTO]

# ---------------------------------------------------------------- modelo
# La configuracion real del modelo en produccion NO esta aqui. Vive en
# modelos/v4/modelo_produccion_ficha.json, que registra con que valores se
# entreno efectivamente, y en los valores por omision de
# evisor_lstm.construir_modelo. Duplicarla aqui solo crea dos verdades.
FRECUENCIA = "10min"
PASOS_HORA = 6
PASOS_DIA = 144
PASOS_SEMANA = 7 * PASOS_DIA
DIAS_VENTANA_DERIVA = 14
SEED = 42

BLOQUES = [
    "B10_ARQ", "B12_DERE", "B15_BIBL", "B17_POLI", "B18_PARQ",
    "B3_RECT", "B4_PRIM", "B5_BACH", "B7_CTIC", "B7_TAC",
    "B8_AA", "B8_CPA", "B8_LABS", "B9_SFA1", "B9_SFA2", "ECOVILLA",
]

# Excluido de los analisis por decision del equipo de lideres, hasta que su
# medidor sea intervenido. Ver el informe metodologico, seccion 4.3.1.
BLOQUES_EXCLUIDOS = ["B8_CPA"]


def nombre_completo(nombre_corto):
    return PREFIJO_MEDIDOR + nombre_corto


def nombre_corto(nombre_completo_):
    return nombre_completo_.replace(PREFIJO_MEDIDOR, "")


# ---------------------------------------------------------------- calidad
# Rangos fisicamente posibles. Fuera de esto, el valor se marca como nulo.
REGLAS_CALIDAD = {
    "totalpowerfactor": (-1, 1),
    "voltaje_promedio": (50, 500),
    "v1": (50, 500),
    "v2": (50, 500),
    "v3": (50, 500),
    "activepower": (0, 2_000_000),
    "activeenergyimport": (0, 500_000_000),
}

# Huecos de hasta 12 horas se interpolan; mas largos quedan como nulos
LIMITE_INTERPOLACION_PASOS = 72

# El contador de energia da la vuelta cerca de 100 millones
TECHO_CONTADOR = 100_000_000
UMBRAL_ROLLOVER = -1_000_000

# ---------------------------------------------------------------- operacion
DIAS_RETENCION_PREDICCIONES = 30   # el vigilante borra lo mas viejo
HORAS_MAX_SIN_DATOS = 2
HORAS_MAX_SIN_PREDICCION = 3

VOLTAJE_NOMINAL = 120.0
TOLERANCIA_VOLTAJE = 0.10          # +-10 % sobre el nominal
FACTOR_POTENCIA_MINIMO = 0.90      # exigido por la regulacion colombiana
