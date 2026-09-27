"""Reloj simulado para probar la actualizacion diaria sin datos nuevos.

Mientras no llegan datos en vivo, cada corrida finge que "hoy" es un dia mas
adelante dentro del historico de julio y agosto. Asi se prueba todo el circuito
con predicciones que si cambian de un dia a otro.

Uso:
    python scripts/reloj_simulado.py leer      imprime el instante simulado actual
    python scripts/reloj_simulado.py avanzar   lo mueve un paso hacia adelante

Cuando lleguen los datos de Sistemas se borra datos/reloj_simulado.txt y el
flujo deja de usar --origen.
"""
import os
import sys

import pandas as pd

ARCHIVO = "datos/reloj_simulado.txt"
# Primer instante con siete dias de historia dentro del semestre 2
INICIO = pd.Timestamp("2026-07-21 06:00")
FIN = pd.Timestamp("2026-08-30 06:00")
PASO = pd.Timedelta(days=1)


def leer():
    if not os.path.isfile(ARCHIVO):
        return INICIO
    with open(ARCHIVO, encoding="utf-8") as f:
        return pd.Timestamp(f.read().strip())


def guardar(t):
    os.makedirs(os.path.dirname(ARCHIVO), exist_ok=True)
    with open(ARCHIVO, "w", encoding="utf-8") as f:
        f.write(f"{t:%Y-%m-%d %H:%M:%S}\n")


if __name__ == "__main__":
    accion = sys.argv[1] if len(sys.argv) > 1 else "leer"
    ahora = leer()
    if accion == "leer":
        print(f"{ahora:%Y-%m-%d %H:%M:%S}")
    elif accion == "avanzar":
        siguiente = ahora + PASO
        # Al llegar al final de los datos vuelve a empezar
        guardar(INICIO if siguiente > FIN else siguiente)
        print(f"{leer():%Y-%m-%d %H:%M:%S}")
    else:
        sys.exit("usa: leer | avanzar")
