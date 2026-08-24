"""Identidad visual del tablero.

Una sola fuente de verdad para los colores. Cuando Comunicaciones entregue el
manual de marca, se cambian estos valores y todas las graficas se actualizan.
"""

# PENDIENTE: confirmar los codigos exactos con el manual de identidad de la UPB.
VERDE_UPB = "#00723F"
VERDE_CLARO = "#4FA97A"
GRIS = "#5F6E67"
AMBAR = "#B8760A"
ROJO = "#9C382C"
AZUL = "#2B5871"

# Paleta para series. Se usa en TODAS las graficas.
PALETA = [VERDE_UPB, AZUL, AMBAR, ROJO, GRIS, VERDE_CLARO]

# Color por severidad. Nunca se usa solo: siempre acompanado de texto, para que
# una persona con daltonismo lea la misma informacion.
SEVERIDAD = {
    "alta": ROJO,
    "media": AMBAR,
    "baja": GRIS,
}

FRESCURA = {
    "fresco": VERDE_UPB,
    "atrasado": AMBAR,
    "viejo": ROJO,
}

NOMBRES_VARIABLES = {
    "activepower": "Potencia activa (W)",
    "totalpowerfactor": "Factor de potencia",
    "voltaje_promedio": "Voltaje promedio (V)",
    "activeenergyimport_absoluto": "Energia acumulada (Wh)",
    "activeenergyimport_incremento": "Consumo por intervalo (Wh)",
}


def etiqueta_frescura(minutos):
    """Traduce la antiguedad del dato a un estado legible."""
    if minutos is None:
        return "viejo", "sin datos"
    if minutos < 30:
        return "fresco", f"hace {int(minutos)} min"
    if minutos < 180:
        return "atrasado", f"hace {int(minutos)} min"
    horas = minutos / 60
    return "viejo", f"hace {horas:.1f} h"
