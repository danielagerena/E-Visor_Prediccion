"""Vigilante del sistema.

Revisa que el pipeline siga vivo. Sin esto, "automatizado" significa "falla en
silencio hasta que alguien lo nota semanas despues".

Tambien poda las predicciones viejas para que la base no crezca sin control:
2304 filas por corrida y una corrida por hora llenan una base gratuita en pocos
meses si nadie borra nada.
"""
import argparse
import datetime as dt
import os
import smtplib
from email.message import EmailMessage

import pandas as pd
from sqlalchemy import text

from . import config, db


def revisar():
    """Devuelve la lista de problemas encontrados. Vacia significa todo bien."""
    problemas = []
    ahora = dt.datetime.now(dt.timezone.utc)

    ultima = db.ultima_medicion()
    if ultima is None:
        problemas.append("No hay ninguna medicion en la base")
    else:
        horas = (pd.Timestamp(ahora).tz_localize(None) - pd.Timestamp(ultima).tz_localize(None)).total_seconds() / 3600
        if horas > config.HORAS_MAX_SIN_DATOS:
            problemas.append(f"La medicion mas reciente tiene {horas:.1f} horas de antiguedad")

    r = db.consultar("SELECT MAX(generado_en) AS t FROM predicciones")
    if r.empty or not r["t"].iloc[0]:
        problemas.append("No hay predicciones generadas")
    else:
        t = pd.Timestamp(r["t"].iloc[0]).tz_localize(None)
        horas = (pd.Timestamp(ahora).tz_localize(None) - t).total_seconds() / 3600
        if horas > config.HORAS_MAX_SIN_PREDICCION:
            problemas.append(f"La ultima prediccion tiene {horas:.1f} horas")

    fallos = db.consultar(
        "SELECT job, COUNT(*) AS n FROM ejecuciones "
        "WHERE estado = 'error' AND inicio >= :d GROUP BY job",
        {"d": ahora - dt.timedelta(days=1)},
    )
    for _, f in fallos.iterrows():
        problemas.append(f"El job '{f['job']}' fallo {int(f['n'])} veces en las ultimas 24 horas")

    bloques = db.consultar(
        "SELECT COUNT(DISTINCT entity_id) AS n FROM predicciones "
        "WHERE generado_en >= :d", {"d": ahora - dt.timedelta(hours=6)}
    )
    n_bloques = int(bloques["n"].iloc[0]) if not bloques.empty else 0
    if 0 < n_bloques < len(config.BLOQUES):
        problemas.append(
            f"Solo {n_bloques} de {len(config.BLOQUES)} bloques tienen prediccion reciente"
        )

    nulos = db.consultar(
        "SELECT AVG(CASE WHEN activepower IS NULL THEN 1.0 ELSE 0.0 END) AS pct "
        "FROM mediciones WHERE ts >= :d", {"d": ahora - dt.timedelta(hours=24)}
    )
    if not nulos.empty and nulos["pct"].iloc[0] is not None and nulos["pct"].iloc[0] > 0.2:
        problemas.append(f"{100 * nulos['pct'].iloc[0]:.0f} % de nulos en las ultimas 24 horas")

    return problemas


def podar_predicciones(dias=None):
    """Borra pronosticos mas viejos que el periodo de retencion."""
    dias = dias or config.DIAS_RETENCION_PREDICCIONES
    corte = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=dias)
    with db.motor().begin() as con:
        r = con.execute(text("DELETE FROM predicciones WHERE ts_origen < :c"), {"c": corte})
    return r.rowcount or 0


def enviar_correo(asunto, cuerpo, destinatarios):
    """Envia por SMTP institucional. Sin credenciales, solo imprime."""
    host = os.getenv("EVISOR_SMTP_HOST")
    usuario = os.getenv("EVISOR_SMTP_USER")
    clave = os.getenv("EVISOR_SMTP_PASS")
    if not (host and usuario and clave and destinatarios):
        print(f"[sin SMTP configurado] {asunto}\n{cuerpo}")
        return False

    msg = EmailMessage()
    msg["Subject"] = asunto
    msg["From"] = usuario
    msg["To"] = ", ".join(destinatarios)
    msg.set_content(cuerpo)
    with smtplib.SMTP(host, int(os.getenv("EVISOR_SMTP_PORT", "587"))) as s:
        s.starttls()
        s.login(usuario, clave)
        s.send_message(msg)
    return True


def main():
    parser = argparse.ArgumentParser(description="Vigilante de E-Visor")
    parser.add_argument("--podar", action="store_true")
    args = parser.parse_args()

    inicio = dt.datetime.now(dt.timezone.utc)
    db.crear_esquema()

    if args.podar:
        print(f"Predicciones podadas: {podar_predicciones()}")

    problemas = revisar()
    if problemas:
        cuerpo = "E-Visor detecto los siguientes problemas:\n\n- " + "\n- ".join(problemas)
        print(cuerpo)
        destinatarios = [d for d in os.getenv("EVISOR_ALERTAS_A", "").split(",") if d]
        enviar_correo("[E-Visor] Revisar el pipeline", cuerpo, destinatarios)
        db.registrar_ejecucion("vigilante", inicio, "error", len(problemas), "; ".join(problemas))
        raise SystemExit(1)

    print("Todo en orden")
    db.registrar_ejecucion("vigilante", inicio, "ok", 0, "sin problemas")


if __name__ == "__main__":
    main()
