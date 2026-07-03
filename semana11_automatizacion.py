# ============================================================================
# CURSO DE TRADING ALGORÍTMICO EN PYTHON — MÓDULO AVANZADO
# SEMANA 11 — AUTOMATIZACIÓN E INFRAESTRUCTURA: EL BOT QUE NO TE DESPIERTA
# ============================================================================
# Objetivos:
#   1. Logging profesional: archivo + consola, con niveles.
#   2. Configuración por variables de entorno (credenciales FUERA del código).
#   3. Descarga con reintentos y VALIDACIÓN de datos antes de operar.
#   4. Kill switch doble: archivo STOP y límite de drawdown.
#   5. Un ciclo diario robusto: si algo falla, NO opera y avisa.
#
# Idea central: la regla nº1 de un bot es "ante la duda, no operar".
# Un día sin operar no cuesta casi nada; una orden con datos rotos, sí.
#
# Material educativo. No constituye asesoramiento financiero.
# ============================================================================

import logging
import os
import time

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# 1) CONFIGURACIÓN POR VARIABLES DE ENTORNO
# ----------------------------------------------------------------------------
# En producción el código es público para tu "yo" del futuro: las claves y los
# parámetros sensibles viven en el entorno, nunca escritos en el script.
#   Linux/Mac:  export BOT_VOL_OBJETIVO=0.15
#   Windows:    $env:BOT_VOL_OBJETIVO = "0.15"
# ----------------------------------------------------------------------------
TICKERS = os.environ.get("BOT_TICKERS", "SPY,QQQ,IWM,EFA,GLD,TLT").split(",")
VOL_OBJETIVO = float(os.environ.get("BOT_VOL_OBJETIVO", "0.15"))
DD_LIMITE = float(os.environ.get("BOT_DD_LIMITE", "-0.20"))   # apagado a -20%
ARCHIVO_STOP = "STOP"          # si este archivo existe, el bot NO opera
CARPETA_LOGS = "logs"
SEED = 42

PARAMS_SINTETICOS = {
    "SPY": ( 1.00, 0.00020, 0.004),
    "QQQ": ( 1.15, 0.00030, 0.006),
    "IWM": ( 1.05, 0.00010, 0.007),
    "EFA": ( 0.85, 0.00005, 0.006),
    "GLD": ( 0.10, 0.00015, 0.008),
    "TLT": (-0.30, 0.00005, 0.006),
}


# ----------------------------------------------------------------------------
# 2) LOGGING: la caja negra del avión
# ----------------------------------------------------------------------------
def configurar_logger():
    os.makedirs(CARPETA_LOGS, exist_ok=True)
    logger = logging.getLogger("bot")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formato = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")
    for handler in (logging.StreamHandler(),
                    logging.FileHandler(os.path.join(CARPETA_LOGS, "bot.log"),
                                        encoding="utf-8")):
        handler.setFormatter(formato)
        logger.addHandler(handler)
    return logger


log = configurar_logger()


# ----------------------------------------------------------------------------
# 3) ALERTAS
# ----------------------------------------------------------------------------
# En producción esto envía un mensaje (webhook, email, bot de mensajería...).
# El patrón: TODA alerta pasa por UNA función. Así cambiar el canal el día de
# mañana es tocar un solo sitio.
#   Ejemplo real (comentado):
#     import requests
#     requests.post(os.environ["WEBHOOK_URL"], json={"text": mensaje}, timeout=10)
# ----------------------------------------------------------------------------
def enviar_alerta(mensaje, nivel="INFO"):
    log.log(logging.ERROR if nivel == "ERROR" else logging.WARNING,
            f"[ALERTA {nivel}] {mensaje}")


# ----------------------------------------------------------------------------
# 4) DESCARGA CON REINTENTOS (backoff exponencial)
# ----------------------------------------------------------------------------
def generar_cierres_sinteticos(n_dias=1200, seed=SEED, romper=False):
    rng = np.random.default_rng(seed)
    mercado = rng.normal(0.0004, 0.010, n_dias)
    mercado += np.sin(np.linspace(0, 10 * np.pi, n_dias)) * 0.0010
    fechas = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_dias)
    cierres = {}
    for t, (beta, deriva, vol_idio) in PARAMS_SINTETICOS.items():
        r = beta * mercado + deriva + rng.normal(0, vol_idio, n_dias)
        cierres[t] = 100.0 * np.exp(np.cumsum(r))
    df = pd.DataFrame(cierres, index=fechas)[TICKERS]
    if romper:                       # para la DEMO de validación de datos
        df.iloc[-1, 0] = df.iloc[-1, 0] * 1.8      # salto absurdo del +80%
        df.iloc[-3, 1] = np.nan                    # hueco
    return df


def descargar_datos(max_intentos=3, romper=False):
    """Intenta datos reales con reintentos; si no hay red, sintéticos.
    El backoff (1s, 2s, 4s...) evita machacar a un servidor con problemas."""
    for intento in range(1, max_intentos + 1):
        try:
            import yfinance as yf

            df = yf.download(TICKERS, period="2y", progress=False, auto_adjust=True)
            cierres = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df[["Close"]]
            cierres = cierres.dropna(how="any")
            if cierres.empty:
                raise ValueError("descarga vacía")
            log.info(f"Datos reales OK en el intento {intento} ({len(cierres)} filas)")
            return cierres
        except Exception as e:
            log.warning(f"Intento {intento}/{max_intentos} fallido: {e}")
            if intento < max_intentos:
                time.sleep(min(2 ** (intento - 1), 4) * 0.05)  # acortado para la demo
    log.info("Sin red: usando datos sintéticos de respaldo.")
    return generar_cierres_sinteticos(romper=romper)


# ----------------------------------------------------------------------------
# 5) VALIDADOR DE DATOS: el portero de la discoteca
# ----------------------------------------------------------------------------
# Nada entra al motor sin pasar tres controles:
#   frescura (¿el último dato es reciente?), completitud (¿hay huecos?) y
#   cordura (¿algún salto diario imposible que huela a dato roto?).
# ----------------------------------------------------------------------------
class ValidadorDatos:
    def __init__(self, max_dias_antiguedad=5, salto_maximo=0.25):
        self.max_dias = max_dias_antiguedad
        self.salto_max = salto_maximo

    def validar(self, cierres):
        errores = []
        antiguedad = (pd.Timestamp.today().normalize() - cierres.index[-1]).days
        if antiguedad > self.max_dias:
            errores.append(f"datos con {antiguedad} días de antigüedad")
        nulos = int(cierres.tail(20).isna().sum().sum())
        if nulos > 0:
            errores.append(f"{nulos} valores nulos en las últimas 20 sesiones")
        saltos = cierres.pct_change().abs().tail(5)
        peor = saltos.max()
        for t, s in peor.items():
            if s > self.salto_max:
                errores.append(f"salto sospechoso en {t}: {s:+.0%} en un día")
        return errores


# ----------------------------------------------------------------------------
# 6) KILL SWITCH
# ----------------------------------------------------------------------------
def kill_switch_activado(curva_capital):
    if os.path.exists(ARCHIVO_STOP):
        return f"archivo {ARCHIVO_STOP} presente (apagado manual)"
    dd = float(curva_capital.iloc[-1] / curva_capital.cummax().iloc[-1] - 1)
    if dd <= DD_LIMITE:
        return f"drawdown {dd:.1%} <= límite {DD_LIMITE:.0%}"
    return None


# ----------------------------------------------------------------------------
# 7) EL CICLO DIARIO COMPLETO
# ----------------------------------------------------------------------------
# Este es el "main" que un programador de tareas lanza una vez al día:
#   - Linux/Mac (cron):        30 22 * * 1-5  cd /ruta/bot && python ciclo.py
#   - Windows (Task Scheduler): acción "python ciclo.py" a las 22:30, L-V.
# Devuelve un código de salida: 0 = OK, 1 = no operó (motivo controlado),
# 2 = error inesperado. El programador puede reintentar o avisar según el código.
# ----------------------------------------------------------------------------
def ciclo_diario(romper_datos=False, forzar_stop=False):
    log.info("===== INICIO DEL CICLO DIARIO =====")
    try:
        cierres = descargar_datos(romper=romper_datos)

        errores = ValidadorDatos().validar(cierres)
        if errores:
            for e in errores:
                enviar_alerta(f"Validación de datos: {e}", "ERROR")
            log.error("Datos NO válidos -> hoy NO se opera. (Regla nº1)")
            return 1

        # Curva de capital del sistema (aquí, simplificada para la demo):
        senal = (cierres.rolling(20).mean() > cierres.rolling(50).mean()).astype(float)
        w = senal / senal.shape[1]
        r = (w.shift(1).fillna(0.0) * cierres.pct_change().fillna(0.0)).sum(axis=1)
        curva = (1 + r).cumprod()

        if forzar_stop:
            open(ARCHIVO_STOP, "w").close()
        motivo = kill_switch_activado(curva)
        if motivo:
            enviar_alerta(f"KILL SWITCH: {motivo}. Bot detenido.", "ERROR")
            return 1

        # ... aquí iría el plan de ejecución de la semana 10 ...
        log.info(f"Señal calculada. Exposición hoy: {w.iloc[-1].sum():.0%}. "
                 "Órdenes enviadas al broker de papel.")
        log.info("===== CICLO COMPLETADO OK =====")
        return 0

    except Exception as e:
        # La red de seguridad final: NADA revienta sin quedar registrado.
        log.exception(f"Error inesperado en el ciclo: {e}")
        enviar_alerta(f"Ciclo diario caído: {e}", "ERROR")
        return 2
    finally:
        if os.path.exists(ARCHIVO_STOP):
            os.remove(ARCHIVO_STOP)          # limpieza solo para la demo


# ----------------------------------------------------------------------------
# PROGRAMA PRINCIPAL: tres ejecuciones de demostración
# ----------------------------------------------------------------------------
def main():
    print("\n" + "=" * 70)
    print("DEMO 1 — Día normal: todo en orden")
    print("=" * 70)
    codigo = ciclo_diario()
    print(f"-> código de salida: {codigo} (0 = OK)")

    print("\n" + "=" * 70)
    print("DEMO 2 — Datos rotos: el validador frena al bot")
    print("=" * 70)
    codigo = ciclo_diario(romper_datos=True)
    print(f"-> código de salida: {codigo} (1 = no operó, motivo controlado)")

    print("\n" + "=" * 70)
    print("DEMO 3 — Kill switch manual: archivo STOP presente")
    print("=" * 70)
    codigo = ciclo_diario(forzar_stop=True)
    print(f"-> código de salida: {codigo} (1 = apagado de emergencia)")

    print("\nRevisa la caja negra completa en logs/bot.log")
    print("\nIDEAS CLAVE DE LA SEMANA 11")
    print("1. Regla nº1: ante la duda (datos, errores), el bot NO opera y avisa.")
    print("2. Logs con niveles = poder reconstruir cualquier día a posteriori.")
    print("3. Credenciales y parámetros en variables de entorno, jamás en el código.")
    print("4. Kill switch doble: manual (archivo STOP) y automático (drawdown).")
    print("5. Códigos de salida claros: el programador de tareas sabe qué pasó.")


if __name__ == "__main__":
    main()
