# ============================================================================
# CURSO DE TRADING ALGORÍTMICO EN PYTHON — MÓDULO AVANZADO
# SEMANA 10 — DEL BACKTEST A LA EJECUCIÓN: PAPER TRADING
# ============================================================================
# Objetivos:
#   1. Convertir pesos objetivo en ÓRDENES concretas (el "puente" a la realidad).
#   2. Introducir el umbral de rebalanceo: no operar por migajas.
#   3. Mantener ESTADO entre días (JSON) y un JOURNAL de órdenes (CSV).
#   4. Medir la diferencia entre backtest teórico y ejecución simulada.
#
# Idea central: en producción no existen "pesos", existen órdenes, estados y
# registros. El que no registra, no aprende.
#
# Material educativo. No constituye asesoramiento financiero.
# Todo lo de esta semana es PAPER TRADING: dinero ficticio, proceso real.
# ============================================================================

import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# CONFIGURACIÓN
# ----------------------------------------------------------------------------
TICKERS = ["SPY", "QQQ", "IWM", "EFA", "GLD", "TLT"]
FECHA_INICIO = "2018-01-01"
DIAS_ANIO = 252
SEED = 42
COSTO_BPS = 5
UMBRAL_REBALANCEO = 0.01     # no tocar posiciones si el desvío es < 1% del capital
CAPITAL_INICIAL = 100_000.0
DIAS_SIMULACION = 120        # cuántas sesiones "vivimos" día a día
CARPETA_GRAFICOS = "graficos"
RUTA_ESTADO = "estado_cartera.json"
RUTA_JOURNAL = "orders_journal.csv"

PARAMS_SINTETICOS = {
    "SPY": ( 1.00, 0.00020, 0.004),
    "QQQ": ( 1.15, 0.00030, 0.006),
    "IWM": ( 1.05, 0.00010, 0.007),
    "EFA": ( 0.85, 0.00005, 0.006),
    "GLD": ( 0.10, 0.00015, 0.008),
    "TLT": (-0.30, 0.00005, 0.006),
}


def generar_cierres_sinteticos(n_dias=1800, seed=SEED):
    rng = np.random.default_rng(seed)
    mercado = rng.normal(0.0004, 0.010, n_dias)
    mercado += np.sin(np.linspace(0, 10 * np.pi, n_dias)) * 0.0010
    fechas = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_dias)
    cierres = {}
    for t, (beta, deriva, vol_idio) in PARAMS_SINTETICOS.items():
        r = beta * mercado + deriva + rng.normal(0, vol_idio, n_dias)
        cierres[t] = 100.0 * np.exp(np.cumsum(r))
    return pd.DataFrame(cierres, index=fechas)


def cargar_cierres(tickers=TICKERS, inicio=FECHA_INICIO):
    try:
        import yfinance as yf

        df = yf.download(tickers, start=inicio, progress=False, auto_adjust=True)
        cierres = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df[["Close"]]
        cierres = cierres.dropna(how="any")
        if cierres.empty:
            raise ValueError("descarga vacía")
        print(f"[OK] Datos reales: {list(cierres.columns)} ({len(cierres)} sesiones)")
        return cierres
    except Exception as e:
        print(f"[AVISO] Sin datos reales ({e}). Usando datos SINTÉTICOS.")
        return generar_cierres_sinteticos()


# ----------------------------------------------------------------------------
# 1) LA ESTRATEGIA (la cesta de cruce de la semana 7, sin cambios)
# ----------------------------------------------------------------------------
def pesos_objetivo_cruce(cierres):
    senal = (cierres.rolling(20).mean() > cierres.rolling(50).mean()).astype(float)
    return senal / senal.shape[1]


# ----------------------------------------------------------------------------
# 2) ESTADO Y JOURNAL: la memoria del sistema
# ----------------------------------------------------------------------------
def cargar_estado(ruta=RUTA_ESTADO):
    if os.path.exists(ruta):
        with open(ruta, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"capital": CAPITAL_INICIAL, "pesos": {t: 0.0 for t in TICKERS}}


def guardar_estado(estado, ruta=RUTA_ESTADO):
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(estado, f, indent=2)


# ----------------------------------------------------------------------------
# 3) DEL PESO A LA ORDEN
# ----------------------------------------------------------------------------
# El plan de ejecución compara pesos objetivo vs. pesos actuales y solo genera
# órdenes cuando el desvío supera el umbral. Dos razones:
#   a) los desvíos pequeños cuestan más en comisiones de lo que aportan;
#   b) menos órdenes = menos puntos de fallo operativos.
# ----------------------------------------------------------------------------
def plan_ejecucion(pesos_obj, pesos_act, capital, precios, umbral=UMBRAL_REBALANCEO):
    ordenes = []
    for t in pesos_obj.index:
        delta = float(pesos_obj[t] - pesos_act.get(t, 0.0))
        if abs(delta) < umbral:
            continue
        nocional = delta * capital
        ordenes.append({
            "ticker": t,
            "lado": "COMPRA" if delta > 0 else "VENTA",
            "delta_peso": round(delta, 4),
            "nocional": round(nocional, 2),
            "titulos_aprox": round(nocional / float(precios[t]), 2),
            "precio_ref": round(float(precios[t]), 2),
        })
    return ordenes


# ----------------------------------------------------------------------------
# 4) SIMULACIÓN DÍA A DÍA (el bucle que en producción corre 1 vez al día)
# ----------------------------------------------------------------------------
def simular_ejecucion(cierres, dias=DIAS_SIMULACION):
    pesos_obj_hist = pesos_objetivo_cruce(cierres)
    r_activos = cierres.pct_change().fillna(0.0)
    fechas = cierres.index[-dias:]

    estado = {"capital": CAPITAL_INICIAL, "pesos": {t: 0.0 for t in TICKERS}}
    journal, curva = [], []

    for fecha in fechas:
        # 1) El mercado mueve las posiciones que YA teníamos (de ayer):
        r_dia = float(sum(estado["pesos"][t] * r_activos.loc[fecha, t] for t in TICKERS))
        estado["capital"] *= (1 + r_dia)

        # 2) Al cierre, la estrategia calcula pesos objetivo con datos de HOY:
        objetivo = pesos_obj_hist.loc[fecha]

        # 3) Se genera y "ejecuta" el plan de órdenes (paper):
        ordenes = plan_ejecucion(objetivo, estado["pesos"], estado["capital"],
                                 cierres.loc[fecha])
        costo_dia = 0.0
        for o in ordenes:
            costo = abs(o["nocional"]) * COSTO_BPS / 10_000
            costo_dia += costo
            estado["pesos"][o["ticker"]] = float(objetivo[o["ticker"]])
            journal.append({"fecha": str(fecha.date()), **o,
                            "costo": round(costo, 2)})
        estado["capital"] -= costo_dia
        curva.append({"fecha": fecha, "capital": estado["capital"],
                      "n_ordenes": len(ordenes)})

    guardar_estado(estado)
    pd.DataFrame(journal).to_csv(RUTA_JOURNAL, index=False)
    return pd.DataFrame(curva).set_index("fecha"), journal, estado


# ----------------------------------------------------------------------------
# 5) ESQUELETO DE CONEXIÓN REAL (NO se ejecuta: es el mapa para el futuro)
# ----------------------------------------------------------------------------
# Con un exchange de cripto (vía la librería ccxt) el patrón sería:
#
#   import ccxt
#   exchange = ccxt.binance({
#       "apiKey": os.environ["API_KEY"],        # credenciales SIEMPRE fuera
#       "secret": os.environ["API_SECRET"],     # del código (semana 11)
#   })
#   exchange.set_sandbox_mode(True)             # 1º SIEMPRE el sandbox/testnet
#   for o in ordenes:
#       exchange.create_order(o["ticker"], "market", o["lado"].lower(),
#                             abs(o["titulos_aprox"]))
#
# Con brokers de acciones/futuros el patrón es idéntico a través de su API
# oficial, empezando SIEMPRE por la cuenta de papel que ofrecen.
# La lógica de esta semana (estado -> plan -> órdenes -> journal) no cambia:
# solo se sustituye la ejecución simulada por llamadas a la API.
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# PROGRAMA PRINCIPAL
# ----------------------------------------------------------------------------
def main():
    os.makedirs(CARPETA_GRAFICOS, exist_ok=True)
    cierres = cargar_cierres()

    # --- Ejecución simulada día a día -------------------------------------------
    curva, journal, estado = simular_ejecucion(cierres)
    print(f"\nSimuladas {len(curva)} sesiones de ejecución diaria (paper).")
    print(f"Órdenes generadas: {len(journal)} | "
          f"Días sin operar: {(curva['n_ordenes'] == 0).sum()} de {len(curva)}")
    print(f"Capital final: {estado['capital']:,.2f} "
          f"(inicial {CAPITAL_INICIAL:,.0f})")
    print(f"[Guardado] {RUTA_JOURNAL} y {RUTA_ESTADO}")

    print("\nÚltimas 5 órdenes del journal:")
    print(pd.DataFrame(journal).tail(5).to_string(index=False))

    # --- Teoría vs. ejecución -----------------------------------------------------
    # El backtest "teórico" rebalancea cada día sin umbral. La ejecución real
    # usa umbral: menos órdenes, menos costos, pequeña desviación de pesos.
    r_activos = cierres.pct_change().fillna(0.0)
    w = pesos_objetivo_cruce(cierres)
    pos = w.shift(1).fillna(0.0)
    rot = w.diff().abs().sum(axis=1).fillna(0.0)
    r_teorico = ((pos * r_activos).sum(axis=1) - rot * COSTO_BPS / 10_000)
    eq_teo = (1 + r_teorico.loc[curva.index]).cumprod() * CAPITAL_INICIAL

    dif = (curva["capital"].iloc[-1] / eq_teo.iloc[-1] - 1) * 100
    print(f"\nTracking teoría vs. ejecución con umbral: {dif:+.2f} % en el periodo")
    print("Un tracking pequeño y explicable = el puente funciona. Si fuera")
    print("grande, hay un bug o costos mal modelados: se investiga ANTES de seguir.")

    fig, ax = plt.subplots(figsize=(11, 5))
    eq_teo.plot(ax=ax, label="Backtest teórico (rebalanceo diario)")
    curva["capital"].plot(ax=ax, label="Ejecución paper (umbral 1%)")
    ax.set_title("Semana 10 — Teoría vs. ejecución simulada")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    ruta = os.path.join(CARPETA_GRAFICOS, "semana10_ejecucion.png")
    fig.savefig(ruta, dpi=110)
    print(f"\n[Gráfico guardado] {ruta}")

    print("\nIDEAS CLAVE DE LA SEMANA 10")
    print("1. Producción = estado + plan de órdenes + journal. Sin registro no hay mejora.")
    print("2. El umbral de rebalanceo filtra órdenes-migaja que solo pagan comisiones.")
    print("3. La señal se calcula al cierre de HOY; se ejecuta a partir de MAÑANA.")
    print("4. Mide siempre el tracking entre teoría y ejecución: es tu detector de bugs.")
    print("5. API real: mismo esquema, credenciales en variables de entorno, sandbox 1º.")


if __name__ == "__main__":
    main()
