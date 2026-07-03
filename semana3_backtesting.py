# ============================================================================
# CURSO DE TRADING ALGORÍTMICO EN PYTHON
# SEMANA 3 — BACKTESTING RIGUROSO
# ============================================================================
# Objetivos de esta semana:
#   1. Construir un motor de backtest vectorizado desde cero.
#   2. Eliminar el sesgo de anticipación (look-ahead) con shift(1).
#   3. Incluir costos de transacción y slippage: sin ellos, el backtest miente.
#   4. Calcular métricas profesionales: CAGR, Sharpe, Sortino, max drawdown,
#      win rate, profit factor, expectativa por operación.
#
# LOS 4 SESGOS QUE DESTRUYEN BACKTESTS (memorízalos):
#   a) Look-ahead: usar hoy información que solo se conoce mañana.
#   b) Supervivencia: probar solo con activos que "sobrevivieron" hasta hoy.
#   c) Costos ignorados: comisiones + spread + slippage comen estrategias enteras.
#   d) Sobreoptimización: lo vemos a fondo en la semana 5.
#
# Material educativo. No constituye asesoramiento financiero.
# ============================================================================

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TICKER = "SPY"
FECHA_INICIO = "2018-01-01"
DIAS_ANIO = 252
CAPITAL_INICIAL = 10_000
COSTO_BPS = 5          # 5 puntos básicos = 0.05% por operación (comisión+slippage)
SEED = 42
CARPETA_GRAFICOS = "graficos"


# --------------------------- datos (autónomo) --------------------------------
def generar_datos_sinteticos(n_dias=1800, precio_inicial=100.0, seed=SEED):
    rng = np.random.default_rng(seed)
    ruido = rng.normal(0.0006, 0.011, n_dias)
    ciclo = np.sin(np.linspace(0, 10 * np.pi, n_dias)) * 0.0010
    close = precio_inicial * np.exp(np.cumsum(ruido + ciclo))
    fechas = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_dias)
    df = pd.DataFrame(index=fechas)
    df["Close"] = close
    df["Open"] = df["Close"].shift(1).fillna(precio_inicial) * (1 + rng.normal(0, 0.001, n_dias))
    rango = np.abs(rng.normal(0, 0.007, n_dias))
    df["High"] = np.maximum(df["Open"], df["Close"]) * (1 + rango)
    df["Low"] = np.minimum(df["Open"], df["Close"]) * (1 - rango)
    df["Volume"] = rng.integers(1_000_000, 5_000_000, n_dias)
    return df[["Open", "High", "Low", "Close", "Volume"]]


def cargar_datos(ticker=TICKER, inicio=FECHA_INICIO):
    try:
        import yfinance as yf
        df = yf.download(ticker, start=inicio, progress=False, auto_adjust=True)
        if df is None or df.empty:
            raise ValueError("descarga vacía")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        print(f"[OK] Datos reales descargados: {ticker} ({len(df)} sesiones)")
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    except Exception as e:
        print(f"[AVISO] Sin datos reales ({e}). Usando datos SINTÉTICOS de práctica.\n")
        return generar_datos_sinteticos()


def sma(serie, n):
    return serie.rolling(n).mean()


def senal_cruce_medias(close, rapida=20, lenta=50):
    s_r, s_l = sma(close, rapida), sma(close, lenta)
    senal = pd.Series(np.where(s_r > s_l, 1, 0), index=close.index)
    senal[s_l.isna()] = 0
    return senal


# ----------------------------------------------------------------------------
# 1) MOTOR DE BACKTEST VECTORIZADO
# ----------------------------------------------------------------------------
# La línea MÁS IMPORTANTE del curso es:   posicion = senal.shift(1)
#
# ¿Por qué? La señal de hoy se calcula con el CIERRE de hoy. Es imposible
# haber comprado hoy con esa información: lo antes que puedes ejecutar es
# mañana. Sin shift(1) tu backtest "ve el futuro" y los resultados son
# ficción pura. Este error está en la mayoría de backtests de internet.
# ----------------------------------------------------------------------------
def backtest(df, senal, costo_bps=COSTO_BPS, capital=CAPITAL_INICIAL):
    ret_activo = df["Close"].pct_change().fillna(0)

    # Ejecutamos la señal al día siguiente (elimina el look-ahead)
    posicion = senal.shift(1).fillna(0)

    # Costos: cada cambio de posición (entrar o salir) paga costo_bps
    rotacion = posicion.diff().abs().fillna(posicion.abs())
    costos = rotacion * (costo_bps / 10_000)

    ret_estrategia = posicion * ret_activo - costos
    equity = capital * (1 + ret_estrategia).cumprod()
    return ret_estrategia, equity, posicion


# ----------------------------------------------------------------------------
# 2) EXTRACCIÓN DE OPERACIONES INDIVIDUALES
# ----------------------------------------------------------------------------
# Para win rate y profit factor necesitamos el retorno de CADA operación
# completa (desde que se entra hasta que se sale), no los retornos diarios.
# ----------------------------------------------------------------------------
def extraer_trades(posicion, ret_activo):
    trades = []
    ret_acum = 0.0
    en_trade = False
    for pos, r in zip(posicion.to_numpy(), ret_activo.to_numpy()):
        if pos != 0:
            en_trade = True
            ret_acum = (1 + ret_acum) * (1 + pos * r) - 1
        elif en_trade:
            trades.append(ret_acum)
            ret_acum, en_trade = 0.0, False
    if en_trade:
        trades.append(ret_acum)
    return pd.Series(trades, dtype=float)


# ----------------------------------------------------------------------------
# 3) MÉTRICAS PROFESIONALES
# ----------------------------------------------------------------------------
def calcular_metricas(retornos, equity, posicion=None, ret_activo=None):
    m = {}
    total = equity.iloc[-1] / equity.iloc[0] - 1
    anios = max(len(retornos) / DIAS_ANIO, 1e-9)

    m["Retorno total %"] = total * 100
    m["CAGR %"] = ((1 + total) ** (1 / anios) - 1) * 100

    vol = retornos.std() * np.sqrt(DIAS_ANIO)
    m["Volatilidad anual %"] = vol * 100

    # Sharpe: retorno por unidad de riesgo total (tasa libre de riesgo ~0 aquí
    # por simplicidad didáctica; en producción réstala).
    m["Sharpe"] = (retornos.mean() * DIAS_ANIO) / vol if vol > 0 else 0.0

    # Sortino: igual, pero penalizando SOLO la volatilidad de los días negativos.
    vol_neg = retornos[retornos < 0].std() * np.sqrt(DIAS_ANIO)
    m["Sortino"] = (retornos.mean() * DIAS_ANIO) / vol_neg if vol_neg > 0 else 0.0

    drawdown = equity / equity.cummax() - 1
    m["Max drawdown %"] = drawdown.min() * 100
    m["Calmar"] = m["CAGR %"] / abs(m["Max drawdown %"]) if m["Max drawdown %"] != 0 else 0.0

    if posicion is not None and ret_activo is not None:
        m["Exposición %"] = (posicion != 0).mean() * 100
        trades = extraer_trades(posicion, ret_activo)
        m["Nº operaciones"] = len(trades)
        if len(trades) > 0:
            ganadoras = trades[trades > 0]
            perdedoras = trades[trades <= 0]
            m["Win rate %"] = len(ganadoras) / len(trades) * 100
            suma_g = ganadoras.sum()
            suma_p = abs(perdedoras.sum())
            m["Profit factor"] = suma_g / suma_p if suma_p > 0 else np.inf
            # Expectativa: cuánto ganas EN PROMEDIO por operación.
            # E = p·GananciaMedia − (1−p)·PérdidaMedia. Si E ≤ 0, no hay edge.
            p = len(ganadoras) / len(trades)
            g_media = ganadoras.mean() if len(ganadoras) else 0.0
            p_media = abs(perdedoras.mean()) if len(perdedoras) else 0.0
            m["Expectativa/op %"] = (p * g_media - (1 - p) * p_media) * 100
    return m, drawdown


def imprimir_comparativa(m_estrategia, m_bh):
    print("=" * 66)
    print(f"{'MÉTRICA':<24}{'ESTRATEGIA':>18}{'BUY & HOLD':>18}")
    print("=" * 66)
    claves = ["Retorno total %", "CAGR %", "Volatilidad anual %", "Sharpe",
              "Sortino", "Max drawdown %", "Calmar", "Exposición %",
              "Nº operaciones", "Win rate %", "Profit factor", "Expectativa/op %"]
    for k in claves:
        v1 = m_estrategia.get(k)
        v2 = m_bh.get(k)
        def fmt(v):
            if not isinstance(v, (int, float)):
                return "—"
            return "n/a" if np.isinf(v) else f"{v:,.2f}"
        f1, f2 = fmt(v1), fmt(v2)
        print(f"{k:<24}{f1:>18}{f2:>18}")
    print("=" * 66)
    print("Cómo leerla: no busques solo más retorno. Busca mejor retorno POR")
    print("unidad de riesgo (Sharpe/Sortino/Calmar) y un drawdown soportable.\n")


# ----------------------------------------------------------------------------
# 4) GRÁFICOS
# ----------------------------------------------------------------------------
def graficar(equity_est, equity_bh, dd_est, dd_bh):
    os.makedirs(CARPETA_GRAFICOS, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    axes[0].plot(equity_est.index, equity_est, color="darkgreen", lw=1.3,
                 label="Estrategia (cruce 20/50, con costos)")
    axes[0].plot(equity_bh.index, equity_bh, color="gray", lw=1.1,
                 label="Buy & hold")
    axes[0].set_yscale("log")  # escala log: compara tasas de crecimiento
    axes[0].set_title("Curvas de capital (escala logarítmica)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].fill_between(dd_est.index, dd_est * 100, 0, color="darkgreen",
                         alpha=0.35, label="Estrategia")
    axes[1].fill_between(dd_bh.index, dd_bh * 100, 0, color="gray",
                         alpha=0.35, label="Buy & hold")
    axes[1].set_title("Drawdown (%)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    ruta = os.path.join(CARPETA_GRAFICOS, "semana3_backtest.png")
    plt.savefig(ruta, dpi=110)
    print(f"[OK] Gráfico guardado en {ruta}")
    plt.show()


# ----------------------------------------------------------------------------
# PROGRAMA PRINCIPAL
# ----------------------------------------------------------------------------
def main():
    print("\nSEMANA 3 — Backtesting riguroso | Material educativo.\n")
    df = cargar_datos()
    ret_activo = df["Close"].pct_change().fillna(0)

    # --- Estrategia: cruce de medias con costos ---
    senal = senal_cruce_medias(df["Close"], 20, 50)
    ret_est, eq_est, pos = backtest(df, senal)
    m_est, dd_est = calcular_metricas(ret_est, eq_est, pos, ret_activo)

    # --- Línea base: buy & hold (posición 1 siempre, un solo costo de entrada) ---
    senal_bh = pd.Series(1, index=df.index)
    ret_bh, eq_bh, pos_bh = backtest(df, senal_bh)
    m_bh, dd_bh = calcular_metricas(ret_bh, eq_bh, pos_bh, ret_activo)

    imprimir_comparativa(m_est, m_bh)

    # --- Demostración: el impacto brutal de los costos ---
    print("SENSIBILIDAD A COSTOS (misma señal, distinto costo por operación):")
    for bps in [0, 5, 15, 30]:
        r, e, _ = backtest(df, senal, costo_bps=bps)
        total = (e.iloc[-1] / e.iloc[0] - 1) * 100
        print(f"  {bps:>3} bps -> retorno total {total:+8.1f} %")
    print("  Lección: una estrategia que solo gana con costos = 0 no es una estrategia.\n")

    graficar(eq_est, eq_bh, dd_est, dd_bh)

    print("TAREA DE LA SEMANA:")
    print("  1. Quita el shift(1) del motor (solo como experimento) y compara métricas.")
    print("     Ese salto de rendimiento 'gratis' es el sesgo de anticipación en acción.")
    print("  2. Prueba la señal RSI de la semana 2 en este motor. ¿Supera al buy & hold?")


if __name__ == "__main__":
    main()
