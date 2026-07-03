# ============================================================================
# CURSO DE TRADING ALGORÍTMICO EN PYTHON
# SEMANA 2 — INDICADORES TÉCNICOS Y SEÑALES
# ============================================================================
# Objetivos de esta semana:
#   1. Construir los indicadores clásicos DESDE CERO (SMA, EMA, RSI,
#      Bandas de Bollinger, ATR). Entenderlos > importarlos de una librería.
#   2. Convertir una hipótesis de mercado en una SEÑAL objetiva y binaria.
#   3. Visualizar señales sobre el precio para validar la lógica a ojo.
#
# Idea central: un indicador NO es una estrategia. Una estrategia es:
#   hipótesis -> regla cuantificable -> señal -> (semana 3) backtest honesto.
#
# Material educativo. No constituye asesoramiento financiero.
# ============================================================================

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TICKER = "SPY"
FECHA_INICIO = "2018-01-01"
SEED = 42
CARPETA_GRAFICOS = "graficos"


# ----------------------------------------------------------------------------
# CARGA DE DATOS (idéntica a la semana 1; cada script es autónomo)
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# 1) MEDIAS MÓVILES — el filtro de tendencia más simple
# ----------------------------------------------------------------------------
def sma(serie, n):
    """Media móvil simple: promedio de los últimos n cierres.
    Suaviza el ruido; cuanto mayor n, más lenta y más 'retraso' tiene."""
    return serie.rolling(n).mean()


def ema(serie, n):
    """Media móvil exponencial: pondera más los datos recientes.
    Reacciona antes que la SMA, a cambio de más señales falsas."""
    return serie.ewm(span=n, adjust=False).mean()


# ----------------------------------------------------------------------------
# 2) RSI — ¿el movimiento reciente está "estirado"?
# ----------------------------------------------------------------------------
def rsi(serie, n=14):
    """RSI de Wilder (0-100). >70 sobrecompra, <30 sobreventa (convención).
    Mide la fuerza relativa de subidas vs. bajadas recientes."""
    delta = serie.diff()
    ganancia = delta.clip(lower=0)
    perdida = -delta.clip(upper=0)
    # Media exponencial de Wilder (alpha = 1/n)
    media_g = ganancia.ewm(alpha=1 / n, adjust=False).mean()
    media_p = perdida.ewm(alpha=1 / n, adjust=False).mean()
    rs = media_g / media_p.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


# ----------------------------------------------------------------------------
# 3) BANDAS DE BOLLINGER — precio en contexto de su propia volatilidad
# ----------------------------------------------------------------------------
def bandas_bollinger(serie, n=20, k=2.0):
    """Media ± k desviaciones estándar. Bandas anchas = alta volatilidad."""
    media = sma(serie, n)
    desv = serie.rolling(n).std()
    return media, media + k * desv, media - k * desv


# ----------------------------------------------------------------------------
# 4) ATR — cuánto se mueve el activo "de verdad" cada día
# ----------------------------------------------------------------------------
def atr(df, n=14):
    """Average True Range. Base para stops y tamaño de posición (semana 4).
    El True Range cubre también los huecos de apertura, no solo High-Low."""
    h_l = df["High"] - df["Low"]
    h_c = (df["High"] - df["Close"].shift(1)).abs()
    l_c = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


# ----------------------------------------------------------------------------
# 5) DE INDICADOR A SEÑAL
# ----------------------------------------------------------------------------
# Hipótesis: "cuando la tendencia de corto plazo supera a la de largo plazo,
# es más probable que el precio siga subiendo" (seguimiento de tendencia).
# Regla objetiva: LONG (1) si SMA_rapida > SMA_lenta; FUERA (0) en caso contrario.
# Nota: este curso opera solo en largo o en liquidez, sin cortos, para
# mantener el foco en el proceso. Añadir cortos es un ejercicio posterior.
# ----------------------------------------------------------------------------
def senal_cruce_medias(close, rapida=20, lenta=50):
    s_r = sma(close, rapida)
    s_l = sma(close, lenta)
    senal = pd.Series(np.where(s_r > s_l, 1, 0), index=close.index)
    # Los primeros 'lenta' días no tienen media completa -> sin señal
    senal[s_l.isna()] = 0
    return senal


def resumen_senal(senal):
    cambios = senal.diff().fillna(0)
    entradas = int((cambios == 1).sum())
    salidas = int((cambios == -1).sum())
    print("=" * 60)
    print("RESUMEN DE LA SEÑAL (cruce SMA 20/50)")
    print("=" * 60)
    print(f"Días totales:            {len(senal)}")
    print(f"Días dentro del mercado: {int(senal.sum())} ({senal.mean() * 100:.1f} %)")
    print(f"Entradas (compras):      {entradas}")
    print(f"Salidas (ventas):        {salidas}")
    print(f"Estado actual:           {'DENTRO (long)' if senal.iloc[-1] == 1 else 'FUERA (liquidez)'}")
    print()
    # Ojo: todavía NO sabemos si esta señal gana dinero. Eso exige un
    # backtest honesto con costos -> semana 3. No te saltes ese paso.


# ----------------------------------------------------------------------------
# 6) VISUALIZACIÓN DE INDICADORES Y SEÑALES
# ----------------------------------------------------------------------------
def graficar(df, senal):
    os.makedirs(CARPETA_GRAFICOS, exist_ok=True)
    close = df["Close"]
    s20, s50 = sma(close, 20), sma(close, 50)
    media_bb, sup_bb, inf_bb = bandas_bollinger(close)
    indicador_rsi = rsi(close)

    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2, 1]})

    # Panel 1: precio + medias + marcas de entrada/salida
    axes[0].plot(close.index, close, color="black", lw=0.9, label="Cierre")
    axes[0].plot(s20.index, s20, color="dodgerblue", lw=1.1, label="SMA 20")
    axes[0].plot(s50.index, s50, color="darkorange", lw=1.1, label="SMA 50")
    cambios = senal.diff().fillna(0)
    entradas = close[cambios == 1]
    salidas = close[cambios == -1]
    axes[0].scatter(entradas.index, entradas, marker="^", color="green", s=70,
                    label="Entrada", zorder=5)
    axes[0].scatter(salidas.index, salidas, marker="v", color="red", s=70,
                    label="Salida", zorder=5)
    axes[0].set_title(f"{TICKER} — Cruce de medias 20/50 con señales")
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.3)

    # Panel 2: Bandas de Bollinger
    axes[1].plot(close.index, close, color="black", lw=0.8)
    axes[1].plot(media_bb.index, media_bb, color="gray", lw=1)
    axes[1].fill_between(close.index, inf_bb, sup_bb, color="steelblue", alpha=0.15)
    axes[1].set_title("Bandas de Bollinger (20, 2σ)")
    axes[1].grid(alpha=0.3)

    # Panel 3: RSI
    axes[2].plot(indicador_rsi.index, indicador_rsi, color="purple", lw=0.9)
    axes[2].axhline(70, color="red", ls="--", lw=0.8)
    axes[2].axhline(30, color="green", ls="--", lw=0.8)
    axes[2].set_ylim(0, 100)
    axes[2].set_title("RSI 14")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    ruta = os.path.join(CARPETA_GRAFICOS, "semana2_indicadores.png")
    plt.savefig(ruta, dpi=110)
    print(f"[OK] Gráfico guardado en {ruta}")
    plt.show()


# ----------------------------------------------------------------------------
# PROGRAMA PRINCIPAL
# ----------------------------------------------------------------------------
def main():
    print("\nSEMANA 2 — Indicadores y señales | Material educativo.\n")
    df = cargar_datos()

    close = df["Close"]
    print("Valores actuales de los indicadores:")
    print(f"  SMA 20:  {sma(close, 20).iloc[-1]:.2f}")
    print(f"  SMA 50:  {sma(close, 50).iloc[-1]:.2f}")
    print(f"  EMA 20:  {ema(close, 20).iloc[-1]:.2f}")
    print(f"  RSI 14:  {rsi(close).iloc[-1]:.1f}")
    print(f"  ATR 14:  {atr(df).iloc[-1]:.2f}\n")

    senal = senal_cruce_medias(close, rapida=20, lenta=50)
    resumen_senal(senal)
    graficar(df, senal)

    print("TAREA DE LA SEMANA:")
    print("  1. Crea una señal alternativa: LONG cuando RSI < 30, salir cuando RSI > 55.")
    print("  2. Compara a ojo (en el gráfico) dónde entra cada señal. ¿Cuál sigue")
    print("     tendencia y cuál compra caídas? Son dos familias distintas de estrategias.")


if __name__ == "__main__":
    main()
