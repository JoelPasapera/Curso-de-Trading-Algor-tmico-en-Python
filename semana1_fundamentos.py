# ============================================================================
# CURSO DE TRADING ALGORÍTMICO EN PYTHON
# SEMANA 1 — FUNDAMENTOS: DATOS, RETORNOS Y LÍNEA BASE
# ============================================================================
# Objetivos de esta semana:
#   1. Preparar el entorno y descargar datos de mercado (OHLCV).
#   2. Entender retornos simples vs. logarítmicos y por qué importan.
#   3. Medir volatilidad y anualizarla correctamente.
#   4. Construir la línea base contra la que se compara TODO: buy & hold.
#
# Filosofía: antes de buscar "la estrategia", hay que saber medir.
# Sin una línea base y sin métricas, cualquier resultado es una anécdota.
#
# Material educativo. No constituye asesoramiento financiero.
# ============================================================================

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# CONFIGURACIÓN
# ----------------------------------------------------------------------------
TICKER = "SPY"            # Cambia aquí el activo: "BTC-USD", "AAPL", "GC=F"...
FECHA_INICIO = "2018-01-01"
DIAS_ANIO = 252           # Días de trading por año (para anualizar métricas)
SEED = 42                 # Semilla fija => resultados reproducibles
CARPETA_GRAFICOS = "graficos"


# ----------------------------------------------------------------------------
# 1) DATOS SINTÉTICOS DE RESPALDO
# ----------------------------------------------------------------------------
# Si no hay internet, generamos una serie de precios artificial pero realista
# (paseo aleatorio con deriva y ciclos suaves). Sirve para practicar toda la
# mecánica del curso. IMPORTANTE: las conclusiones de mercado solo valen con
# datos reales.
# ----------------------------------------------------------------------------
def generar_datos_sinteticos(n_dias=1800, precio_inicial=100.0, seed=SEED):
    rng = np.random.default_rng(seed)

    # Retornos diarios: media ligeramente positiva + ruido + ciclos de tendencia
    ruido = rng.normal(loc=0.0006, scale=0.011, size=n_dias)
    ciclo = np.sin(np.linspace(0, 10 * np.pi, n_dias)) * 0.0010
    retornos = ruido + ciclo

    close = precio_inicial * np.exp(np.cumsum(retornos))
    fechas = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_dias)

    df = pd.DataFrame(index=fechas)
    df["Close"] = close
    df["Open"] = df["Close"].shift(1).fillna(precio_inicial) * (
        1 + rng.normal(0, 0.001, n_dias)
    )
    rango = np.abs(rng.normal(0, 0.007, n_dias))
    df["High"] = np.maximum(df["Open"], df["Close"]) * (1 + rango)
    df["Low"] = np.minimum(df["Open"], df["Close"]) * (1 - rango)
    df["Volume"] = rng.integers(1_000_000, 5_000_000, n_dias)
    return df[["Open", "High", "Low", "Close", "Volume"]]


# ----------------------------------------------------------------------------
# 2) CARGA DE DATOS (REALES SI ES POSIBLE, SINTÉTICOS SI NO)
# ----------------------------------------------------------------------------
def cargar_datos(ticker=TICKER, inicio=FECHA_INICIO):
    try:
        import yfinance as yf

        df = yf.download(ticker, start=inicio, progress=False, auto_adjust=True)
        if df is None or df.empty:
            raise ValueError("descarga vacía")
        # yfinance a veces devuelve columnas MultiIndex (ticker, campo)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        print(f"[OK] Datos reales descargados: {ticker} ({len(df)} sesiones)")
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    except Exception as e:
        print(f"[AVISO] No se pudieron descargar datos reales ({e}).")
        print("        Usando datos SINTÉTICOS de práctica.\n")
        return generar_datos_sinteticos()


# ----------------------------------------------------------------------------
# 3) EXPLORACIÓN BÁSICA
# ----------------------------------------------------------------------------
# Siempre inspecciona los datos antes de calcular nada: rango de fechas,
# huecos, valores nulos. La mitad de los "bugs" en trading algorítmico
# son en realidad problemas de datos.
# ----------------------------------------------------------------------------
def explorar_datos(df):
    print("=" * 60)
    print("EXPLORACIÓN DE DATOS")
    print("=" * 60)
    print(f"Filas (sesiones):      {len(df)}")
    print(f"Rango de fechas:       {df.index[0].date()} -> {df.index[-1].date()}")
    print(f"Valores nulos totales: {int(df.isna().sum().sum())}")
    print(f"Último cierre:         {df['Close'].iloc[-1]:.2f}")
    print("\nPrimeras filas:")
    print(df.head(3).round(2))
    print()


# ----------------------------------------------------------------------------
# 4) RETORNOS: SIMPLES VS. LOGARÍTMICOS
# ----------------------------------------------------------------------------
# Retorno simple:      r_t = P_t / P_{t-1} - 1
#   -> Es lo que gana tu cuenta. Se COMPONE multiplicando: (1+r1)(1+r2)...
# Retorno logarítmico: l_t = ln(P_t / P_{t-1})
#   -> Se SUMA en el tiempo, cómodo para estadística y agregación temporal.
# Regla práctica del curso: log-retornos para analizar, retornos simples
# para calcular el dinero real de la estrategia.
# ----------------------------------------------------------------------------
def calcular_retornos(df):
    df = df.copy()
    df["ret_simple"] = df["Close"].pct_change()
    df["ret_log"] = np.log(df["Close"] / df["Close"].shift(1))
    return df.dropna()


def estadisticas_basicas(df):
    r = df["ret_simple"]
    vol_diaria = r.std()
    vol_anual = vol_diaria * np.sqrt(DIAS_ANIO)  # la vol escala con sqrt(t)
    media_anual = r.mean() * DIAS_ANIO

    print("=" * 60)
    print("ESTADÍSTICAS DE RETORNOS")
    print("=" * 60)
    print(f"Retorno medio diario:        {r.mean() * 100:+.4f} %")
    print(f"Retorno medio anualizado:    {media_anual * 100:+.2f} %")
    print(f"Volatilidad diaria:          {vol_diaria * 100:.2f} %")
    print(f"Volatilidad anualizada:      {vol_anual * 100:.2f} %")
    print(f"Mejor día:                   {r.max() * 100:+.2f} %  ({r.idxmax().date()})")
    print(f"Peor día:                    {r.min() * 100:+.2f} %  ({r.idxmin().date()})")
    print(f"% de días positivos:         {(r > 0).mean() * 100:.1f} %")
    print()
    # Lección clave: incluso un activo alcista tiene ~45-55% de días verdes.
    # El "edge" diario es pequeño; por eso el proceso importa más que el acierto puntual.


# ----------------------------------------------------------------------------
# 5) LÍNEA BASE: BUY & HOLD
# ----------------------------------------------------------------------------
# Comprar y mantener es el rival a batir. Si tu estrategia no supera
# (en retorno ajustado a riesgo) a simplemente mantener el activo,
# la complejidad extra no se justifica.
# ----------------------------------------------------------------------------
def linea_base_buy_hold(df, capital_inicial=10_000):
    equity = capital_inicial * (1 + df["ret_simple"]).cumprod()
    retorno_total = equity.iloc[-1] / capital_inicial - 1
    anios = len(df) / DIAS_ANIO
    cagr = (1 + retorno_total) ** (1 / anios) - 1

    # Máximo drawdown: la peor caída desde un máximo previo.
    # Es LA métrica de dolor: define si un humano aguantaría la estrategia.
    maximos = equity.cummax()
    drawdown = equity / maximos - 1
    max_dd = drawdown.min()

    print("=" * 60)
    print("LÍNEA BASE — BUY & HOLD")
    print("=" * 60)
    print(f"Capital inicial:   {capital_inicial:,.0f}")
    print(f"Capital final:     {equity.iloc[-1]:,.0f}")
    print(f"Retorno total:     {retorno_total * 100:+.1f} %")
    print(f"CAGR:              {cagr * 100:+.2f} %  (crecimiento anual compuesto)")
    print(f"Máximo drawdown:   {max_dd * 100:.1f} %")
    print()
    return equity, drawdown


# ----------------------------------------------------------------------------
# 6) VISUALIZACIÓN
# ----------------------------------------------------------------------------
def graficar(df, equity, drawdown):
    os.makedirs(CARPETA_GRAFICOS, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

    axes[0].plot(df.index, df["Close"], color="steelblue", lw=1.2)
    axes[0].set_title(f"{TICKER} — Precio de cierre")
    axes[0].grid(alpha=0.3)

    axes[1].plot(equity.index, equity, color="darkgreen", lw=1.2)
    axes[1].set_title("Curva de capital buy & hold (10.000 iniciales)")
    axes[1].grid(alpha=0.3)

    axes[2].fill_between(drawdown.index, drawdown * 100, 0, color="firebrick", alpha=0.4)
    axes[2].set_title("Drawdown (%) — la métrica del dolor")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    ruta = os.path.join(CARPETA_GRAFICOS, "semana1_panorama.png")
    plt.savefig(ruta, dpi=110)
    print(f"[OK] Gráfico guardado en {ruta}")
    plt.show()


# ----------------------------------------------------------------------------
# PROGRAMA PRINCIPAL
# ----------------------------------------------------------------------------
def main():
    print("\nSEMANA 1 — Fundamentos | Material educativo, no es asesoramiento financiero.\n")
    df = cargar_datos()
    explorar_datos(df)
    df = calcular_retornos(df)
    estadisticas_basicas(df)
    equity, drawdown = linea_base_buy_hold(df)
    graficar(df, equity, drawdown)

    print("TAREA DE LA SEMANA:")
    print("  1. Cambia TICKER por otro activo y compara vol anualizada y max drawdown.")
    print("  2. Responde: ¿qué activo te dejaría dormir tranquilo? ¿Por qué?")


if __name__ == "__main__":
    main()
