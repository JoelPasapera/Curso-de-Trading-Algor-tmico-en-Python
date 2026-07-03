# ============================================================================
# CURSO DE TRADING ALGORÍTMICO EN PYTHON
# SEMANA 4 — GESTIÓN DE RIESGO Y TAMAÑO DE POSICIÓN
# ============================================================================
# Objetivos de esta semana:
#   1. Entender que el TAMAÑO de la posición importa tanto como la señal.
#   2. Implementar tres métodos: riesgo fijo con ATR, volatility targeting
#      y Kelly fraccionado.
#   3. Comparar la MISMA señal con y sin gestión de riesgo.
#
# Idea central: dos traders con la misma señal pueden acabar uno rico y otro
# quebrado. La diferencia es cuánto arriesgan por operación. Una pérdida del
# 50% exige un +100% solo para recuperar: la asimetría de las pérdidas es
# la razón matemática por la que la gestión de riesgo no es opcional.
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
COSTO_BPS = 5
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


def atr(df, n=14):
    h_l = df["High"] - df["Low"]
    h_c = (df["High"] - df["Close"].shift(1)).abs()
    l_c = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def senal_cruce_medias(close, rapida=20, lenta=50):
    s_r, s_l = sma(close, rapida), sma(close, lenta)
    senal = pd.Series(np.where(s_r > s_l, 1, 0), index=close.index)
    senal[s_l.isna()] = 0
    return senal


# ----------------------------------------------------------------------------
# 1) MOTOR DE BACKTEST GENERALIZADO
# ----------------------------------------------------------------------------
# Mejora respecto a la semana 3: ahora aceptamos PESOS (0.0 a 1.0), no solo
# señales binarias 0/1. Un peso de 0.4 significa invertir el 40% del capital.
# Esto permite modular la exposición según el riesgo del momento.
# ----------------------------------------------------------------------------
def backtest_pesos(df, pesos, costo_bps=COSTO_BPS, capital=CAPITAL_INICIAL):
    ret_activo = df["Close"].pct_change().fillna(0)
    posicion = pesos.shift(1).fillna(0)                      # ejecución al día siguiente
    rotacion = posicion.diff().abs().fillna(posicion.abs())  # los ajustes de peso también pagan costo
    costos = rotacion * (costo_bps / 10_000)
    ret = posicion * ret_activo - costos
    equity = capital * (1 + ret).cumprod()
    return ret, equity, posicion


def metricas_resumen(retornos, equity):
    total = equity.iloc[-1] / equity.iloc[0] - 1
    anios = max(len(retornos) / DIAS_ANIO, 1e-9)
    cagr = (1 + total) ** (1 / anios) - 1
    vol = retornos.std() * np.sqrt(DIAS_ANIO)
    sharpe = (retornos.mean() * DIAS_ANIO) / vol if vol > 0 else 0.0
    dd = (equity / equity.cummax() - 1).min()
    calmar = (cagr * 100) / abs(dd * 100) if dd != 0 else 0.0
    return {
        "CAGR %": cagr * 100,
        "Vol anual %": vol * 100,
        "Sharpe": sharpe,
        "Max DD %": dd * 100,
        "Calmar": calmar,
    }


# ----------------------------------------------------------------------------
# 2) MÉTODO A — RIESGO FIJO POR OPERACIÓN CON ATR
# ----------------------------------------------------------------------------
# Regla profesional clásica: "no arriesgar más del 1% del capital por operación".
# Si colocamos el stop a k·ATR del precio de entrada:
#     pérdida si salta el stop = exposición · (k·ATR / precio)
# Despejando la exposición que hace esa pérdida igual al 1%:
#     peso = (riesgo_pct · precio) / (k · ATR)
# Cuando el mercado está nervioso (ATR alto) el peso baja solo. Elegante.
# ----------------------------------------------------------------------------
def pesos_riesgo_atr(df, senal, riesgo_pct=0.01, k_atr=2.0, peso_max=1.0):
    a = atr(df, 14)
    peso = (riesgo_pct * df["Close"]) / (k_atr * a)
    peso = peso.clip(upper=peso_max).fillna(0)   # sin apalancamiento en este curso
    return senal * peso


# ----------------------------------------------------------------------------
# 3) MÉTODO B — VOLATILITY TARGETING
# ----------------------------------------------------------------------------
# Fijamos una volatilidad objetivo para la cartera (ej. 15% anual) y ajustamos
# la exposición cada día:  peso = vol_objetivo / vol_realizada_reciente.
# Resultado: riesgo estable en el tiempo. Es la base de muchos fondos
# sistemáticos profesionales.
# ----------------------------------------------------------------------------
def pesos_vol_objetivo(df, senal, vol_objetivo=0.15, ventana=20, peso_max=1.0):
    ret = df["Close"].pct_change()
    vol_realizada = ret.rolling(ventana).std() * np.sqrt(DIAS_ANIO)
    peso = (vol_objetivo / vol_realizada).clip(upper=peso_max).fillna(0)
    return senal * peso


# ----------------------------------------------------------------------------
# 4) MÉTODO C — CRITERIO DE KELLY (Y POR QUÉ SE USA FRACCIONADO)
# ----------------------------------------------------------------------------
# Kelly maximiza el crecimiento a largo plazo:  f* = p − (1−p)/b
#   p = probabilidad de ganar, b = ganancia media / pérdida media.
# Problema: el Kelly completo produce drawdowns intolerables y asume que
# conoces p y b con exactitud (nunca es cierto). En la práctica se usa
# 1/4 o 1/2 de Kelly como TECHO de exposición, no como objetivo.
# ----------------------------------------------------------------------------
def kelly(p, b):
    if b <= 0:
        return 0.0
    return max(0.0, p - (1 - p) / b)


def demo_kelly(posicion, ret_activo):
    # Reconstruimos operaciones para estimar p y b de nuestra señal
    trades = []
    ret_acum, en_trade = 0.0, False
    for pos, r in zip(posicion.to_numpy(), ret_activo.to_numpy()):
        if pos != 0:
            en_trade = True
            ret_acum = (1 + ret_acum) * (1 + r) - 1  # señal binaria: exposición completa
        elif en_trade:
            trades.append(ret_acum)
            ret_acum, en_trade = 0.0, False
    if en_trade:
        trades.append(ret_acum)
    trades = pd.Series(trades, dtype=float)
    if len(trades) < 5:
        print("  (Muy pocas operaciones para estimar Kelly con seriedad)\n")
        return

    ganadoras = trades[trades > 0]
    perdedoras = trades[trades <= 0]
    p = len(ganadoras) / len(trades)
    g = ganadoras.mean() if len(ganadoras) else 0.0
    q = abs(perdedoras.mean()) if len(perdedoras) else 1e-9
    b = g / q
    f = kelly(p, b)
    expectativa = p * g - (1 - p) * q

    print("=" * 60)
    print("CRITERIO DE KELLY (estimado sobre las operaciones de la señal)")
    print("=" * 60)
    print(f"  Operaciones:            {len(trades)}")
    print(f"  p (prob. de ganar):     {p * 100:.1f} %")
    print(f"  b (ganancia/pérdida):   {b:.2f}")
    print(f"  Expectativa por op.:    {expectativa * 100:+.2f} %")
    print(f"  Kelly completo f*:      {f * 100:.1f} % del capital")
    print(f"  Kelly 1/4 (prudente):   {f / 4 * 100:.1f} % del capital")
    print("  Nota: con pocas operaciones, p y b tienen mucho error de estimación.")
    print("  Por eso Kelly fraccionado es un techo, nunca una promesa.\n")


# ----------------------------------------------------------------------------
# PROGRAMA PRINCIPAL — LA MISMA SEÑAL, TRES PERFILES DE RIESGO
# ----------------------------------------------------------------------------
def main():
    print("\nSEMANA 4 — Gestión de riesgo | Material educativo.\n")
    df = cargar_datos()
    ret_activo = df["Close"].pct_change().fillna(0)
    senal = senal_cruce_medias(df["Close"], 20, 50)

    variantes = {
        "A) Señal pura (todo o nada)": senal.astype(float),
        "B) Riesgo fijo 1% con ATR": pesos_riesgo_atr(df, senal),
        "C) Vol objetivo 15%": pesos_vol_objetivo(df, senal),
    }

    resultados, curvas = {}, {}
    for nombre, pesos in variantes.items():
        ret, eq, pos = backtest_pesos(df, pesos)
        resultados[nombre] = metricas_resumen(ret, eq)
        curvas[nombre] = eq

    tabla = pd.DataFrame(resultados).T.round(2)
    print("COMPARATIVA — misma señal, distinto tamaño de posición:")
    print(tabla.to_string())
    print()
    print("Cómo leerla: fíjate en el Max DD y el Calmar. La gestión de riesgo")
    print("no busca ganar más, busca perder MENOS y de forma más predecible.")
    print("Un drawdown menor = estrategia que un humano puede seguir sin abandonarla.\n")

    # Kelly informativo (sobre la señal binaria)
    _, _, pos_binaria = backtest_pesos(df, senal.astype(float))
    demo_kelly(pos_binaria, ret_activo)

    # Gráfico comparativo
    os.makedirs(CARPETA_GRAFICOS, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    colores = {"A) Señal pura (todo o nada)": "gray",
               "B) Riesgo fijo 1% con ATR": "darkorange",
               "C) Vol objetivo 15%": "darkgreen"}
    for nombre, eq in curvas.items():
        axes[0].plot(eq.index, eq, lw=1.2, label=nombre, color=colores[nombre])
        dd = eq / eq.cummax() - 1
        axes[1].plot(dd.index, dd * 100, lw=1.0, color=colores[nombre])
    axes[0].set_title("Misma señal, tres perfiles de riesgo — curvas de capital")
    axes[0].set_yscale("log")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].set_title("Drawdown (%)")
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    ruta = os.path.join(CARPETA_GRAFICOS, "semana4_riesgo.png")
    plt.savefig(ruta, dpi=110)
    print(f"[OK] Gráfico guardado en {ruta}")
    plt.show()

    print("\nTAREA DE LA SEMANA:")
    print("  1. Prueba vol objetivo del 10% y del 20%. ¿Cómo cambian CAGR y Max DD?")
    print("  2. Escribe tu regla personal: ¿qué drawdown máximo tolerarías sin")
    print("     abandonar el sistema? Esa cifra manda sobre todo lo demás.")


if __name__ == "__main__":
    main()
