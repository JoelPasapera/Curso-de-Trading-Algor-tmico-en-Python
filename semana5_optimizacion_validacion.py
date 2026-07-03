# ============================================================================
# CURSO DE TRADING ALGORÍTMICO EN PYTHON
# SEMANA 5 — OPTIMIZACIÓN Y VALIDACIÓN: EL ENEMIGO ES EL OVERFITTING
# ============================================================================
# Objetivos de esta semana:
#   1. Hacer un grid search de parámetros y visualizarlo como mapa de calor.
#   2. Entender el overfitting: ajustar la estrategia al RUIDO del pasado.
#   3. Separar in-sample (IS) y out-of-sample (OOS) y medir la degradación.
#   4. Implementar walk-forward: el estándar profesional de validación.
#   5. Simulación Monte Carlo: convertir "un resultado" en "una distribución
#      de resultados posibles" — así se piensa en probabilidades.
#
# Regla de oro: tu mejor backtest es probablemente el más falso. Si de 200
# combinaciones eliges la mejor, has seleccionado en parte suerte pasada.
# Lo que buscamos son MESETAS (zonas amplias de parámetros que funcionan),
# no picos aislados.
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
COSTO_BPS = 5
SEED = 42
CARPETA_GRAFICOS = "graficos"

RAPIDAS = [5, 10, 15, 20, 30, 40]
LENTAS = [50, 75, 100, 125, 150, 200]


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


# ------------------------ estrategia y motor base ----------------------------
def senal_cruce(close, rapida, lenta):
    s_r = close.rolling(rapida).mean()
    s_l = close.rolling(lenta).mean()
    senal = pd.Series(np.where(s_r > s_l, 1, 0), index=close.index)
    senal[s_l.isna()] = 0
    return senal


def backtest(df, senal, costo_bps=COSTO_BPS):
    ret_activo = df["Close"].pct_change().fillna(0)
    posicion = senal.shift(1).fillna(0)
    rotacion = posicion.diff().abs().fillna(posicion.abs())
    ret = posicion * ret_activo - rotacion * (costo_bps / 10_000)
    return ret


def sharpe(retornos):
    vol = retornos.std()
    if vol == 0 or np.isnan(vol):
        return 0.0
    return (retornos.mean() / vol) * np.sqrt(DIAS_ANIO)


def cagr(retornos):
    equity = (1 + retornos).cumprod()
    total = equity.iloc[-1] - 1
    anios = max(len(retornos) / DIAS_ANIO, 1e-9)
    return ((1 + total) ** (1 / anios) - 1) * 100


# ----------------------------------------------------------------------------
# 1) GRID SEARCH + MAPA DE CALOR
# ----------------------------------------------------------------------------
def grid_search(df, rapidas=RAPIDAS, lentas=LENTAS):
    matriz = np.full((len(lentas), len(rapidas)), np.nan)
    filas = []
    for i, l in enumerate(lentas):
        for j, r in enumerate(rapidas):
            if r >= l:
                continue  # una "rápida" más lenta que la "lenta" no tiene sentido
            s = senal_cruce(df["Close"], r, l)
            sh = sharpe(backtest(df, s))
            matriz[i, j] = sh
            filas.append({"rapida": r, "lenta": l, "sharpe": sh})
    tabla = pd.DataFrame(filas).sort_values("sharpe", ascending=False)
    return matriz, tabla


def mapa_calor(matriz, titulo, nombre_archivo):
    os.makedirs(CARPETA_GRAFICOS, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(matriz, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(RAPIDAS)), RAPIDAS)
    ax.set_yticks(range(len(LENTAS)), LENTAS)
    ax.set_xlabel("SMA rápida")
    ax.set_ylabel("SMA lenta")
    ax.set_title(titulo)
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            if not np.isnan(matriz[i, j]):
                ax.text(j, i, f"{matriz[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, label="Sharpe")
    plt.tight_layout()
    ruta = os.path.join(CARPETA_GRAFICOS, nombre_archivo)
    plt.savefig(ruta, dpi=110)
    print(f"[OK] Mapa de calor guardado en {ruta}")
    plt.show()


def analizar_meseta(matriz, tabla):
    """Robustez: compara el mejor Sharpe con el Sharpe medio de sus vecinos.
    Si los vecinos son malos, el 'óptimo' es un pico frágil (probable suerte)."""
    mejor = tabla.iloc[0]
    i = LENTAS.index(int(mejor["lenta"]))
    j = RAPIDAS.index(int(mejor["rapida"]))
    vecinos = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            ni, nj = i + di, j + dj
            if 0 <= ni < matriz.shape[0] and 0 <= nj < matriz.shape[1]:
                v = matriz[ni, nj]
                if not np.isnan(v):
                    vecinos.append(v)
    media_vecinos = float(np.mean(vecinos)) if vecinos else np.nan
    print("ANÁLISIS DE MESETA (robustez del óptimo):")
    print(f"  Mejor combinación: SMA {int(mejor['rapida'])}/{int(mejor['lenta'])} "
          f"con Sharpe {mejor['sharpe']:.2f}")
    print(f"  Sharpe medio de sus vecinos: {media_vecinos:.2f}")
    if mejor["sharpe"] <= 0 or np.isnan(media_vecinos):
        print("  El mejor Sharpe no es positivo: no hay edge que analizar en")
        print("  estos datos. Veredicto directo: NO-GO con esta estrategia/activo.\n")
        return
    ratio = media_vecinos / mejor["sharpe"]
    print(f"  Ratio vecinos/óptimo: {ratio:.2f}  "
          f"({'meseta sólida' if ratio > 0.6 else 'pico frágil: desconfía'})\n")


# ----------------------------------------------------------------------------
# 2) IN-SAMPLE VS OUT-OF-SAMPLE
# ----------------------------------------------------------------------------
def is_vs_oos(df, pct_train=0.70):
    corte = int(len(df) * pct_train)
    df_is, df_oos = df.iloc[:corte], df.iloc[corte:]

    _, tabla_is = grid_search(df_is)
    mejor = tabla_is.iloc[0]
    r, l = int(mejor["rapida"]), int(mejor["lenta"])

    sh_is = mejor["sharpe"]
    # Para el tramo OOS calculamos la señal con historia previa incluida
    # (las medias necesitan "calentarse"), pero medimos SOLO el tramo OOS.
    s_total = senal_cruce(df["Close"], r, l)
    ret_total = backtest(df, s_total)
    sh_oos = sharpe(ret_total.iloc[corte:])

    print("=" * 60)
    print("IN-SAMPLE vs OUT-OF-SAMPLE")
    print("=" * 60)
    print(f"  Parámetros elegidos SOLO con el 70% inicial: SMA {r}/{l}")
    print(f"  Sharpe in-sample (donde se eligió):  {sh_is:+.2f}")
    print(f"  Sharpe out-of-sample (datos nuevos): {sh_oos:+.2f}")
    if sh_is > 0:
        print(f"  Retención OOS/IS: {sh_oos / sh_is * 100:.0f} %  "
              f"(por debajo de ~50% es mala señal)")
    print("  La degradación IS->OOS es normal; que sea moderada es lo que")
    print("  distingue una estrategia real de una casualidad optimizada.\n")
    return corte


# ----------------------------------------------------------------------------
# 3) WALK-FORWARD — el estándar profesional
# ----------------------------------------------------------------------------
# Simula lo que harías en la vida real: cada cierto tiempo reoptimizas con el
# pasado reciente y operas el siguiente tramo con esos parámetros. Todo el
# resultado final es, por construcción, out-of-sample.
# ----------------------------------------------------------------------------
def walk_forward(df, n_train=504, n_test=126):
    ret_oos, ventanas = [], []
    i = n_train
    while i < len(df) - 5:
        df_train = df.iloc[i - n_train:i]
        _, tabla = grid_search(df_train)
        if tabla.empty:
            break
        mejor = tabla.iloc[0]
        r, l = int(mejor["rapida"]), int(mejor["lenta"])

        fin = min(i + n_test, len(df))
        # Calentamos las medias con datos previos y medimos solo el tramo test
        df_ventana = df.iloc[max(0, i - n_train):fin]
        s = senal_cruce(df_ventana["Close"], r, l)
        ret = backtest(df_ventana, s)
        tramo = ret.iloc[-(fin - i):]
        ret_oos.append(tramo)
        ventanas.append({"desde": df.index[i].date(), "hasta": df.index[fin - 1].date(),
                         "rapida": r, "lenta": l, "sharpe_train": round(mejor["sharpe"], 2)})
        i = fin
    ret_wf = pd.concat(ret_oos)
    return ret_wf, pd.DataFrame(ventanas)


# ----------------------------------------------------------------------------
# 4) MONTE CARLO — pensar en distribuciones, no en un solo número
# ----------------------------------------------------------------------------
# Remuestreamos (bootstrap) los retornos diarios OOS miles de veces para ver
# el ABANICO de trayectorias compatibles con nuestra estrategia. Así
# respondemos preguntas de probabilidad: "¿qué drawdown debo esperar en el
# 5% de los peores escenarios?".
# ----------------------------------------------------------------------------
def monte_carlo(retornos, n_sims=1000, seed=7):
    rng = np.random.default_rng(seed)
    arr = retornos.to_numpy()
    n = len(arr)
    finales = np.empty(n_sims)
    max_dds = np.empty(n_sims)
    for k in range(n_sims):
        sim = rng.choice(arr, size=n, replace=True)
        eq = np.cumprod(1 + sim)
        finales[k] = eq[-1] - 1
        max_dds[k] = (eq / np.maximum.accumulate(eq) - 1).min()
    return finales, max_dds


def informe_monte_carlo(finales, max_dds):
    print("=" * 60)
    print("MONTE CARLO (1.000 remuestreos de los retornos walk-forward)")
    print("=" * 60)
    print(f"  Retorno total  p5 / mediana / p95: "
          f"{np.percentile(finales, 5) * 100:+.1f}% / "
          f"{np.percentile(finales, 50) * 100:+.1f}% / "
          f"{np.percentile(finales, 95) * 100:+.1f}%")
    print(f"  Max drawdown   p5 / mediana / p95: "
          f"{np.percentile(max_dds, 5) * 100:.1f}% / "
          f"{np.percentile(max_dds, 50) * 100:.1f}% / "
          f"{np.percentile(max_dds, 95) * 100:.1f}%")
    print(f"  Probabilidad de acabar en pérdidas: {(finales < 0).mean() * 100:.1f} %")
    print("  Si el drawdown del p5 te resulta insoportable, reduce exposición")
    print("  (semana 4) ANTES de operar, no después de sufrirlo.\n")


# ----------------------------------------------------------------------------
# PROGRAMA PRINCIPAL
# ----------------------------------------------------------------------------
def main():
    print("\nSEMANA 5 — Optimización y validación | Material educativo.\n")
    df = cargar_datos()

    # 1) Grid search sobre TODO el histórico (para ilustrar el peligro)
    matriz, tabla = grid_search(df)
    print("TOP 5 combinaciones (optimizadas sobre TODO el histórico — ilusión):")
    print(tabla.head(5).to_string(index=False))
    print()
    analizar_meseta(matriz, tabla)
    mapa_calor(matriz, "Sharpe por parámetros (todo el histórico)",
               "semana5_heatmap.png")

    # 2) IS vs OOS
    is_vs_oos(df)

    # 3) Walk-forward
    ret_wf, ventanas = walk_forward(df)
    print("VENTANAS WALK-FORWARD (parámetros reoptimizados cada ~6 meses):")
    print(ventanas.to_string(index=False))
    print()
    print(f"  Sharpe walk-forward (100% out-of-sample): {sharpe(ret_wf):+.2f}")
    print(f"  CAGR walk-forward:                        {cagr(ret_wf):+.2f} %")
    print(f"  Comparar con el mejor Sharpe 'de laboratorio': "
          f"{tabla.iloc[0]['sharpe']:+.2f}")
    print("  La diferencia entre ambos es el precio de la honestidad estadística.\n")

    # Curva de capital walk-forward
    os.makedirs(CARPETA_GRAFICOS, exist_ok=True)
    eq_wf = (1 + ret_wf).cumprod()
    plt.figure(figsize=(11, 4.5))
    plt.plot(eq_wf.index, eq_wf, color="darkgreen", lw=1.2)
    plt.title("Curva de capital walk-forward (todo out-of-sample)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    ruta = os.path.join(CARPETA_GRAFICOS, "semana5_walkforward.png")
    plt.savefig(ruta, dpi=110)
    print(f"[OK] Gráfico guardado en {ruta}")
    plt.show()

    # 4) Monte Carlo sobre los retornos walk-forward
    finales, max_dds = monte_carlo(ret_wf)
    informe_monte_carlo(finales, max_dds)

    plt.figure(figsize=(11, 4))
    plt.hist(finales * 100, bins=50, color="steelblue", alpha=0.8)
    plt.axvline(0, color="red", ls="--")
    plt.title("Monte Carlo: distribución del retorno total (1.000 simulaciones)")
    plt.xlabel("Retorno total (%)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    ruta = os.path.join(CARPETA_GRAFICOS, "semana5_montecarlo.png")
    plt.savefig(ruta, dpi=110)
    print(f"[OK] Gráfico guardado en {ruta}")
    plt.show()

    print("\nCRITERIOS GO / NO-GO antes de pasar a la semana 6:")
    print("  [ ] El óptimo vive en una meseta, no en un pico aislado.")
    print("  [ ] Retención OOS/IS razonable (orientativo: > 50%).")
    print("  [ ] Sharpe walk-forward positivo y estable entre ventanas.")
    print("  [ ] El drawdown p95 del Monte Carlo es tolerable para ti.")
    print("  Si falla alguno: se rediseña la estrategia, no se fuerzan los datos.")

    print("\nTAREA DE LA SEMANA:")
    print("  1. Repite el walk-forward con ventanas train=756/test=252. ¿Cambian")
    print("     mucho los parámetros elegidos? La estabilidad también es información.")


if __name__ == "__main__":
    main()
