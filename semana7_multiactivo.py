# ============================================================================
# CURSO DE TRADING ALGORÍTMICO EN PYTHON — MÓDULO AVANZADO
# SEMANA 7 — MULTI-ACTIVO: CORRELACIONES Y DIVERSIFICACIÓN
# ============================================================================
# Objetivos:
#   1. Trabajar con una cesta de activos en vez de uno solo.
#   2. Medir correlaciones y entender por qué son el motor de la diversificación.
#   3. Aplicar la señal de cruce (semana 2) a cada activo y construir una cartera.
#   4. Comparar equiponderado vs. ponderado por volatilidad inversa.
#
# Idea central: la diversificación es "el único almuerzo gratis" — mismo
# retorno medio, menos riesgo, SI las correlaciones no son perfectas.
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
TICKERS = ["SPY", "QQQ", "IWM", "EFA", "GLD", "TLT"]  # acciones, oro, bonos
FECHA_INICIO = "2018-01-01"
DIAS_ANIO = 252
SEED = 42
COSTO_BPS = 5            # costo por unidad de rotación (comisión + slippage)
SMA_RAPIDA, SMA_LENTA = 20, 50
VENTANA_VOL = 60         # ventana para volatilidad inversa (sin look-ahead)
CARPETA_GRAFICOS = "graficos"


# ----------------------------------------------------------------------------
# 1) DATOS SINTÉTICOS MULTI-ACTIVO DE RESPALDO
# ----------------------------------------------------------------------------
# Modelo de UN factor: cada activo = beta * mercado + deriva propia + ruido
# propio. Así las correlaciones entre activos son realistas: los índices de
# acciones se parecen entre sí, el oro va por libre y los bonos a la contra.
# ----------------------------------------------------------------------------
PARAMS_SINTETICOS = {
    #          beta   deriva    vol_idiosincrática
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
    """Cierres ajustados de varios activos: reales si hay red, sintéticos si no."""
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
# 2) MÉTRICAS (versión compacta del motor de la semana 3)
# ----------------------------------------------------------------------------
def metricas(r, nombre):
    r = r.dropna()
    eq = (1 + r).cumprod()
    n = len(r)
    cagr = eq.iloc[-1] ** (DIAS_ANIO / n) - 1
    vol = r.std() * np.sqrt(DIAS_ANIO)
    sharpe = r.mean() / r.std() * np.sqrt(DIAS_ANIO) if r.std() > 0 else 0.0
    dd = (eq / eq.cummax() - 1).min()
    return {
        "Cartera": nombre,
        "CAGR %": round(cagr * 100, 2),
        "Vol %": round(vol * 100, 2),
        "Sharpe": round(sharpe, 2),
        "Max DD %": round(dd * 100, 2),
        "Calmar": round((cagr / abs(dd)) if dd != 0 else 0.0, 2),
    }


def backtest_pesos(pesos, r_activos, costo_bps=COSTO_BPS):
    """Motor multi-activo: pesos de HOY se ejecutan MAÑANA (shift) y la
    rotación total de la cartera paga costos."""
    pos = pesos.shift(1).fillna(0.0)
    r_bruto = (pos * r_activos).sum(axis=1)
    rotacion = pesos.diff().abs().sum(axis=1).fillna(0.0)
    return r_bruto - rotacion * costo_bps / 10_000


# ----------------------------------------------------------------------------
# 3) ANÁLISIS DE CORRELACIONES
# ----------------------------------------------------------------------------
def analizar_correlaciones(r_activos):
    corr = r_activos.corr()
    print("\nMatriz de correlaciones (retornos diarios):")
    print(corr.round(2).to_string())
    media_fuera_diag = corr.values[~np.eye(len(corr), dtype=bool)].mean()
    print(f"\nCorrelación media entre activos: {media_fuera_diag:.2f}")
    print("Lectura: cuanto más baja, más 'gratis' sale la diversificación.")
    print("Ojo: en las crisis las correlaciones de los activos de riesgo SUBEN")
    print("justo cuando más falta hace que no lo hagan. Los refugios (bonos,")
    print("oro) están en la cesta precisamente por eso.")
    return corr


# ----------------------------------------------------------------------------
# 4) SEÑALES Y CARTERAS
# ----------------------------------------------------------------------------
def senales_cruce(cierres):
    """Señal de cruce SMA por activo: 1 dentro, 0 fuera (semana 2, en cesta)."""
    rapida = cierres.rolling(SMA_RAPIDA).mean()
    lenta = cierres.rolling(SMA_LENTA).mean()
    return (rapida > lenta).astype(float)


def pesos_equiponderados(senales):
    """Reparte 1/N entre TODOS los activos de la cesta. Si un activo está
    fuera de señal, su parte queda en liquidez (no se redistribuye)."""
    return senales / senales.shape[1]


def pesos_vol_inversa(senales, r_activos):
    """Más peso a lo tranquilo, menos a lo violento. La volatilidad se estima
    con ventana PASADA (rolling) — nada de mirar el futuro."""
    vol = r_activos.rolling(VENTANA_VOL).std()
    inv = (1.0 / vol).replace([np.inf, -np.inf], np.nan)
    base = inv.div(inv.sum(axis=1), axis=0)          # reparto proporcional a 1/vol
    return (senales * base).fillna(0.0)


# ----------------------------------------------------------------------------
# PROGRAMA PRINCIPAL
# ----------------------------------------------------------------------------
def main():
    os.makedirs(CARPETA_GRAFICOS, exist_ok=True)
    cierres = cargar_cierres()
    r_activos = cierres.pct_change().fillna(0.0)

    # --- Paso 1: correlaciones -------------------------------------------------
    corr = analizar_correlaciones(r_activos)

    # --- Paso 2: el almuerzo gratis en números ---------------------------------
    vols = r_activos.std() * np.sqrt(DIAS_ANIO) * 100
    print("\nVolatilidad anual por activo (%):")
    print(vols.round(1).to_string())
    r_ew_bh = r_activos.mean(axis=1)   # buy & hold equiponderado, sin señal
    print(f"Media simple de las vols: {vols.mean():.1f} %")
    print(f"Vol de la cesta equiponderada: {r_ew_bh.std() * np.sqrt(DIAS_ANIO) * 100:.1f} %")
    print("=> La cesta es MENOS volátil que la media de sus piezas: eso ES la")
    print("   diversificación. La diferencia sale de las correlaciones < 1.")

    # --- Paso 3: carteras con señal --------------------------------------------
    senales = senales_cruce(cierres)
    carteras = {
        "Mejor activo solo (cruce)": None,   # se rellena abajo
        "Cesta cruce equiponderada": backtest_pesos(pesos_equiponderados(senales), r_activos),
        "Cesta cruce vol inversa": backtest_pesos(pesos_vol_inversa(senales, r_activos), r_activos),
        "Buy & hold cesta (referencia)": r_ew_bh,
    }

    # El "mejor activo en solitario" se elige por Sharpe A POSTERIORI: es una
    # referencia deliberadamente tramposa (nadie sabe cuál será por adelantado).
    mejor_t, mejor_r, mejor_s = None, None, -np.inf
    for t in cierres.columns:
        r_t = backtest_pesos(senales[[t]], r_activos[[t]])
        s = r_t.mean() / r_t.std() * np.sqrt(DIAS_ANIO) if r_t.std() > 0 else 0
        if s > mejor_s:
            mejor_t, mejor_r, mejor_s = t, r_t, s
    carteras[f"Mejor activo solo (cruce)"] = mejor_r
    print(f"\n[Referencia] Mejor activo individual a posteriori: {mejor_t}")

    tabla = pd.DataFrame([metricas(r, n) for n, r in carteras.items()])
    print("\nCOMPARATIVA DE CARTERAS:")
    print(tabla.to_string(index=False))
    print("\nCómo leerla: la cesta no gana por CAGR — gana por Max DD y Calmar.")
    print("Y recuerda: el 'mejor activo solo' solo se conoce mirando atrás.")

    # --- Paso 4: gráfico --------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    im = axes[0].imshow(corr, vmin=-1, vmax=1, cmap="RdYlGn")
    axes[0].set_xticks(range(len(corr)), corr.columns, rotation=45)
    axes[0].set_yticks(range(len(corr)), corr.columns)
    axes[0].set_title("Correlaciones")
    fig.colorbar(im, ax=axes[0], shrink=0.8)
    for nombre in ["Cesta cruce equiponderada", "Cesta cruce vol inversa",
                   "Buy & hold cesta (referencia)"]:
        (1 + carteras[nombre]).cumprod().plot(ax=axes[1], label=nombre)
    axes[1].set_title("Curvas de capital")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    ruta = os.path.join(CARPETA_GRAFICOS, "semana7_diversificacion.png")
    fig.savefig(ruta, dpi=110)
    print(f"\n[Gráfico guardado] {ruta}")

    print("\nIDEAS CLAVE DE LA SEMANA 7")
    print("1. Correlación < 1 => la cartera arriesga menos que la suma de partes.")
    print("2. La señal se aplica POR ACTIVO; la cartera agrega las posiciones.")
    print("3. Vol inversa: presupuesto de riesgo repartido, no capital repartido.")
    print("4. Elegir 'el mejor activo' a posteriori es autoengaño: cesta > pálpito.")


if __name__ == "__main__":
    main()
