# ============================================================================
# CURSO DE TRADING ALGORÍTMICO EN PYTHON — MÓDULO AVANZADO
# SEMANA 8 — DOS MOTORES CLÁSICOS: MOMENTUM Y REVERSIÓN A LA MEDIA
# ============================================================================
# Objetivos:
#   1. Implementar momentum cross-sectional: comprar los MEJORES de la cesta.
#   2. Implementar reversión a la media con z-score e histéresis vectorizada.
#   3. Medir la correlación ENTRE ESTRATEGIAS (no entre activos).
#   4. Combinarlas y comprobar el efecto en el riesgo.
#
# Idea central: diversificar entre ESTRATEGIAS poco correlacionadas es tan
# potente como diversificar entre activos — o más.
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
TICKERS = ["SPY", "QQQ", "IWM", "EFA", "GLD", "TLT"]
FECHA_INICIO = "2018-01-01"
DIAS_ANIO = 252
SEED = 42
COSTO_BPS = 5
VENTANA_MOM = 126        # ~6 meses de retorno para el ranking
CADA_REB = 21            # rebalanceo mensual (21 sesiones)
TOP_N = 3                # cuántos activos compra el momentum
VENTANA_Z = 20           # ventana del z-score de reversión
Z_ENTRADA, Z_SALIDA = -1.0, 0.0
CARPETA_GRAFICOS = "graficos"

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


def metricas(r, nombre):
    r = r.dropna()
    eq = (1 + r).cumprod()
    cagr = eq.iloc[-1] ** (DIAS_ANIO / len(r)) - 1
    sharpe = r.mean() / r.std() * np.sqrt(DIAS_ANIO) if r.std() > 0 else 0.0
    dd = (eq / eq.cummax() - 1).min()
    return {"Estrategia": nombre, "CAGR %": round(cagr * 100, 2),
            "Vol %": round(r.std() * np.sqrt(DIAS_ANIO) * 100, 2),
            "Sharpe": round(sharpe, 2), "Max DD %": round(dd * 100, 2)}


def backtest_pesos(pesos, r_activos, costo_bps=COSTO_BPS):
    pos = pesos.shift(1).fillna(0.0)
    r_bruto = (pos * r_activos).sum(axis=1)
    rotacion = pesos.diff().abs().sum(axis=1).fillna(0.0)
    return r_bruto - rotacion * costo_bps / 10_000


# ----------------------------------------------------------------------------
# 1) MOMENTUM CROSS-SECTIONAL
# ----------------------------------------------------------------------------
# La pregunta del momentum de serie temporal (semanas 2-6) era: "¿este activo
# sube?". La del momentum CROSS-SECTIONAL es distinta: "¿QUIÉNES suben MÁS que
# el resto?". Se compra el Top N del ranking y se revisa una vez al mes.
# Es uno de los efectos más documentados de la literatura financiera.
# ----------------------------------------------------------------------------
def pesos_momentum(cierres):
    mom = cierres.pct_change(VENTANA_MOM)          # retorno de ~6 meses
    fechas_reb = cierres.index[VENTANA_MOM::CADA_REB]

    w_reb = pd.DataFrame(0.0, index=fechas_reb, columns=cierres.columns)
    for f in fechas_reb:
        ranking = mom.loc[f].dropna()
        if ranking.empty:
            continue
        top = ranking.nlargest(TOP_N).index
        w_reb.loc[f, top] = 1.0 / TOP_N
    # Entre rebalanceos, los pesos objetivo se mantienen constantes:
    return w_reb.reindex(cierres.index).ffill().fillna(0.0)


# ----------------------------------------------------------------------------
# 2) REVERSIÓN A LA MEDIA CON Z-SCORE (histéresis sin bucles)
# ----------------------------------------------------------------------------
# z = cuántas desviaciones típicas está el precio de su media reciente.
# Regla: ENTRAR largo si z < -1 (caída exagerada), SALIR cuando z > 0
# (ya volvió a su media). Entre ambos umbrales se MANTIENE lo que hubiera.
# El truco vectorizado: 1 en entradas, 0 en salidas, NaN en tierra de nadie,
# y un ffill() propaga el último estado. Histéresis limpia, sin bucles.
# ----------------------------------------------------------------------------
def pesos_reversion(cierres):
    media = cierres.rolling(VENTANA_Z).mean()
    desv = cierres.rolling(VENTANA_Z).std()
    z = (cierres - media) / desv

    estado = pd.DataFrame(
        np.where(z < Z_ENTRADA, 1.0, np.where(z > Z_SALIDA, 0.0, np.nan)),
        index=cierres.index, columns=cierres.columns,
    )
    senal = estado.ffill().fillna(0.0)
    return senal / senal.shape[1]                  # 1/N por activo en señal


# ----------------------------------------------------------------------------
# PROGRAMA PRINCIPAL
# ----------------------------------------------------------------------------
def main():
    os.makedirs(CARPETA_GRAFICOS, exist_ok=True)
    cierres = cargar_cierres()
    r_activos = cierres.pct_change().fillna(0.0)

    # --- Las dos estrategias ---------------------------------------------------
    w_mom = pesos_momentum(cierres)
    w_rev = pesos_reversion(cierres)
    r_mom = backtest_pesos(w_mom, r_activos)
    r_rev = backtest_pesos(w_rev, r_activos)
    r_bh = r_activos.mean(axis=1)

    rot_mom = w_mom.diff().abs().sum(axis=1).mean() * DIAS_ANIO
    rot_rev = w_rev.diff().abs().sum(axis=1).mean() * DIAS_ANIO
    print(f"\nRotación anual — momentum: {rot_mom:.1f}x | reversión: {rot_rev:.1f}x")
    print("La reversión opera mucho más: sus costos pesan más. Vigílalo siempre.")

    # --- Correlación ENTRE estrategias ------------------------------------------
    correl = r_mom.corr(r_rev)
    print(f"\nCorrelación momentum <-> reversión: {correl:.2f}")
    if correl < 0.3:
        print("Correlación BAJA: candidatas ideales a combinarse.")
    elif correl < 0.7:
        print("Correlación MODERADA: diversifican en parte, pero comparten el")
        print("motor común del mercado. La mejora al combinarlas será parcial.")
    else:
        print("Correlación ALTA: combinarlas apenas diversifica. Buscar motores")
        print("más distintos (otros plazos, otros activos, otras familias).")
    print("La cifra exacta depende de los datos: mídela SIEMPRE, no la supongas.")

    # --- Combinación 50/50 -------------------------------------------------------
    # Combinar RETORNOS 50/50 equivale a repartir el capital entre ambas.
    r_combo = 0.5 * r_mom + 0.5 * r_rev

    tabla = pd.DataFrame([
        metricas(r_mom, "Momentum top-3 mensual"),
        metricas(r_rev, "Reversión z-score"),
        metricas(r_combo, "Combo 50/50"),
        metricas(r_bh, "Buy & hold cesta"),
    ])
    print("\nCOMPARATIVA:")
    print(tabla.to_string(index=False))
    print("\nCómo leerla: compara el combo con su pieza MÁS arriesgada. El")
    print("suavizado (menos vol y DD que esa pieza) nace de la correlación")
    print("imperfecta, no de que una estrategia sea mejor que otra.")

    # --- Gráfico -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5))
    for r, nombre in [(r_mom, "Momentum"), (r_rev, "Reversión"),
                      (r_combo, "Combo 50/50"), (r_bh, "Buy & hold cesta")]:
        (1 + r).cumprod().plot(ax=ax, label=nombre)
    ax.set_title("Semana 8 — Momentum vs. reversión vs. combinación")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    ruta = os.path.join(CARPETA_GRAFICOS, "semana8_estrategias.png")
    fig.savefig(ruta, dpi=110)
    print(f"\n[Gráfico guardado] {ruta}")

    print("\nIDEAS CLAVE DE LA SEMANA 8")
    print("1. Momentum cross-sectional: rankear y comprar lo MEJOR de la cesta.")
    print("2. Reversión: comprar excesos a la baja con entrada/salida asimétrica.")
    print("3. La métrica nueva de hoy: correlación ENTRE estrategias.")
    print("4. Un combo de motores mediocres poco correlacionados puede batir a")
    print("   un motor brillante en solitario. Ese es el negocio real.")


if __name__ == "__main__":
    main()
