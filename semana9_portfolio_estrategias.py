# ============================================================================
# CURSO DE TRADING ALGORÍTMICO EN PYTHON — MÓDULO AVANZADO
# SEMANA 9 — PORTFOLIO DE ESTRATEGIAS: EL VERDADERO PRODUCTO
# ============================================================================
# Objetivos:
#   1. Tratar cada estrategia como una "línea de negocio" con sus retornos.
#   2. Asignar capital entre estrategias: equiponderado vs. vol inversa.
#   3. Aplicar volatility targeting AL CONJUNTO (nivel portfolio).
#   4. Medir la contribución al riesgo de cada estrategia.
#
# Idea central: los profesionales no gestionan "una estrategia"; gestionan un
# PORTFOLIO de estrategias con un presupuesto de riesgo global.
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
VOL_OBJETIVO = 0.15          # 15% anual para el portfolio completo
VENTANA_VOL = 60
EXPOSICION_MAX = 1.0         # sin apalancamiento, regla del curso
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


def backtest_pesos(pesos, r_activos, costo_bps=COSTO_BPS):
    pos = pesos.shift(1).fillna(0.0)
    r_bruto = (pos * r_activos).sum(axis=1)
    rotacion = pesos.diff().abs().sum(axis=1).fillna(0.0)
    return r_bruto - rotacion * costo_bps / 10_000


def metricas(r, nombre):
    r = r.dropna()
    eq = (1 + r).cumprod()
    cagr = eq.iloc[-1] ** (DIAS_ANIO / len(r)) - 1
    sharpe = r.mean() / r.std() * np.sqrt(DIAS_ANIO) if r.std() > 0 else 0.0
    dd = (eq / eq.cummax() - 1).min()
    return {"Cartera": nombre, "CAGR %": round(cagr * 100, 2),
            "Vol %": round(r.std() * np.sqrt(DIAS_ANIO) * 100, 2),
            "Sharpe": round(sharpe, 2), "Max DD %": round(dd * 100, 2),
            "Calmar": round((cagr / abs(dd)) if dd != 0 else 0.0, 2)}


# ----------------------------------------------------------------------------
# 1) LAS TRES "LÍNEAS DE NEGOCIO" (resumen de las semanas 7 y 8)
# ----------------------------------------------------------------------------
def estrategia_cruce(cierres, r_activos):
    senal = (cierres.rolling(20).mean() > cierres.rolling(50).mean()).astype(float)
    return backtest_pesos(senal / senal.shape[1], r_activos)


def estrategia_momentum(cierres, r_activos, ventana=126, cada=21, top_n=3):
    mom = cierres.pct_change(ventana)
    fechas_reb = cierres.index[ventana::cada]
    w = pd.DataFrame(0.0, index=fechas_reb, columns=cierres.columns)
    for f in fechas_reb:
        ranking = mom.loc[f].dropna()
        if not ranking.empty:
            w.loc[f, ranking.nlargest(top_n).index] = 1.0 / top_n
    return backtest_pesos(w.reindex(cierres.index).ffill().fillna(0.0), r_activos)


def estrategia_reversion(cierres, r_activos, ventana=20):
    z = (cierres - cierres.rolling(ventana).mean()) / cierres.rolling(ventana).std()
    estado = pd.DataFrame(np.where(z < -1, 1.0, np.where(z > 0, 0.0, np.nan)),
                          index=cierres.index, columns=cierres.columns)
    senal = estado.ffill().fillna(0.0)
    return backtest_pesos(senal / senal.shape[1], r_activos)


# ----------------------------------------------------------------------------
# 2) ASIGNACIÓN ENTRE ESTRATEGIAS
# ----------------------------------------------------------------------------
def combinar(retornos_estrategias, modo="equiponderado", ventana=VENTANA_VOL):
    """Combina los retornos de N estrategias en un portfolio.
    - equiponderado: 1/N fijo.
    - vol_inversa: pesos ~ 1/vol reciente de CADA estrategia (rolling, shift
      de 1 día para no usar información del propio día)."""
    R = retornos_estrategias
    if modo == "equiponderado":
        pesos = pd.DataFrame(1.0 / R.shape[1], index=R.index, columns=R.columns)
    else:
        vol = R.rolling(ventana).std().shift(1)
        inv = (1.0 / vol).replace([np.inf, -np.inf], np.nan)
        pesos = inv.div(inv.sum(axis=1), axis=0).fillna(1.0 / R.shape[1])
    return (pesos * R).sum(axis=1), pesos


def vol_targeting(r_portfolio, objetivo=VOL_OBJETIVO, ventana=VENTANA_VOL):
    """Escala la exposición del PORTFOLIO para apuntar a una vol estable.
    Exposición de hoy decidida con vol de AYER hacia atrás (shift)."""
    vol_real = r_portfolio.rolling(ventana).std().shift(1) * np.sqrt(DIAS_ANIO)
    exposicion = (objetivo / vol_real).clip(upper=EXPOSICION_MAX).fillna(0.0)
    return r_portfolio * exposicion, exposicion


# ----------------------------------------------------------------------------
# 3) CONTRIBUCIÓN AL RIESGO
# ----------------------------------------------------------------------------
# El peso en capital NO es el peso en riesgo. Una estrategia con 33% del
# capital puede aportar el 70% del riesgo si es volátil y correlacionada.
# Contribución_i = w_i * (Cov · w)_i / vol_portfolio  (suman la vol total).
# ----------------------------------------------------------------------------
def contribucion_riesgo(R, w):
    cov = R.cov() * DIAS_ANIO
    w = np.asarray(w)
    vol_p = float(np.sqrt(w @ cov.values @ w))
    contrib = w * (cov.values @ w) / vol_p
    return pd.Series(contrib / vol_p, index=R.columns), vol_p  # en % de la vol


# ----------------------------------------------------------------------------
# PROGRAMA PRINCIPAL
# ----------------------------------------------------------------------------
def main():
    os.makedirs(CARPETA_GRAFICOS, exist_ok=True)
    cierres = cargar_cierres()
    r_activos = cierres.pct_change().fillna(0.0)

    R = pd.DataFrame({
        "Cruce": estrategia_cruce(cierres, r_activos),
        "Momentum": estrategia_momentum(cierres, r_activos),
        "Reversion": estrategia_reversion(cierres, r_activos),
    }).dropna()

    print("\nCorrelaciones ENTRE estrategias:")
    print(R.corr().round(2).to_string())

    r_ew, _ = combinar(R, "equiponderado")
    r_iv, _ = combinar(R, "vol_inversa")
    r_vt, exposicion = vol_targeting(r_iv)

    tabla = pd.DataFrame([
        metricas(R["Cruce"], "Solo cruce"),
        metricas(R["Momentum"], "Solo momentum"),
        metricas(R["Reversion"], "Solo reversión"),
        metricas(r_ew, "Portfolio equiponderado"),
        metricas(r_iv, "Portfolio vol inversa"),
        metricas(r_vt, "Portfolio vol inversa + target 15%"),
    ])
    print("\nCOMPARATIVA DE PORTFOLIOS:")
    print(tabla.to_string(index=False))
    print("\nCómo leerla: el objetivo del portfolio no es el CAGR máximo, es el")
    print("viaje más estable (vol y DD contenidos) para poder MANTENERLO años.")

    contrib, vol_p = contribucion_riesgo(R, [1 / 3] * 3)
    print(f"\nVol anual del portfolio equiponderado: {vol_p * 100:.1f} %")
    print("Contribución al riesgo con capital 1/3 - 1/3 - 1/3:")
    for k, v in contrib.items():
        print(f"  {k:<10} capital 33% -> riesgo {v * 100:5.1f} %")
    print("Lectura: si una línea domina el riesgo, tu 'diversificación' es de")
    print("etiqueta, no real. La vol inversa corrige justo esto.")

    exp_media = exposicion.replace(0, np.nan).mean()
    print(f"\nExposición media con vol targeting: {exp_media:.2f} (tope 1.0)")
    if exp_media > 0.95:
        print("Nota honesta: esta cartera ya es más tranquila que el objetivo,")
        print("y sin apalancamiento el target solo puede RECORTAR, no amplificar.")
        print("Su valor aparece en las crisis: cuando la vol se dispara, reduce")
        print("exposición automáticamente. Que hoy no actúe no significa que sobre.")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for col in R.columns:
        (1 + R[col]).cumprod().plot(ax=axes[0], alpha=0.6, label=col)
    (1 + r_vt).cumprod().plot(ax=axes[0], lw=2.2, color="black",
                              label="Portfolio final (IV + target)")
    axes[0].set_title("Estrategias sueltas vs. portfolio")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    contrib.mul(100).plot.bar(ax=axes[1], color="#4477aa")
    axes[1].set_title("Contribución al riesgo (capital 1/3 cada una)")
    axes[1].set_ylabel("% de la vol del portfolio")
    axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    ruta = os.path.join(CARPETA_GRAFICOS, "semana9_portfolio.png")
    fig.savefig(ruta, dpi=110)
    print(f"\n[Gráfico guardado] {ruta}")

    print("\nIDEAS CLAVE DE LA SEMANA 9")
    print("1. El producto profesional es el PORTFOLIO, no la estrategia estrella.")
    print("2. Asignar por vol inversa iguala lo que cada línea aporta al riesgo.")
    print("3. El vol targeting global fija la experiencia de riesgo del conjunto.")
    print("4. Mide contribución al riesgo: el capital repartido puede engañar.")


if __name__ == "__main__":
    main()
