# ============================================================================
# CURSO DE TRADING ALGORÍTMICO EN PYTHON — MÓDULO AVANZADO
# SEMANA 12 — MONITOREO, GOBERNANZA Y MEJORA CONTINUA
# ============================================================================
# Objetivos:
#   1. Comparar el rendimiento EN VIVO contra el abanico Monte Carlo esperado.
#   2. Vigilar la salud del sistema: rolling Sharpe y drawdown vs. límites.
#   3. Definir un semáforo de degradación (VERDE/ÁMBAR/ROJO) con reglas escritas.
#   4. Fijar la gobernanza: cuándo se reoptimiza, cuándo se apaga, quién decide.
#
# Idea central: "¿está funcionando mi sistema?" es una pregunta ESTADÍSTICA.
# Se responde con el abanico de resultados esperado, no con el ánimo del mes.
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
DIAS_ANIO = 252
SEED = 42
COSTO_BPS = 5
DIAS_VIVO = 126                 # los últimos ~6 meses hacen de "periodo en vivo"
N_MONTECARLO = 1000
DD_LIMITE = -0.20               # el límite ESCRITO en el plan (semanas 4-6)
VENTANA_SHARPE = 126
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


def cargar_cierres():
    try:
        import yfinance as yf

        df = yf.download(TICKERS, start="2018-01-01", progress=False,
                         auto_adjust=True)
        cierres = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df[["Close"]]
        cierres = cierres.dropna(how="any")
        if cierres.empty:
            raise ValueError("descarga vacía")
        print(f"[OK] Datos reales: {list(cierres.columns)} ({len(cierres)} sesiones)")
        return cierres
    except Exception as e:
        print(f"[AVISO] Sin datos reales ({e}). Usando datos SINTÉTICOS.")
        return generar_cierres_sinteticos()


def retornos_sistema(cierres):
    """El sistema de referencia del módulo: cesta de cruce 20/50 con costos."""
    senal = (cierres.rolling(20).mean() > cierres.rolling(50).mean()).astype(float)
    w = senal / senal.shape[1]
    r_activos = cierres.pct_change().fillna(0.0)
    pos = w.shift(1).fillna(0.0)
    rot = w.diff().abs().sum(axis=1).fillna(0.0)
    return (pos * r_activos).sum(axis=1) - rot * COSTO_BPS / 10_000


# ----------------------------------------------------------------------------
# 1) EL ABANICO ESPERADO: MONTE CARLO SOBRE EL HISTÓRICO "PRE-VIVO"
# ----------------------------------------------------------------------------
# Antes de salir a producción, congelamos la expectativa: remuestreamos los
# retornos del backtest y generamos 1.000 caminos de la misma longitud que el
# periodo en vivo. Ese abanico ES el contrato con nosotros mismos.
# ----------------------------------------------------------------------------
def abanico_montecarlo(r_backtest, n_dias, n_sims=N_MONTECARLO, seed=SEED):
    rng = np.random.default_rng(seed)
    base = r_backtest.dropna().values
    caminos = np.empty((n_sims, n_dias))
    for i in range(n_sims):
        muestra = rng.choice(base, size=n_dias, replace=True)
        caminos[i] = np.cumprod(1 + muestra)
    return caminos


def diagnostico_vivo(caminos, eq_vivo):
    finales = caminos[:, -1]
    percentil = float((finales < eq_vivo.iloc[-1]).mean() * 100)
    print(f"\nResultado en vivo tras {len(eq_vivo)} sesiones: "
          f"{(eq_vivo.iloc[-1] - 1) * 100:+.1f} %")
    print(f"Percentil dentro del abanico esperado: {percentil:.0f}")
    print("Lectura del percentil:")
    print("  ~50   -> el sistema hace EXACTAMENTE lo que prometió.")
    print("  <10   -> peor que 9 de cada 10 escenarios: revisar en serio.")
    print("  >90   -> mejor de lo esperado: agradecer, NO extrapolar ni subir riesgo.")
    return percentil


# ----------------------------------------------------------------------------
# 2) SEMÁFORO DE DEGRADACIÓN (reglas escritas ANTES de operar)
# ----------------------------------------------------------------------------
def semaforo(percentil, dd_vivo, sharpe_rolling_actual):
    razones = []
    estado = "VERDE"
    if percentil < 25 or sharpe_rolling_actual < 0:
        estado = "AMBAR"
        if percentil < 25:
            razones.append(f"percentil vivo bajo ({percentil:.0f} < 25)")
        if sharpe_rolling_actual < 0:
            razones.append(f"rolling Sharpe negativo ({sharpe_rolling_actual:.2f})")
    if percentil < 10 or dd_vivo <= DD_LIMITE:
        estado = "ROJO"
        if percentil < 10:
            razones.append(f"percentil vivo crítico ({percentil:.0f} < 10)")
        if dd_vivo <= DD_LIMITE:
            razones.append(f"drawdown {dd_vivo:.1%} <= límite {DD_LIMITE:.0%}")
    acciones = {
        "VERDE": "seguir el plan sin tocar NADA hasta la revisión mensual.",
        "AMBAR": "reducir exposición al 50% y abrir investigación documentada.",
        "ROJO": "APAGAR el sistema (kill switch) y volver al laboratorio.",
    }
    return estado, razones, acciones[estado]


# ----------------------------------------------------------------------------
# 3) GOBERNANZA: el reglamento que te protege de ti mismo
# ----------------------------------------------------------------------------
CALENDARIO_GOBERNANZA = """
CALENDARIO DE GOBERNANZA DEL SISTEMA
------------------------------------
DIARIO      El bot corre solo (semana 11). El humano solo lee alertas.
SEMANAL     15 min: revisar journal de órdenes y tracking teoría vs. ejecución.
MENSUAL     1 h: informe de este script -> percentil, semáforo, decisión escrita.
TRIMESTRAL  Reoptimización SOLO en las fechas planificadas del walk-forward
            (semana 5). Nunca tras una mala semana: eso es overfitting emocional.
ANUAL       Auditoría completa: costos reales vs. modelados, hipótesis de cada
            estrategia, ¿sigue existiendo la razón económica del edge?

REGLAS INNEGOCIABLES
--------------------
1. Toda decisión (seguir/reducir/apagar) se ESCRIBE con fecha y motivo.
2. Los umbrales del semáforo se fijaron ANTES de operar y no se mueven
   durante un drawdown. Cambiarlos exige pasar por el laboratorio.
3. Subir el riesgo tras una buena racha requiere el MISMO proceso que
   cualquier otro cambio. Las rachas buenas también son ruido.
"""


# ----------------------------------------------------------------------------
# PROGRAMA PRINCIPAL
# ----------------------------------------------------------------------------
def main():
    os.makedirs(CARPETA_GRAFICOS, exist_ok=True)
    cierres = cargar_cierres()
    r = retornos_sistema(cierres).dropna()

    # Partición honesta: todo menos los últimos DIAS_VIVO días = "backtest";
    # los últimos DIAS_VIVO días hacen el papel de "operativa en vivo".
    r_backtest, r_vivo = r.iloc[:-DIAS_VIVO], r.iloc[-DIAS_VIVO:]
    eq_vivo = (1 + r_vivo).cumprod()

    caminos = abanico_montecarlo(r_backtest, len(r_vivo))
    percentil = diagnostico_vivo(caminos, eq_vivo)

    dd_vivo = float((eq_vivo / eq_vivo.cummax() - 1).min())
    sharpe_roll = (r.rolling(VENTANA_SHARPE).mean()
                   / r.rolling(VENTANA_SHARPE).std() * np.sqrt(DIAS_ANIO))
    sharpe_actual = float(sharpe_roll.iloc[-1])
    print(f"\nDrawdown del periodo en vivo: {dd_vivo:.1%} (límite {DD_LIMITE:.0%})")
    print(f"Rolling Sharpe ({VENTANA_SHARPE} sesiones) actual: {sharpe_actual:.2f}")

    estado, razones, accion = semaforo(percentil, dd_vivo, sharpe_actual)
    print(f"\nSEMÁFORO DEL SISTEMA: {estado}")
    for rz in razones:
        print(f"  - {rz}")
    print(f"Acción según el reglamento: {accion}")

    print(CALENDARIO_GOBERNANZA)

    # --- Gráfico: abanico esperado vs. realidad + rolling Sharpe ------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(caminos.shape[1])
    for p_lo, p_hi, alpha in [(5, 95, 0.15), (25, 75, 0.25)]:
        axes[0].fill_between(x, np.percentile(caminos, p_lo, axis=0),
                             np.percentile(caminos, p_hi, axis=0),
                             color="tab:blue", alpha=alpha,
                             label=f"Abanico p{p_lo}-p{p_hi}")
    axes[0].plot(x, np.percentile(caminos, 50, axis=0), "b--", lw=1,
                 label="Mediana esperada")
    axes[0].plot(x, eq_vivo.values, "k", lw=2, label="Realidad (vivo)")
    axes[0].set_title(f"Vivo vs. esperado — percentil {percentil:.0f}")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    sharpe_roll.plot(ax=axes[1], color="tab:purple")
    axes[1].axhline(0, color="red", ls="--", lw=1)
    axes[1].set_title("Rolling Sharpe (salud del sistema)")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    ruta = os.path.join(CARPETA_GRAFICOS, "semana12_monitoreo.png")
    fig.savefig(ruta, dpi=110)
    print(f"[Gráfico guardado] {ruta}")

    print("\nIDEAS CLAVE DE LA SEMANA 12 — Y CIERRE DEL PROGRAMA")
    print("1. El abanico Monte Carlo congelado ANTES de operar es tu contrato.")
    print("2. El percentil en vivo convierte '¿va bien?' en un número objetivo.")
    print("3. Semáforo con umbrales escritos = decisiones sin pánico ni euforia.")
    print("4. Reoptimizar tiene calendario; el dolor de una mala semana, no.")
    print("5. Fin del programa: ya tienes laboratorio (semanas 1-6) y fábrica")
    print("   (semanas 7-12). El resto es disciplina, registro y paciencia.")


if __name__ == "__main__":
    main()
