# ============================================================================
# CURSO DE TRADING ALGORÍTMICO EN PYTHON
# SEMANA 6 — SISTEMA COMPLETO: DE SCRIPT A ARQUITECTURA PROFESIONAL
# ============================================================================
# Objetivos de esta semana:
#   1. Refactorizar todo lo aprendido en una arquitectura por clases:
#      Datos -> Estrategia -> Riesgo -> Backtest -> Validación -> Informe.
#   2. Ejecutar el pipeline completo de extremo a extremo con un solo comando.
#   3. Definir el checklist de pre-producción y el camino a paper trading.
#
# Por qué clases: separar responsabilidades permite cambiar UNA pieza
# (otra estrategia, otro gestor de riesgo) sin romper el resto. Es la
# diferencia entre un experimento y un sistema mantenible.
#
# Material educativo. No constituye asesoramiento financiero.
# ============================================================================

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DIAS_ANIO = 252


# ============================================================================
# CONFIGURACIÓN CENTRAL — un único lugar para todos los parámetros
# ============================================================================
@dataclass
class Config:
    ticker: str = "SPY"
    fecha_inicio: str = "2018-01-01"
    capital_inicial: float = 10_000.0
    costo_bps: float = 5.0            # comisión + slippage por operación
    # Estrategia
    sma_rapida: int = 20
    sma_lenta: int = 50
    sma_regimen: int = 200            # filtro de régimen: solo largos sobre la SMA200
    # Riesgo
    vol_objetivo: float = 0.15        # 15% anual
    ventana_vol: int = 20
    peso_max: float = 1.0             # sin apalancamiento
    # Walk-forward
    n_train: int = 504
    n_test: int = 126
    grid_rapidas: tuple = (10, 20, 30)
    grid_lentas: tuple = (50, 100, 150)
    carpeta_salida: str = "graficos"
    seed: int = 42


# ============================================================================
# CAPA 1 — DATOS
# ============================================================================
class CargadorDatos:
    """Obtiene datos reales (yfinance) o sintéticos de respaldo."""

    def __init__(self, config: Config):
        self.config = config

    def obtener(self) -> pd.DataFrame:
        c = self.config
        try:
            import yfinance as yf
            df = yf.download(c.ticker, start=c.fecha_inicio,
                             progress=False, auto_adjust=True)
            if df is None or df.empty:
                raise ValueError("descarga vacía")
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            print(f"[OK] Datos reales: {c.ticker} ({len(df)} sesiones)")
            return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        except Exception as e:
            print(f"[AVISO] Sin datos reales ({e}). Usando datos sintéticos.\n")
            return self._sinteticos()

    def _sinteticos(self, n_dias=1800, precio_inicial=100.0) -> pd.DataFrame:
        rng = np.random.default_rng(self.config.seed)
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


# ============================================================================
# CAPA 2 — ESTRATEGIA (interfaz + implementación)
# ============================================================================
class Estrategia(ABC):
    """Contrato: toda estrategia recibe un DataFrame OHLCV y devuelve una
    señal 0/1 alineada al índice. Añadir una estrategia nueva = una clase
    nueva, sin tocar el resto del sistema."""

    @abstractmethod
    def generar_senales(self, df: pd.DataFrame) -> pd.Series:
        ...

    @abstractmethod
    def nombre(self) -> str:
        ...


class CruceMediasConRegimen(Estrategia):
    """Cruce SMA rápida/lenta + filtro de régimen (precio > SMA200).
    El filtro reduce operaciones en mercados bajistas: menos señales,
    de mejor calidad media. Es un ejemplo de cómo COMBINAR condiciones."""

    def __init__(self, rapida: int, lenta: int, regimen: int):
        self.rapida, self.lenta, self.regimen = rapida, lenta, regimen

    def generar_senales(self, df: pd.DataFrame) -> pd.Series:
        close = df["Close"]
        s_r = close.rolling(self.rapida).mean()
        s_l = close.rolling(self.lenta).mean()
        s_reg = close.rolling(self.regimen).mean()
        cond = (s_r > s_l) & (close > s_reg)
        senal = pd.Series(np.where(cond, 1, 0), index=close.index)
        senal[s_reg.isna() | s_l.isna()] = 0
        return senal

    def nombre(self) -> str:
        return f"Cruce SMA {self.rapida}/{self.lenta} + régimen SMA{self.regimen}"


# ============================================================================
# CAPA 3 — GESTIÓN DE RIESGO
# ============================================================================
class GestorRiesgo:
    """Convierte una señal 0/1 en PESOS de cartera mediante volatility
    targeting (semana 4). La señal dice CUÁNDO; el gestor dice CUÁNTO."""

    def __init__(self, config: Config):
        self.config = config

    def aplicar(self, df: pd.DataFrame, senal: pd.Series) -> pd.Series:
        c = self.config
        ret = df["Close"].pct_change()
        vol = ret.rolling(c.ventana_vol).std() * np.sqrt(DIAS_ANIO)
        peso = (c.vol_objetivo / vol).clip(upper=c.peso_max).fillna(0)
        return senal * peso


# ============================================================================
# CAPA 4 — BACKTESTER (motor + métricas)
# ============================================================================
class Backtester:
    def __init__(self, config: Config):
        self.config = config

    def ejecutar(self, df: pd.DataFrame, pesos: pd.Series) -> dict:
        c = self.config
        ret_activo = df["Close"].pct_change().fillna(0)
        posicion = pesos.shift(1).fillna(0)                       # sin look-ahead
        rotacion = posicion.diff().abs().fillna(posicion.abs())
        ret = posicion * ret_activo - rotacion * (c.costo_bps / 10_000)
        equity = c.capital_inicial * (1 + ret).cumprod()
        return {"retornos": ret, "equity": equity, "posicion": posicion}

    @staticmethod
    def metricas(retornos: pd.Series, equity: pd.Series) -> dict:
        total = equity.iloc[-1] / equity.iloc[0] - 1
        anios = max(len(retornos) / DIAS_ANIO, 1e-9)
        vol = retornos.std() * np.sqrt(DIAS_ANIO)
        dd = (equity / equity.cummax() - 1)
        vol_neg = retornos[retornos < 0].std() * np.sqrt(DIAS_ANIO)
        cagr = ((1 + total) ** (1 / anios) - 1) * 100
        max_dd = dd.min() * 100
        return {
            "Retorno total %": round(total * 100, 2),
            "CAGR %": round(cagr, 2),
            "Vol anual %": round(vol * 100, 2),
            "Sharpe": round((retornos.mean() * DIAS_ANIO) / vol, 2) if vol > 0 else 0.0,
            "Sortino": round((retornos.mean() * DIAS_ANIO) / vol_neg, 2) if vol_neg > 0 else 0.0,
            "Max DD %": round(max_dd, 2),
            "Calmar": round(cagr / abs(max_dd), 2) if max_dd != 0 else 0.0,
        }


# ============================================================================
# CAPA 5 — VALIDADOR WALK-FORWARD
# ============================================================================
class ValidadorWalkForward:
    """Reoptimiza los parámetros del cruce en cada ventana de entrenamiento
    y mide SOLO en las ventanas de test: resultado 100% out-of-sample."""

    def __init__(self, config: Config, backtester: Backtester,
                 gestor: GestorRiesgo):
        self.c = config
        self.bt = backtester
        self.gestor = gestor

    def _sharpe(self, ret: pd.Series) -> float:
        vol = ret.std()
        return (ret.mean() / vol) * np.sqrt(DIAS_ANIO) if vol > 0 else 0.0

    def _mejores_parametros(self, df_train: pd.DataFrame):
        mejor, mejor_sh = None, -np.inf
        for r in self.c.grid_rapidas:
            for l in self.c.grid_lentas:
                if r >= l:
                    continue
                est = CruceMediasConRegimen(r, l, self.c.sma_regimen)
                senal = est.generar_senales(df_train)
                pesos = self.gestor.aplicar(df_train, senal)
                res = self.bt.ejecutar(df_train, pesos)
                sh = self._sharpe(res["retornos"])
                if sh > mejor_sh:
                    mejor, mejor_sh = (r, l), sh
        return mejor, mejor_sh

    def ejecutar(self, df: pd.DataFrame):
        c = self.c
        tramos, registros = [], []
        i = c.n_train
        while i < len(df) - 5:
            df_train = df.iloc[i - c.n_train:i]
            params, sh_train = self._mejores_parametros(df_train)
            if params is None:
                break
            r, l = params
            fin = min(i + c.n_test, len(df))
            # Ventana con historia previa para "calentar" las medias
            df_ventana = df.iloc[max(0, i - c.n_train):fin]
            est = CruceMediasConRegimen(r, l, c.sma_regimen)
            senal = est.generar_senales(df_ventana)
            pesos = self.gestor.aplicar(df_ventana, senal)
            res = self.bt.ejecutar(df_ventana, pesos)
            tramos.append(res["retornos"].iloc[-(fin - i):])
            registros.append({"desde": df.index[i].date(), "rapida": r,
                              "lenta": l, "sharpe_train": round(sh_train, 2)})
            i = fin
        ret_oos = pd.concat(tramos)
        return ret_oos, pd.DataFrame(registros)


# ============================================================================
# CAPA 6 — INFORME EJECUTIVO
# ============================================================================
class Informe:
    def __init__(self, config: Config):
        self.c = config
        os.makedirs(config.carpeta_salida, exist_ok=True)

    def generar(self, nombre_estrategia: str, res_bt: dict, m_bt: dict,
                m_bh: dict, ret_wf: pd.Series, m_wf: dict,
                ventanas: pd.DataFrame):
        print("\n" + "=" * 68)
        print("INFORME EJECUTIVO DEL SISTEMA")
        print("=" * 68)
        print(f"Activo:      {self.c.ticker}")
        print(f"Estrategia:  {nombre_estrategia}")
        print(f"Riesgo:      Vol targeting {self.c.vol_objetivo * 100:.0f}% "
              f"(peso máx {self.c.peso_max:.0%})")
        print(f"Costos:      {self.c.costo_bps} bps por operación\n")

        tabla = pd.DataFrame({
            "Sistema (histórico)": m_bt,
            "Buy & hold": m_bh,
            "Sistema (walk-forward)": m_wf,
        })
        print(tabla.to_string())
        print("\nLa columna que manda es WALK-FORWARD: es la única 100% fuera")
        print("de muestra y la mejor estimación honesta de expectativas futuras.")

        print("\nParámetros elegidos en cada ventana walk-forward:")
        print(ventanas.to_string(index=False))

        # Guardar métricas y gráficos
        ruta_csv = os.path.join(self.c.carpeta_salida, "semana6_metricas.csv")
        tabla.to_csv(ruta_csv)

        equity = res_bt["equity"]
        eq_wf = self.c.capital_inicial * (1 + ret_wf).cumprod()
        dd = equity / equity.cummax() - 1

        fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=False)
        axes[0].plot(equity.index, equity, color="darkgreen", lw=1.2,
                     label="Sistema (histórico completo)")
        axes[0].plot(eq_wf.index, eq_wf, color="darkorange", lw=1.2,
                     label="Sistema (walk-forward, OOS)")
        axes[0].set_yscale("log")
        axes[0].set_title("Curvas de capital del sistema")
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        axes[1].fill_between(dd.index, dd * 100, 0, color="firebrick", alpha=0.4)
        axes[1].set_title("Drawdown del sistema (%)")
        axes[1].grid(alpha=0.3)
        plt.tight_layout()
        ruta_png = os.path.join(self.c.carpeta_salida, "semana6_sistema.png")
        plt.savefig(ruta_png, dpi=110)
        plt.show()

        print(f"\n[OK] Métricas guardadas en {ruta_csv}")
        print(f"[OK] Gráfico guardado en  {ruta_png}")


# ============================================================================
# CHECKLIST DE PRE-PRODUCCIÓN
# ============================================================================
CHECKLIST = """
CHECKLIST ANTES DE ARRIESGAR UN SOLO EURO/DÓLAR REAL
-----------------------------------------------------
[ ] Walk-forward positivo y estable (no depende de una sola ventana buena).
[ ] Drawdown p95 del Monte Carlo (semana 5) dentro de tu tolerancia escrita.
[ ] Costos y slippage del backtest >= los reales de tu bróker.
[ ] Mínimo 3 meses de PAPER TRADING con ejecución idéntica al plan.
[ ] Diario de operaciones: cada divergencia plan-vs-ejecución, documentada.
[ ] Límite de pérdida diaria y mensual definidos POR ESCRITO de antemano.
[ ] Regla de apagado: qué drawdown detiene el sistema para revisión.
[ ] Capital asignado que puedes permitirte perder al 100%.

PRÓXIMOS PASOS TÉCNICOS (fuera del alcance de este curso)
-----------------------------------------------------
- Conexión a bróker/exchange vía API (p. ej. ccxt para cripto, IBKR para
  acciones/futuros) empezando SIEMPRE en cuenta de papel.
- Programar la ejecución diaria (tarea programada / cron) con logs.
- Alertas automáticas (correo/mensajería) de órdenes, errores y drawdown.
- Revisión mensual: métricas reales vs. esperadas; nunca reoptimizar tras
  una semana mala — solo en las fechas planificadas del walk-forward.
"""


# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================
def main():
    print("\nSEMANA 6 — Sistema completo | Material educativo, no es asesoramiento financiero.")
    config = Config()

    # 1) Datos
    df = CargadorDatos(config).obtener()

    # 2) Estrategia -> señal
    estrategia = CruceMediasConRegimen(config.sma_rapida, config.sma_lenta,
                                       config.sma_regimen)
    senal = estrategia.generar_senales(df)

    # 3) Riesgo -> pesos
    gestor = GestorRiesgo(config)
    pesos = gestor.aplicar(df, senal)

    # 4) Backtest histórico completo
    bt = Backtester(config)
    res = bt.ejecutar(df, pesos)
    m_sistema = bt.metricas(res["retornos"], res["equity"])

    # Referencia buy & hold
    pesos_bh = pd.Series(1.0, index=df.index)
    res_bh = bt.ejecutar(df, pesos_bh)
    m_bh = bt.metricas(res_bh["retornos"], res_bh["equity"])

    # 5) Validación walk-forward (la cifra que importa)
    validador = ValidadorWalkForward(config, bt, gestor)
    ret_wf, ventanas = validador.ejecutar(df)
    eq_wf = config.capital_inicial * (1 + ret_wf).cumprod()
    m_wf = bt.metricas(ret_wf, eq_wf)

    # 6) Informe ejecutivo
    Informe(config).generar(estrategia.nombre(), res, m_sistema, m_bh,
                            ret_wf, m_wf, ventanas)

    print(CHECKLIST)
    print("FIN DEL CURSO — el edge no es un secreto: es un proceso disciplinado")
    print("de señal validada + costos honestos + riesgo controlado, repetido")
    print("con paciencia. Tu tarea final: escribe tu plan de trading en 1 página.")


if __name__ == "__main__":
    main()
