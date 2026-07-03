# Curso de Trading Algorítmico en Python — De Cero a Avanzado (6 Semanas)

Programa práctico y sistemático para aprender a diseñar, probar y validar estrategias de trading con Python. El objetivo no es "predecir el mercado", sino construir un **proceso** que aumente las probabilidades a tu favor: señales cuantificables, costos realistas, gestión de riesgo y validación fuera de muestra.

> **Descargo de responsabilidad:** material exclusivamente educativo. No constituye asesoramiento financiero ni de inversión. Operar en mercados implica riesgo real de pérdida de capital. Los resultados de un backtest no garantizan resultados futuros.

---

## ¿Para quién es este curso?

- Personas con conocimientos básicos de Python (variables, funciones, listas).
- Traders discrecionales que quieren sistematizar sus decisiones.
- Programadores que quieren entrar al mundo cuantitativo con bases sólidas.

## Instalación

```bash
pip install -r requirements.txt
```

Cada script intenta descargar datos reales con `yfinance` (por defecto SPY, el ETF del S&P 500). Si no hay conexión, genera automáticamente **datos sintéticos de práctica**, así que todo el curso funciona incluso sin internet. Puedes cambiar el ticker en la variable `TICKER` de cada script (ej. `BTC-USD`, `AAPL`, `GC=F`).

## Estructura del repositorio

```
curso_trading_algoritmico/
├── README.md
├── requirements.txt
├── semana1_fundamentos.py
├── semana2_indicadores_senales.py
├── semana3_backtesting.py
├── semana4_gestion_riesgo.py
├── semana5_optimizacion_validacion.py
├── semana6_sistema_completo.py
└── guiones/
    ├── semana1_guion.md ... semana6_guion.md   (guiones de video de 30 min)
```

Cada script guarda sus gráficos en la carpeta `graficos/` al ejecutarse.

## Temario

| Semana | Tema | Qué aprendes | Código | Guion |
|---|---|---|---|---|
| 1 | Fundamentos | Datos OHLCV, retornos simples vs logarítmicos, volatilidad, línea base buy & hold | `semana1_fundamentos.py` | `guiones/semana1_guion.md` |
| 2 | Indicadores y señales | SMA, EMA, RSI, Bollinger, ATR construidos desde cero; convertir una idea en una señal | `semana2_indicadores_senales.py` | `guiones/semana2_guion.md` |
| 3 | Backtesting riguroso | Motor vectorizado, sesgo de anticipación (look-ahead), costos y slippage, métricas profesionales | `semana3_backtesting.py` | `guiones/semana3_guion.md` |
| 4 | Gestión de riesgo | Riesgo fijo por operación con ATR, volatility targeting, Kelly fraccionado, expectativa matemática | `semana4_gestion_riesgo.py` | `guiones/semana4_guion.md` |
| 5 | Optimización y validación | Grid search, overfitting, in-sample vs out-of-sample, walk-forward, Monte Carlo | `semana5_optimizacion_validacion.py` | `guiones/semana5_guion.md` |
| 6 | Sistema completo | Arquitectura profesional por clases, pipeline de extremo a extremo, informe ejecutivo, checklist pre-producción | `semana6_sistema_completo.py` | `guiones/semana6_guion.md` |

## Filosofía del curso

Una estrategia con expectativa positiva no nace de un indicador "mágico". Nace de la suma de cuatro capas, y cada semana construye una:

1. **Señal cuantificable** — una hipótesis de mercado convertida en regla objetiva (semanas 1–2).
2. **Medición honesta** — backtest sin sesgos y con costos reales (semana 3).
3. **Gestión de riesgo** — sobrevivir a las rachas malas es requisito para capturar las buenas (semana 4).
4. **Validación robusta** — si solo funciona con un juego exacto de parámetros, no funciona (semana 5).

La semana 6 integra todo en un sistema mantenible y define el puente hacia paper trading y, eventualmente, ejecución real.

## Cómo estudiar cada semana

1. Ve el video de 30 minutos siguiendo el guion.
2. Ejecuta el script completo: `python semanaX_*.py`.
3. Lee el código de arriba abajo; los comentarios son parte del material.
4. Haz la tarea indicada al final de cada guion antes de pasar a la siguiente semana.

## Reglas de oro (resumen del curso)

- Ejecuta las señales **al día siguiente** de generarse (`shift(1)`): sin esto, tu backtest miente.
- Un backtest sin costos de transacción es marketing, no ciencia.
- Prefiere **mesetas** de parámetros, no picos: la robustez vale más que el mejor Sharpe in-sample.
- El tamaño de posición importa tanto como la señal.
- Nada pasa a dinero real sin walk-forward positivo y un mínimo de 3 meses de paper trading.

# Curso de Trading Algorítmico en Python — Módulo Avanzado (Semanas 7–12)

Continuación directa del curso base (semanas 1–6). Si el módulo base construía
**el laboratorio** (medir, señalar, backtestear, dimensionar, validar), este
módulo construye **la fábrica**: carteras multi-activo, varias estrategias
conviviendo, ejecución real en papel, automatización robusta y gobernanza.

> **Material educativo.** Nada de este curso constituye asesoramiento
> financiero. Operar conlleva riesgo real de pérdida. Todo el trabajo práctico
> se realiza en simulación y paper trading.

## Requisitos

- Haber completado (o dominar) las semanas 1–6 del curso base.
- Python 3.10+ con las mismas dependencias del módulo base:

```bash
pip install -r requirements.txt
```

Cada script descarga datos reales con `yfinance` (cesta por defecto: SPY, QQQ,
IWM, EFA, GLD, TLT). Sin conexión, genera automáticamente datos **sintéticos
multi-activo** con un modelo de un factor (correlaciones realistas), así que
todo el curso funciona offline. Las conclusiones de mercado, como siempre,
solo valen con datos reales.

## Temario

| Semana | Script | Tema |
|-------:|--------|------|
| 7 | `semana7_multiactivo.py` | Cesta de activos, correlaciones, diversificación, pesos por volatilidad inversa |
| 8 | `semana8_momentum_reversion.py` | Momentum cross-sectional (ranking top-N), reversión con z-score e histéresis, correlación entre estrategias |
| 9 | `semana9_portfolio_estrategias.py` | Portfolio de estrategias, asignación de capital, contribución al riesgo, vol targeting global |
| 10 | `semana10_ejecucion_paper.py` | Del peso a la orden: plan de ejecución, umbral de rebalanceo, estado (JSON), journal (CSV), tracking teoría vs. ejecución |
| 11 | `semana11_automatizacion.py` | Logging, config por variables de entorno, reintentos, validación de datos, kill switch, ciclo diario con códigos de salida |
| 12 | `semana12_monitoreo_gobernanza.py` | Abanico Monte Carlo como contrato, percentil en vivo, semáforo de degradación, calendario de gobernanza |

Cada semana incluye su guion de video de 30 minutos en `guiones/`.

## Cómo usar el material

1. Ejecuta el script de la semana: `python semana7_multiactivo.py`
2. Lee la salida completa: los scripts **enseñan mientras corren** (tablas
   comentadas, lecturas honestas de los resultados, ideas clave al final).
3. Los gráficos se guardan en `graficos/`. La semana 10 genera además
   `orders_journal.csv` y `estado_cartera.json`; la 11, `logs/bot.log`.
4. Haz la tarea del guion antes de pasar a la siguiente semana.

## Filosofía del módulo (las reglas de oro siguen vigentes)

1. **Sin look-ahead, nunca**: cada peso decidido hoy se ejecuta mañana
   (`shift(1)`), también en cartera.
2. **Los costos siempre cuentan**: la rotación de la cartera completa paga.
3. **Diversificar es repartir riesgo, no capital**: correlaciones y
   contribución al riesgo por delante del reparto ingenuo.
4. **Ante la duda, el bot no opera**: datos validados o no hay órdenes.
5. **Los umbrales se escriben antes de operar**: semáforo, drawdown límite y
   calendario de reoptimización no se negocian durante un drawdown.
6. **Sin apalancamiento** en todo el curso: primero sobrevivir.

 
