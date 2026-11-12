# TradeGate

**An ML "gate" that decides which breakout trades on large NSE stocks are worth taking after costs.**
AstraQuant X 2026 | Team TradeGate

## Quick start

**1. Install** (Python 3.10 or newer)

```bash
pip install -r requirements.txt
```

If `pip` is not recognised, use `python -m pip install -r requirements.txt`.

**2. Run the engine**

```bash
python tradegate.py > results.txt
```

On the first run it downloads daily data for 10 NSE stocks from Yahoo Finance and caches it in `./data`. Later runs use the cached files and work offline. It prints the tuned parameters, performance of the gated, ungated and buy-and-hold strategies, classification scores, trade statistics, the shuffle (leakage) test, latency numbers and a cost sensitivity sweep.

**3. Run the dashboard**

```bash
streamlit run app.py
```

(If the command is not found, use `python -m streamlit run app.py`.)

**Optional: check the code on fake data** (results are meaningless, never quote them)

```bash
python tradegate.py --synthetic
```

## The problem

A trading rule that looks profitable on a chart often loses money once transaction costs, slippage and bad market conditions are counted. A simple 20-day breakout rule fires many times, but only some of those trades are worth taking.

> Given a simple, transparent breakout rule that proposes long trades on liquid NSE stocks, can a machine-learning model, using only information available at decision time, identify which proposed trades are likely to remain profitable after transaction costs, and does filtering and risk-sizing trades this way improve net performance and reduce drawdown compared with taking every signal or simply holding?

## How it works

1. **Primary rule:** a 20-day breakout proposes a long trade (enter at the next day's open, hold 5 trading days).
2. **Features:** 14 inputs known at the day's close, such as returns over several windows, volatility, RSI, distance from the 50-day average, breakout strength, volume z-score, ATR and market-regime features.
3. **Gate model:** a classifier (gradient boosting, with logistic regression as the simpler comparison) predicts the probability that the trade is profitable after costs.
4. **Trade filter:** take the trade only if the probability is at least the threshold (default 0.55), otherwise skip.
5. **Risk sizing:** smaller positions for more volatile stocks (target daily risk 0.25%, maximum weight 25%).
6. **Walk-forward backtest:** the model is trained on past years only and tested on the next year, with costs and slippage charged on every trade (default 15 bps one way).

Because costs are built into the label, the model learns to predict profit after costs, not just price direction.

## Honesty checks built in

- **Walk-forward testing:** no future data is used for training.
- **Tuning on pre-test years only:** 9 gradient-boosting settings (depth 2/3/4 times learning rate 0.03/0.05/0.1) are compared on years before the first test year.
- **Shuffle test:** with shuffled labels the model should show no skill (accuracy near the base rate). If it still "works", something leaks.
- **Cost sweep:** shows how performance changes as costs rise and where the strategy stops working.
- **Latency test:** training time, single-prediction latency (median and 95th percentile) and batch throughput.

## Files

| File | What it does |
|---|---|
| `tradegate.py` | Data loading, features, gate model, walk-forward testing, backtest, metrics, shuffle test and latency test |
| `app.py` | Streamlit dashboard: equity curves, metrics, prediction quality, latency and latest TAKE/SKIP signals |
| `requirements.txt` | Python packages (pandas, numpy, scikit-learn, yfinance, streamlit, matplotlib) |
| `results.txt` | Output of the final run (add after the final run) |

## Settings

Defined at the top of `tradegate.py`. Do not change them after seeing test results; change them only for labelled what-if experiments.

| Setting | Default | Meaning |
|---|---|---|
| `HOLD_DAYS` | 5 | Days each trade is held |
| `BREAKOUT_N` | 20 | Breakout lookback in days |
| `COST_BPS` | 15 | One-way cost and slippage in basis points |
| `TARGET_DAILY_RISK` / `MAX_WEIGHT` | 0.0025 / 0.25 | Position sizing |
| threshold | 0.55 | Minimum probability to take a trade |

## Results

> Fill this table with numbers printed by your own final run of `python tradegate.py`. Do not copy numbers from a synthetic run.

| Strategy | Net profit % | Sharpe | Max drawdown % | Profit factor |
|---|---|---|---|---|
| TradeGate (gated) | [FILL] | [FILL] | [FILL] | [FILL] |
| Ungated breakout | [FILL] | [FILL] | [FILL] | [FILL] |
| Buy and hold | [FILL] | [FILL] | [FILL] | [FILL] |

Shuffle test: [FILL]. Latency (median ms per prediction): [FILL].

## Limitations

- Daily data on 10 large NSE stocks only, over a limited history.
- Yahoo Finance data can differ slightly between downloads, so the team shares one cached `data` folder.
- Costs are a simple per-trade assumption, not a full market-impact model.
- Past performance does not guarantee future results. This project is for education and is not financial advice.

## Team

| Name | Role |
|---|---|
| Niyati | Team lead and model engineer |
| Shirley | Data and feature engineer |
| Sameeha | Strategy, backtest and risk analyst; repo owner |
| Pranitha | Research, maths, dashboard and presentation lead |
