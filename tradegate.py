"""
TradeGate - an ML "gatekeeper" that decides WHICH trades are worth taking after costs.

Pipeline (all decisions use only information available at that day's close):
  1. Primary rule  : 20-day breakout proposes a LONG trade (enter next day's open).
  2. Features      : returns, volatility, RSI, volume, market-regime features.
  3. Gate model    : classifier predicts P(trade is profitable AFTER costs).
  4. Trade filter  : take the trade only if probability >= threshold.
  5. Risk sizing   : smaller position when the stock is more volatile.
  6. Backtest      : walk-forward (train on past only), costs + slippage charged.
"""
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

HOLD_DAYS = 5          # each trade is held for 5 trading days
BREAKOUT_N = 20        # breakout lookback
COST_BPS = 15.0        # ONE-WAY cost+slippage in basis points (1 bp = 0.01%). Edit to test.
TARGET_DAILY_RISK = 0.0025
MAX_WEIGHT = 0.25

FEATURES = ["ret_1", "ret_5", "ret_10", "ret_20", "vol_20", "vol_ratio", "rsi_14",
            "dist_ma50", "breakout_strength", "volume_z", "atr_pct",
            "mkt_vol_20", "mkt_ret_20", "mkt_trend"]

# ---------------------------------------------------------------- data
def load_yfinance(tickers, start="2015-01-01", end=None):
    import yfinance as yf
    out = {}
    for t in tickers:
        df = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        df.columns = [c.lower() for c in df.columns]
        out[t] = df
    return out

def load_csv_folder(folder):
    import glob, os
    out = {}
    for f in glob.glob(os.path.join(folder, "*.csv")):
        df = pd.read_csv(f, parse_dates=["date"], index_col="date").sort_index()
        df.columns = [c.lower() for c in df.columns]
        out[os.path.basename(f)[:-4]] = df[["open", "high", "low", "close", "volume"]].dropna()
    return out

def load_real(tickers, cache="data"):
    """Use cached CSVs in ./data if present (works offline at the venue); otherwise download
    from Yahoo Finance and cache them. Never silently falls back to fake data."""
    import os, glob
    if glob.glob(os.path.join(cache, "*.csv")):
        return load_csv_folder(cache)
    data = load_yfinance(tickers)
    os.makedirs(cache, exist_ok=True)
    for k, df in data.items():
        df.to_csv(os.path.join(cache, k + ".csv"), index_label="date")
    return data

NSE10 = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
         "ITC.NS", "LT.NS", "SBIN.NS", "BHARTIARTL.NS", "HINDUNILVR.NS"]

def make_synthetic(n_stocks=10, n_days=2500, seed=7):
    """Fake market for SMOKE-TESTING THE CODE ONLY. Results on this are meaningless."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2014-01-01", periods=n_days)
    regime = np.zeros(n_days, int)
    for i in range(1, n_days):
        regime[i] = regime[i-1] if rng.random() > 0.01 else 1 - regime[i-1]
    mkt_vol = np.where(regime == 0, 0.007, 0.018)
    mkt = rng.normal(0.0003, 1, n_days) * mkt_vol
    out = {}
    for s in range(n_stocks):
        idio = rng.normal(0, 0.009, n_days)
        r = 0.8 * mkt + idio
        # weak trend persistence only in calm regime (so the gate has something to learn in the test)
        for i in range(1, n_days):
            if regime[i] == 0:
                r[i] += 0.04 * r[i-1]
        close = 100 * np.exp(np.cumsum(r))
        openp = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.002, n_days))
        high = np.maximum(openp, close) * (1 + np.abs(rng.normal(0, 0.004, n_days)))
        low = np.minimum(openp, close) * (1 - np.abs(rng.normal(0, 0.004, n_days)))
        vol = rng.integers(1_000_000, 5_000_000, n_days) * (1 + 3 * np.abs(r) / 0.01)
        out[f"SYN{s}"] = pd.DataFrame({"open": openp, "high": high, "low": low,
                                       "close": close, "volume": vol}, index=idx)
    return out

# ---------------------------------------------------------------- features
def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + up / (dn + 1e-12))

def build_panel(data):
    closes = pd.DataFrame({k: v["close"] for k, v in data.items()}).dropna()
    mret = closes.pct_change().mean(axis=1)             # equal-weight market proxy (no extra download)
    mkt = pd.DataFrame({"mkt_vol_20": mret.rolling(20).std(),
                        "mkt_ret_20": mret.rolling(20).sum(),
                        "mkt_trend": (1 + mret).cumprod() / (1 + mret).cumprod().rolling(50).mean() - 1})
    frames = []
    for k, df in data.items():
        df = df.loc[closes.index].copy()
        c, o, h, l, v = df["close"], df["open"], df["high"], df["low"], df["volume"]
        r1 = c.pct_change()
        f = pd.DataFrame(index=df.index)
        f["ticker"] = k
        f["open"], f["close"] = o, c
        f["ret_1"] = r1
        f["ret_5"], f["ret_10"], f["ret_20"] = c.pct_change(5), c.pct_change(10), c.pct_change(20)
        f["vol_20"] = r1.rolling(20).std()
        f["vol_ratio"] = r1.rolling(5).std() / (f["vol_20"] + 1e-12)
        f["rsi_14"] = rsi(c)
        f["dist_ma50"] = c / c.rolling(50).mean() - 1
        prior_max = c.shift(1).rolling(BREAKOUT_N).max()
        f["breakout_strength"] = c / prior_max - 1
        f["is_signal"] = (c > prior_max).astype(int)       # primary rule: 20-day breakout
        f["volume_z"] = (v - v.rolling(20).mean()) / (v.rolling(20).std() + 1e-9)
        tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        f["atr_pct"] = tr.rolling(14).mean() / c
        f = f.join(mkt)
        # LABEL: enter at NEXT open, exit at close HOLD_DAYS later, minus round-trip cost.
        entry = o.shift(-1)
        exit_ = c.shift(-HOLD_DAYS)
        f["net_ret"] = exit_ / entry - 1 - 2 * COST_BPS / 1e4
        f["label"] = (f["net_ret"] > 0).astype(float)
        f.loc[f["net_ret"].isna(), "label"] = np.nan
        frames.append(f)
    return pd.concat(frames).reset_index().rename(columns={"index": "date", "Date": "date"})

# ---------------------------------------------------------------- model
GBM_PARAMS = dict(max_depth=3, learning_rate=0.05, max_iter=150)

def tune(panel, min_train_years=3):
    """Light hyper-parameter search using ONLY data before the first walk-forward test year
    (train on older years, validate on the last pre-test year). The test years are never touched."""
    from sklearn.metrics import log_loss
    ev = panel[panel.is_signal == 1].dropna(subset=FEATURES + ["label"]).copy()
    ev["year"] = ev.date.dt.year
    years = sorted(ev.year.unique())
    first_test = years[min_train_years]
    val_year = first_test - 1
    val_start = pd.Timestamp(f"{val_year}-01-01")
    tr = ev[ev.date < val_start - pd.tseries.offsets.BDay(HOLD_DAYS + 1)]
    va = ev[ev.year == val_year]
    best = None
    for d in (2, 3, 4):
        for lr in (0.03, 0.05, 0.1):
            m = HistGradientBoostingClassifier(max_depth=d, learning_rate=lr, max_iter=150,
                    min_samples_leaf=50, l2_regularization=1.0, random_state=0).fit(tr[FEATURES], tr.label)
            ll = log_loss(va.label, m.predict_proba(va[FEATURES])[:, 1])
            if best is None or ll < best[0]:
                best = (ll, dict(max_depth=d, learning_rate=lr, max_iter=150))
    GBM_PARAMS.update(best[1])
    return best

def make_model(kind="gbm"):
    if kind == "logit":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=500))
    return HistGradientBoostingClassifier(min_samples_leaf=50, l2_regularization=1.0,
                                          random_state=0, **GBM_PARAMS)

def walk_forward(panel, kind="gbm", first_test_year=None, min_train_years=3):
    """Expanding-window walk-forward. For each test year, train ONLY on earlier data,
    with a HOLD_DAYS gap so no training label overlaps the test period (purging)."""
    ev = panel[(panel.is_signal == 1)].dropna(subset=FEATURES + ["label"]).copy()
    ev["year"] = ev["date"].dt.year
    years = sorted(ev.year.unique())
    first = first_test_year or years[min_train_years]
    ev["proba"] = np.nan
    for y in [y for y in years if y >= first]:
        test_start = pd.Timestamp(f"{y}-01-01")
        cutoff = test_start - pd.tseries.offsets.BDay(HOLD_DAYS + 1)
        train = ev[ev.date < cutoff]
        test = ev[ev.year == y]
        if len(train) < 200 or train.label.nunique() < 2 or len(test) == 0:
            continue
        m = make_model(kind).fit(train[FEATURES], train["label"])
        ev.loc[test.index, "proba"] = m.predict_proba(test[FEATURES])[:, 1]
    return ev.dropna(subset=["proba"])

# ---------------------------------------------------------------- backtest
def backtest(panel, events, threshold=0.55, cost_bps=COST_BPS, use_gate=True):
    """Daily portfolio simulation. Returns daily net return series + trade list."""
    sel = events[events.proba >= threshold] if use_gate else events
    dates = pd.DatetimeIndex(sorted(panel.date.unique()))
    pos = {t: pd.Series(0.0, index=dates) for t in panel.ticker.unique()}
    cost = pd.Series(0.0, index=dates)
    px = {t: g.set_index("date") for t, g in panel.groupby("ticker")}
    n_trades = 0
    for _, e in sel.iterrows():
        i = dates.get_loc(e.date)
        if i + HOLD_DAYS >= len(dates):
            continue
        w = min(MAX_WEIGHT, TARGET_DAILY_RISK / max(e.vol_20, 1e-4))
        d_in, d_out = dates[i + 1], dates[i + HOLD_DAYS]
        pos[e.ticker].loc[d_in:d_out] = np.maximum(pos[e.ticker].loc[d_in:d_out], w)
        cost.loc[d_in] += w * cost_bps / 1e4
        cost.loc[d_out] += w * cost_bps / 1e4
        n_trades += 1
    ret = pd.Series(0.0, index=dates)
    gross = pd.Series(0.0, index=dates)
    for t, p in pos.items():
        g = px[t]
        prev_close = g.close.shift(1).reindex(dates)
        o, c = g.open.reindex(dates), g.close.reindex(dates)
        was_flat = p.shift(1, fill_value=0.0) == 0
        base = np.where(was_flat, o, prev_close)          # first day return measured from the open
        r = (c / base - 1).fillna(0.0)
        ret += p * r
        gross += p
    scale = np.where(gross > 1, 1 / gross, 1.0)           # never exceed 100% invested
    net = ret * scale - cost * scale
    first, last = sel.date.min() if len(sel) else dates[0], dates[-1]
    return net.loc[first:last], n_trades

def buy_and_hold(panel, start, end):
    c = panel.pivot(index="date", columns="ticker", values="close").loc[start:end]
    return c.pct_change().mean(axis=1).fillna(0.0)

def metrics(daily):
    daily = daily.fillna(0.0)
    eq = (1 + daily).cumprod()
    yrs = max(len(daily) / 252, 1e-9)
    ann_vol = daily.std() * np.sqrt(252)
    sharpe = daily.mean() / (daily.std() + 1e-12) * np.sqrt(252)
    mdd = (eq / eq.cummax() - 1).min()
    active = daily[daily != 0]
    wins, losses = active[active > 0].sum(), -active[active < 0].sum()
    return {"Net profit %": 100 * (eq.iloc[-1] - 1), "CAGR %": 100 * (eq.iloc[-1] ** (1 / yrs) - 1),
            "Sharpe": sharpe, "Max drawdown %": 100 * mdd, "Volatility % (ann)": 100 * ann_vol,
            "Win rate % (days)": 100 * (active > 0).mean() if len(active) else 0.0,
            "Profit factor": wins / (losses + 1e-12)}

def classification_report_(ev, threshold=0.55):
    y, p = ev.label.astype(int), (ev.proba >= threshold).astype(int)
    return {"Accuracy": accuracy_score(y, p), "Precision": precision_score(y, p, zero_division=0),
            "Recall": recall_score(y, p, zero_division=0), "F1": f1_score(y, p, zero_division=0),
            "Base rate (all breakouts profitable)": y.mean()}

def trade_stats(events, threshold=0.55, use_gate=True):
    """Trade-level stats (each breakout = one trade, cost already included in net_ret)."""
    sel = events[events.proba >= threshold] if use_gate else events
    r = sel.net_ret
    if len(r) == 0:
        return {"Trades": 0}
    return {"Trades": len(r), "Win rate % (trades)": 100 * (r > 0).mean(),
            "Avg net return per trade %": 100 * r.mean(),
            "Profit factor (trades)": r[r > 0].sum() / (-r[r < 0].sum() + 1e-12)}

def shuffle_test(panel, kind="gbm", seed=0):
    """LEAKAGE / OVERFIT SANITY CHECK: shuffle the labels. A sound pipeline must now
    show NO skill (accuracy ~ base rate, Sharpe ~ 0). If it still 'works', something leaks."""
    p = panel.copy()
    rng = np.random.default_rng(seed)
    m = p.label.notna()
    p.loc[m, "label"] = rng.permutation(p.loc[m, "label"].values)
    ev = walk_forward(p, kind)
    y = ev.label.astype(int)
    return {"accuracy_shuffled": accuracy_score(y, (ev.proba >= 0.5).astype(int)),
            "base_rate": y.mean()}

def latency_test(panel, events, kind="gbm", n=500):
    """Measure real inference latency + training time."""
    ev = events.dropna(subset=FEATURES + ["label"])
    t0 = time.perf_counter(); m = make_model(kind).fit(ev[FEATURES], ev["label"]); train_s = time.perf_counter() - t0
    X = ev[FEATURES].iloc[:n]
    lat = []
    for i in range(len(X)):
        t0 = time.perf_counter(); m.predict_proba(X.iloc[[i]]); lat.append((time.perf_counter() - t0) * 1e3)
    t0 = time.perf_counter(); m.predict_proba(X); batch = time.perf_counter() - t0
    return {"train_seconds": train_s, "single_pred_ms_median": float(np.median(lat)),
            "single_pred_ms_p95": float(np.percentile(lat, 95)),
            "batch_throughput_pred_per_sec": len(X) / max(batch, 1e-9)}

def run_all(data, kind="gbm", threshold=0.55, cost_bps=COST_BPS, tune_first=True):
    global COST_BPS
    COST_BPS = cost_bps
    panel = build_panel(data)
    if kind == "gbm" and tune_first:
        tune(panel)
    ev = walk_forward(panel, kind)
    gated, n_g = backtest(panel, ev, threshold, cost_bps, True)
    ungated, n_u = backtest(panel, ev, threshold, cost_bps, False)
    bh = buy_and_hold(panel, ev.date.min(), panel.date.max())
    return dict(panel=panel, events=ev, gated=gated, ungated=ungated, bh=bh, n_g=n_g, n_u=n_u)

if __name__ == "__main__":
    import sys
    if "--synthetic" in sys.argv:
        data = make_synthetic(); src = "SYNTHETIC (code smoke-test only - results are meaningless)"
    else:
        data = load_real(NSE10); src = "REAL NSE daily data (cached in ./data)"
    print("DATA:", src)
    print("(light hyper-parameter tuning runs on pre-test years only)")
    R = run_all(data)
    print("Chosen GBM params:", GBM_PARAMS)
    print("\nGated    :", {k: round(float(v), 2) for k, v in metrics(R["gated"]).items()}, "trades", R["n_g"])
    print("Ungated  :", {k: round(float(v), 2) for k, v in metrics(R["ungated"]).items()}, "trades", R["n_u"])
    print("BuyHold  :", {k: round(float(v), 2) for k, v in metrics(R["bh"]).items()})
    print("Classif  :", {k: round(float(v), 3) for k, v in classification_report_(R["events"]).items()})
    print("TradeStat:", {k: round(float(v), 3) for k, v in trade_stats(R["events"]).items()})
    print("ShuffleT :", {k: round(float(v), 3) for k, v in shuffle_test(R["panel"]).items()})
    print("Latency  :", {k: round(float(v), 3) for k, v in latency_test(R["panel"], R["events"]).items()})
    print("\nCost sensitivity (gated net profit %, Sharpe):")
    for c in [0, 5, 10, 15, 20, 30]:
        Rc = run_all(data, "gbm", 0.55, c, tune_first=False)
        m = metrics(Rc["gated"]); print(f"  {c:>2} bps -> {m['Net profit %']:.1f}%  Sharpe {m['Sharpe']:.2f}")
