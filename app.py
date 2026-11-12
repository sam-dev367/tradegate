# Dashboard:  streamlit run app.py
import streamlit as st, pandas as pd
import tradegate as tg

st.set_page_config(page_title="TradeGate", layout="wide")
st.title("TradeGate: does the ML gate beat blind breakout trading after costs?")

src = st.sidebar.radio("Data", ["Real NSE (yfinance)", "Synthetic (demo only)"])
cost = st.sidebar.slider("One-way cost + slippage (bps)", 0, 40, 15)
thr = st.sidebar.slider("Gate threshold (min probability to trade)", 0.40, 0.75, 0.55, 0.01)
kind = st.sidebar.selectbox("Model", ["gbm", "logit"])

@st.cache_data(show_spinner=True)
def run(src, cost, thr, kind):
    if src.startswith("Real"):
        data = tg.load_real(tg.NSE10)
    else:
        data = tg.make_synthetic()
    R = tg.run_all(data, kind, thr, cost)
    return R, tg.latency_test(R["panel"], R["events"], kind)

R, lat = run(src, cost, thr, kind)
eq = pd.DataFrame({"TradeGate (gated)": (1+R["gated"]).cumprod(),
                   "Ungated breakout": (1+R["ungated"]).cumprod(),
                   "Buy & hold": (1+R["bh"]).cumprod()}).ffill()
st.subheader("Equity curves (net of costs)"); st.line_chart(eq)
c1, c2 = st.columns(2)
with c1:
    st.subheader("Performance")
    st.dataframe(pd.DataFrame({"Gated": tg.metrics(R["gated"]), "Ungated": tg.metrics(R["ungated"]),
                               "Buy&Hold": tg.metrics(R["bh"])}).round(2))
    st.write(tg.trade_stats(R["events"], thr))
with c2:
    st.subheader("Prediction quality"); st.write(tg.classification_report_(R["events"], thr))
    st.subheader("Latency / efficiency"); st.write(lat)
st.subheader("Latest signals")
last = R["events"].sort_values("date").tail(15)[["date","ticker","proba"]]
last["decision"] = (last.proba >= thr).map({True: "TAKE TRADE", False: "SKIP"})
st.dataframe(last)
