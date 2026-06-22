import streamlit as st
import pandas as pd
import json
from huggingface_hub import HfFileSystem
import config
from us_calendar import next_trading_day

st.set_page_config(page_title="Quantile Regression Forest Engine", layout="wide")

st.markdown("""
<style>
.main-header { font-size:2.4rem; font-weight:700; color:#1e1b4b; margin-bottom:0.3rem; }
.sub-header  { font-size:1.1rem; color:#555; margin-bottom:1.5rem; }
.uni-title   { font-size:1.4rem; font-weight:600; margin-top:1rem; margin-bottom:0.8rem;
               padding-left:0.5rem; border-left:5px solid #4f46e5; }
.etf-card    { background:linear-gradient(135deg,#1e1b4b 0%,#4f46e5 100%); color:white;
               border-radius:14px; padding:1rem; margin:0.4rem; text-align:center;
               box-shadow:0 4px 6px rgba(0,0,0,0.2); }
.win-card    { background:linear-gradient(135deg,#1e1b4b 0%,#3730a3 100%); color:white;
               border-radius:14px; padding:1rem; margin:0.4rem; text-align:center;
               box-shadow:0 4px 6px rgba(0,0,0,0.2); }
.etf-ticker  { font-size:1.3rem; font-weight:bold; }
.etf-score   { font-size:0.88rem; margin-top:0.25rem; opacity:0.9; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🌲 Quantile Regression Forest Engine</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Meinshausen (2006) QRF · '
    'Full conditional return distribution · '
    'Non-parametric · No distributional assumptions · '
    'Median + CVaR + P(r>0) composite score</div>',
    unsafe_allow_html=True)

st.sidebar.markdown("## 🌲 QRF Engine")
st.sidebar.markdown(f"**Next Trading Day:** `{next_trading_day()}`")
st.sidebar.markdown(f"**Windows:** {config.WINDOWS}")
st.sidebar.markdown(
    f"**Forest:** {config.QRF_N_TREES} trees | "
    f"depth={config.QRF_MAX_DEPTH} | "
    f"min_leaf={config.QRF_MIN_SAMPLES}")
st.sidebar.markdown(
    f"**Weights:** Median {config.WEIGHT_MEDIAN:.0%} | "
    f"P(r>0) {config.WEIGHT_PROB:.0%} | "
    f"CVaR {config.WEIGHT_CVAR:.0%}")

HF_TOKEN    = config.HF_TOKEN
OUTPUT_REPO = config.OUTPUT_REPO


@st.cache_data(ttl=3600)
def list_repo_files():
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        return [f["name"] for f in fs.ls(f"datasets/{OUTPUT_REPO}",
                                          detail=True, recursive=True)
                if f["type"] == "file"]
    except Exception as e:
        return [f"Error: {e}"]


def find_latest(files, prefix):
    matches = sorted([f for f in files if f.endswith(".json") and prefix in f],
                     reverse=True)
    return matches[0] if matches else None


@st.cache_data(ttl=3600)
def load_json(path):
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        with fs.open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}


files     = list_repo_files()
tab1_path = find_latest(files, "qrf_engine_2")
tab2_path = find_latest(files, "qrf_engine_windows_")

if not tab1_path:
    st.error("No results found. Run trainer.py first.")
    st.stop()

data1 = load_json(tab1_path)
if "error" in data1:
    st.error(f"Error loading data: {data1['error']}")
    st.stop()

data2      = load_json(tab2_path) if tab2_path else None
universes1 = data1["universes"]
universes2 = data2["universes"] if data2 and "error" not in data2 else None

st.sidebar.markdown(f"**Run date:** `{data1.get('run_date','?')}`")

tab1, tab2 = st.tabs(["🏆 Best Window per ETF", "🔍 Explore by Window"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("🏆 Top ETFs — Quantile Regression Forest Signal")

    with st.expander("📖 QRF Methodology (Meinshausen 2006)", expanded=True):
        st.markdown(f"""
Standard random forests predict E[Y|X] by averaging leaf values. **Quantile
Regression Forests** generalise this — each leaf stores the full empirical
distribution of training Y values, giving access to the entire conditional CDF:

```
F̂(y|X=x) = (1/B) Σ_b (1/|Lᵦ(x)|) Σ_{{i∈Lᵦ(x)}} 1[Yᵢ ≤ y]
```

This means one QRF gives **all quantiles simultaneously** — no need to train
separate models for each quantile level (unlike quantile regression).

**Features:**
```
x = [ret_1d, ret_5d, ret_21d,   ← momentum
     vol_21d, vol_63d,            ← realised volatility
     skew_21d, kurt_21d,          ← higher moments
     VIX, DXY, T10Y2Y, IG, HY]  ← macro signals
```

**Three quantile-based score components:**

| Component | Quantile | Weight | Signal |
|-----------|---------|--------|--------|
| Median | Q(0.50\|x) | {config.WEIGHT_MEDIAN:.0%} | Conditional median return — direction |
| P(r>0) | P(Y>0\|x) | {config.WEIGHT_PROB:.0%} | Probability of positive return |
| −CVaR | −Q(0.10\|x) | {config.WEIGHT_CVAR:.0%} | Negative tail risk (10th pct) |

**Advantages over parametric models:**
- No distributional assumption — captures skew, fat tails, bimodality
- Consistent convergence to true conditional quantile (Meinshausen 2006)
- No gradient, no learning rate, no training instability
- One forest → full conditional distribution
        """)

    for universe_name, uni_data in universes1.items():
        top_etfs = uni_data.get("top_etfs", [])
        if not top_etfs:
            continue
        st.markdown(
            f'<div class="uni-title">{universe_name.replace("_"," ").title()}</div>',
            unsafe_allow_html=True)
        cols = st.columns(3)
        for idx, etf in enumerate(top_etfs):
            with cols[idx]:
                st.markdown(f"""
<div class="etf-card">
  <div class="etf-ticker">{etf['ticker']}</div>
  <div class="etf-score">QRF score = {etf['qrf_score']:.4f}</div>
  <div class="etf-score">best window = {etf.get('best_window','N/A')}d</div>
</div>
""", unsafe_allow_html=True)

        with st.expander(f"📋 Full ranking — {universe_name}"):
            full = uni_data.get("full_scores", {})
            if full:
                rows = []
                for t, info in full.items():
                    score = info.get("score", info) if isinstance(info, dict) else info
                    win   = info.get("best_window", "N/A") if isinstance(info, dict) else "N/A"
                    rows.append({"ETF": t, "QRF Score": score, "Best Window (d)": win})
                df = pd.DataFrame(rows).sort_values("QRF Score", ascending=False)
                st.dataframe(df, use_container_width=True, hide_index=True)
        st.divider()

    st.caption(
        f"Run date: {data1.get('run_date','?')} · "
        "Meinshausen (2006) QRF · Breiman (2001) Random Forests · "
        "Scores are cross-sectional z-scores.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("🔍 Explore QRF Rankings by Window")

    if not universes2:
        st.warning("Window-level detail not found. Re-run trainer.")
        st.stop()

    all_wins = set()
    for ud in universes2.values():
        all_wins.update(ud.get("windows", {}).keys())
    win_options = sorted([int(w) for w in all_wins])

    if not win_options:
        st.error("No window data available.")
        st.stop()

    default_idx  = win_options.index(252) if 252 in win_options else 0
    selected_win = st.selectbox(
        "Select lookback window",
        options=win_options,
        index=default_idx,
        format_func=lambda w: f"{w}d  (~{round(w/21)} months)",
    )
    win_key = str(selected_win)

    with st.expander("ℹ️ Window guidance", expanded=False):
        st.markdown("""
- **63d** — short-term distribution; captures current regime; smaller training set
- **126d** — 6-month window; good balance of recency and leaf population size
- **252d** — 1-year window; most stable quantile estimates; recommended primary
- **504d** — 2-year window; structural return distribution; slow-moving signal
        """)

    st.markdown(f"### QRF Rankings at **{selected_win}d** window")

    for universe_name in ["FI_COMMODITIES", "EQUITY_SECTORS", "COMBINED"]:
        label = {
            "FI_COMMODITIES": "🏦 FI & Commodities",
            "EQUITY_SECTORS": "📈 Equity Sectors",
            "COMBINED":       "🌐 Combined",
        }.get(universe_name, universe_name)

        st.markdown(f'<div class="uni-title">{label}</div>', unsafe_allow_html=True)

        uni_data = universes2.get(universe_name, {})
        win_data = uni_data.get("windows", {}).get(win_key)

        if not win_data:
            st.info(f"No data for {universe_name} at {selected_win}d.")
            st.divider()
            continue

        cols = st.columns(3)
        for idx, etf in enumerate(win_data.get("top_etfs", [])):
            with cols[idx]:
                st.markdown(f"""
<div class="win-card">
  <div class="etf-ticker">{etf['ticker']}</div>
  <div class="etf-score">QRF score = {etf['qrf_score']:.4f}</div>
  <div class="etf-score">window = {selected_win}d</div>
</div>
""", unsafe_allow_html=True)

        with st.expander(f"📋 Full ranking — {label} @ {selected_win}d"):
            rows = win_data.get("full_ranking", [])
            if rows:
                df = pd.DataFrame(rows, columns=["ETF", "QRF Score"])
                df.insert(0, "Rank", range(1, len(df) + 1))
                st.dataframe(df, use_container_width=True, hide_index=True)

        st.divider()

    st.caption(f"Window: {selected_win}d · Run date: {data2.get('run_date','?')}")
