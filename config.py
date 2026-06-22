import os

HF_TOKEN    = os.environ.get("HF_TOKEN", "")
DATA_REPO   = "P2SAMAPA/fi-etf-macro-signal-master-data"
OUTPUT_REPO = "P2SAMAPA/p2-etf-qrf-results"

UNIVERSES = {
    "FI_COMMODITIES": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"],
    "EQUITY_SECTORS": [
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI",
        "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
    "COMBINED": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI",
        "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
}

MACRO_COLS_CORE     = ["VIX", "DXY", "T10Y2Y"]
MACRO_COLS_EXTENDED = ["IG_SPREAD", "HY_SPREAD"]

# ── Rolling windows (trading days) ────────────────────────────────────────────
WINDOWS = [63, 126, 252, 504]

# ── Feature construction ──────────────────────────────────────────────────────
# Features fed into each tree at time t:
#   [ret_1d, ret_5d, ret_21d,        ← return momentum
#    vol_21d, vol_63d,                ← realised volatility
#    skew_21d, kurt_21d,              ← higher moments
#    macro_1, ..., macro_M]           ← normalised macro signals
N_RET_LAGS  = [1, 5, 21]     # momentum lookbacks
VOL_WINDOWS = [21, 63]        # vol lookbacks

# Prediction horizon: forward log-return target
PRED_HORIZON = 21             # ~1 month

# ── Random Forest hyperparameters ─────────────────────────────────────────────
QRF_N_TREES     = 200         # number of trees
QRF_MAX_DEPTH   = 6           # max depth per tree
QRF_MIN_SAMPLES = 5           # min samples per leaf
QRF_MAX_FEATURES = 0.6        # fraction of features considered per split
QRF_BOOTSTRAP   = True        # bootstrap samples per tree

# ── Quantile targets for scoring ──────────────────────────────────────────────
# QRF predicts the full conditional CDF, not just the mean.
# We extract three quantile-based signals:
#   q_median  : Q(0.50 | X)  — conditional median (direction)
#   q_cvar    : Q(0.10 | X)  — 10th percentile (tail risk / CVaR proxy)
#   q_prob    : P(r > 0 | X) — probability of positive return

QUANTILE_MEDIAN = 0.50
QUANTILE_TAIL   = 0.10

# Score = weighted combination of quantile signals
WEIGHT_MEDIAN  = 0.50   # conditional median return
WEIGHT_PROB    = 0.30   # P(r > 0) − 0.5
WEIGHT_CVAR    = 0.20   # −CVaR (negative tail risk as positive signal)

# Minimum training samples
MIN_TRAIN_SAMPLES = 30

TOP_N = 3
