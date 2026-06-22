# 🌲 P2-ETF-QUANTILE-REGRESSION-FOREST

**Quantile Regression Forest Engine — Meinshausen (2006)**

Part of the **P2Quant Engine Suite** · [P2SAMAPA](https://github.com/P2SAMAPA)

---

## What This Engine Does

This engine applies **Quantile Regression Forests (QRF)** to estimate the
full conditional return distribution for each ETF — not just the expected
return, but the entire distribution F(y|X=x_today).

From this distribution it extracts three complementary signals: the conditional
median (direction), the probability of positive return, and the 10th percentile
(tail risk). Combined, these give a richer, distribution-aware ranking than any
mean-only predictor.

---

## Theory

### Random Forests (Breiman 2001)

An ensemble of B decision trees, each trained on a bootstrap sample with
random feature subsets per split. Ensemble variance reduction via averaging.

### Quantile Regression Forests (Meinshausen 2006)

Standard RF: predict E[Y|X] by averaging leaf node values.

QRF: keep the **full empirical distribution** of training Y-values in each leaf.
The conditional CDF is estimated as:

```
F̂(y|X=x) = (1/B) Σ_b  (1/|Lᵦ(x)|)  Σ_{i∈Lᵦ(x)}  1[Yᵢ ≤ y]
```

The q-th conditional quantile Q(q|x) is the q-th quantile of this weighted
empirical distribution.

**Key properties:**
- **Non-parametric** — no Gaussian or parametric assumption on Y|X
- **Consistent** — converges to true conditional quantile as N → ∞
- **Full distribution** — one forest gives all quantiles simultaneously
- **No gradient** — no learning rate, no numerical instability
- **Fast** — O(N log N) training, O(depth) inference

---

## Features

```
x_t = [ret_1d,    ret_5d,   ret_21d,     ← momentum (1d, 5d, 21d)
        vol_21d,   vol_63d,               ← realised volatility
        skew_21d,  kurt_21d,              ← higher moments
        VIX,       DXY,      T10Y2Y,      ← core macro (normalised)
        IG_SPREAD, HY_SPREAD]             ← credit macro
```

Target: `y_t = mean(log_return_{t+1:t+21})` — 21-day forward log return

---

## Score Construction

```
score = 0.50 · Q(0.50|x)  +  0.30 · (P(r>0|x) − 0.5)  +  0.20 · (−Q(0.10|x))
```

| Component | Formula | Weight | Meaning |
|-----------|---------|--------|---------|
| Median | Q(0.50\|x) | 50% | Conditional median return — primary direction signal |
| Prob positive | P(Y>0\|x) − 0.5 | 30% | Probability of positive return (centred at 0) |
| Neg tail | −Q(0.10\|x) | 20% | Negative 10th percentile = positive tail risk signal |

Final score: **cross-sectional z-score** per universe per window.

---

## Distinction from Other Suite Engines

| Engine | Method | Output |
|--------|--------|--------|
| CAUSAL-FOREST | Heterogeneous treatment effects | CATE estimate |
| DISTRIBUTIONAL-RL | Temporal difference quantile learning | Value distribution |
| FACTOR-ZOO | OLS factor regression | Mean return |
| **QRF (this engine)** | **Non-parametric conditional quantiles** | **Full CDF** |
| CFM | Generative flow model | Sampled distribution |

QRF is the simplest, most robust, and fastest full-distribution engine in the suite.

---

## Forest Hyperparameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `QRF_N_TREES` | 200 | Number of trees |
| `QRF_MAX_DEPTH` | 6 | Max tree depth |
| `QRF_MIN_SAMPLES` | 5 | Min samples per leaf |
| `QRF_MAX_FEATURES` | 0.6 | Feature fraction per split |
| `QRF_BOOTSTRAP` | True | Bootstrap sampling |
| `PRED_HORIZON` | 21d | Forward return target |

---

## Universes & Windows

| Universe | Tickers |
|---|---|
| FI_COMMODITIES | TLT, VCIT, LQD, HYG, VNQ, GLD, SLV |
| EQUITY_SECTORS | SPY, QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, GDX, XME, IWF, XSD, XBI, IWM, IWD, IWO, XLB, XLRE |
| COMBINED | All of the above |

**Windows:** `63d · 126d · 252d · 504d`

---

## Repository Structure

```
P2-ETF-QUANTILE-REGRESSION-FOREST/
├── config.py          # Universes, QRF hyperparameters, score weights
├── data_manager.py    # HuggingFace loader → (prices, macro) DataFrames
├── qrf_engine.py      # Core: CART trees, QRF, quantile prediction
├── trainer.py         # Orchestrator: load → train → score → JSON → upload
├── push_results.py    # HfApi.upload_file wrapper
├── streamlit_app.py   # Two-tab Streamlit dashboard
├── us_calendar.py     # US trading calendar helper
├── requirements.txt
└── .github/
    └── workflows/
        └── daily.yml  # Single job (no parallel needed)
```

---

## Setup

```bash
git clone https://github.com/P2SAMAPA/P2-ETF-QUANTILE-REGRESSION-FOREST
cd P2-ETF-QUANTILE-REGRESSION-FOREST
pip install -r requirements.txt

export HF_TOKEN=hf_...
python trainer.py
streamlit run streamlit_app.py
```

**Required GitHub secret:** `HF_TOKEN`

**Required HuggingFace dataset repo:** `P2SAMAPA/p2-etf-qrf-results`

---

## References

- Meinshausen, N. (2006). Quantile Regression Forests. *Journal of Machine
  Learning Research*, 7, 983–999.
- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
- Athey, S., Tibshirani, J. & Wager, S. (2019). Generalized Random Forests.
  *Annals of Statistics*, 47(2), 1148–1178.
- Koenker, R. & Bassett, G. (1978). Regression Quantiles. *Econometrica*,
  46(1), 33–50.
