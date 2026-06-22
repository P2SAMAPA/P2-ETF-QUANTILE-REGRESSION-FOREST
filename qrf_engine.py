"""
qrf_engine.py — Quantile Regression Forest Engine
===================================================

Theory
------
**Random Forests (Breiman 2001)**

A random forest is an ensemble of decision trees, each trained on a bootstrap
sample of the data with random feature subsets at each split. The ensemble
prediction averages over all trees, reducing variance while maintaining low bias.

**Quantile Regression Forests (Meinshausen 2006)**

Standard random forests predict E[Y|X] by averaging leaf node values. QRF
generalises this: instead of averaging, it keeps the full empirical distribution
of training Y values that fall into each leaf. The conditional quantile Q(q|X)
is then estimated as the q-th quantile of this weighted empirical distribution.

Formally, for a query point x:

    F̂(y|X=x) = (1/B) Σ_b (1/|L_b(x)|) Σ_{i ∈ L_b(x)} 1[Y_i ≤ y]

Where:
    B          : number of trees
    L_b(x)    : leaf node in tree b that contains x
    Y_i        : training target for sample i
    |L_b(x)|  : number of training samples in that leaf

The q-th conditional quantile is the q-th quantile of this weighted distribution.

**Key properties:**
- Non-parametric: no distributional assumption on Y|X
- Consistent: converges to true conditional quantile as N → ∞ (Meinshausen 2006)
- Full distribution: one forest gives ALL quantiles simultaneously
- Fast: O(N log N) training, O(depth) inference — much faster than KRR or CFM
- Robust: no gradient, no learning rate, no numerical instability

**Application to ETF Ranking**

Features at time t (for ETF i):
    x_t = [ret_1d, ret_5d, ret_21d, vol_21d, vol_63d, skew_21d, kurt_21d,
            VIX, DXY, T10Y2Y, IG_SPREAD, HY_SPREAD]

Target: y_t = mean(log_return_{t+1:t+PRED_HORIZON})  (forward log return)

From QRF predictions on today's feature vector x_today:
    q_median = Q(0.50 | x_today)   → conditional median return
    q_tail   = Q(0.10 | x_today)   → 10th percentile (CVaR proxy)
    q_prob   = P(r > 0 | x_today)  → P(positive return)

Composite score = 0.50·q_median + 0.30·(q_prob − 0.5) + 0.20·(−q_tail)
Cross-sectionally z-scored per universe per window.

**Distinction from CAUSAL-FOREST and DISTRIBUTIONAL-RL (in suite):**
    - Causal Forest: estimates heterogeneous treatment effects (CATE)
    - Distributional RL: learns quantile functions via temporal difference
    - QRF: non-parametric conditional quantile estimation via tree ensembles
      — no causality assumption, no RL, just weighted empirical distribution

References
----------
- Meinshausen, N. (2006). Quantile Regression Forests. Journal of Machine
  Learning Research, 7, 983–999.
- Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32.
- Athey, S., Tibshirani, J. & Wager, S. (2019). Generalized Random Forests.
  Annals of Statistics, 47(2), 1148–1178.
- Meinshausen, N. & Ridgeway, G. (2006). Quantile Regression Forests.
  Journal of Machine Learning Research, 7, 983–999.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

import config


# ── Decision tree node ────────────────────────────────────────────────────────

class _Node:
    __slots__ = ["feat", "thresh", "left", "right", "leaf_indices"]

    def __init__(self):
        self.feat         = None
        self.thresh       = None
        self.left         = None
        self.right        = None
        self.leaf_indices = None   # training sample indices in this leaf


# ── Single regression tree ────────────────────────────────────────────────────

class _RegressionTree:
    """
    Single CART tree with random feature subsampling per split.
    Stores training sample indices at each leaf for QRF inference.
    """
    def __init__(self, max_depth: int, min_samples: int,
                 max_features: float, rng: np.random.Generator):
        self.max_depth   = max_depth
        self.min_samples = min_samples
        self.max_features = max_features
        self.rng         = rng
        self.root        = None
        self._Y_train    = None   # reference to training targets

    def fit(self, X: np.ndarray, Y: np.ndarray,
            sample_indices: np.ndarray) -> None:
        """
        X: (N, p) features, Y: (N,) targets
        sample_indices: bootstrap indices into X, Y
        """
        self._Y_train = Y
        self._n_features = X.shape[1]
        self.root = self._build(X, Y, sample_indices, depth=0)

    def _build(self, X, Y, idx, depth):
        node = _Node()

        # Leaf conditions
        if depth >= self.max_depth or len(idx) <= self.min_samples:
            node.leaf_indices = idx
            return node

        # Random feature subset
        n_feats = max(1, int(self._n_features * self.max_features))
        feats   = self.rng.choice(self._n_features, size=n_feats, replace=False)

        best_feat, best_thresh, best_gain = None, None, -np.inf
        Y_idx = Y[idx]
        var_parent = Y_idx.var() * len(idx)

        for f in feats:
            vals   = X[idx, f]
            threshs = np.unique(vals)
            if len(threshs) < 2:
                continue
            # Try midpoints between unique values (subsample for speed)
            mids = (threshs[:-1] + threshs[1:]) / 2
            if len(mids) > 20:
                mids = self.rng.choice(mids, size=20, replace=False)

            for t in mids:
                left_mask  = vals <= t
                right_mask = ~left_mask
                n_l, n_r   = left_mask.sum(), right_mask.sum()
                if n_l < self.min_samples or n_r < self.min_samples:
                    continue
                var_l = Y_idx[left_mask].var() * n_l
                var_r = Y_idx[right_mask].var() * n_r
                gain  = var_parent - var_l - var_r
                if gain > best_gain:
                    best_gain  = gain
                    best_feat  = f
                    best_thresh = t

        if best_feat is None:
            node.leaf_indices = idx
            return node

        node.feat   = best_feat
        node.thresh = best_thresh
        left_idx  = idx[X[idx, best_feat] <= best_thresh]
        right_idx = idx[X[idx, best_feat] >  best_thresh]
        node.left  = self._build(X, Y, left_idx,  depth+1)
        node.right = self._build(X, Y, right_idx, depth+1)
        return node

    def get_leaf_indices(self, x: np.ndarray) -> np.ndarray:
        """Return training indices in the leaf that x falls into."""
        node = self.root
        while node.leaf_indices is None:
            if x[node.feat] <= node.thresh:
                node = node.left
            else:
                node = node.right
        return node.leaf_indices

    def get_leaf_indices_batch(self, X: np.ndarray) -> List[np.ndarray]:
        """Return leaf indices for each row in X."""
        return [self.get_leaf_indices(X[i]) for i in range(len(X))]


# ── Quantile Regression Forest ────────────────────────────────────────────────

class QuantileRegressionForest:
    """
    QRF: ensemble of regression trees storing leaf training indices.
    Prediction: weighted empirical quantile across all trees.
    """
    def __init__(self, n_trees: int, max_depth: int, min_samples: int,
                 max_features: float, rng: np.random.Generator):
        self.n_trees     = n_trees
        self.max_depth   = max_depth
        self.min_samples = min_samples
        self.max_features = max_features
        self.rng         = rng
        self.trees: List[_RegressionTree] = []
        self._Y_train    = None

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        X: (N, p), Y: (N,) — training features and targets.
        Fits n_trees regression trees on bootstrap samples.
        """
        N = len(X)
        self._Y_train = Y.copy()
        self.trees    = []

        for _ in range(self.n_trees):
            tree = _RegressionTree(
                max_depth   = self.max_depth,
                min_samples = self.min_samples,
                max_features = self.max_features,
                rng         = self.rng,
            )
            if config.QRF_BOOTSTRAP:
                boot_idx = self.rng.integers(0, N, size=N)
            else:
                boot_idx = np.arange(N)
            tree.fit(X, Y, boot_idx)
            self.trees.append(tree)

    def predict_quantiles(self, x: np.ndarray,
                          quantiles: List[float]) -> np.ndarray:
        """
        Predict conditional quantiles for a single query point x.

        Aggregates leaf training Y-values across all trees (equal weight per tree),
        then computes empirical quantiles of the resulting weighted distribution.

        x: (p,) feature vector
        quantiles: list of quantile levels in [0,1]
        Returns: np.ndarray of shape (len(quantiles),)
        """
        # Collect all leaf Y-values across trees
        all_Y = []
        for tree in self.trees:
            leaf_idx = tree.get_leaf_indices(x)
            all_Y.append(self._Y_train[leaf_idx])

        # Pool all leaf Y-values (equal weight per tree, uniform within leaf)
        pooled = np.concatenate(all_Y)
        return np.quantile(pooled, quantiles)

    def predict_prob_positive(self, x: np.ndarray) -> float:
        """P(Y > 0 | X=x) — probability of positive return."""
        all_Y = []
        for tree in self.trees:
            leaf_idx = tree.get_leaf_indices(x)
            all_Y.append(self._Y_train[leaf_idx])
        pooled = np.concatenate(all_Y)
        return float(np.mean(pooled > 0))


# ── Feature construction ──────────────────────────────────────────────────────

def _build_features(
    log_ret:    np.ndarray,
    macro_norm: np.ndarray,
    t:          int,
) -> Optional[np.ndarray]:
    """
    Build feature vector at time t:
    [ret_1d, ret_5d, ret_21d, vol_21d, vol_63d, skew_21d, kurt_21d, macro...]
    """
    max_lag = max(config.N_RET_LAGS + config.VOL_WINDOWS)
    if t < max_lag:
        return None

    feats = []

    # Momentum features
    for lag in config.N_RET_LAGS:
        feats.append(log_ret[t-lag:t].sum())

    # Volatility features
    for w in config.VOL_WINDOWS:
        feats.append(log_ret[t-w:t].std() * np.sqrt(252))

    # Higher moments (21d)
    w21 = log_ret[t-21:t]
    mu21 = w21.mean()
    std21 = w21.std() + 1e-10
    skew = float(np.mean(((w21 - mu21)/std21)**3))
    kurt = float(np.mean(((w21 - mu21)/std21)**4) - 3)
    feats.append(np.clip(skew, -5, 5))
    feats.append(np.clip(kurt, -5, 10))

    # Macro features
    if macro_norm.shape[1] > 0 and t < len(macro_norm):
        feats.extend(macro_norm[t].tolist())

    arr = np.array(feats, dtype=np.float64)
    if np.isnan(arr).any():
        return None
    return arr


def _build_dataset(
    log_ret:    np.ndarray,
    macro_norm: np.ndarray,
    window:     int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (X, Y) training dataset over a rolling window.
    X: (N, p) features, Y: (N,) forward log returns.
    """
    T      = len(log_ret)
    start  = max(T - window, max(config.N_RET_LAGS + config.VOL_WINDOWS) + 1)
    end    = T - config.PRED_HORIZON

    X_rows, Y_rows = [], []
    for t in range(start, end):
        x = _build_features(log_ret, macro_norm, t)
        if x is None:
            continue
        fwd = log_ret[t:t+config.PRED_HORIZON].mean()
        if np.isnan(fwd):
            continue
        X_rows.append(x)
        Y_rows.append(fwd)

    if not X_rows:
        return np.empty((0, 0)), np.empty(0)

    return np.array(X_rows), np.array(Y_rows)


# ── Main scoring function ─────────────────────────────────────────────────────

def compute_qrf_scores(
    prices:    pd.DataFrame,
    macro_df:  pd.DataFrame,
    tickers:   List[str],
    window:    int,
) -> pd.Series:
    """
    Train a QRF per ETF and return quantile-based cross-sectional z-scores.

    Parameters
    ----------
    prices   : DataFrame of closing prices, DatetimeIndex
    macro_df : DataFrame of macro signal levels, DatetimeIndex
    tickers  : list of ETF tickers in this universe
    window   : lookback window in trading days

    Returns
    -------
    pd.Series indexed by ticker, values = composite QRF z-score
    """
    avail = [t for t in tickers if t in prices.columns]
    if not avail:
        return pd.Series(dtype=float)

    max_lag  = max(config.N_RET_LAGS + config.VOL_WINDOWS)
    min_rows = window + config.PRED_HORIZON + max_lag + 5
    if len(prices) < min_rows:
        return pd.Series(dtype=float)

    # Align macro
    common    = prices.index.intersection(macro_df.index) if not macro_df.empty else prices.index
    prices_a  = prices.loc[common]
    macro_a   = macro_df.loc[common] if not macro_df.empty else pd.DataFrame(index=common)

    macro_vals = macro_a.values.astype(np.float64) if not macro_a.empty else np.zeros((len(common), 0))
    if macro_vals.shape[1] > 0:
        m_mu       = np.nanmean(macro_vals, axis=0, keepdims=True)
        m_std      = np.nanstd(macro_vals,  axis=0, keepdims=True) + 1e-8
        macro_norm = np.nan_to_num((macro_vals - m_mu) / m_std, 0.0)
    else:
        macro_norm = macro_vals

    rng        = np.random.default_rng(42)
    raw_scores = {}

    for ticker in avail:
        price_series = prices_a[ticker].dropna()
        if len(price_series) < min_rows:
            continue

        log_ret = np.log(price_series / price_series.shift(1)).dropna().values
        mac     = macro_norm[-len(log_ret):]
        if len(mac) < len(log_ret):
            log_ret = log_ret[-len(mac):]

        X, Y = _build_dataset(log_ret, mac, window)

        if len(X) < config.MIN_TRAIN_SAMPLES:
            print(f"    {ticker}: only {len(X)} samples, skipping")
            continue

        print(f"    Training QRF for {ticker} "
              f"(N={len(X)}, p={X.shape[1]}, trees={config.QRF_N_TREES})")

        # Train QRF
        qrf = QuantileRegressionForest(
            n_trees      = config.QRF_N_TREES,
            max_depth    = config.QRF_MAX_DEPTH,
            min_samples  = config.QRF_MIN_SAMPLES,
            max_features = config.QRF_MAX_FEATURES,
            rng          = rng,
        )
        try:
            qrf.fit(X, Y)
        except Exception as e:
            print(f"    Training failed {ticker}: {e}")
            continue

        # Build today's feature vector
        x_today = _build_features(log_ret, mac, len(log_ret) - 1)
        if x_today is None:
            continue

        # Predict quantiles
        try:
            qs = qrf.predict_quantiles(
                x_today,
                quantiles=[config.QUANTILE_TAIL, config.QUANTILE_MEDIAN],
            )
            q_tail   = float(qs[0])   # 10th percentile
            q_median = float(qs[1])   # 50th percentile
            q_prob   = qrf.predict_prob_positive(x_today)
        except Exception as e:
            print(f"    Inference failed {ticker}: {e}")
            continue

        # Composite score
        composite = (
            config.WEIGHT_MEDIAN * q_median
            + config.WEIGHT_PROB  * (q_prob - 0.5)
            + config.WEIGHT_CVAR  * (-q_tail)    # negative tail = positive signal
        )

        print(f"    {ticker}: q50={q_median:.4f}  q10={q_tail:.4f}  "
              f"P(r>0)={q_prob:.3f}  score={composite:.4f}")

        raw_scores[ticker] = composite

    if not raw_scores:
        return pd.Series(dtype=float)

    scores = pd.Series(raw_scores)
    mu, std = scores.mean(), scores.std()
    if std < 1e-10:
        return pd.Series(0.0, index=scores.index)
    return (scores - mu) / std
