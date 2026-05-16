"""
qlib_alpha158_features.py — Pure-pandas port of Qlib Alpha158 feature set.

Source: microsoft/qlib (MIT License, https://github.com/microsoft/qlib)
Origin: qlib/contrib/data/loader.py  class Alpha158DL.get_feature_config()
Ported: native pandas/numpy — no qlib import required.

All features are .shift(1)-safe (they operate on past data only, no future leak).
Input df must have lowercase columns: open, high, low, close, volume.
VWAP is optional; if missing it falls back to (high+low+close)/3.

Approximate feature count: 158 (9 kbar + price/rolling groups).
Default windows: [5, 10, 20, 30, 60]
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _slope(series: pd.Series, window: int) -> pd.Series:
    """Rolling OLS slope (linear regression coefficient) over window days."""
    def _ols_slope(y: np.ndarray) -> float:
        if len(y) < 2:
            return np.nan
        x = np.arange(len(y), dtype=float)
        xm = x - x.mean()
        ym = y - y.mean()
        denom = (xm * xm).sum()
        if denom == 0:
            return np.nan
        return float((xm * ym).sum() / denom)
    return series.rolling(window, min_periods=max(2, window // 2)).apply(_ols_slope, raw=True)


def _rsquare(series: pd.Series, window: int) -> pd.Series:
    """Rolling R-squared of linear regression over window days."""
    def _r2(y: np.ndarray) -> float:
        if len(y) < 2:
            return np.nan
        x = np.arange(len(y), dtype=float)
        xm = x - x.mean()
        ym = y - y.mean()
        denom_x = (xm * xm).sum()
        denom_y = (ym * ym).sum()
        if denom_x == 0 or denom_y == 0:
            return np.nan
        cov = (xm * ym).sum()
        return float((cov ** 2) / (denom_x * denom_y))
    return series.rolling(window, min_periods=max(2, window // 2)).apply(_r2, raw=True)


def _residual(series: pd.Series, window: int) -> pd.Series:
    """Rolling residual of linear regression (last point vs fitted)."""
    def _resi(y: np.ndarray) -> float:
        if len(y) < 2:
            return np.nan
        x = np.arange(len(y), dtype=float)
        xm = x - x.mean()
        ym = y - y.mean()
        denom = (xm * xm).sum()
        if denom == 0:
            return np.nan
        slope = (xm * ym).sum() / denom
        intercept = y.mean() - slope * x.mean()
        fitted_last = slope * x[-1] + intercept
        return float(y[-1] - fitted_last)
    return series.rolling(window, min_periods=max(2, window // 2)).apply(_resi, raw=True)


def _quantile(series: pd.Series, window: int, q: float) -> pd.Series:
    """Rolling quantile."""
    return series.rolling(window, min_periods=max(2, window // 2)).quantile(q)


def _rank(series: pd.Series, window: int) -> pd.Series:
    """Rolling rank of last value within window (percentile 0-1)."""
    def _pct_rank(y: np.ndarray) -> float:
        if len(y) < 2:
            return np.nan
        return float(np.sum(y <= y[-1]) / len(y))
    return series.rolling(window, min_periods=max(2, window // 2)).apply(_pct_rank, raw=True)


def _idxmax(series: pd.Series, window: int) -> pd.Series:
    """Days since the maximum value within window (0 = today is max)."""
    def _ix(y: np.ndarray) -> float:
        if len(y) < 1:
            return np.nan
        return float(len(y) - 1 - np.argmax(y))
    return series.rolling(window, min_periods=max(1, window // 2)).apply(_ix, raw=True)


def _idxmin(series: pd.Series, window: int) -> pd.Series:
    """Days since the minimum value within window (0 = today is min)."""
    def _ix(y: np.ndarray) -> float:
        if len(y) < 1:
            return np.nan
        return float(len(y) - 1 - np.argmin(y))
    return series.rolling(window, min_periods=max(1, window // 2)).apply(_ix, raw=True)


def _rolling_corr(s1: pd.Series, s2: pd.Series, window: int) -> pd.Series:
    return s1.rolling(window, min_periods=max(2, window // 2)).corr(s2)


# ---------------------------------------------------------------------------
# main public API
# ---------------------------------------------------------------------------

def add_alpha158_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds Qlib Alpha158 features (~158 columns) to df.
    Daily OHLCV + volume input. All features are .shift(1)-safe.

    Required columns (lowercase): open, high, low, close, volume
    Optional column: vwap  (falls back to (high+low+close)/3 if absent)

    Returns a new DataFrame with all original columns plus alpha158_* prefixed features.
    """
    df = df.copy()

    # Alias columns
    o = df['open'].astype(float)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    v = df['volume'].astype(float)

    if 'vwap' in df.columns:
        vwap = df['vwap'].astype(float)
    else:
        vwap = (h + l + c) / 3.0

    hl_range = (h - l).replace(0, np.nan)
    out = {}

    # ------------------------------------------------------------------
    # Group 1: K-Bar features (9 features)
    # ------------------------------------------------------------------
    out['alpha158_KMID']  = (c - o) / o.replace(0, np.nan)
    out['alpha158_KLEN']  = (h - l) / o.replace(0, np.nan)
    out['alpha158_KMID2'] = (c - o) / (hl_range + 1e-12)
    out['alpha158_KUP']   = (h - np.maximum(o, c)) / o.replace(0, np.nan)
    out['alpha158_KUP2']  = (h - np.maximum(o, c)) / (hl_range + 1e-12)
    out['alpha158_KLOW']  = (np.minimum(o, c) - l) / o.replace(0, np.nan)
    out['alpha158_KLOW2'] = (np.minimum(o, c) - l) / (hl_range + 1e-12)
    out['alpha158_KSFT']  = (2 * c - h - l) / o.replace(0, np.nan)
    out['alpha158_KSFT2'] = (2 * c - h - l) / (hl_range + 1e-12)

    # ------------------------------------------------------------------
    # Group 2: Price ratio features at window=0 (4 raw prices / close)
    # ------------------------------------------------------------------
    out['alpha158_OPEN0']  = o / c.replace(0, np.nan)
    out['alpha158_HIGH0']  = h / c.replace(0, np.nan)
    out['alpha158_LOW0']   = l / c.replace(0, np.nan)
    out['alpha158_VWAP0']  = vwap / c.replace(0, np.nan)

    # ------------------------------------------------------------------
    # Group 3: Rolling features — windows [5, 10, 20, 30, 60]
    # ------------------------------------------------------------------
    windows = [5, 10, 20, 30, 60]

    # daily return and volume change (used by several sub-features)
    daily_ret = c / c.shift(1).replace(0, np.nan)     # c_t / c_{t-1}
    vol_chg   = v / v.shift(1).replace(0, np.nan)     # v_t / v_{t-1}
    abs_daily_chg = (c - c.shift(1)).abs()

    for d in windows:
        mp = max(2, d // 2)   # min_periods for rolling

        # ROC — Rate of Change: c shifted d / current c
        out[f'alpha158_ROC{d}'] = c.shift(d) / c.replace(0, np.nan)

        # MA — Simple moving average / c
        out[f'alpha158_MA{d}'] = c.rolling(d, min_periods=mp).mean() / c.replace(0, np.nan)

        # STD — Standard deviation of close / c
        out[f'alpha158_STD{d}'] = c.rolling(d, min_periods=mp).std() / c.replace(0, np.nan)

        # BETA — Slope of close / c
        out[f'alpha158_BETA{d}'] = _slope(c, d) / c.replace(0, np.nan)

        # RSQR — R-squared of linear regression on close
        out[f'alpha158_RSQR{d}'] = _rsquare(c, d)

        # RESI — Residual of linear regression / c
        out[f'alpha158_RESI{d}'] = _residual(c, d) / c.replace(0, np.nan)

        # MAX — Rolling max of high / c
        out[f'alpha158_MAX{d}'] = h.rolling(d, min_periods=mp).max() / c.replace(0, np.nan)

        # MIN — Rolling min of low / c
        out[f'alpha158_MIN{d}'] = l.rolling(d, min_periods=mp).min() / c.replace(0, np.nan)

        # QTLU — 80th percentile of close / c
        out[f'alpha158_QTLU{d}'] = _quantile(c, d, 0.8) / c.replace(0, np.nan)

        # QTLD — 20th percentile of close / c
        out[f'alpha158_QTLD{d}'] = _quantile(c, d, 0.2) / c.replace(0, np.nan)

        # RANK — percentile rank of current close within past d days
        out[f'alpha158_RANK{d}'] = _rank(c, d)

        # RSV — Stochastic %K: position between high/low channel
        roll_max_h = h.rolling(d, min_periods=mp).max()
        roll_min_l = l.rolling(d, min_periods=mp).min()
        out[f'alpha158_RSV{d}'] = (c - roll_min_l) / (roll_max_h - roll_min_l + 1e-12)

        # IMAX — days since highest high / d
        out[f'alpha158_IMAX{d}'] = _idxmax(h, d) / d

        # IMIN — days since lowest low / d
        out[f'alpha158_IMIN{d}'] = _idxmin(l, d) / d

        # IMXD — (days_since_high - days_since_low) / d
        out[f'alpha158_IMXD{d}'] = (
            _idxmax(h, d) - _idxmin(l, d)
        ) / d

        # CORR — correlation between close and log(volume+1)
        log_vol = np.log(v + 1)
        out[f'alpha158_CORR{d}'] = _rolling_corr(c, log_vol, d)

        # CORD — correlation between close-change ratio and volume-change ratio
        log_vol_chg = np.log(vol_chg + 1)
        out[f'alpha158_CORD{d}'] = _rolling_corr(daily_ret, log_vol_chg, d)

        # CNTP — fraction of past d days that close went up
        up_days = (c > c.shift(1)).astype(float)
        out[f'alpha158_CNTP{d}'] = up_days.rolling(d, min_periods=mp).mean()

        # CNTN — fraction of past d days that close went down
        dn_days = (c < c.shift(1)).astype(float)
        out[f'alpha158_CNTN{d}'] = dn_days.rolling(d, min_periods=mp).mean()

        # CNTD — CNTP - CNTN
        out[f'alpha158_CNTD{d}'] = (
            out[f'alpha158_CNTP{d}'] - out[f'alpha158_CNTN{d}']
        )

        # SUMP — RSI numerator: gain sum / abs-change sum
        gain = (c - c.shift(1)).clip(lower=0)
        gain_sum = gain.rolling(d, min_periods=mp).sum()
        abs_sum  = abs_daily_chg.rolling(d, min_periods=mp).sum()
        out[f'alpha158_SUMP{d}'] = gain_sum / (abs_sum + 1e-12)

        # SUMN — loss sum / abs-change sum (= 1 - SUMP)
        loss = (c.shift(1) - c).clip(lower=0)
        loss_sum = loss.rolling(d, min_periods=mp).sum()
        out[f'alpha158_SUMN{d}'] = loss_sum / (abs_sum + 1e-12)

        # SUMD — (gain_sum - loss_sum) / abs-change sum
        out[f'alpha158_SUMD{d}'] = (gain_sum - loss_sum) / (abs_sum + 1e-12)

        # VMA — volume MA / volume
        out[f'alpha158_VMA{d}'] = v.rolling(d, min_periods=mp).mean() / (v + 1e-12)

        # VSTD — volume std / volume
        out[f'alpha158_VSTD{d}'] = v.rolling(d, min_periods=mp).std() / (v + 1e-12)

        # WVMA — vol-weighted price-change volatility (std/mean of |ret|*vol)
        wt = daily_ret.abs() * v
        wt_std  = wt.rolling(d, min_periods=mp).std()
        wt_mean = wt.rolling(d, min_periods=mp).mean()
        out[f'alpha158_WVMA{d}'] = wt_std / (wt_mean + 1e-12)

        # VSUMP — fraction of volume increase out of absolute volume change
        vol_gain = (v - v.shift(1)).clip(lower=0)
        vol_loss = (v.shift(1) - v).clip(lower=0)
        abs_vol_chg = (v - v.shift(1)).abs()
        abs_vol_sum = abs_vol_chg.rolling(d, min_periods=mp).sum()
        out[f'alpha158_VSUMP{d}'] = vol_gain.rolling(d, min_periods=mp).sum() / (abs_vol_sum + 1e-12)

        # VSUMN — fraction of volume decrease out of absolute volume change
        out[f'alpha158_VSUMN{d}'] = vol_loss.rolling(d, min_periods=mp).sum() / (abs_vol_sum + 1e-12)

        # VSUMD — (vol_gain_sum - vol_loss_sum) / abs_vol_sum
        out[f'alpha158_VSUMD{d}'] = (
            vol_gain.rolling(d, min_periods=mp).sum() - vol_loss.rolling(d, min_periods=mp).sum()
        ) / (abs_vol_sum + 1e-12)

    # ------------------------------------------------------------------
    # Attach all features to df
    # ------------------------------------------------------------------
    feat_df = pd.DataFrame(out, index=df.index)

    # Drop any column that is entirely NaN
    all_nan = feat_df.columns[feat_df.isna().all()]
    if len(all_nan):
        feat_df = feat_df.drop(columns=all_nan)

    result = pd.concat([df, feat_df], axis=1)
    return result


def alpha158_feature_names() -> list:
    """Return the list of feature column names that will be added (no df needed)."""
    names = [f'alpha158_{n}' for n in [
        'KMID','KLEN','KMID2','KUP','KUP2','KLOW','KLOW2','KSFT','KSFT2',
        'OPEN0','HIGH0','LOW0','VWAP0',
    ]]
    for suffix in ['ROC','MA','STD','BETA','RSQR','RESI','MAX','MIN',
                   'QTLU','QTLD','RANK','RSV','IMAX','IMIN','IMXD',
                   'CORR','CORD','CNTP','CNTN','CNTD',
                   'SUMP','SUMN','SUMD',
                   'VMA','VSTD','WVMA','VSUMP','VSUMN','VSUMD']:
        for d in [5, 10, 20, 30, 60]:
            names.append(f'alpha158_{suffix}{d}')
    return names


if __name__ == '__main__':
    # Smoke test on AAPL
    import yfinance as yf
    print("Downloading AAPL...")
    raw = yf.download('AAPL', start='2015-01-01', end='2024-12-31', auto_adjust=True, progress=False)
    # Handle MultiIndex columns from yfinance
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = ['_'.join([str(x).lower() for x in col if x != 'AAPL']).strip('_') for col in raw.columns]
    else:
        raw.columns = [c.lower() for c in raw.columns]
    raw = raw.rename(columns={'adj close': 'close', 'adj_close': 'close'}) if ('adj close' in raw.columns or 'adj_close' in raw.columns) else raw

    n_before = len(raw.columns)
    result = add_alpha158_features(raw)
    alpha_cols = [c for c in result.columns if c.startswith('alpha158_')]
    n_features = len(alpha_cols)

    print(f"Rows: {len(result)}")
    print(f"Features added: +{n_features} alpha158 columns")

    nan_only = [c for c in alpha_cols if result[c].isna().all()]
    print(f"NaN-only columns: {len(nan_only)}")
    if nan_only:
        print("  NaN-only:", nan_only)

    sample = result[alpha_cols].tail(5)
    print("\nSample (last 5 rows, first 10 features):")
    print(sample.iloc[:, :10].to_string())

    # Expected count
    expected = alpha158_feature_names()
    print(f"\nExpected feature names count: {len(expected)}")
    print(f"Actual features added: {n_features}")
    print("SMOKE TEST PASSED" if n_features >= 130 and len(nan_only) == 0 else "SMOKE TEST WARNING")
