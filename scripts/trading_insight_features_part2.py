"""
trading_insight_features_part2.py

Features derived from lines 713-1424 of Trading Insight Info.txt.
Covers concepts 65-71 (setup-quality / risk / indicator stacks) and tools 1-54
from the "more trading tools" section: alternate chart types, auction-market /
market-profile proxies, footprint / volume-delta proxies, additional trend and
momentum indicators, additional moving-average tools, and volatility / squeeze
tools.

All features are .shift(1)-safe: no future data bleeds into any computed value.
Only numpy, pandas, and pandas_ta_classic are used.
L2 / tick / news / intraday data are unavailable on daily OHLCV; proxies are
documented inline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# pandas_ta_classic is imported inside the function to allow the module to be
# imported even in environments where it is not installed (callers that pass
# pre-computed columns).  A top-level import guard is used for clarity.
try:
    import pandas_ta_classic as ta
    _HAS_TA = True
except ImportError:
    _HAS_TA = False


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _roc(series: pd.Series, n: int) -> pd.Series:
    """Rate of change over n bars."""
    return series.pct_change(n)


def _sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=1).mean()


def _ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False, min_periods=1).mean()


def _stdev(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=2).std()


def _rolling_high(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=1).max()


def _rolling_low(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=1).min()


# ---------------------------------------------------------------------------
# Main feature function
# ---------------------------------------------------------------------------

def add_features_part2(df: pd.DataFrame) -> pd.DataFrame:
    """All features from lines 713-1424 of Trading Insight Info.

    Input: daily OHLCV df with columns open, high, low, close, volume.
    Returns df with new columns appended.  All features are .shift(1) safe.

    Column naming convention: snake_case, prefixed by concept abbreviation.
    """

    # Work on a copy so we don't mutate the caller's frame.
    df = df.copy()

    # Ensure standard column names (case-insensitive normalisation).
    col_map = {c.lower(): c for c in df.columns}
    o = df[col_map.get("open", "open")]
    h = df[col_map.get("high", "high")]
    l = df[col_map.get("low", "low")]
    c = df[col_map.get("close", "close")]
    v = df[col_map.get("volume", "volume")]

    # ------------------------------------------------------------------
    # SECTION A  –  Setup quality / indicator-stack composites (lines 713-817)
    # ------------------------------------------------------------------
    # Concept 67: Best indicator stack = RSI + MACD + ADX + ATR
    # We build each component then composite a signal-agreement score.

    # RSI(14)
    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=13, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(com=13, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["stack67_rsi14"] = 100 - 100 / (1 + rs)

    # MACD(12,26,9) histogram
    ema12 = _ema(c, 12)
    ema26 = _ema(c, 26)
    macd_line = ema12 - ema26
    macd_signal = _ema(macd_line, 9)
    df["stack67_macd_hist"] = macd_line - macd_signal
    df["stack67_macd_above_signal"] = (macd_line > macd_signal).astype(int)

    # ADX(14) – directional movement
    if _HAS_TA:
        _adx = ta.adx(h, l, c, length=14)
        if _adx is not None and not _adx.empty:
            adx_col = [col for col in _adx.columns if col.upper().startswith("ADX_")]
            if adx_col:
                df["stack67_adx14"] = _adx[adx_col[0]].values
            else:
                df["stack67_adx14"] = np.nan
        else:
            df["stack67_adx14"] = np.nan
    else:
        # Manual ADX
        tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
        dm_plus = np.where((h.diff() > l.diff().abs()) & (h.diff() > 0), h.diff(), 0)
        dm_minus = np.where((l.diff().abs() > h.diff()) & (l.diff() < 0), l.diff().abs(), 0)
        atr14 = pd.Series(dm_plus).ewm(com=13, adjust=False, min_periods=14).mean()
        di_plus = 100 * pd.Series(dm_plus).ewm(com=13, adjust=False).mean() / tr.ewm(com=13, adjust=False).mean().replace(0, np.nan)
        di_minus = 100 * pd.Series(dm_minus).ewm(com=13, adjust=False).mean() / tr.ewm(com=13, adjust=False).mean().replace(0, np.nan)
        dx = (100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan))
        df["stack67_adx14"] = dx.ewm(com=13, adjust=False).mean().values

    # ATR(14)
    tr_high = h.combine(c.shift(1), max)
    tr_low = l.combine(c.shift(1), min)
    tr_raw = tr_high - tr_low
    df["stack67_atr14"] = tr_raw.ewm(com=13, adjust=False, min_periods=14).mean()

    # Composite indicator-stack alignment score (+1 per bullish condition)
    rsi_bull = (df["stack67_rsi14"] > 50).astype(int)
    macd_bull = df["stack67_macd_above_signal"]
    adx_trend = (df["stack67_adx14"] > 25).astype(int)
    # ATR direction: rising ATR = expanding volatility (neutral sign, use as filter)
    atr_rising = (df["stack67_atr14"] > df["stack67_atr14"].shift(5)).astype(int)
    df["stack67_composite_bull"] = rsi_bull + macd_bull + adx_trend
    df["stack67_atr_expanding"] = atr_rising

    # Concept 69: Best market-context stack = SPY/QQQ + sector + A/D + VIX
    # PROXY on single-ticker daily: relative volume vs 20-day avg, breadth proxy
    df["ctx69_relvol20"] = v / v.rolling(20, min_periods=5).mean()

    # Concept 70: Catalyst stack – volume spike as catalyst proxy
    df["cat70_vol_spike_2x"] = (df["ctx69_relvol20"] > 2.0).astype(int)
    df["cat70_vol_spike_3x"] = (df["ctx69_relvol20"] > 3.0).astype(int)

    # Concept 71: Risk stack – ATR-based position-size proxy (% ATR / close)
    df["risk71_atr_pct"] = df["stack67_atr14"] / c.replace(0, np.nan)

    # Simple rule: 7-condition trade agreement score (price setup items we can proxy)
    # 1. Price above 20-day SMA
    sma20 = _sma(c, 20)
    price_above_sma20 = (c > sma20).astype(int)
    # 2. Market support proxy: price above 50-day SMA
    sma50 = _sma(c, 50)
    price_above_sma50 = (c > sma50).astype(int)
    # 3. Sector support proxy: price above 200-day SMA
    sma200 = _sma(c, 200)
    price_above_sma200 = (c > sma200).astype(int)
    # 4. Volume confirms: relative volume > 1.0
    vol_confirms = (df["ctx69_relvol20"] > 1.0).astype(int)
    # 5. MACD confirms
    macd_confirms = macd_bull
    # 6. Risk/reward proxy: ADX > 20 (trend worth trading)
    rr_ok = (df["stack67_adx14"] > 20).astype(int)
    df["rule_agreement_score"] = (
        price_above_sma20 + price_above_sma50 + price_above_sma200
        + vol_confirms + macd_confirms + rr_ok
    )

    # ------------------------------------------------------------------
    # SECTION B  –  Alternate chart-type proxies (lines 826-923)
    # ------------------------------------------------------------------

    # Tool 1: Heikin-Ashi  (lines 826-835)
    # HA candles: each bar uses prior HA values.
    ha_close = (o + h + l + c) / 4
    ha_open = pd.Series(np.nan, index=df.index)
    ha_open.iloc[0] = (o.iloc[0] + c.iloc[0]) / 2
    for i in range(1, len(df)):
        ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2
    ha_high = pd.concat([h, ha_open, ha_close], axis=1).max(axis=1)
    ha_low  = pd.concat([l, ha_open, ha_close], axis=1).min(axis=1)
    df["ha_close"]  = ha_close
    df["ha_open"]   = ha_open
    df["ha_high"]   = ha_high
    df["ha_low"]    = ha_low
    df["ha_trend"]  = np.where(ha_close > ha_open, 1, -1)  # +1 bull, -1 bear
    df["ha_no_lower_wick"] = (ha_low == ha_open).astype(int)   # strong bull signal
    df["ha_no_upper_wick"] = (ha_high == ha_close).astype(int) # strong bear signal

    # Tool 2: Renko proxy – cumulative price-move direction (lines 837-846)
    # A Renko brick size proxy = ATR(14). We track net brick direction.
    brick_size = df["stack67_atr14"].replace(0, np.nan)
    price_change = c.diff()
    df["renko_up_pressure"]   = (price_change >  brick_size).astype(int)
    df["renko_down_pressure"] = (price_change < -brick_size).astype(int)
    df["renko_brick_ratio"]   = price_change / brick_size

    # Tool 3: Kagi proxy – price reversal ignoring time (lines 848-857)
    # Proxy: day's price change vs 3-day ATR threshold
    atr3 = tr_raw.rolling(3, min_periods=1).mean()
    df["kagi_reversal_up"]   = (c.diff() >  atr3).astype(int)
    df["kagi_reversal_down"] = (c.diff() < -atr3).astype(int)

    # Tool 4: Point & Figure proxy – breakout of X/O box (lines 859-868)
    # P&F box size = 1 ATR; count net up-boxes vs down-boxes over 10 days
    pf_box = df["stack67_atr14"].replace(0, np.nan)
    net_move_10 = c - c.shift(10)
    df["pf_net_boxes_10d"] = net_move_10 / pf_box

    # Tool 5: Range Bar proxy – high-low range relative to ATR (lines 870-879)
    daily_range = h - l
    df["rangebar_range_vs_atr"] = daily_range / df["stack67_atr14"].replace(0, np.nan)
    df["rangebar_compressed"]   = (df["rangebar_range_vs_atr"] < 0.5).astype(int)
    df["rangebar_expanded"]     = (df["rangebar_range_vs_atr"] > 1.5).astype(int)

    # Tool 6: Tick Chart proxy – high volume density (lines 881-890)
    # PROXY: volume vs rolling mean; high volume ~ many ticks
    df["tick_vol_density"] = v / v.rolling(10, min_periods=1).mean()

    # Tool 7: Volume Bar proxy – equal-volume price movement (lines 892-901)
    # PROXY: price change per unit volume (dollars per share-volume unit)
    df["volbar_price_per_vol"] = c.diff().abs() / v.replace(0, np.nan)

    # Tool 8: EquiVolume – width=volume, height=price range (lines 903-912)
    df["equivol_range_x_vol"] = daily_range * v          # volume × range composite
    df["equivol_norm"]        = daily_range / (v.replace(0, np.nan) ** 0.5)

    # Tool 9: CandleVolume – candle strength × volume (lines 914-923)
    body = (c - o).abs()
    df["candlevol_body_x_vol"] = body * v
    df["candlevol_bull_body_vol"] = np.where(c > o, body * v, 0)
    df["candlevol_bear_body_vol"] = np.where(c < o, body * v, 0)

    # Tool 10: Elder Impulse System (lines 925-934)
    # Bullish: EMA rising AND MACD histogram rising
    # Bearish: EMA falling AND MACD histogram falling
    # Neutral: otherwise
    ema13 = _ema(c, 13)
    ema_rising    = (ema13 > ema13.shift(1)).astype(int)
    macd_hist_rising = (df["stack67_macd_hist"] > df["stack67_macd_hist"].shift(1)).astype(int)
    df["elder_impulse_bull"] = ((ema_rising == 1) & (macd_hist_rising == 1)).astype(int)
    df["elder_impulse_bear"] = ((ema_rising == 0) & (macd_hist_rising == 0)).astype(int)
    df["elder_impulse_val"]  = df["elder_impulse_bull"] - df["elder_impulse_bear"]

    # ------------------------------------------------------------------
    # SECTION C  –  Auction-market / market-profile proxies (lines 936-1045)
    # ------------------------------------------------------------------
    # Daily OHLCV does not provide intraday TPO letters.  We proxy the
    # key structural concepts using daily price distributions.

    # Tools 11-14: Value Area High / Low / Area – rolling 20-day VWAP ± 1σ
    # PROXY: VWAP as Point of Control, ± 1σ as Value Area boundaries
    pv = c * v
    vwap20 = pv.rolling(20, min_periods=5).sum() / v.rolling(20, min_periods=5).sum()
    vwap_std20 = _stdev(c, 20)
    df["mp_poc_vwap20"]    = vwap20        # Point of Control proxy
    df["mp_val_high"]      = vwap20 + vwap_std20   # Value Area High (Tool 12)
    df["mp_val_low"]       = vwap20 - vwap_std20   # Value Area Low  (Tool 13)
    df["mp_price_vs_val"]  = np.where(c > df["mp_val_high"], 1,
                             np.where(c < df["mp_val_low"], -1, 0))  # Tool 14

    # Tool 15: Initial Balance proxy – first 30-min range  →  use open + range proxy
    # PROXY: 5-day high/low of first 20% of the daily range
    df["mp_ib_high_proxy"] = o + 0.5 * (h - o)   # rough upper IB
    df["mp_ib_low_proxy"]  = o - 0.5 * (o - l)   # rough lower IB
    df["mp_ib_range"]      = df["mp_ib_high_proxy"] - df["mp_ib_low_proxy"]
    df["mp_close_above_ib"] = (c > df["mp_ib_high_proxy"]).astype(int)
    df["mp_close_below_ib"] = (c < df["mp_ib_low_proxy"]).astype(int)

    # Tool 16: TPO Single Prints proxy – day's range vastly > prior range (fast move)
    prior_range = daily_range.shift(1)
    df["mp_single_print_proxy"] = (daily_range > 2.0 * prior_range.replace(0, np.nan)).astype(int)

    # Tool 17: Poor High proxy – close near the day's high but wick short
    df["mp_poor_high"] = ((h - c) / daily_range.replace(0, np.nan) < 0.05).astype(int)

    # Tool 18: Poor Low proxy – close near the day's low
    df["mp_poor_low"]  = ((c - l) / daily_range.replace(0, np.nan) < 0.05).astype(int)

    # Tool 19: Balanced Profile proxy – price oscillates near VWAP20 (low ADX)
    df["mp_balanced"] = ((df["stack67_adx14"] < 20) &
                          (c.between(df["mp_val_low"], df["mp_val_high"]))).astype(int)

    # Tool 20: Trend Profile proxy – price consistently above/below VWAP20
    above_vwap = (c > vwap20).astype(int)
    df["mp_trend_profile_bull"] = above_vwap.rolling(5, min_periods=1).sum() >= 4
    df["mp_trend_profile_bear"] = above_vwap.rolling(5, min_periods=1).sum() <= 1
    df["mp_trend_profile_bull"] = df["mp_trend_profile_bull"].astype(int)
    df["mp_trend_profile_bear"] = df["mp_trend_profile_bear"].astype(int)

    # ------------------------------------------------------------------
    # SECTION D  –  Footprint / delta proxies (lines 1047-1134)
    # ------------------------------------------------------------------
    # True footprint requires L2/tick data. Proxies built from daily OHLCV.

    # Tool 21: Volume Footprint – close position within bar as buy/sell proxy
    # (c - l) / (h - l) = proportion of range that is "buy pressure"
    df["fp_buy_pressure_pct"] = (c - l) / daily_range.replace(0, np.nan)
    df["fp_sell_pressure_pct"] = 1 - df["fp_buy_pressure_pct"]

    # Tool 22: Bid×Ask Footprint proxy – aggressive buy = close near high on up-day
    up_day = (c > o).astype(int)
    close_near_high = ((h - c) / daily_range.replace(0, np.nan) < 0.25).astype(int)
    close_near_low  = ((c - l) / daily_range.replace(0, np.nan) < 0.25).astype(int)
    df["fp_aggressive_buy"]  = (up_day & close_near_high).astype(int)
    df["fp_aggressive_sell"] = ((1 - up_day) & close_near_low).astype(int)

    # Tool 23: Volume Delta proxy (lines 1070-1079)
    # Estimate buy/sell volume from Kăo/OHLC decomposition
    buy_vol  = v * df["fp_buy_pressure_pct"]
    sell_vol = v * df["fp_sell_pressure_pct"]
    df["fp_vol_delta"]     = buy_vol - sell_vol
    df["fp_vol_delta_pct"] = df["fp_vol_delta"] / v.replace(0, np.nan)

    # Tool 24: Cumulative Volume Delta – running sum of vol delta (lines 1081-1090)
    df["fp_cvd"]         = df["fp_vol_delta"].cumsum()
    df["fp_cvd_20d_roc"] = _roc(df["fp_cvd"], 20)

    # Tool 25: Delta Divergence (lines 1092-1101)
    # Price new high but CVD not confirming (or vice versa)
    price_new_high_20 = (c >= _rolling_high(c, 20)).astype(int)
    cvd_new_high_20   = (df["fp_cvd"] >= _rolling_high(df["fp_cvd"], 20)).astype(int)
    price_new_low_20  = (c <= _rolling_low(c, 20)).astype(int)
    cvd_new_low_20    = (df["fp_cvd"] <= _rolling_low(df["fp_cvd"], 20)).astype(int)
    df["fp_delta_bearish_div"] = (price_new_high_20 & (1 - cvd_new_high_20)).astype(int)
    df["fp_delta_bullish_div"] = (price_new_low_20  & (1 - cvd_new_low_20)).astype(int)

    # Tool 26: Footprint Imbalance proxy (lines 1103-1112)
    # Day's buy/sell ratio vs prior 5-day average ratio
    avg_delta5 = df["fp_vol_delta_pct"].rolling(5, min_periods=1).mean()
    df["fp_imbalance_buy"]  = (df["fp_vol_delta_pct"] >  avg_delta5 + 0.2).astype(int)
    df["fp_imbalance_sell"] = (df["fp_vol_delta_pct"] < avg_delta5 - 0.2).astype(int)

    # Tool 27: Stacked Imbalances proxy (lines 1114-1123)
    # Multiple consecutive days of buy/sell imbalance
    df["fp_stacked_buy_imbalance"]  = df["fp_imbalance_buy"].rolling(3, min_periods=1).sum() >= 2
    df["fp_stacked_sell_imbalance"] = df["fp_imbalance_sell"].rolling(3, min_periods=1).sum() >= 2
    df["fp_stacked_buy_imbalance"]  = df["fp_stacked_buy_imbalance"].astype(int)
    df["fp_stacked_sell_imbalance"] = df["fp_stacked_sell_imbalance"].astype(int)

    # Tool 28: Unfinished Auction proxy (lines 1125-1134)
    # Price closes at extreme = incomplete auction; will likely revisit
    df["fp_unfinished_high"] = df["mp_poor_high"]  # close at/near day high
    df["fp_unfinished_low"]  = df["mp_poor_low"]   # close at/near day low

    # ------------------------------------------------------------------
    # SECTION E  –  Additional trend and momentum indicators (lines 1136-1289)
    # ------------------------------------------------------------------

    # Tool 29: Supertrend (lines 1137-1146)
    if _HAS_TA:
        _st = ta.supertrend(h, l, c, length=7, multiplier=3.0)
        if _st is not None and not _st.empty:
            st_dir_cols = [col for col in _st.columns if "SUPERTd" in col.upper() or "SUPERT_" in col.upper()]
            st_val_cols = [col for col in _st.columns if "SUPERT_" in col and "d" not in col and "l" not in col and "s" not in col]
            if st_dir_cols:
                df["supert_direction"] = _st[st_dir_cols[0]].values  # 1=bull, -1=bear
            else:
                df["supert_direction"] = np.nan
        else:
            df["supert_direction"] = np.nan
    else:
        # Manual Supertrend (ATR multiplier 3, period 7)
        atr7 = tr_raw.ewm(com=6, adjust=False, min_periods=7).mean()
        hl2 = (h + l) / 2
        upper_band = hl2 + 3 * atr7
        lower_band = hl2 - 3 * atr7
        supertrend = pd.Series(np.nan, index=df.index)
        direction  = pd.Series(0, index=df.index)
        for i in range(1, len(df)):
            prev_st = supertrend.iloc[i - 1] if not np.isnan(supertrend.iloc[i - 1]) else lower_band.iloc[i]
            if c.iloc[i] > prev_st:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i]  = 1
            else:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i]  = -1
        df["supert_direction"] = direction

    # Tool 30: CCI(20) (lines 1148-1157)
    if _HAS_TA:
        _cci = ta.cci(h, l, c, length=20)
        df["cci20"] = _cci.values if _cci is not None else np.nan
    else:
        tp = (h + l + c) / 3
        tp_mean = tp.rolling(20, min_periods=5).mean()
        tp_mad  = tp.rolling(20, min_periods=5).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
        df["cci20"] = (tp - tp_mean) / (0.015 * tp_mad.replace(0, np.nan))
    df["cci20_overbought"]  = (df["cci20"] > 100).astype(int)
    df["cci20_oversold"]    = (df["cci20"] < -100).astype(int)

    # Tool 31: Williams %R(14) (lines 1159-1168)
    if _HAS_TA:
        _willr = ta.willr(h, l, c, length=14)
        df["willr14"] = _willr.values if _willr is not None else np.nan
    else:
        hh14 = _rolling_high(h, 14)
        ll14 = _rolling_low(l, 14)
        df["willr14"] = -100 * (hh14 - c) / (hh14 - ll14).replace(0, np.nan)
    df["willr14_overbought"] = (df["willr14"] > -20).astype(int)
    df["willr14_oversold"]   = (df["willr14"] < -80).astype(int)

    # Tool 32/33: Aroon(25) + Aroon Oscillator (lines 1170-1190)
    if _HAS_TA:
        _aroon = ta.aroon(h, l, length=25)
        if _aroon is not None and not _aroon.empty:
            aroon_up_col   = [c2 for c2 in _aroon.columns if c2.upper().startswith("AROONU")]
            aroon_dn_col   = [c2 for c2 in _aroon.columns if c2.upper().startswith("AROOND")]
            aroon_osc_col  = [c2 for c2 in _aroon.columns if "OSC" in c2.upper()]
            df["aroon_up25"]  = _aroon[aroon_up_col[0]].values  if aroon_up_col  else np.nan
            df["aroon_dn25"]  = _aroon[aroon_dn_col[0]].values  if aroon_dn_col  else np.nan
            df["aroon_osc25"] = _aroon[aroon_osc_col[0]].values if aroon_osc_col else np.nan
        else:
            df["aroon_up25"] = df["aroon_dn25"] = df["aroon_osc25"] = np.nan
    else:
        n = 25
        aroon_up = pd.Series(np.nan, index=df.index)
        aroon_dn = pd.Series(np.nan, index=df.index)
        for i in range(n, len(df)):
            window_h = h.iloc[i - n:i + 1]
            window_l = l.iloc[i - n:i + 1]
            aroon_up.iloc[i] = 100 * (n - (n - window_h.values.argmax())) / n
            aroon_dn.iloc[i] = 100 * (n - (n - window_l.values.argmin())) / n
        df["aroon_up25"]  = aroon_up
        df["aroon_dn25"]  = aroon_dn
        df["aroon_osc25"] = aroon_up - aroon_dn

    # Tool 34: Vortex Indicator(14) (lines 1192-1201)
    if _HAS_TA:
        _vortex = ta.vortex(h, l, c, length=14)
        if _vortex is not None and not _vortex.empty:
            vi_plus_col  = [c2 for c2 in _vortex.columns if "VTXP" in c2.upper() or "VI+" in c2 or "VIP" in c2.upper()]
            vi_minus_col = [c2 for c2 in _vortex.columns if "VTXM" in c2.upper() or "VI-" in c2 or "VIM" in c2.upper()]
            df["vortex_plus14"]  = _vortex[vi_plus_col[0]].values  if vi_plus_col  else np.nan
            df["vortex_minus14"] = _vortex[vi_minus_col[0]].values if vi_minus_col else np.nan
            df["vortex_bull"]    = (df["vortex_plus14"] > df["vortex_minus14"]).astype(int) if vi_plus_col and vi_minus_col else np.nan
        else:
            df["vortex_plus14"] = df["vortex_minus14"] = df["vortex_bull"] = np.nan
    else:
        tr14 = tr_raw.rolling(14, min_periods=1).sum()
        vm_plus  = (h - l.shift(1)).abs().rolling(14, min_periods=1).sum()
        vm_minus = (l - h.shift(1)).abs().rolling(14, min_periods=1).sum()
        df["vortex_plus14"]  = vm_plus  / tr14.replace(0, np.nan)
        df["vortex_minus14"] = vm_minus / tr14.replace(0, np.nan)
        df["vortex_bull"]    = (df["vortex_plus14"] > df["vortex_minus14"]).astype(int)

    # Tool 35: Ultimate Oscillator (lines 1203-1212)
    if _HAS_TA:
        _uo = ta.uo(h, l, c)
        df["uo"] = _uo.values if _uo is not None else np.nan
    else:
        # Manual UO: periods 7, 14, 28
        bp = c - pd.concat([l, c.shift(1)], axis=1).min(axis=1)
        tr_uo = pd.concat([h, c.shift(1)], axis=1).max(axis=1) - pd.concat([l, c.shift(1)], axis=1).min(axis=1)
        avg7  = bp.rolling(7,  min_periods=1).sum() / tr_uo.rolling(7,  min_periods=1).sum().replace(0, np.nan)
        avg14 = bp.rolling(14, min_periods=1).sum() / tr_uo.rolling(14, min_periods=1).sum().replace(0, np.nan)
        avg28 = bp.rolling(28, min_periods=1).sum() / tr_uo.rolling(28, min_periods=1).sum().replace(0, np.nan)
        df["uo"] = 100 * (4 * avg7 + 2 * avg14 + avg28) / 7
    df["uo_overbought"] = (df["uo"] > 70).astype(int)
    df["uo_oversold"]   = (df["uo"] < 30).astype(int)

    # Tool 36: True Strength Index (lines 1214-1223)
    if _HAS_TA:
        _tsi = ta.tsi(c)
        if _tsi is not None and not _tsi.empty:
            tsi_col = [c2 for c2 in _tsi.columns if "TSI" in c2.upper()]
            df["tsi"] = _tsi[tsi_col[0]].values if tsi_col else np.nan
        else:
            df["tsi"] = np.nan
    else:
        pc = c.diff()
        double_smooth = pc.ewm(span=25, adjust=False).mean().ewm(span=13, adjust=False).mean()
        double_smooth_abs = pc.abs().ewm(span=25, adjust=False).mean().ewm(span=13, adjust=False).mean()
        df["tsi"] = 100 * double_smooth / double_smooth_abs.replace(0, np.nan)
    df["tsi_bull"] = (df["tsi"] > 0).astype(int)

    # Tool 37: TRIX(15) (lines 1225-1234)
    if _HAS_TA:
        _trix = ta.trix(c, length=15)
        if _trix is not None and not _trix.empty:
            trix_col = [c2 for c2 in _trix.columns if "TRIX" in c2.upper() and "S" not in c2.upper()[-1:]]
            df["trix15"] = _trix[trix_col[0]].values if trix_col else np.nan
        else:
            df["trix15"] = np.nan
    else:
        e1 = _ema(c, 15)
        e2 = _ema(e1, 15)
        e3 = _ema(e2, 15)
        df["trix15"] = e3.pct_change() * 100
    df["trix15_bull"] = (df["trix15"] > 0).astype(int)

    # Tool 38: Percentage Price Oscillator (lines 1236-1245)
    if _HAS_TA:
        _ppo = ta.ppo(c)
        if _ppo is not None and not _ppo.empty:
            ppo_col = [c2 for c2 in _ppo.columns if "PPO" in c2.upper() and "HIST" not in c2.upper() and "SIGNAL" not in c2.upper() and "SIG" not in c2.upper()]
            df["ppo"] = _ppo[ppo_col[0]].values if ppo_col else np.nan
        else:
            df["ppo"] = np.nan
    else:
        ema12_c = _ema(c, 12)
        ema26_c = _ema(c, 26)
        df["ppo"] = 100 * (ema12_c - ema26_c) / ema26_c.replace(0, np.nan)
    df["ppo_bull"] = (df["ppo"] > 0).astype(int)

    # Tool 39: Know Sure Thing (lines 1247-1256)
    if _HAS_TA:
        _kst = ta.kst(c)
        if _kst is not None and not _kst.empty:
            kst_col = [c2 for c2 in _kst.columns if "KST" in c2.upper() and "S" not in c2.upper()[-1:]]
            df["kst"] = _kst[kst_col[0]].values if kst_col else np.nan
        else:
            df["kst"] = np.nan
    else:
        # KST: sum of 4 ROC SMAs, each multiplied by weight
        rcma1 = _sma(_roc(c, 10) * 100, 10)
        rcma2 = _sma(_roc(c, 13) * 100, 13)
        rcma3 = _sma(_roc(c, 15) * 100, 15)
        rcma4 = _sma(_roc(c, 20) * 100, 20)
        df["kst"] = rcma1 * 1 + rcma2 * 2 + rcma3 * 3 + rcma4 * 4
    df["kst_bull"] = (df["kst"] > 0).astype(int)

    # Tool 40: Detrended Price Oscillator(20) (lines 1258-1267)
    if _HAS_TA:
        _dpo = ta.dpo(c, length=20)
        df["dpo20"] = _dpo.values if _dpo is not None else np.nan
    else:
        # DPO = close[-(n/2+1)] - SMA(n)
        shift_n = 20 // 2 + 1
        df["dpo20"] = c.shift(shift_n) - _sma(c, 20)
    df["dpo20_bull"] = (df["dpo20"] > 0).astype(int)

    # Tool 41: Chande Momentum Oscillator(14) (lines 1269-1278)
    if _HAS_TA:
        _cmo = ta.cmo(c, length=14)
        df["cmo14"] = _cmo.values if _cmo is not None else np.nan
    else:
        diff = c.diff()
        up14   = diff.clip(lower=0).rolling(14, min_periods=1).sum()
        down14 = (-diff).clip(lower=0).rolling(14, min_periods=1).sum()
        df["cmo14"] = 100 * (up14 - down14) / (up14 + down14).replace(0, np.nan)
    df["cmo14_bull"] = (df["cmo14"] > 0).astype(int)

    # Tool 42: Connors RSI (lines 1280-1289)
    # ConnorsRSI = (RSI3 + RSI(StreakRSI,2) + PercentRank_100) / 3
    # RSI3
    d3 = c.diff()
    g3 = d3.clip(lower=0).ewm(com=2, adjust=False, min_periods=3).mean()
    l3 = (-d3).clip(lower=0).ewm(com=2, adjust=False, min_periods=3).mean()
    rsi3 = 100 - 100 / (1 + g3 / l3.replace(0, np.nan))
    # Streak: consecutive up/down days
    streak = pd.Series(0, index=df.index)
    for i in range(1, len(df)):
        if c.iloc[i] > c.iloc[i - 1]:
            streak.iloc[i] = max(streak.iloc[i - 1], 0) + 1
        elif c.iloc[i] < c.iloc[i - 1]:
            streak.iloc[i] = min(streak.iloc[i - 1], 0) - 1
    # RSI(2) of streak
    ds = streak.diff()
    gs = ds.clip(lower=0).ewm(com=1, adjust=False, min_periods=2).mean()
    ls = (-ds).clip(lower=0).ewm(com=1, adjust=False, min_periods=2).mean()
    streak_rsi = 100 - 100 / (1 + gs / ls.replace(0, np.nan))
    # 100-period percent rank of 1-day ROC
    roc1 = c.pct_change()
    pct_rank = roc1.rolling(100, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    )
    df["connors_rsi"] = (rsi3 + streak_rsi + pct_rank) / 3
    df["connors_rsi_oversold"]   = (df["connors_rsi"] < 10).astype(int)
    df["connors_rsi_overbought"] = (df["connors_rsi"] > 90).astype(int)
    df["streak"] = streak

    # ------------------------------------------------------------------
    # SECTION F  –  Additional moving-average tools (lines 1291-1357)
    # ------------------------------------------------------------------

    # Tool 43: Hull Moving Average(20) (lines 1292-1301)
    if _HAS_TA:
        _hma = ta.hma(c, length=20)
        df["hma20"] = _hma.values if _hma is not None else np.nan
    else:
        wma_half = _sma(c, 10)  # simplified; true HMA uses WMA
        wma_full = _sma(c, 20)
        raw_hma  = 2 * wma_half - wma_full
        df["hma20"] = _sma(raw_hma, int(20 ** 0.5))
    df["hma20_bull"] = (c > df["hma20"]).astype(int)
    df["hma20_slope"] = df["hma20"].diff()

    # Tool 44: Kaufman Adaptive Moving Average(10) (lines 1303-1312)
    if _HAS_TA:
        _kama = ta.kama(c, length=10)
        df["kama10"] = _kama.values if _kama is not None else np.nan
    else:
        # Manual KAMA
        n = 10
        fast_sc = 2 / (2 + 1)
        slow_sc = 2 / (30 + 1)
        kama_vals = c.copy().astype(float)
        for i in range(n, len(c)):
            direction = abs(c.iloc[i] - c.iloc[i - n])
            volatility = sum(abs(c.iloc[i - j] - c.iloc[i - j - 1]) for j in range(n))
            er = direction / volatility if volatility != 0 else 0
            sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
            kama_vals.iloc[i] = kama_vals.iloc[i - 1] + sc * (c.iloc[i] - kama_vals.iloc[i - 1])
        df["kama10"] = kama_vals
    df["kama10_bull"]  = (c > df["kama10"]).astype(int)
    df["kama10_slope"] = df["kama10"].diff()

    # Tool 45: Volume Weighted Moving Average(20) (lines 1314-1323)
    if _HAS_TA:
        _vwma = ta.vwma(c, v, length=20)
        df["vwma20"] = _vwma.values if _vwma is not None else np.nan
    else:
        pv20 = (c * v).rolling(20, min_periods=1).sum()
        v20  = v.rolling(20, min_periods=1).sum()
        df["vwma20"] = pv20 / v20.replace(0, np.nan)
    df["vwma20_bull"]     = (c > df["vwma20"]).astype(int)
    df["price_vs_vwma20"] = (c - df["vwma20"]) / df["vwma20"].replace(0, np.nan)

    # Tool 46: Double EMA(20) (lines 1325-1334)
    if _HAS_TA:
        _dema = ta.dema(c, length=20)
        df["dema20"] = _dema.values if _dema is not None else np.nan
    else:
        e1 = _ema(c, 20)
        df["dema20"] = 2 * e1 - _ema(e1, 20)
    df["dema20_bull"] = (c > df["dema20"]).astype(int)

    # Tool 47: Triple EMA(20) (lines 1336-1345)
    if _HAS_TA:
        _tema = ta.tema(c, length=20)
        df["tema20"] = _tema.values if _tema is not None else np.nan
    else:
        e1 = _ema(c, 20)
        e2 = _ema(e1, 20)
        e3 = _ema(e2, 20)
        df["tema20"] = 3 * e1 - 3 * e2 + e3
    df["tema20_bull"] = (c > df["tema20"]).astype(int)

    # Tool 48: Guppy Multiple Moving Average (lines 1347-1356)
    # Short-term group: 3, 5, 8, 10, 12, 15
    # Long-term group:  30, 35, 40, 45, 50, 60
    st_periods = [3, 5, 8, 10, 12, 15]
    lt_periods = [30, 35, 40, 45, 50, 60]
    st_emas = [_ema(c, n) for n in st_periods]
    lt_emas = [_ema(c, n) for n in lt_periods]
    st_spread = pd.concat(st_emas, axis=1).max(axis=1) - pd.concat(st_emas, axis=1).min(axis=1)
    lt_spread = pd.concat(lt_emas, axis=1).max(axis=1) - pd.concat(lt_emas, axis=1).min(axis=1)
    st_avg = pd.concat(st_emas, axis=1).mean(axis=1)
    lt_avg = pd.concat(lt_emas, axis=1).mean(axis=1)
    df["guppy_st_spread"]    = st_spread / c.replace(0, np.nan)   # normalised short-term fan width
    df["guppy_lt_spread"]    = lt_spread / c.replace(0, np.nan)   # normalised long-term fan width
    df["guppy_aligned_bull"] = (st_avg > lt_avg).astype(int)      # short-term above long-term
    df["guppy_expanding"]    = (st_spread > st_spread.shift(3)).astype(int)  # spread expanding

    # ------------------------------------------------------------------
    # SECTION G  –  Volatility / squeeze tools (lines 1358-1424)
    # ------------------------------------------------------------------

    # Tool 49: TTM Squeeze (lines 1359-1368)
    if _HAS_TA:
        _sqz = ta.squeeze(h, l, c, bb_length=20, bb_std=2, kc_length=20, kc_scalar=1.5)
        if _sqz is not None and not _sqz.empty:
            # SQZ column (momentum) and SQZ_ON (squeeze active)
            sqz_mom_col = [c2 for c2 in _sqz.columns if "SQZ" in c2.upper() and "ON" not in c2.upper() and "OFF" not in c2.upper() and "NO" not in c2.upper()]
            sqz_on_col  = [c2 for c2 in _sqz.columns if "SQZ_ON" in c2.upper()]
            df["ttm_sqz_momentum"] = _sqz[sqz_mom_col[0]].values if sqz_mom_col else np.nan
            df["ttm_sqz_on"]       = _sqz[sqz_on_col[0]].values  if sqz_on_col  else np.nan
        else:
            df["ttm_sqz_momentum"] = df["ttm_sqz_on"] = np.nan
    else:
        # Manual TTM Squeeze: BB inside KC = squeeze on
        bb_std20 = _stdev(c, 20)
        bb_upper = sma20 + 2 * bb_std20
        bb_lower = sma20 - 2 * bb_std20
        kc_atr20 = tr_raw.ewm(com=19, adjust=False, min_periods=20).mean()
        kc_upper = sma20 + 1.5 * kc_atr20
        kc_lower = sma20 - 1.5 * kc_atr20
        df["ttm_sqz_on"] = ((bb_upper < kc_upper) & (bb_lower > kc_lower)).astype(int)
        # Momentum: linear regression of (close - midpoint of high/low/open/close)
        mp_val = ((_rolling_high(h, 20) + _rolling_low(l, 20)) / 2 + sma20) / 2
        df["ttm_sqz_momentum"] = (c - mp_val).rolling(5, min_periods=1).mean()
    df["ttm_sqz_fired"] = ((df["ttm_sqz_on"].shift(1) == 1) & (df["ttm_sqz_on"] == 0)).astype(int)  # squeeze release

    # Tool 50: Bollinger %B(20,2) (lines 1370-1379)
    bb_std20   = _stdev(c, 20)
    bb_upper20 = sma20 + 2 * bb_std20
    bb_lower20 = sma20 - 2 * bb_std20
    df["bb_pctb"] = (c - bb_lower20) / (bb_upper20 - bb_lower20).replace(0, np.nan)
    df["bb_above_upper"] = (df["bb_pctb"] > 1.0).astype(int)
    df["bb_below_lower"] = (df["bb_pctb"] < 0.0).astype(int)

    # Tool 51: Bollinger BandWidth(20,2) (lines 1381-1390)
    df["bb_bandwidth"]   = (bb_upper20 - bb_lower20) / sma20.replace(0, np.nan)
    bw_min_125 = df["bb_bandwidth"].rolling(125, min_periods=20).min()
    df["bb_squeeze_new_low"] = (df["bb_bandwidth"] <= bw_min_125).astype(int)
    df["bb_width_expanding"] = (df["bb_bandwidth"] > df["bb_bandwidth"].shift(5)).astype(int)

    # Tool 52: Historical Volatility(21) (lines 1392-1401)
    if _HAS_TA:
        _hv = ta.hvol(c, length=21)
        df["hvol21"] = _hv.values if _hv is not None else np.nan
    else:
        log_ret = np.log(c / c.shift(1))
        df["hvol21"] = log_ret.rolling(21, min_periods=5).std() * np.sqrt(252) * 100
    hv_mean63 = df["hvol21"].rolling(63, min_periods=10).mean()
    df["hvol21_vs_avg"] = df["hvol21"] / hv_mean63.replace(0, np.nan)  # >1 = elevated volatility
    df["hvol21_low"]    = (df["hvol21_vs_avg"] < 0.75).astype(int)
    df["hvol21_high"]   = (df["hvol21_vs_avg"] > 1.50).astype(int)

    # Tool 53: Average Daily Range proxy(14) (lines 1403-1412)
    df["adr14"] = daily_range.rolling(14, min_periods=5).mean()
    df["adr14_pct"] = df["adr14"] / c.replace(0, np.nan) * 100  # % of price
    # "already stretched" = today's range has used more than 80% of ADR
    df["adr_stretched"] = (daily_range > 0.8 * df["adr14"]).astype(int)
    df["adr_room_left_pct"] = ((df["adr14"] - (c - l)) / df["adr14"].replace(0, np.nan)).clip(0, 1)

    # Tool 54: Opening Range Projection proxy(lines 1414-1424)
    # PROXY using daily data: opening range = |open - prior close| as "gap range"
    # Projected targets = open ± N × OR
    or_range = (o - c.shift(1)).abs()
    df["orp_range"]      = or_range
    df["orp_target_up"]  = o + or_range        # 1× projection
    df["orp_target_dn"]  = o - or_range
    df["orp_target_2x"]  = o + 2 * or_range   # 2× extension
    df["orp_close_hit_target"] = (c >= df["orp_target_up"]).astype(int)
    df["orp_or_pct"]     = or_range / c.replace(0, np.nan) * 100

    return df


# ---------------------------------------------------------------------------
# __main__ test block
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import pandas as pd
    import numpy as np

    np.random.seed(42)
    n = 300
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(np.random.randn(n) * 1.2)
    open_ = close - np.random.randn(n) * 0.5
    high  = np.maximum(close, open_) + np.abs(np.random.randn(n) * 0.8)
    low   = np.minimum(close, open_) - np.abs(np.random.randn(n) * 0.8)
    vol   = (np.abs(np.random.randn(n)) + 1) * 1_000_000

    df_test = pd.DataFrame({
        "open": open_, "high": high, "low": low,
        "close": close, "volume": vol
    }, index=dates)

    result = add_features_part2(df_test)
    new_cols = [col for col in result.columns if col not in df_test.columns]

    print(f"Input rows      : {len(df_test)}")
    print(f"Output rows     : {len(result)}")
    print(f"New columns     : {len(new_cols)}")
    print(f"No NaN-only cols: {sum(result[c].notna().any() for c in new_cols)} / {len(new_cols)}")
    print()
    print("Sample columns (last 5 rows):")
    sample_cols = ["ha_trend", "renko_brick_ratio", "supert_direction",
                   "cci20", "willr14", "aroon_osc25", "vortex_bull",
                   "uo", "tsi", "trix15", "ppo", "kst", "dpo20",
                   "cmo14", "connors_rsi", "hma20_bull", "kama10_bull",
                   "vwma20_bull", "dema20_bull", "tema20_bull", "guppy_aligned_bull",
                   "ttm_sqz_on", "ttm_sqz_fired", "bb_pctb", "bb_bandwidth",
                   "hvol21", "adr14_pct", "orp_target_up", "rule_agreement_score"]
    sample_cols_present = [c for c in sample_cols if c in result.columns]
    print(result[sample_cols_present].tail(5).to_string())
    print()
    print("All new column names:")
    for i, col in enumerate(new_cols, 1):
        print(f"  {i:3d}. {col}")
