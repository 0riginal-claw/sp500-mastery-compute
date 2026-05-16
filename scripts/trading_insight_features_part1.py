"""
trading_insight_features_part1.py

Features derived from lines 1-712 of 'Trading Insight Info.txt'.
Covers: RSI, MACD, Stochastic, StochRSI, ADX/DMI, ATR, Relative Volume,
OBV, Accumulation/Distribution, Chaikin Money Flow, MFI, Rate-of-Change,
Momentum, Standard Deviation, market-internals proxies (A/D breadth,
TICK, TRIN, Up/Down vol, New Highs/Lows, VIX), relative-strength proxies,
sector-comparison proxies, options proxies (IV, P/C ratio, Gamma),
scanner-style flags (gap, float-liquidity, large-prints proxy), and
risk/trade-management metrics (ATR-based position sizing, drawdown,
R-multiple, profit-factor, expectancy, rolling Sharpe/Sortino,
max-drawdown, daily-loss-limit flag, time-stop proxy, setup quality).

All features use .shift(1) on final outputs so bar t relies only on
data through bar t-1 (point-in-time safe).

Input DataFrame requirements
-----------------------------
- DatetimeIndex (any timezone)
- Columns: open, high, low, close, volume  (lowercase)
"""

import math
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder smoothing (used in RSI, ATR, ADX)."""
    return series.ewm(alpha=1.0 / period, adjust=False).mean()


def _true_range(df: pd.DataFrame) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"] - df["close"].shift(1)).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1)


# ---------------------------------------------------------------------------
# Main feature function
# ---------------------------------------------------------------------------

def add_features_part1(df: pd.DataFrame) -> pd.DataFrame:
    """All features from lines 1-712 of Trading Insight Info.

    Input df must have: open, high, low, close, volume columns + datetime index.
    All features are point-in-time safe (use .shift(1) on outputs).
    Returns df with new columns added.
    """
    df = df.copy()
    c = df["close"]
    o = df["open"]
    h = df["high"]
    lo = df["low"]
    v = df["volume"]

    # -----------------------------------------------------------------------
    # 1. RSI (lines 9-18)
    # Momentum oscillator 0-100; >70 overbought, <30 oversold.
    # -----------------------------------------------------------------------
    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = _rma(gain, period)
        avg_loss = _rma(loss, period)
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - 100 / (1 + rs)

    rsi14 = _rsi(c, 14)
    df["feat_rsi_14"] = rsi14.shift(1)
    df["feat_rsi_ob"] = (rsi14 > 70).astype(int).shift(1)   # overbought flag
    df["feat_rsi_os"] = (rsi14 < 30).astype(int).shift(1)   # oversold flag

    # RSI divergence proxy: price made new 20-bar high but RSI did not
    price_new_high_20 = (c == c.rolling(20).max()).astype(int)
    rsi_new_high_20 = (rsi14 == rsi14.rolling(20).max()).astype(int)
    df["feat_rsi_bear_div"] = ((price_new_high_20 == 1) & (rsi_new_high_20 == 0)).astype(int).shift(1)

    price_new_low_20 = (c == c.rolling(20).min()).astype(int)
    rsi_new_low_20 = (rsi14 == rsi14.rolling(20).min()).astype(int)
    df["feat_rsi_bull_div"] = ((price_new_low_20 == 1) & (rsi_new_low_20 == 0)).astype(int).shift(1)

    # -----------------------------------------------------------------------
    # 2. MACD (lines 20-29)
    # Two EMAs + histogram; crossovers and histogram sign shifts signal momentum.
    # -----------------------------------------------------------------------
    ema12 = _ema(c, 12)
    ema26 = _ema(c, 26)
    macd_line = ema12 - ema26
    signal_line = _ema(macd_line, 9)
    macd_hist = macd_line - signal_line

    df["feat_macd_line"] = macd_line.shift(1)
    df["feat_macd_signal"] = signal_line.shift(1)
    df["feat_macd_hist"] = macd_hist.shift(1)
    # Bullish crossover: MACD crossed above signal on previous bar
    df["feat_macd_bull_cross"] = ((macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))).astype(int).shift(1)
    df["feat_macd_bear_cross"] = ((macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))).astype(int).shift(1)
    # Histogram direction change (momentum shift)
    df["feat_macd_hist_turn_up"] = ((macd_hist > macd_hist.shift(1)) & (macd_hist.shift(1) < macd_hist.shift(2))).astype(int).shift(1)

    # MACD divergence: price new 20-bar high but MACD line did not
    macd_new_high_20 = (macd_line == macd_line.rolling(20).max()).astype(int)
    df["feat_macd_bear_div"] = ((price_new_high_20 == 1) & (macd_new_high_20 == 0)).astype(int).shift(1)

    # -----------------------------------------------------------------------
    # 3. Stochastic Oscillator (lines 31-40)
    # %K = (close - low_n) / (high_n - low_n) * 100; %D = SMA(%K, 3).
    # -----------------------------------------------------------------------
    stoch_n = 14
    low_n = lo.rolling(stoch_n).min()
    high_n = h.rolling(stoch_n).max()
    stoch_k = 100 * (c - low_n) / (high_n - low_n).replace(0, np.nan)
    stoch_d = stoch_k.rolling(3).mean()

    df["feat_stoch_k"] = stoch_k.shift(1)
    df["feat_stoch_d"] = stoch_d.shift(1)
    df["feat_stoch_ob"] = (stoch_k > 80).astype(int).shift(1)
    df["feat_stoch_os"] = (stoch_k < 20).astype(int).shift(1)
    # %K crossing above %D in oversold zone
    df["feat_stoch_bull_cross"] = ((stoch_k > stoch_d) & (stoch_k.shift(1) <= stoch_d.shift(1)) & (stoch_k < 30)).astype(int).shift(1)

    # -----------------------------------------------------------------------
    # 4. StochRSI (lines 42-51)
    # Applies Stochastic formula to RSI values; faster overbought/oversold.
    # -----------------------------------------------------------------------
    stochrsi_n = 14
    rsi14_min = rsi14.rolling(stochrsi_n).min()
    rsi14_max = rsi14.rolling(stochrsi_n).max()
    stochrsi = (rsi14 - rsi14_min) / (rsi14_max - rsi14_min).replace(0, np.nan)
    stochrsi_k = stochrsi * 100
    stochrsi_d = stochrsi_k.rolling(3).mean()

    df["feat_stochrsi_k"] = stochrsi_k.shift(1)
    df["feat_stochrsi_d"] = stochrsi_d.shift(1)
    df["feat_stochrsi_ob"] = (stochrsi_k > 80).astype(int).shift(1)
    df["feat_stochrsi_os"] = (stochrsi_k < 20).astype(int).shift(1)

    # -----------------------------------------------------------------------
    # 5 & 6. ADX and DMI (lines 53-73)
    # ADX measures trend strength (directionless); +DI vs -DI gives direction.
    # ADX > 25 = trending, < 20 = chopping.
    # -----------------------------------------------------------------------
    adx_period = 14
    tr = _true_range(df)
    up_move = h.diff()
    down_move = -(lo.diff())
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm_s = pd.Series(plus_dm, index=df.index)
    minus_dm_s = pd.Series(minus_dm, index=df.index)

    atr14 = _rma(tr, adx_period)
    plus_di = 100 * _rma(plus_dm_s, adx_period) / atr14.replace(0, np.nan)
    minus_di = 100 * _rma(minus_dm_s, adx_period) / atr14.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = _rma(dx, adx_period)

    df["feat_adx"] = adx.shift(1)
    df["feat_plus_di"] = plus_di.shift(1)
    df["feat_minus_di"] = minus_di.shift(1)
    df["feat_adx_trending"] = (adx > 25).astype(int).shift(1)   # >25 = trending
    df["feat_adx_choppy"] = (adx < 20).astype(int).shift(1)     # <20 = choppy
    df["feat_dmi_bull"] = (plus_di > minus_di).astype(int).shift(1)
    df["feat_dmi_bear"] = (minus_di > plus_di).astype(int).shift(1)

    # -----------------------------------------------------------------------
    # 7. ATR (lines 75-84)
    # Average True Range: volatility measure used for stop sizing and target sizing.
    # -----------------------------------------------------------------------
    atr_vals = _rma(tr, 14)
    df["feat_atr_14"] = atr_vals.shift(1)
    # ATR as % of close (normalized)
    df["feat_atr_pct"] = (atr_vals / c.replace(0, np.nan) * 100).shift(1)
    # Price is stretched: more than 2 ATR from 20-day SMA
    sma20 = c.rolling(20).mean()
    df["feat_price_stretched"] = ((c - sma20).abs() > 2 * atr_vals).astype(int).shift(1)

    # -----------------------------------------------------------------------
    # 8. Relative Volume (lines 86-95)
    # Today's volume vs. rolling average; >1.5 = unusual attention.
    # -----------------------------------------------------------------------
    avg_vol_20 = v.rolling(20).mean()
    rel_vol = v / avg_vol_20.replace(0, np.nan)
    df["feat_rel_vol_20"] = rel_vol.shift(1)
    df["feat_high_rel_vol"] = (rel_vol > 1.5).astype(int).shift(1)
    df["feat_very_high_rel_vol"] = (rel_vol > 2.5).astype(int).shift(1)

    # -----------------------------------------------------------------------
    # 9. OBV (lines 97-106)
    # Cumulative volume added on up-days, subtracted on down-days.
    # OBV trend vs. price trend divergence is a signal.
    # -----------------------------------------------------------------------
    price_dir = np.sign(c.diff())
    obv = (price_dir * v).fillna(0).cumsum()
    df["feat_obv"] = obv.shift(1)
    # OBV z-score over 20 bars (normalised trend strength)
    obv_z = (obv - obv.rolling(20).mean()) / obv.rolling(20).std().replace(0, np.nan)
    df["feat_obv_z20"] = obv_z.shift(1)
    # OBV divergence: price new 20-bar high but OBV did not
    obv_new_high_20 = (obv == obv.rolling(20).max()).astype(int)
    df["feat_obv_bear_div"] = ((price_new_high_20 == 1) & (obv_new_high_20 == 0)).astype(int).shift(1)

    # -----------------------------------------------------------------------
    # 10. Accumulation / Distribution (lines 108-117)
    # Weighted volume-based line; adds or subtracts based on close position in range.
    # -----------------------------------------------------------------------
    clv = ((c - lo) - (h - c)) / (h - lo).replace(0, np.nan)  # close location value -1 to +1
    ad_line = (clv * v).fillna(0).cumsum()
    df["feat_ad_line"] = ad_line.shift(1)
    # A/D 14-day slope (rising = accumulation)
    ad_slope = ad_line.diff(14)
    df["feat_ad_slope_14"] = ad_slope.shift(1)

    # -----------------------------------------------------------------------
    # 11. Chaikin Money Flow (lines 119-128)
    # Sum of CLV*vol over N bars / sum of vol over N bars.
    # Positive = buying pressure, negative = selling pressure.
    # -----------------------------------------------------------------------
    cmf_period = 20
    cmf = (clv * v).rolling(cmf_period).sum() / v.rolling(cmf_period).sum().replace(0, np.nan)
    df["feat_cmf_20"] = cmf.shift(1)
    df["feat_cmf_bull"] = (cmf > 0).astype(int).shift(1)
    df["feat_cmf_bear"] = (cmf < 0).astype(int).shift(1)

    # -----------------------------------------------------------------------
    # 12. Money Flow Index (lines 130-139)
    # Volume-weighted RSI; >80 overbought, <20 oversold.
    # -----------------------------------------------------------------------
    typical_price = (h + lo + c) / 3
    raw_money_flow = typical_price * v
    tp_up = np.where(typical_price > typical_price.shift(1), raw_money_flow, 0.0)
    tp_dn = np.where(typical_price < typical_price.shift(1), raw_money_flow, 0.0)
    tp_up_s = pd.Series(tp_up, index=df.index)
    tp_dn_s = pd.Series(tp_dn, index=df.index)
    mf_ratio = tp_up_s.rolling(14).sum() / tp_dn_s.rolling(14).sum().replace(0, np.nan)
    mfi = 100 - 100 / (1 + mf_ratio)
    df["feat_mfi_14"] = mfi.shift(1)
    df["feat_mfi_ob"] = (mfi > 80).astype(int).shift(1)
    df["feat_mfi_os"] = (mfi < 20).astype(int).shift(1)

    # -----------------------------------------------------------------------
    # 13. Rate of Change (lines 141-150)
    # (close_t - close_{t-n}) / close_{t-n} * 100; acceleration/deceleration.
    # -----------------------------------------------------------------------
    roc10 = c.pct_change(10) * 100
    roc5 = c.pct_change(5) * 100
    df["feat_roc_10"] = roc10.shift(1)
    df["feat_roc_5"] = roc5.shift(1)
    # ROC acceleration: ROC_5 - ROC_5 one week ago
    df["feat_roc_accel"] = (roc5 - roc5.shift(5)).shift(1)

    # -----------------------------------------------------------------------
    # 14. Momentum Indicator (lines 152-161)
    # Raw price difference over N periods; confirms breakout force.
    # -----------------------------------------------------------------------
    mom10 = c - c.shift(10)
    df["feat_momentum_10"] = mom10.shift(1)
    df["feat_momentum_positive"] = (mom10 > 0).astype(int).shift(1)

    # -----------------------------------------------------------------------
    # 15. Standard Deviation (lines 163-172)
    # Rolling std of close; measures volatility and abnormal price movement.
    # -----------------------------------------------------------------------
    std20 = c.rolling(20).std()
    df["feat_std_20"] = std20.shift(1)
    # Normalised: std as % of rolling mean
    df["feat_std_pct_20"] = (std20 / sma20.replace(0, np.nan) * 100).shift(1)
    # Bollinger Band width (proxy for squeeze)
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    bb_width = (bb_upper - bb_lower) / sma20.replace(0, np.nan)
    df["feat_bb_width"] = bb_width.shift(1)
    # Price Z-score vs 20-bar mean
    c_zscore = (c - sma20) / std20.replace(0, np.nan)
    df["feat_close_zscore_20"] = c_zscore.shift(1)
    # Bollinger Band %B: position of close within bands
    bb_pct_b = (c - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)
    df["feat_bb_pct_b"] = bb_pct_b.shift(1)

    # -----------------------------------------------------------------------
    # 16-24. Order-flow tools proxies (lines 174-261)
    # Time & Sales, Level2, Bid/Ask, Order Book Imbalance, Tape Speed,
    # Large Prints, Absorption — all require tick/L2 data.
    # PROXY: use daily volume patterns as substitutes.
    # -----------------------------------------------------------------------

    # Large-prints proxy (line 230-239): volume spike on single bar
    vol_z = (v - v.rolling(20).mean()) / v.rolling(20).std().replace(0, np.nan)
    df["feat_large_print_proxy"] = vol_z.shift(1)  # high = possible large institutional print

    # Absorption proxy (lines 241-250): high volume but small price move
    daily_range_pct = ((h - lo) / c.replace(0, np.nan)) * 100
    df["feat_absorption_proxy"] = (vol_z / daily_range_pct.replace(0, np.nan)).shift(1)

    # Bid/ask spread proxy (lines 197-206): (high - low) / close as daily spread proxy
    daily_spread_proxy = (h - lo) / c.replace(0, np.nan)
    df["feat_spread_proxy"] = daily_spread_proxy.shift(1)

    # Tape speed proxy (lines 219-228): volume per unit price range (urgency)
    tape_speed_proxy = v / (h - lo + 0.001).replace(0, np.nan)
    df["feat_tape_speed_proxy"] = tape_speed_proxy.shift(1)

    # Order book imbalance proxy (lines 208-217):
    # Close above midpoint = buyer pressure; below = seller pressure
    midpoint = (h + lo) / 2
    book_imbal = (c - midpoint) / (h - lo + 0.001).replace(0, np.nan)   # -0.5 to +0.5
    df["feat_book_imbal_proxy"] = book_imbal.shift(1)

    # -----------------------------------------------------------------------
    # 25. Advance/Decline Line proxy (lines 275-284)
    # True A/D requires market-wide data. PROXY: rolling fraction of up-days.
    # -----------------------------------------------------------------------
    up_days = (c.diff() > 0).astype(int)
    df["feat_ad_breadth_20"] = up_days.rolling(20).mean().shift(1)  # 0-1 breadth proxy

    # -----------------------------------------------------------------------
    # 26. TICK proxy (lines 286-295)
    # Intraday up-tick minus down-tick. PROXY: intraday close vs open direction.
    # -----------------------------------------------------------------------
    tick_proxy = np.sign(c - o)
    df["feat_tick_proxy"] = tick_proxy.shift(1)
    df["feat_tick_cumsum_5"] = tick_proxy.rolling(5).sum().shift(1)

    # -----------------------------------------------------------------------
    # 27. TRIN / Arms Index proxy (lines 297-306)
    # TRIN = (adv/dec) / (adv_vol/dec_vol). PROXY: up-vol ratio.
    # -----------------------------------------------------------------------
    up_vol = np.where(c.diff() > 0, v, 0.0)
    dn_vol = np.where(c.diff() < 0, v, 0.0)
    up_vol_s = pd.Series(up_vol, index=df.index)
    dn_vol_s = pd.Series(dn_vol, index=df.index)
    # Ratio of up-volume to down-volume over 5 days
    trin_proxy = up_vol_s.rolling(5).sum() / dn_vol_s.rolling(5).sum().replace(0, np.nan)
    df["feat_trin_proxy_5"] = trin_proxy.shift(1)

    # -----------------------------------------------------------------------
    # 28. Up Volume vs Down Volume (lines 308-317)
    # -----------------------------------------------------------------------
    upvol_frac = up_vol_s.rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)
    df["feat_upvol_frac_20"] = upvol_frac.shift(1)

    # -----------------------------------------------------------------------
    # 29. New Highs / New Lows proxy (lines 319-328)
    # Requires cross-sectional data. PROXY: rolling 52-week high/low flags.
    # -----------------------------------------------------------------------
    df["feat_new_high_252"] = (c == c.rolling(252).max()).astype(int).shift(1)
    df["feat_new_low_252"] = (c == c.rolling(252).min()).astype(int).shift(1)
    df["feat_new_high_20"] = (c == c.rolling(20).max()).astype(int).shift(1)
    df["feat_new_low_20"] = (c == c.rolling(20).min()).astype(int).shift(1)

    # -----------------------------------------------------------------------
    # 30. VIX proxy (lines 330-339)
    # VIX requires options market data. PROXY: 20-day rolling std of daily returns.
    # -----------------------------------------------------------------------
    daily_ret = c.pct_change()
    vix_proxy = daily_ret.rolling(20).std() * math.sqrt(252) * 100  # annualised vol %
    df["feat_vix_proxy"] = vix_proxy.shift(1)
    df["feat_high_vol_regime"] = (vix_proxy > vix_proxy.rolling(60).mean()).astype(int).shift(1)

    # -----------------------------------------------------------------------
    # 31. Relative Strength vs SPY (lines 342-350)
    # Requires SPY data. PROXY: 20-day return vs rolling average (cross-sectional
    # signal cannot be computed without multiple stocks; single-stock proxy used).
    # -----------------------------------------------------------------------
    ret_20 = c.pct_change(20)
    df["feat_ret_20d"] = ret_20.shift(1)
    # Rank within own history as RS proxy (percentile of 20d return)
    df["feat_rs_pct_rank_252"] = ret_20.rolling(252).rank(pct=True).shift(1)

    # -----------------------------------------------------------------------
    # 32 & 33. Relative Strength vs QQQ / Sector ETF (lines 352-373)
    # Proxy: 10-day vs 60-day return spread (momentum vs base)
    # -----------------------------------------------------------------------
    ret_10 = c.pct_change(10)
    ret_60 = c.pct_change(60)
    df["feat_ret_10d"] = ret_10.shift(1)
    df["feat_ret_60d"] = ret_60.shift(1)
    df["feat_momentum_vs_base"] = (ret_10 - ret_60).shift(1)

    # -----------------------------------------------------------------------
    # 34. Leader / Laggard tracking (lines 375-384)
    # PROXY: 5-day return percentile rank within own rolling history
    # -----------------------------------------------------------------------
    ret_5 = c.pct_change(5)
    df["feat_ret_5d"] = ret_5.shift(1)
    df["feat_leader_proxy"] = ret_5.rolling(252).rank(pct=True).shift(1)

    # -----------------------------------------------------------------------
    # 35. Correlation check (lines 386-395)
    # PROXY: rolling autocorrelation of daily returns (trend persistence)
    # -----------------------------------------------------------------------
    df["feat_autocorr_5"] = daily_ret.rolling(20).apply(lambda x: x.autocorr(lag=5) if len(x) >= 6 else np.nan, raw=False).shift(1)

    # -----------------------------------------------------------------------
    # 36-39. News / catalyst proxies (lines 397-440)
    # Earnings, 8-K, Form 4 events require external data.
    # PROXY: abnormal volume + abnormal return on same day = possible catalyst
    # -----------------------------------------------------------------------
    abnormal_vol = (vol_z > 2.0).astype(int)
    abnormal_ret = (daily_ret.abs() > 2 * daily_ret.rolling(20).std()).astype(int)
    df["feat_catalyst_proxy"] = (abnormal_vol & abnormal_ret).astype(int).shift(1)
    # Pre-event quiet: vol below 20-day avg for 5 days (coiling)
    df["feat_vol_quiet_5"] = (v.rolling(5).mean() < avg_vol_20).astype(int).shift(1)

    # -----------------------------------------------------------------------
    # 40. Insider buying proxy (Form 4, lines 441-451)
    # No insider data. PROXY: sustained price rise on above-avg vol
    # -----------------------------------------------------------------------
    df["feat_insider_buy_proxy"] = ((ret_10 > 0.05) & (rel_vol > 1.2)).astype(int).shift(1)

    # -----------------------------------------------------------------------
    # 41. Analyst rating proxy (lines 453-462)
    # Upgrades/downgrades require external feed.
    # PROXY: sudden volume + gap in same direction
    # -----------------------------------------------------------------------
    gap_pct = (o - c.shift(1)) / c.shift(1).replace(0, np.nan) * 100
    df["feat_gap_pct"] = gap_pct.shift(1)
    df["feat_gap_up"] = (gap_pct > 1.0).astype(int).shift(1)
    df["feat_gap_down"] = (gap_pct < -1.0).astype(int).shift(1)

    # -----------------------------------------------------------------------
    # 42. Economic calendar / macro event proxy (lines 464-473)
    # PROXY: end-of-month / beginning-of-month calendar effect flag
    # -----------------------------------------------------------------------
    if hasattr(df.index, "day"):
        day_of_month = pd.Series(df.index.day, index=df.index)
        df["feat_month_end_flag"] = ((day_of_month >= 28) | (day_of_month <= 3)).astype(int).shift(1)
    else:
        df["feat_month_end_flag"] = 0

    # -----------------------------------------------------------------------
    # 43. Options volume proxy (lines 476-485)
    # IV / call-put ratio require options data.
    # PROXY: 5-day realised vol vs 20-day realised vol (vol term structure)
    # -----------------------------------------------------------------------
    rv5 = daily_ret.rolling(5).std() * math.sqrt(252) * 100
    rv20 = daily_ret.rolling(20).std() * math.sqrt(252) * 100
    df["feat_rv5"] = rv5.shift(1)
    df["feat_rv20"] = rv20.shift(1)
    df["feat_rv_term_structure"] = (rv5 / rv20.replace(0, np.nan)).shift(1)  # >1 = short vol elevated

    # -----------------------------------------------------------------------
    # 44. Open Interest proxy (lines 487-496)
    # PROXY: rolling volume 20-day avg (stable high OI = high avg vol)
    # -----------------------------------------------------------------------
    df["feat_avg_vol_20"] = avg_vol_20.shift(1)

    # -----------------------------------------------------------------------
    # 45. Implied Volatility proxy (lines 498-507)
    # PROXY: 10-day Parkinson vol estimator (uses high-low range)
    # -----------------------------------------------------------------------
    parkinson_vol = (1.0 / (4 * math.log(2))) * ((np.log(h / lo.replace(0, np.nan))) ** 2)
    iv_proxy = np.sqrt(parkinson_vol.rolling(10).mean() * 252) * 100
    df["feat_iv_proxy_parkinson"] = iv_proxy.shift(1)
    df["feat_iv_high"] = (iv_proxy > iv_proxy.rolling(60).quantile(0.75)).astype(int).shift(1)
    df["feat_iv_low"] = (iv_proxy < iv_proxy.rolling(60).quantile(0.25)).astype(int).shift(1)

    # -----------------------------------------------------------------------
    # 46. Put/Call Ratio proxy (lines 509-518)
    # PROXY: negative return days / positive return days over 20 bars
    # -----------------------------------------------------------------------
    put_proxy = (daily_ret < 0).astype(int).rolling(20).sum()
    call_proxy = (daily_ret > 0).astype(int).rolling(20).sum()
    pc_proxy = put_proxy / call_proxy.replace(0, np.nan)
    df["feat_put_call_proxy"] = pc_proxy.shift(1)
    df["feat_bearish_sentiment"] = (pc_proxy > 1.2).astype(int).shift(1)

    # -----------------------------------------------------------------------
    # 47. Gamma levels proxy (lines 520-529)
    # Dealer hedging gamma requires options chain data.
    # PROXY: distance to nearest round-number price level (psychological S/R)
    # -----------------------------------------------------------------------
    round_level = (c / 10).round() * 10
    gamma_dist = ((c - round_level) / c.replace(0, np.nan) * 100).abs()
    df["feat_near_round_level"] = (gamma_dist < 1.0).astype(int).shift(1)
    df["feat_gamma_dist_pct"] = gamma_dist.shift(1)

    # -----------------------------------------------------------------------
    # 48. Top Gainers scanner proxy (lines 532-541)
    # Requires cross-sectional data. PROXY: 1-day return percentile in own history
    # -----------------------------------------------------------------------
    df["feat_top_gainer_proxy"] = daily_ret.rolling(252).rank(pct=True).shift(1)

    # -----------------------------------------------------------------------
    # 49. Top Losers scanner proxy (lines 543-552)
    # -----------------------------------------------------------------------
    df["feat_top_loser_proxy"] = (-daily_ret).rolling(252).rank(pct=True).shift(1)

    # -----------------------------------------------------------------------
    # 50. High Relative Volume Scanner (lines 554-563) — already captured above
    # Additional: dollar volume spike
    # -----------------------------------------------------------------------
    dollar_vol = c * v
    avg_dollar_vol = dollar_vol.rolling(20).mean()
    df["feat_dollar_vol_rel"] = (dollar_vol / avg_dollar_vol.replace(0, np.nan)).shift(1)

    # -----------------------------------------------------------------------
    # 51. Gap Scanner (lines 565-574)
    # Open significantly above/below prior close; already computed gap_pct above.
    # Additional: gap fill detection (price returned to prior close intraday)
    # -----------------------------------------------------------------------
    df["feat_gap_filled"] = (
        (gap_pct > 0) & (lo <= c.shift(1)) |
        (gap_pct < 0) & (h >= c.shift(1))
    ).astype(int).shift(1)

    # -----------------------------------------------------------------------
    # 52. Halt scanner proxy (lines 576-585)
    # PROXY: extreme intraday range > 5 ATR (halt-then-resume volatility)
    # -----------------------------------------------------------------------
    df["feat_halt_proxy"] = ((h - lo) > 5 * atr_vals).astype(int).shift(1)

    # -----------------------------------------------------------------------
    # 53. Float scanner proxy (lines 587-596)
    # PROXY: average daily dollar turnover proxy (inverse float proxy)
    # -----------------------------------------------------------------------
    turnover_proxy = v / avg_vol_20.replace(0, np.nan)  # relative float velocity
    df["feat_float_velocity"] = turnover_proxy.shift(1)

    # -----------------------------------------------------------------------
    # 54. Liquidity scanner (lines 598-607)
    # PROXY: average dollar volume > threshold flag
    # -----------------------------------------------------------------------
    df["feat_liquid_flag"] = (avg_dollar_vol > avg_dollar_vol.rolling(60).median()).astype(int).shift(1)

    # -----------------------------------------------------------------------
    # 55. Position Size Calculator proxy (lines 610-619)
    # Risk-per-trade based on ATR stop: shares = max_risk / ATR
    # Output: ATR-based risk unit (normalised)
    # -----------------------------------------------------------------------
    df["feat_risk_unit_atr"] = atr_vals.shift(1)  # stop-size proxy per share

    # -----------------------------------------------------------------------
    # 56. R-Multiple tracking (lines 621-630)
    # R = realised return / risk (ATR). Forward-looking so we compute trailing
    # realised R over last 5 bars as feature.
    # -----------------------------------------------------------------------
    ret_5_raw = c.diff(5)
    r_multiple = ret_5_raw / atr_vals.replace(0, np.nan)
    df["feat_r_multiple_5"] = r_multiple.shift(1)

    # -----------------------------------------------------------------------
    # 57. Win Rate (lines 632-641)
    # Rolling 20-bar win rate (fraction of up-close days)
    # -----------------------------------------------------------------------
    df["feat_win_rate_20"] = up_days.rolling(20).mean().shift(1)

    # -----------------------------------------------------------------------
    # 58. Profit Factor (lines 643-650)
    # Gross wins / gross losses over rolling window
    # -----------------------------------------------------------------------
    wins = daily_ret.clip(lower=0)
    losses = (-daily_ret).clip(lower=0)
    profit_factor = wins.rolling(20).sum() / losses.rolling(20).sum().replace(0, np.nan)
    df["feat_profit_factor_20"] = profit_factor.shift(1)

    # -----------------------------------------------------------------------
    # 59. Expectancy (lines 652-663)
    # (win_rate * avg_win) - (loss_rate * avg_loss) over 20 bars
    # -----------------------------------------------------------------------
    avg_win = wins.rolling(20).mean()
    avg_loss = losses.rolling(20).mean()
    win_rate = up_days.rolling(20).mean()
    loss_rate = 1 - win_rate
    expectancy = win_rate * avg_win - loss_rate * avg_loss
    df["feat_expectancy_20"] = expectancy.shift(1)

    # -----------------------------------------------------------------------
    # 60. Max Drawdown (lines 665-674)
    # Rolling 60-bar drawdown from peak
    # -----------------------------------------------------------------------
    roll_max = c.rolling(60, min_periods=1).max()
    drawdown = (c - roll_max) / roll_max.replace(0, np.nan)
    df["feat_drawdown_60"] = drawdown.shift(1)
    df["feat_max_drawdown_60"] = drawdown.rolling(60).min().shift(1)

    # -----------------------------------------------------------------------
    # 61. Daily Loss Limit flag proxy (lines 676-685)
    # PROXY: flag when today's return < -2 * ATR% (hard-stop triggered)
    # -----------------------------------------------------------------------
    atr_stop = 2 * (atr_vals / c.replace(0, np.nan))
    df["feat_daily_loss_limit_hit"] = (daily_ret < -atr_stop).astype(int).shift(1)

    # -----------------------------------------------------------------------
    # 62. Time Stop proxy (lines 687-696)
    # PROXY: flag bars where price is within ±0.5 ATR of entry 5 bars ago (stalled)
    # -----------------------------------------------------------------------
    price_change_5 = (c - c.shift(5)).abs()
    df["feat_trade_stalled_5"] = (price_change_5 < 0.5 * atr_vals).astype(int).shift(1)

    # -----------------------------------------------------------------------
    # 63-64. Trade Journal / Setup Grading proxy (lines 698-712)
    # Composite "setup quality" score: combines ATR trend, relative volume,
    # MACD alignment, RSI not extreme, ADX trending.
    # -----------------------------------------------------------------------
    setup_score = (
        df["feat_adx_trending"].shift(-1)        # already shifted; undo then redo cleanly
        .fillna(0)
    )
    # Recompute cleanly on raw (pre-shift) series
    setup_components = (
        (adx > 25).astype(int)
        + (rel_vol > 1.5).astype(int)
        + (macd_line > signal_line).astype(int)
        + ((rsi14 > 40) & (rsi14 < 70)).astype(int)
        + (c > sma20).astype(int)
    )
    df["feat_setup_quality"] = setup_components.shift(1)  # 0-5 score

    # -----------------------------------------------------------------------
    # Rolling Sharpe (bonus: implied by profit-factor / expectancy discussion)
    # -----------------------------------------------------------------------
    rolling_sharpe = (daily_ret.rolling(20).mean() / daily_ret.rolling(20).std().replace(0, np.nan)) * math.sqrt(252)
    df["feat_rolling_sharpe_20"] = rolling_sharpe.shift(1)

    # Rolling Sortino
    downside = daily_ret.clip(upper=0)
    sortino = (daily_ret.rolling(20).mean() / downside.rolling(20).std().replace(0, np.nan)) * math.sqrt(252)
    df["feat_rolling_sortino_20"] = sortino.shift(1)

    return df


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import pandas as pd
    import numpy as np

    np.random.seed(42)
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='B', tz='UTC')
    df = pd.DataFrame({
        'open':   100 + np.random.randn(len(dates)).cumsum(),
        'high':   100 + np.random.randn(len(dates)).cumsum() + 0.5,
        'low':    100 + np.random.randn(len(dates)).cumsum() - 0.5,
        'close':  100 + np.random.randn(len(dates)).cumsum(),
        'volume': np.random.randint(1_000_000, 100_000_000, len(dates)),
    }, index=dates)
    df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
    df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)

    out = add_features_part1(df)
    new = [c for c in out.columns if c not in df.columns]
    print(f'added {len(new)} features')
    print('sample feature non-zero pct:')
    for col in new[:20]:
        if pd.api.types.is_numeric_dtype(out[col]):
            nn = out[col].dropna()
            nz = (nn != 0).mean() * 100
            sample = nn.iloc[-3:].tolist()
            print(f'  {col}: {nz:.0f}% non-zero, sample={sample}')
