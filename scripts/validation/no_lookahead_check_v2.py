"""
no_lookahead_check_v2.py — upgraded per DeepSeek Wave 0 review.

Improvements over v1:
    - Uses a 500-row multi-session sample (spans ~5 sessions vs v1's 50-row single-session)
    - VWAP reset check can now actually verify cross-session reset
    - RSI/ATR correlation test has statistical power (n=500 → 95% CI for ρ=0 ≈ ±0.09)
"""

import argparse, sys
import pandas as pd
import numpy as np

THRESHOLDS = {
    'rsi_range_corr_max': 0.30,  # at n=500, |corr|>0.30 is genuinely suspicious
    'atr_range_corr_max': 0.45,  # ATR can correlate more naturally with current range due to rolling
}

def check_no_inf(df):
    issues = []
    for col in df.columns:
        if col in ['session_date', 'session_minute']: continue
        if not pd.api.types.is_numeric_dtype(df[col]): continue
        n_inf = np.isinf(df[col]).sum()
        if n_inf > 0:
            issues.append(f"{col}: {n_inf} inf values")
    return issues

def check_sorted_unique_index(df):
    issues = []
    if not df.index.is_monotonic_increasing:
        issues.append("index not monotonically increasing")
    if df.index.has_duplicates:
        issues.append(f"index has {df.index.duplicated().sum()} duplicates")
    return issues

def check_vwap_resets(df):
    issues = []
    if 'vwap_session' not in df.columns or 'session_date' not in df.columns:
        return ["vwap_session or session_date missing"]
    sessions = df['session_date'].unique()
    if len(sessions) < 3:
        return [f"only {len(sessions)} sessions in sample; need ≥3 to verify reset (sample too narrow)"]
    # for each session: first bar's VWAP should be ~= first bar's close (single-bar avg)
    grouped = df.groupby('session_date')
    n_inconsistent = 0
    for sess, grp in grouped:
        if len(grp) < 1: continue
        first = grp.iloc[0]
        # at the FIRST bar of each session, VWAP should equal that bar's typical price
        typical = (first['high'] + first['low'] + first['close']) / 3
        if abs(first['vwap_session'] - typical) > 0.001 * typical:
            n_inconsistent += 1
    if n_inconsistent > 0:
        issues.append(f"{n_inconsistent}/{len(sessions)} sessions where first-bar VWAP != bar typical (VWAP may not be resetting)")
    # also: VWAP at end of session N+1 should not equal VWAP at end of session N (different price domains)
    last_per_sess = grouped['vwap_session'].last()
    if len(last_per_sess) >= 2:
        # check that consecutive sessions have meaningfully different end-VWAP
        if (last_per_sess.diff().dropna().abs() < 0.001).all():
            issues.append("end-of-session VWAP identical across sessions — suspicious")
    return issues

def check_pdh_pdl_constant_per_session(df):
    issues = []
    if 'pdh' not in df.columns:
        return ["pdh missing"]
    grouped = df.groupby('session_date')
    n_inconsistent = sum(1 for _, grp in grouped if grp['pdh'].nunique(dropna=False) > 1)
    if n_inconsistent > 0:
        issues.append(f"{n_inconsistent} sessions with non-constant PDH (violates B1)")
    if 'pdl' in df.columns:
        n2 = sum(1 for _, grp in grouped if grp['pdl'].nunique(dropna=False) > 1)
        if n2 > 0:
            issues.append(f"{n2} sessions with non-constant PDL")
    return issues

def check_pdh_pdl_advance(df):
    """PDH at session N must equal HIGH at session N-1."""
    issues = []
    if 'pdh' not in df.columns:
        return ["pdh missing"]
    # build per-session high
    grouped = df.groupby('session_date')
    sess_high = grouped['high'].max()
    sess_pdh = grouped['pdh'].first()
    # for session N (N>0): pdh should equal sess_high.shift(1)
    expected_pdh = sess_high.shift(1)
    diff = (sess_pdh - expected_pdh).abs()
    # tolerate first session (where pdh should be NaN) and rounding
    mismatch = (diff > 0.01).sum()
    if mismatch > 1:  # allow 1 for the first session's NaN
        issues.append(f"{mismatch} sessions where pdh != prior session's high (potential lookahead leak)")
    return issues

def check_feature_correlations(df):
    """RSI/ATR shouldn't correlate strongly with CURRENT bar's high-low range if properly shifted."""
    issues = []
    if 'rsi_14' not in df.columns:
        return ["rsi_14 missing"]
    cur_range = df['high'] - df['low']
    # use only the rows where RSI is non-null (post warmup)
    valid = df['rsi_14'].notna()
    if valid.sum() < 50:
        return [f"only {valid.sum()} valid rsi rows; cannot test correlation"]
    rsi_corr = df.loc[valid, 'rsi_14'].corr(cur_range[valid], method='spearman')
    atr_corr = df.loc[valid, 'atr_14'].corr(cur_range[valid], method='spearman')
    if abs(rsi_corr) > THRESHOLDS['rsi_range_corr_max']:
        issues.append(f"rsi_14 has |spearman|={abs(rsi_corr):.3f} (>{THRESHOLDS['rsi_range_corr_max']}) with current bar's range — n={valid.sum()}, suspicious")
    if abs(atr_corr) > THRESHOLDS['atr_range_corr_max']:
        issues.append(f"atr_14 has |spearman|={abs(atr_corr):.3f} (>{THRESHOLDS['atr_range_corr_max']}) with current bar's range — n={valid.sum()}, suspicious")
    return issues

def check_n_vwap_bars_monotone(df):
    """n_vwap_bars should increase within a session and reset at session start."""
    issues = []
    if 'n_vwap_bars' not in df.columns:
        return []  # optional column
    for sess, grp in df.groupby('session_date'):
        if len(grp) < 2: continue
        diffs = grp['n_vwap_bars'].diff().dropna()
        if (diffs != 1).any():
            issues.append(f"n_vwap_bars not strictly +1 within session {sess}")
    return issues

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features-parquet', required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.features_parquet)
    n_sessions = df['session_date'].nunique() if 'session_date' in df.columns else 0
    print(f"loaded features: {df.shape}, sessions: {n_sessions}")

    checks = [
        ("A4 — VWAP per-session reset (cross-session)", check_vwap_resets(df)),
        ("B1 — PDH/PDL constant within session", check_pdh_pdl_constant_per_session(df)),
        ("C1 — PDH = prior session's high (no leak)", check_pdh_pdl_advance(df)),
        ("E1 — index sorted + unique", check_sorted_unique_index(df)),
        ("H1 — no Inf in feature columns", check_no_inf(df)),
        ("A2 — features uncorrelated with current bar range", check_feature_correlations(df)),
        ("V1 — n_vwap_bars monotone within session", check_n_vwap_bars_monotone(df)),
    ]
    print()
    n_failed = 0
    for name, issues in checks:
        if issues:
            print(f"  ❌ {name}")
            for i in issues: print(f"      • {i}")
            n_failed += 1
        else:
            print(f"  ✅ {name}")
    print()
    if n_failed == 0:
        print("ALL CHECKS PASSED")
        sys.exit(0)
    else:
        print(f"{n_failed}/{len(checks)} CHECKS FAILED")
        sys.exit(1)

if __name__ == '__main__':
    main()
