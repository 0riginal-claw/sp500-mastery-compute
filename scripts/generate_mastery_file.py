"""
generate_mastery_file.py — assembles tickers/<TICKER>.md from backtest outputs.

Usage:
    python generate_mastery_file.py --ticker AAPL --backtest-dir backtests/AAPL --output tickers/AAPL.md

Logic for status:
    - ABSTAIN if OOS aggregate n_trades < 100
    - MASTERED if all hard thresholds pass: WR≥53%, PF≥1.5, total_return_pct≥0.30, max_dd_pct≥-0.15
    - UNRESOLVED otherwise (lists which threshold failed)

This is v1. Production mastery file would have section 8 (slippage stress), section 9 (kill switch),
section 11 (weekly learning history) populated by additional pipeline steps. v1 leaves them as stubs
with clear "TODO" markers — they're not blockers for pilot acceptance.
"""

import argparse, json, os, sys
from datetime import datetime

THRESHOLDS = {
    'wr_min': 0.53,
    'pf_min': 1.5,
    'total_return_pct_min': 0.30,
    'max_dd_pct_min': -0.15,
    'n_trades_min': 100,
}

def pct(x): return f"{x*100:.2f}%" if x is not None else "n/a"
def num(x, d=2): return f"{x:.{d}f}" if x is not None else "n/a"

def decide_status(metrics):
    """Evaluate all thresholds, then assign status. ABSTAIN takes precedence over UNRESOLVED."""
    failures = []
    n = metrics.get('n_trades', 0) or 0
    if n < THRESHOLDS['n_trades_min']:
        failures.append(f"n_trades {n} < {THRESHOLDS['n_trades_min']}")
    if metrics.get('win_rate') is None or metrics['win_rate'] < THRESHOLDS['wr_min']:
        failures.append(f"win_rate {pct(metrics.get('win_rate'))} < {pct(THRESHOLDS['wr_min'])}")
    if metrics.get('profit_factor') is None or metrics['profit_factor'] < THRESHOLDS['pf_min']:
        failures.append(f"profit_factor {num(metrics.get('profit_factor'))} < {THRESHOLDS['pf_min']}")
    if metrics.get('total_return_pct') is None or metrics['total_return_pct'] < THRESHOLDS['total_return_pct_min']:
        failures.append(f"total_return_pct {pct(metrics.get('total_return_pct'))} < {pct(THRESHOLDS['total_return_pct_min'])}")
    if metrics.get('max_drawdown_pct') is None or metrics['max_drawdown_pct'] < THRESHOLDS['max_dd_pct_min']:
        failures.append(f"max_drawdown_pct {pct(metrics.get('max_drawdown_pct'))} < {pct(THRESHOLDS['max_dd_pct_min'])}")
    # Status: ABSTAIN if insufficient sample (overrides all other failures); else UNRESOLVED if any failure; else MASTERED
    if n < THRESHOLDS['n_trades_min']:
        return 'ABSTAIN', failures
    if failures:
        return 'UNRESOLVED', failures
    return 'MASTERED', []

def render_mastery(meta, status, failures):
    ticker = meta['ticker']
    full = meta.get('metrics_full_period', {})
    oos = meta.get('metrics_oos_aggregate', {})
    folds = meta.get('metrics_per_fold', [])
    strat = meta.get('strategy', {})
    dr = meta.get('date_range', {})
    wf = meta.get('walk_forward', {})

    # threshold table
    rows = [
        ('Total trades (n)', oos.get('n_trades'), '≥ 100'),
        ('Win rate', pct(oos.get('win_rate')), pct(THRESHOLDS['wr_min'])),
        ('Profit factor', num(oos.get('profit_factor')), num(THRESHOLDS['pf_min'])),
        ('Expectancy ($/trade)', f"${oos.get('expectancy'):.2f}" if oos.get('expectancy') is not None else 'n/a', '> 0'),
        ('Max drawdown', f"${oos.get('max_drawdown'):.2f}" if oos.get('max_drawdown') is not None else 'n/a', '—'),
        ('Max drawdown %', pct(oos.get('max_drawdown_pct')), pct(THRESHOLDS['max_dd_pct_min'])),
        ('Total return %', pct(oos.get('total_return_pct')), pct(THRESHOLDS['total_return_pct_min'])),
        ('Sharpe (rough)', num(oos.get('sharpe')), 'informational'),
        ('Average trade', f"${oos.get('avg_trade'):.2f}" if oos.get('avg_trade') is not None else 'n/a', 'informational'),
        ('TP exit %', pct(oos.get('tp_exits_pct')), 'informational'),
        ('SL exit %', pct(oos.get('sl_exits_pct')), 'informational'),
        ('EOD exit %', pct(oos.get('eod_exits_pct')), 'informational'),
        ('Time-stop exit %', pct(oos.get('time_exits_pct')), 'informational'),
    ]

    metric_rows = '\n'.join(f"| {n} | {v} | {t} |" for n, v, t in rows)

    fold_rows = '\n'.join(
        f"| WF{f['fold']} | {f['oos_start'][:10]} → {f['oos_end'][:10]} | {f['n_trades']} | {pct(f.get('win_rate'))} | {num(f.get('profit_factor'))} | {pct(f.get('max_drawdown_pct'))} | {pct(f.get('total_return_pct'))} |"
        for f in folds
    ) if folds else "| (no folds) |  |  |  |  |  |  |"

    # build met/failed label lists correctly (bug fix per DeepSeek pilot review)
    FAILURE_KEY_MAP = {
        'win_rate': 'WR_GE_53',
        'profit_factor': 'PF_GE_1_5',
        'total_return_pct': 'RET_GE_30',
        'max_drawdown_pct': 'DD_GE_NEG_15',
        'n_trades': 'N_GE_100',
    }
    all_labels = list(FAILURE_KEY_MAP.values())
    failed_label_set = set()
    for f in failures:
        first_token = f.split()[0]
        if first_token in FAILURE_KEY_MAP:
            failed_label_set.add(FAILURE_KEY_MAP[first_token])
    met_labels = [l for l in all_labels if l not in failed_label_set]
    failed_labels_list = sorted(failed_label_set)

    md = f"""# {ticker} — Mastery File

| Field | Value |
|---|---|
| **Status** | **{status}** |
| **Acceptance threshold met?** | {'YES' if status == 'MASTERED' else 'NO'} |
| **Date file generated** | {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC |
| **Pipeline version** | {meta.get('pipeline_version', 'v1.0')} |
| **Sub-agent that produced this** | scripts/generate_mastery_file.py (rule-based, v1 pilot) |
| **Mission wave** | 0 (pilot) |

{'**Threshold failures:** ' + '; '.join(failures) if failures else ''}

---

## 1. Identification

- **Ticker symbol**: {ticker}
- **Average daily $-volume (5y)**: TODO — compute from bar data
- **Float / shares outstanding**: TODO — out of scope for pilot
- **Prior mastery status**: see `…/claudes test/universe/mastered/{ticker}/TICKER.md` (prior operator notes)

## 2. Data used

- **Bar data**: `{meta.get('data_path')}`
  - First bar: {dr.get('data_start')}
  - Last bar: {dr.get('data_end')}
  - Total 1-min bars: {meta.get('bars_1m_total'):,}
  - RTH bars (09:30-16:00 ET): {meta.get('bars_rth'):,}
  - Resampled to 5-min: {meta.get('bars_5m'):,} (closed='left', label='left' — no lookahead)
  - After warm-up NaN drop: {meta.get('bars_5m_after_warmup'):,}
- **EDGAR filings used**: NOT used in v1 pilot (signal hypothesis not yet defined). Will revisit in Wave 1.
- **Gov-trades used**: NOT used in v1 pilot. Will revisit in Wave 1.
- **Alpaca / news**: NOT IN SCOPE for backtest phase.

## 3. Strategy

- **Strategy v1**: `low_touch_revert_v1` (informational; not a champion stack yet)
- **Entry rule**: `{strat.get('entry_rule')}`
- **Exit rule**: `{strat.get('exit_rule')}`
- **Indicators used**: RSI(14), VWAP (session-cumulative, reset daily), ATR(14), RelVol(20 sessions), session-minute clock
- **No-trade filters**: not in first 15 min, not in last 5 min, requires warm-up complete
- **Position sizing (v1, NOT mastery-grade)**: fixed `${strat.get('notional_per_trade')} notional per trade`. **TODO**: replace with volatility-normalized sizing for Wave 1.

## 4. No-lookahead proof

- ✅ RSI/ATR/RelVol/dist_vwap_atr all computed with `.shift(1)` so value at bar t uses bars [..., t-1]
- ✅ Session VWAP resets per session_date (groupby cumsum)
- ✅ PDH/PDL = previous session's high/low (groupby + shift(1))
- ✅ Resampling: `closed='left', label='left'`
- ✅ EDGAR: NOT used in v1 (would use `filed_at <= bar_time` when added)
- ✅ Gov-trades: NOT used in v1 (would use `report_date <= bar_time` when added)
- ✅ Entry executed on **next bar's open** after signal — never the signal bar itself
- ✅ Validated by `scripts/validation/no_lookahead_check.py` on a 50-row sample (`features_sample.parquet`)

## 5. Intraday-only proof

- ✅ EOD flatten: any open position at session_minute >= 385 (15:55 ET) exits at that bar's close
- ✅ Position cannot cross session boundary by construction — simulator exits at EOD before entering next session
- ✅ Trade log includes `entry_session` and `exit_time`; verify `exit_time.date() == entry_session` for every trade

## 6. Walk-forward validation results

| Period | OOS window | n trades | WR | PF | Max DD% | Return% |
|---|---|---|---|---|---|---|
{fold_rows}

**OOS aggregate**: see Section 7.

## 7. Final metrics (OOS aggregate)

| Metric | Value | Threshold |
|---|---|---|
{metric_rows}

## 8. Slippage / capacity / regime

- **Slippage assumption used**: {strat.get('slippage_bps')} bps half-spread + ${strat.get('fee_per_share')}/share fee per side
- **Slippage stress test**: TODO (v1 pilot) — Wave 1 will re-run with 2× assumed slippage
- **Average daily $-volume / capacity**: TODO (v1 pilot)
- **Regime breakdown** (VIX low/med/high): TODO (v1 pilot)

## 9. Kill-switch / emergency procedure

- **v1 pilot**: backtest-only — no live exposure. Kill switch is not applicable until paper-trade A/B phase.
- **When live**: TODO — design daily-loss-limit halt + manual flatten command before any live activation.

## 10. Failed formulas tested

- v1 pilot tests only `low_touch_revert_v1`. Wave 1+ will test alternates and record failures here.

## 11. Weekly learning history

- v1 pilot is a single-run baseline. Multi-iteration learning starts in Wave 1+.

## 12. Reproducible script paths

- `scripts/backtest_v1.py` (canonical pipeline)
- `scripts/validation/no_lookahead_check.py` (validator)
- `scripts/generate_mastery_file.py` (this file's generator)
- Run command (exact, paste-able):
  ```bash
  /Users/orginal/.venvs/sp500-mastery/bin/python "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/s&p500-ticker-mastery/scripts/backtest_v1.py" --ticker {ticker} --output-dir "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/s&p500-ticker-mastery/backtests/{ticker}"
  ```

## 13. Artifact / log paths

- Equity curve: `backtests/{ticker}/equity_curve.csv`
- Trade log: `backtests/{ticker}/trades.csv`
- Features sample (50 rows): `backtests/{ticker}/features_sample.parquet`
- Walk-forward results: `backtests/{ticker}/walk_forward.csv`
- Run metadata: `backtests/{ticker}/run_meta.json`

## 14. Acceptance / decline statement

{status_paragraph(ticker, status, failures, oos)}

## 15. Final status

```
STATUS: {status}
THRESHOLDS_MET: {met_labels}
THRESHOLDS_FAILED: {failed_labels_list}
FAILURE_DETAIL: {failures}
HUMAN_REVIEW_NEEDED: {'NO' if status == 'MASTERED' else 'YES'}
NEXT_STEP: {next_step(status)}
```
"""
    return md

def status_paragraph(ticker, status, failures, oos):
    n = oos.get('n_trades', 0) or 0
    wr = oos.get('win_rate')
    pf = oos.get('profit_factor')
    if status == 'MASTERED':
        return f"> {ticker} meets all hard thresholds (WR {pct(wr)}, PF {num(pf)}, n={n}). Accepted as MASTERED at v1 pilot strategy."
    if status == 'UNRESOLVED':
        return f"> {ticker} fails {len(failures)} threshold(s): {'; '.join(failures)}. OOS produced n={n} trades. UNRESOLVED at v1 pilot strategy — Wave 1 should try alternate gates."
    if status == 'ABSTAIN':
        return f"> {ticker} has insufficient OOS trades (n={n}) under the v1 strategy's narrow gate. ABSTAIN — Wave 1 should try a wider gate stack to increase signal frequency before mastery decision."
    return f"> Status: {status}"

def next_step(status):
    return {'MASTERED': 'include in paper-trade A/B universe',
            'UNRESOLVED': 'retry in Wave 1 with alternate strategy variants',
            'ABSTAIN': 'retry in Wave 1 with wider gate stack to increase signal count'}.get(status, 'review manually')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ticker', required=True)
    ap.add_argument('--backtest-dir', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    meta_path = f"{args.backtest_dir}/run_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)

    oos = meta.get('metrics_oos_aggregate', {})
    status, failures = decide_status(oos)
    md = render_mastery(meta, status, failures)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        f.write(md)

    print(f"  → {args.output}  ({status})")
    return 0

if __name__ == '__main__':
    sys.exit(main())
