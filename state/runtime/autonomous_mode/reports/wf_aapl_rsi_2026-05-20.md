# Walk-forward AAPL daily RSI<30 mean-reversion, hold 21d

- Source: `AAPL_v10_full_0175c4585644fd24.parquet`
- Data range: 2021-04-21 .. 2026-04-20 (1213 bars)
- Cost: 1.0 bp/side round-trip

| Fold | OOS start  | OOS end    | n_trades | WR     | PF     | Ret%   |
|------|------------|------------|----------|--------|--------|--------|
| 1 | 2025-04-21 | 2026-04-20 | 1 | 100.0% | — | 5.56% |
| 2 | 2024-04-21 | 2025-04-20 | 1 | 0.0% | 0.00 | -5.29% |
| 3 | 2023-04-21 | 2024-04-20 | 3 | 66.7% | 2.07 | 1.16% |
| 4 | 2022-04-21 | 2023-04-20 | 0 | — | — | 0.00% |
| **agg** | — | — | 5 | 60.0% | 1.22 | 1.42% |

- Promising folds (WR≥52% & PF≥1.10): [3]
- Aggregate promising: True