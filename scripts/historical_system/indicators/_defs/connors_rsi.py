"""Connors RSI — average of RSI(close, r_len), RSI(streak, s_len), percent-rank ROC."""
from __future__ import annotations
import numpy as np, pandas as pd
from historical_system.indicators.base import Indicator, register
from historical_system.indicators import kernels as K

@register
class ConnorsRSI(Indicator):
    name = "connors_rsi"
    outputs = ("connors_rsi",)
    params = {"rsi_len": 3, "streak_len": 2, "pr_len": 100, "source": "close"}
    deps = ("close",)
    def compute(self, df, rsi_len=3, streak_len=2, pr_len=100, source="close"):
        c = df[source].to_numpy(dtype=np.float64)
        r1 = K.rsi(c, rsi_len)
        # Streak: consecutive up/down closes
        change = np.diff(c, prepend=c[0])
        streak = np.zeros_like(c)
        for i in range(1, len(c)):
            if change[i] > 0:
                streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
            elif change[i] < 0:
                streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        r2 = K.rsi(streak, streak_len)
        # Percent-rank of 1-bar ROC over last pr_len bars
        roc = pd.Series(c).pct_change().fillna(0.0)
        pr = roc.rolling(pr_len, min_periods=pr_len).apply(
            lambda w: (w.rank().iloc[-1] - 1) / (len(w) - 1) * 100.0, raw=False
        ).to_numpy()
        return {"connors_rsi": (r1 + r2 + pr) / 3.0}
