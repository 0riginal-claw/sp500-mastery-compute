"""Accumulative Swing Index — cumulative Welles Wilder SI.

Wilder's formula (from *New Concepts in Technical Trading Systems*, 1978):

    SI = 50 * (N / R) * (K / T)

    N = (C - prevC) + 0.5*(C - O) + 0.25*(prevC - prevO)

    K = max(|H - prevC|, |L - prevC|)
    T = limit_move  (user-defined; commodity daily limit or 1 for equities)
    TR = max(|H - L|, |H - prevC|, |L - prevC|)
    SH = prevC - prevO

    ER (expansion) is conditional on where prevC sits vs the current bar:
      if prevC > H:       ER = H - prevC    (negative)
      if L <= prevC <= H: ER = 0
      if prevC < L:       ER = L - prevC    (negative)

    R = TR - 0.5 * |ER| + 0.25 * |SH|

(In Wilder's original ER and SH enter R as absolute values; sources
differ on signs but the magnitude is what matters for R.)

ASI = cumulative sum of SI.
"""
from __future__ import annotations
import numpy as np
from historical_system.indicators.base import Indicator, register

@register
class AccumulativeSwingIndex(Indicator):
    name = "accumulative_swing_index"
    outputs = ("asi",)
    params = {"limit_move": 1.0}
    deps = ("open", "high", "low", "close")
    def compute(self, df, limit_move=1.0):
        o = df["open"].to_numpy(dtype=np.float64)
        h = df["high"].to_numpy(dtype=np.float64)
        l = df["low"].to_numpy(dtype=np.float64)
        c = df["close"].to_numpy(dtype=np.float64)
        n = len(c)
        si = np.zeros(n, dtype=np.float64)
        if n < 2 or limit_move == 0:
            return {"asi": si}
        for i in range(1, n):
            prev_c = c[i-1]; prev_o = o[i-1]
            # K — the "move" factor
            K = max(abs(h[i] - prev_c), abs(l[i] - prev_c))
            # TR and SH
            TR = max(abs(h[i] - l[i]), abs(h[i] - prev_c), abs(l[i] - prev_c))
            SH = abs(prev_c - prev_o)
            # ER: conditional on prevC relative to current range
            if prev_c > h[i]:
                ER = h[i] - prev_c     # negative
            elif prev_c < l[i]:
                ER = l[i] - prev_c     # negative
            else:
                ER = 0.0
            R = TR - 0.5 * abs(ER) + 0.25 * SH
            if R == 0 or TR == 0:
                si[i] = 0.0
                continue
            # Numerator N
            N = (c[i] - prev_c) + 0.5 * (c[i] - o[i]) + 0.25 * (prev_c - prev_o)
            si[i] = 50.0 * (N / R) * (K / limit_move)
        return {"asi": np.cumsum(si)}
