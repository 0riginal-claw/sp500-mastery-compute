"""
mythos_features.py — OpenMythos financial embedding inference module.

Public API:
    compute_mythos_embedding(ticker, end_date) -> np.ndarray[256] or np.ndarray[0]

Environment variables:
    MYTHOS_CHECKPOINT_PATH  -- path to the trained checkpoint .pt file.
                               Defaults to the path in mythos_features_contract.json.
    MYTHOS_DISABLED         -- if "1"/"true"/"yes" (default since 2026-05-21),
                               disable the transformer entirely. The function
                               returns an empty (0,) array and downstream feature
                               builders skip the 256-col concat step. OpenClaw
                               audit (2026-05-21) ranked Mythos #2 for removal —
                               every training row was wasting 256 zero-fill cols
                               with no signal added.

Caching:
    Results are written to /tmp/mythos_emb_cache/<ticker>_<end_date>.npy.
    On re-call for the same (ticker, end_date) the cached array is returned
    without a model forward pass. Cache is bypassed when MYTHOS_DISABLED=1.

Graceful degradation:
    If the checkpoint is missing or fails to load, a zero vector of shape (256,)
    is returned and a WARNING is logged. This prevents the XGBoost pipeline from
    crashing when the checkpoint is not yet available.

Restoration:
    To re-enable Mythos: set MYTHOS_DISABLED=0 in env (or unset). The on-disk
    checkpoint + parquet cache are preserved at /tmp/mythos_emb_cache/ and
    AI-Tools/s&p500-ticker-mastery/paper_trade/mythos_embeddings/ for rebuild.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Master disable switch (2026-05-21, OpenClaw audit rank #2 — drop transformer)
# ---------------------------------------------------------------------------
# When True (default since 2026-05-21), compute_mythos_embedding returns an
# empty array of shape (0,), signaling downstream feature builders to SKIP
# the 256-col concat step. This frees those 256 columns for future wired
# features (TabTransformer/FT-Transformer or backlog stubs).
#
# Override:
#   export MYTHOS_DISABLED=0          # re-enable transformer (restoration)
#   export MYTHOS_DISABLED=1          # explicit disable (matches default)
_MYTHOS_DISABLED_RAW = os.environ.get("MYTHOS_DISABLED", "1").strip().lower()
MYTHOS_DISABLED: bool = _MYTHOS_DISABLED_RAW in ("1", "true", "yes", "on")
if MYTHOS_DISABLED:
    logger.info(
        "[mythos] MYTHOS_DISABLED=1 — transformer dropped per OC audit 2026-05-21; "
        "emitting empty (0,) embeddings."
    )

# ---------------------------------------------------------------------------
# Ensure open_mythos is importable (installed in external-repos/OpenMythos).
# The package requires Python >=3.10 in its pyproject.toml, but the code is
# compatible with 3.9. We add the repo root to sys.path so import works
# regardless of whether the package was pip-installed.
# ---------------------------------------------------------------------------
_OPENMYTHOS_REPO = (
    "/Users/orginal/Library/CloudStorage/"
    "GoogleDrive-zachgladstone@gmail.com/My Drive/"
    "AI-Tools/external-repos/OpenMythos"
)
if _OPENMYTHOS_REPO not in sys.path:
    sys.path.insert(0, _OPENMYTHOS_REPO)

# ---------------------------------------------------------------------------
# Constants (must match mythos_features_contract.json)
# ---------------------------------------------------------------------------

EMBEDDING_DIM: int = 256
INPUT_WINDOW_BARS: int = 60
N_FEATURES_PER_BAR: int = 4  # log_return, signed_volume_ratio, range_pct, time_of_day_fraction
MAX_RTH_BARS: int = 390  # 9:30–15:59 ET inclusive = 390 bars
RTH_START_HOUR_UTC: int = 14
RTH_START_MINUTE_UTC: int = 30
RTH_END_HOUR_UTC: int = 21  # exclusive

DRIVE_ROOT = (
    "/Users/orginal/Library/CloudStorage/"
    "GoogleDrive-zachgladstone@gmail.com/My Drive"
)
DEFAULT_CHECKPOINT = os.path.join(
    DRIVE_ROOT,
    "AI-Tools/checkpoints/mythos_financial_v0.pt",
)
DEFAULT_PARQUET_DIR = os.path.join(
    DRIVE_ROOT,
    "claudes test/data/timeframes/S&P500 5 Year Historical Data"
    "/Minutes TimeFrames/1Min_merged",
)
CACHE_DIR = Path("/tmp/mythos_emb_cache")

# ---------------------------------------------------------------------------
# Alpaca SIP fallback config — used when the parquet is missing or stale
# (no RTH bars for the requested session date). Reads paper-account creds
# from /Users/orginal/.config/auto_signup/alpaca.env on first call.
# ---------------------------------------------------------------------------

_ALPACA_ENV_PATH = "/Users/orginal/.config/auto_signup/alpaca.env"
_ALPACA_CREDS_LOADED: bool = False


def _load_alpaca_creds() -> bool:
    """Load Alpaca paper-account creds into os.environ once.

    Returns:
        True if either ALPACA_PAPER_API_KEY/APCA_API_KEY_ID is now set in env,
        False if the env file is missing and no key was present.
    """
    global _ALPACA_CREDS_LOADED
    if _ALPACA_CREDS_LOADED:
        return True
    # Already in env (e.g. shell-sourced) — no need to read file.
    if os.environ.get("ALPACA_PAPER_API_KEY") or os.environ.get("APCA_API_KEY_ID"):
        _ALPACA_CREDS_LOADED = True
        return True
    if not os.path.exists(_ALPACA_ENV_PATH):
        return False
    try:
        with open(_ALPACA_ENV_PATH) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                if line.startswith("export "):
                    line = line[len("export "):]
                k, _, v = line.partition("=")
                v = v.strip().strip('"').strip("'")
                os.environ.setdefault(k.strip(), v)
        _ALPACA_CREDS_LOADED = True
        return True
    except Exception as exc:
        logger.warning("Failed to load Alpaca creds from %s: %s", _ALPACA_ENV_PATH, exc)
        return False


def _fetch_alpaca_sip_minute_bars(ticker: str, end_date: str):
    """Fetch 1-min RTH bars from Alpaca SIP for ``ticker`` on ``end_date``.

    Returns:
        pandas.DataFrame with columns [timestamp, open, high, low, close, volume]
        (timestamp tz-aware UTC), sorted ascending. Empty DataFrame on failure.
    """
    import pandas as pd

    if not _load_alpaca_creds():
        logger.warning(
            "Alpaca creds unavailable (env-file missing at %s) — cannot fallback for %s/%s.",
            _ALPACA_ENV_PATH, ticker, end_date,
        )
        return pd.DataFrame()

    try:
        from datetime import datetime, timezone
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        from alpaca.data.enums import DataFeed

        key = os.environ.get("ALPACA_PAPER_API_KEY") or os.environ.get("APCA_API_KEY_ID")
        secret = (
            os.environ.get("ALPACA_PAPER_SECRET_KEY")
            or os.environ.get("APCA_API_SECRET_KEY")
        )
        if not key or not secret:
            logger.warning(
                "Alpaca creds incomplete after env-load — cannot fallback for %s/%s.",
                ticker, end_date,
            )
            return pd.DataFrame()

        # 09:30-16:00 ET == 13:30-20:00 UTC (DST) / 14:30-21:00 UTC (std).
        # Request a wide window (13:00-21:00 UTC) and let _filter_rth_bars trim.
        y, m, d = (int(p) for p in end_date.split("-"))
        start_utc = datetime(y, m, d, 13, 0, tzinfo=timezone.utc)
        end_utc = datetime(y, m, d, 21, 30, tzinfo=timezone.utc)

        client = StockHistoricalDataClient(key, secret)
        req = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Minute,
            start=start_utc,
            end=end_utc,
            feed=DataFeed.SIP,
        )
        df = client.get_stock_bars(req).df
        if df is None or len(df) == 0:
            logger.warning(
                "Alpaca SIP returned 0 bars for %s on %s.", ticker, end_date,
            )
            return pd.DataFrame()

        # Drop the symbol level — keep timestamp index, then reset to column.
        if "symbol" in df.index.names:
            df = df.droplevel("symbol")
        df = df.reset_index().rename(columns={"timestamp": "timestamp"})
        # Ensure tz-aware UTC (alpaca returns tz-aware).
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
        else:
            df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")
        keep = [c for c in ["timestamp", "open", "high", "low", "close", "volume"] if c in df.columns]
        return df[keep].sort_values("timestamp").reset_index(drop=True)

    except Exception as exc:
        logger.warning(
            "Alpaca SIP fetch failed for %s on %s: %s",
            ticker, end_date, exc,
        )
        return pd.DataFrame()

# ---------------------------------------------------------------------------
# Lazy-loaded model cache (module-level singleton; populated by _get_model())
# ---------------------------------------------------------------------------

_MODEL_CACHE: Optional["FinancialMythos"] = None
_CHECKPOINT_PATH_LOADED: Optional[str] = None


# ---------------------------------------------------------------------------
# OpenMythos TINY financial config & wrapper
# ---------------------------------------------------------------------------

def _build_financial_config():
    """Build the TINY financial MythosConfig (dim=256, seq_len=60).

    Returns:
        MythosConfig instance with financial-specific hyperparameters.

    Raises:
        ImportError: if open_mythos is not installed.
    """
    from open_mythos.main import MythosConfig  # type: ignore[import]

    return MythosConfig(
        vocab_size=1,        # unused; continuous embedding replaces embed
        dim=EMBEDDING_DIM,
        n_heads=4,
        n_kv_heads=4,
        max_seq_len=64,
        max_loop_iters=2,
        prelude_layers=1,
        coda_layers=1,
        attn_type="gqa",
        # MLA params unused (attn_type=gqa) but must satisfy defaults
        kv_lora_rank=32,
        q_lora_rank=64,
        qk_rope_head_dim=16,
        qk_nope_head_dim=16,
        v_head_dim=16,
        n_experts=4,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=256,
        act_threshold=0.99,
        rope_theta=10000.0,
        lora_rank=8,
        dropout=0.0,
    )


class ContinuousInputProjection:
    """Linear(4 → dim) + LayerNorm — replaces token embedding for continuous input.

    Implemented as a plain PyTorch module and attached to FinancialMythos.
    Each bar's 4-dimensional feature vector is projected to ``dim`` dimensions,
    giving the model a learned tokenizer that operates in continuous space
    rather than discrete vocabulary bins.
    """

    pass  # defined inside _build_financial_mythos() to defer torch import


def _build_financial_mythos():
    """Construct the FinancialMythos model class and return an instance.

    Defers torch/open_mythos imports so the module can be imported without
    a GPU or PyTorch installed (graceful degradation path returns zeros before
    reaching here).

    Returns:
        FinancialMythos instance in eval mode.
    """
    import torch
    import torch.nn as nn
    from open_mythos.main import OpenMythos  # type: ignore[import]

    cfg = _build_financial_config()

    class _ContinuousInputProjection(nn.Module):
        """Projects (B, T, 4) continuous bar features to (B, T, dim)."""

        def __init__(self, n_features: int, dim: int) -> None:
            super().__init__()
            self.proj = nn.Linear(n_features, dim, bias=True)
            self.norm = nn.LayerNorm(dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Args:
                x: (B, T, n_features) float32 bar features
            Returns:
                (B, T, dim) token embeddings
            """
            return self.norm(self.proj(x))

    class _ProjectionHead(nn.Module):
        """Linear(dim → dim) + tanh — maps last-token hidden to bounded embedding."""

        def __init__(self, dim: int) -> None:
            super().__init__()
            self.linear = nn.Linear(dim, dim, bias=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Args:
                x: (B, dim) last-token hidden state
            Returns:
                (B, dim) embedding in (-1, 1)
            """
            return torch.tanh(self.linear(x))

    class _RegressionHead(nn.Module):
        """Linear(dim → 1) — next-bar log-return predictor (training only)."""

        def __init__(self, dim: int) -> None:
            super().__init__()
            self.linear = nn.Linear(dim, 1, bias=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Args:
                x: (B, T, dim) all-token hidden states
            Returns:
                (B, T, 1) per-bar log-return predictions
            """
            return self.linear(x)

    class FinancialMythos(nn.Module):
        """OpenMythos wrapper for financial sequence embedding.

        Architecture:
            ContinuousInputProjection (B,60,4) → (B,60,256)
            OpenMythos backbone (prelude + recurrent + coda)
            Last-token readout [:, -1, :]
            ProjectionHead → (B, 256) in (-1, 1)

        Training also exposes RegressionHead for next-bar log-return MSE loss.
        The regression head is ignored at inference time.

        Args:
            cfg: MythosConfig with dim=256, attn_type="gqa", prelude/coda=1.
        """

        def __init__(self, cfg) -> None:
            super().__init__()
            self.cfg = cfg
            self.input_proj = _ContinuousInputProjection(N_FEATURES_PER_BAR, cfg.dim)
            self.backbone = OpenMythos(cfg)
            self.proj_head = _ProjectionHead(cfg.dim)
            self.regression_head = _RegressionHead(cfg.dim)
            # We need to intercept the backbone to bypass its embed layer.
            # Store freqs_cis as backbone has them as buffers.
            self._init_weights()

        def _init_weights(self) -> None:
            """Initialize projection layers with N(0, 0.02)."""
            nn.init.normal_(self.input_proj.proj.weight, std=0.02)
            nn.init.zeros_(self.input_proj.proj.bias)
            nn.init.normal_(self.proj_head.linear.weight, std=0.02)
            nn.init.zeros_(self.proj_head.linear.bias)
            nn.init.normal_(self.regression_head.linear.weight, std=0.02)
            nn.init.zeros_(self.regression_head.linear.bias)

        def forward(
            self,
            bars: torch.Tensor,
            n_loops: Optional[int] = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Full forward pass for training.

            Args:
                bars: (B, T, 4) float32 bar features; T = INPUT_WINDOW_BARS
                n_loops: recurrent depth override (defaults to cfg.max_loop_iters)

            Returns:
                embedding: (B, 256) last-token projection, values in (-1, 1)
                log_return_preds: (B, T, 1) per-bar next-bar predictions
            """
            B, T, _ = bars.shape
            device = bars.device
            dtype = bars.dtype

            # Project continuous features to hidden dim
            x = self.input_proj(bars)  # (B, T, dim)

            # Run through backbone internals (bypass embed layer).
            # We replicate OpenMythos.forward() but inject x directly
            # after the embed step.
            backbone = self.backbone
            freqs_cis = (
                backbone.freqs_cis_mla
                if backbone.cfg.attn_type == "mla"
                else backbone.freqs_cis
            )[:T]
            mask = backbone._causal_mask(T, device, dtype) if T > 1 else None

            for i, layer in enumerate(backbone.prelude):
                x = layer(x, freqs_cis, mask, kv_cache=None, cache_key=f"prelude_{i}")

            e = x
            x = backbone.recurrent(x, e, freqs_cis, mask, n_loops, kv_cache=None)

            for i, layer in enumerate(backbone.coda):
                x = layer(x, freqs_cis, mask, kv_cache=None, cache_key=f"coda_{i}")

            x = backbone.norm(x)  # (B, T, dim)

            # Regression head: predict next-bar log-return from each position
            log_return_preds = self.regression_head(x)  # (B, T, 1)

            # Embedding: last token → projection head
            last_hidden = x[:, -1, :]  # (B, dim)
            embedding = self.proj_head(last_hidden)  # (B, dim)

            return embedding, log_return_preds

        @torch.no_grad()
        def embed(self, bars: torch.Tensor) -> torch.Tensor:
            """Inference-only forward pass returning only the embedding.

            Args:
                bars: (B, T, 4) float32 bar features

            Returns:
                (B, 256) embedding tensor, values in (-1, 1)
            """
            embedding, _ = self.forward(bars)
            return embedding

    cfg_instance = _build_financial_config()
    model = FinancialMythos(cfg_instance)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Feature computation helpers
# ---------------------------------------------------------------------------


def _filter_rth_bars(df) -> "pd.DataFrame":
    """Filter a parquet DataFrame to RTH bars (9:30–15:59 ET = 14:30–20:59 UTC).

    Args:
        df: DataFrame with a tz-aware 'timestamp' column (UTC).

    Returns:
        Filtered DataFrame sorted by timestamp ascending.
    """
    ts = df["timestamp"]
    hour_utc = ts.dt.hour
    minute_utc = ts.dt.minute

    # 9:30 ET = 14:30 UTC; 16:00 ET = 21:00 UTC (exclusive)
    after_open = (hour_utc > RTH_START_HOUR_UTC) | (
        (hour_utc == RTH_START_HOUR_UTC) & (minute_utc >= RTH_START_MINUTE_UTC)
    )
    before_close = hour_utc < RTH_END_HOUR_UTC

    rth = df[after_open & before_close].copy()
    return rth.sort_values("timestamp")


def _compute_bar_features(rth_df) -> "np.ndarray":
    """Compute 4 features per bar from RTH DataFrame.

    Features:
        log_return:           ln(close_t / close_{t-1}); first bar uses open
        signed_volume_ratio:  sign(log_ret) * ln(1 + vol / session_avg_vol)
        range_pct:            (high - low) / close
        time_of_day_fraction: bar_index_in_session / (MAX_RTH_BARS - 1)

    Args:
        rth_df: RTH-filtered DataFrame with columns [open, high, low, close, volume].
                Must be sorted ascending by timestamp.

    Returns:
        np.ndarray of shape (n_bars, 4) with float32 values.
        NaN/Inf values are clipped to 0.
    """
    close = rth_df["close"].values.astype(np.float64)
    high = rth_df["high"].values.astype(np.float64)
    low = rth_df["low"].values.astype(np.float64)
    volume = rth_df["volume"].values.astype(np.float64)
    open_ = rth_df["open"].values.astype(np.float64)
    n = len(close)

    # log_return: first bar uses open as prior close
    prior_close = np.empty(n, dtype=np.float64)
    prior_close[0] = open_[0]
    prior_close[1:] = close[:-1]
    log_ret = np.log(np.clip(close / np.where(prior_close > 0, prior_close, 1e-8), 1e-10, 1e10))

    # signed_volume_ratio
    session_avg_vol = np.nanmean(volume) if np.nanmean(volume) > 0 else 1.0
    vol_ratio = np.log1p(volume / session_avg_vol)
    signed_vol_ratio = np.sign(log_ret) * vol_ratio

    # range_pct
    range_pct = (high - low) / np.where(close > 0, close, 1.0)

    # time_of_day_fraction: position within the session (0-indexed)
    # We assign fractional position based on actual bar index, not UTC time,
    # so it's robust to gaps (early close, data gaps).
    tod_frac = np.arange(n, dtype=np.float64) / (MAX_RTH_BARS - 1)

    features = np.stack([log_ret, signed_vol_ratio, range_pct, tod_frac], axis=1)

    # Replace NaN/Inf with 0 (safe fallback)
    features = np.where(np.isfinite(features), features, 0.0)
    return features.astype(np.float32)


def _pad_or_trim(features: "np.ndarray", window: int) -> "np.ndarray":
    """Pad (at front with zeros) or trim (from front) to exactly ``window`` bars.

    Args:
        features: (n_bars, 4) float32 feature array.
        window:   target number of bars (INPUT_WINDOW_BARS = 60).

    Returns:
        (window, 4) float32 array.
    """
    n = len(features)
    if n >= window:
        return features[-window:]  # take last ``window`` bars
    # Front-pad with zeros
    pad = np.zeros((window - n, features.shape[1]), dtype=np.float32)
    return np.concatenate([pad, features], axis=0)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def _get_checkpoint_path() -> str:
    """Resolve checkpoint path from env var or default."""
    return os.environ.get("MYTHOS_CHECKPOINT_PATH", DEFAULT_CHECKPOINT)


def _get_model() -> Optional["FinancialMythos"]:
    """Return the cached FinancialMythos model, loading it on first call.

    Uses module-level singleton ``_MODEL_CACHE`` to avoid repeated disk reads.
    Calls ``torch.set_num_threads(4)`` and ``model.eval()`` exactly once at
    first load.

    Returns:
        FinancialMythos in eval mode, or None if checkpoint missing/broken.
    """
    global _MODEL_CACHE, _CHECKPOINT_PATH_LOADED

    ckpt_path = _get_checkpoint_path()

    # Return cached model if same checkpoint is already loaded
    if _MODEL_CACHE is not None and _CHECKPOINT_PATH_LOADED == ckpt_path:
        return _MODEL_CACHE

    if not os.path.exists(ckpt_path):
        logger.warning(
            "MYTHOS_CHECKPOINT_PATH not found: %s — returning zero embeddings.",
            ckpt_path,
        )
        return None

    try:
        import torch

        # Limit CPU parallelism once at first load (avoids thread-oversubscription
        # when many processes or threads call this concurrently).
        torch.set_num_threads(4)

        model = _build_financial_mythos()
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Support both raw state_dict and wrapped checkpoint formats
        if "model" in ckpt:
            state = ckpt["model"]
        else:
            state = ckpt

        model.load_state_dict(state, strict=False)
        model.eval()  # set once here; never toggled again for inference
        _MODEL_CACHE = model
        _CHECKPOINT_PATH_LOADED = ckpt_path
        logger.info("FinancialMythos loaded from %s", ckpt_path)
        return model

    except Exception as exc:
        logger.warning(
            "Failed to load FinancialMythos checkpoint from %s: %s — returning zeros.",
            ckpt_path,
            exc,
        )
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_mythos_embedding(
    ticker: str,
    end_date: str,
    parquet_dir: Optional[str] = None,
    force_recompute: bool = False,
) -> np.ndarray:
    """Compute the 256-dim OpenMythos embedding for a (ticker, session).

    Reads the ticker's 1-min parquet file, extracts the RTH bars for
    ``end_date``, computes 4 features per bar, pads/trims to 60 bars,
    runs a forward pass through FinancialMythos, and returns the
    256-dimensional embedding.

    Results are cached to ``/tmp/mythos_emb_cache/<ticker>_<end_date>.npy``
    so subsequent calls for the same (ticker, end_date) are near-instant.

    Args:
        ticker: Ticker symbol, e.g. "AAPL". Case-sensitive; must match
                the parquet filename without extension.
        end_date: Session date in "YYYY-MM-DD" format. The RTH bars for
                  this calendar date are used (last 60 bars of the session).
        parquet_dir: Directory containing ``<ticker>.parquet`` files.
                     Defaults to the S&P 500 data directory on Google Drive.
        force_recompute: If True, ignore the on-disk cache and recompute.

    Returns:
        np.ndarray of shape (256,) and dtype float32 in approximately (-1, 1).
        Returns zeros with a WARNING log if the checkpoint or parquet is missing.

    Example:
        >>> emb = compute_mythos_embedding("AAPL", "2024-03-15")
        >>> # default (MYTHOS_DISABLED=1): shape == (0,)
        >>> # re-enabled (MYTHOS_DISABLED=0): shape == (256,), dtype float32
    """
    # Master disable switch — return empty array so downstream concat sees
    # zero new columns rather than 256 zero-fill columns.
    if MYTHOS_DISABLED:
        return np.zeros(0, dtype=np.float32)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{ticker}_{end_date}.npy"

    # Cache hit
    if not force_recompute and cache_file.exists():
        try:
            emb = np.load(str(cache_file))
            if emb.shape == (EMBEDDING_DIM,) and emb.dtype == np.float32:
                logger.debug("Cache hit: %s", cache_file)
                return emb
        except Exception:
            pass  # corrupt cache — fall through to recompute

    # Resolve parquet path
    pq_dir = parquet_dir or DEFAULT_PARQUET_DIR
    pq_path = os.path.join(pq_dir, f"{ticker}.parquet")

    if not os.path.exists(pq_path):
        logger.warning(
            "Parquet not found for ticker %s at %s — returning zeros.", ticker, pq_path
        )
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    # Load and filter data
    try:
        import pandas as pd

        df = pd.read_parquet(pq_path)
        df_rth = _filter_rth_bars(df)

        # Filter to the target session date (ET date matches UTC date for 14:30-21:00)
        session_date = pd.Timestamp(end_date).date()
        session_df = df_rth[df_rth["timestamp"].dt.date == session_date]

        if len(session_df) == 0:
            logger.warning(
                "No RTH bars in parquet for %s on %s (parquet may be stale) — "
                "falling back to Alpaca SIP.",
                ticker, end_date,
            )
            alpaca_df = _fetch_alpaca_sip_minute_bars(ticker, end_date)
            if len(alpaca_df) == 0:
                logger.warning(
                    "Alpaca SIP fallback also empty for %s on %s — returning zeros.",
                    ticker, end_date,
                )
                return np.zeros(EMBEDDING_DIM, dtype=np.float32)
            session_df = _filter_rth_bars(alpaca_df)
            if len(session_df) == 0:
                logger.warning(
                    "Alpaca SIP rows present but none in RTH for %s on %s — returning zeros.",
                    ticker, end_date,
                )
                return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    except Exception as exc:
        logger.warning(
            "Error loading parquet for %s: %s — returning zeros.", ticker, exc
        )
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    # Compute features
    features = _compute_bar_features(session_df)       # (n_bars, 4)
    features = _pad_or_trim(features, INPUT_WINDOW_BARS)  # (60, 4)

    # Load model (lazy singleton — loads from disk only on first call)
    model = _get_model()
    if model is None:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    # Forward pass
    try:
        import torch

        with torch.no_grad():
            x = torch.from_numpy(features).unsqueeze(0)  # (1, 60, 4)
            emb_tensor = model.embed(x)                  # (1, 256)
            emb = emb_tensor.squeeze(0).cpu().numpy().astype(np.float32)  # (256,)

    except Exception as exc:
        logger.warning(
            "Forward pass failed for %s/%s: %s — returning zeros.", ticker, end_date, exc
        )
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    # Write to cache
    try:
        np.save(str(cache_file), emb)
        logger.debug("Cached embedding → %s", cache_file)
    except Exception as exc:
        logger.debug("Cache write failed (non-fatal): %s", exc)

    return emb


def get_feature_names() -> list[str]:
    """Return the ordered list of embedding feature names.

    When MYTHOS_DISABLED=1 (default since 2026-05-21), returns an empty list
    so downstream feature builders emit zero Mythos columns. When re-enabled,
    returns ["mythos_emb_0", ..., "mythos_emb_255"].

    Returns:
        List of feature-name strings (length 0 when disabled, 256 when active).
    """
    if MYTHOS_DISABLED:
        return []
    return [f"mythos_emb_{i}" for i in range(EMBEDDING_DIM)]


def compute_mythos_embeddings_batch(
    ticker_date_pairs: list[tuple[str, str]],
    parquet_dir: Optional[str] = None,
    force_recompute: bool = False,
) -> "np.ndarray":
    """Compute embeddings for multiple (ticker, date) pairs.

    Shares the loaded model across all calls for efficiency.

    Args:
        ticker_date_pairs: List of (ticker, end_date) tuples.
        parquet_dir: Optional parquet directory override.
        force_recompute: If True, ignore cache for all pairs.

    Returns:
        np.ndarray of shape (N, 256) where N = len(ticker_date_pairs).
        When MYTHOS_DISABLED=1 returns shape (N, 0) — caller's column concat
        must tolerate empty column blocks (build_v9_features already does).
        Rows correspond to input order; failed rows are zeros.
    """
    if MYTHOS_DISABLED:
        return np.zeros((len(ticker_date_pairs), 0), dtype=np.float32)
    results = np.zeros((len(ticker_date_pairs), EMBEDDING_DIM), dtype=np.float32)
    for i, (ticker, end_date) in enumerate(ticker_date_pairs):
        results[i] = compute_mythos_embedding(
            ticker, end_date, parquet_dir=parquet_dir, force_recompute=force_recompute
        )
    return results


def compute_mythos_embedding_batch(
    tickers: "list[str]",
    end_date: str,
    parquet_dir: Optional[str] = None,
    force_recompute: bool = False,
) -> "dict[str, np.ndarray]":
    """Compute 256-dim embeddings for multiple tickers on a shared session date.

    Loads the model ONCE via ``_get_model()`` and executes a single batched
    forward pass over all valid tickers, making this significantly faster than
    calling ``compute_mythos_embedding`` in a loop.

    Per-ticker failures (missing parquet, no RTH bars for the session, data
    errors) are caught individually — that ticker is skipped with a WARNING and
    the rest of the batch is unaffected.  Tickers whose results already exist in
    the on-disk cache are returned from cache and are NOT included in the batched
    forward pass.

    Args:
        tickers:        Sequence of ticker symbols, e.g. ["AAPL", "MSFT", "GOOGL"].
        end_date:       Session date shared by all tickers, in "YYYY-MM-DD" format.
        parquet_dir:    Directory containing ``<ticker>.parquet`` files.
                        Defaults to the S&P 500 1-min data directory on Google Drive.
        force_recompute: If True, ignore the on-disk cache for all tickers.

    Returns:
        dict mapping each successfully embedded ticker to a ``np.ndarray`` of
        shape (256,) and dtype float32 with values approximately in (-1, 1).
        Tickers that fail for any reason are absent from the returned dict.
        Results are written to ``/tmp/mythos_emb_cache/<ticker>_<end_date>.npy``
        (same cache as ``compute_mythos_embedding``).

    Example:
        >>> result = compute_mythos_embedding_batch(["AAPL", "MSFT"], "2024-03-15")
        >>> assert result["AAPL"].shape == (256,)
        >>> assert result["AAPL"].dtype == np.float32
    """
    if MYTHOS_DISABLED:
        # Master switch: return empty dict — caller treats missing tickers as
        # "no embedding produced" and downstream concat sees zero new columns.
        return {}

    import pandas as pd
    import torch

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    result: dict[str, np.ndarray] = {}

    # --- Phase 1: cache hits (skip forward pass for already-cached tickers) ---
    needs_compute: list[str] = []
    for ticker in tickers:
        cache_file = CACHE_DIR / f"{ticker}_{end_date}.npy"
        if not force_recompute and cache_file.exists():
            try:
                emb = np.load(str(cache_file))
                if emb.shape == (EMBEDDING_DIM,) and emb.dtype == np.float32:
                    logger.debug("Cache hit (batch): %s", cache_file)
                    result[ticker] = emb
                    continue
            except Exception:
                pass  # corrupt cache entry — fall through to recompute
        needs_compute.append(ticker)

    if not needs_compute:
        return result

    # --- Phase 2: load model once for the entire batch ---
    model = _get_model()
    if model is None:
        logger.warning(
            "Model unavailable — skipping batched compute for %d tickers on %s.",
            len(needs_compute), end_date,
        )
        return result

    # --- Phase 3: build per-ticker feature arrays (failures are caught individually) ---
    pq_dir = parquet_dir or DEFAULT_PARQUET_DIR
    batch_features: list[np.ndarray] = []
    batch_tickers: list[str] = []

    try:
        session_date = pd.Timestamp(end_date).date()
    except Exception as exc:
        logger.warning("Invalid end_date %r: %s — returning empty.", end_date, exc)
        return result

    for ticker in needs_compute:
        try:
            pq_path = os.path.join(pq_dir, f"{ticker}.parquet")
            if not os.path.exists(pq_path):
                logger.warning(
                    "Parquet not found for %s at %s — skipping.", ticker, pq_path
                )
                continue

            df = pd.read_parquet(pq_path)
            df_rth = _filter_rth_bars(df)
            session_df = df_rth[df_rth["timestamp"].dt.date == session_date]

            if len(session_df) == 0:
                logger.warning(
                    "No RTH bars in parquet for %s on %s (parquet may be stale) — "
                    "falling back to Alpaca SIP.",
                    ticker, end_date,
                )
                alpaca_df = _fetch_alpaca_sip_minute_bars(ticker, end_date)
                if len(alpaca_df) == 0:
                    logger.warning(
                        "Alpaca SIP fallback empty for %s on %s — skipping.",
                        ticker, end_date,
                    )
                    continue
                session_df = _filter_rth_bars(alpaca_df)
                if len(session_df) == 0:
                    logger.warning(
                        "Alpaca SIP rows present but none in RTH for %s on %s — skipping.",
                        ticker, end_date,
                    )
                    continue

            features = _compute_bar_features(session_df)         # (n_bars, 4)
            features = _pad_or_trim(features, INPUT_WINDOW_BARS) # (60, 4)
            batch_features.append(features)
            batch_tickers.append(ticker)

        except Exception as exc:
            logger.warning(
                "Failed to build features for %s/%s: %s — skipping.", ticker, end_date, exc
            )

    if not batch_tickers:
        return result

    # --- Phase 4: single batched forward pass ---
    try:
        batch_array = np.stack(batch_features, axis=0)  # (B, 60, 4)
        x = torch.from_numpy(batch_array)               # (B, 60, 4)
        with torch.no_grad():
            emb_tensor = model.embed(x)                 # (B, 256)
        emb_batch = emb_tensor.cpu().numpy().astype(np.float32)  # (B, 256)

    except Exception as exc:
        logger.warning(
            "Batched forward pass failed for %d tickers on %s: %s — all skipped.",
            len(batch_tickers), end_date, exc,
        )
        return result

    # --- Phase 5: unpack results and write to cache ---
    for i, ticker in enumerate(batch_tickers):
        emb = emb_batch[i]  # (256,)
        result[ticker] = emb
        cache_file = CACHE_DIR / f"{ticker}_{end_date}.npy"
        try:
            np.save(str(cache_file), emb)
            logger.debug("Cached embedding (batch) → %s", cache_file)
        except Exception as exc:
            logger.debug("Cache write failed for %s (non-fatal): %s", ticker, exc)

    return result


# ---------------------------------------------------------------------------
# Smoke test (run directly: python mythos_features.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Smoke-test compute_mythos_embedding")
    parser.add_argument("--ticker", default="AAPL", help="Ticker symbol")
    parser.add_argument("--date", default="2024-06-03", help="Session date YYYY-MM-DD")
    parser.add_argument("--force", action="store_true", help="Force recompute (ignore cache)")
    args = parser.parse_args()

    logger.info("Computing embedding for %s on %s ...", args.ticker, args.date)
    emb = compute_mythos_embedding(args.ticker, args.date, force_recompute=args.force)
    logger.info("Shape: %s  dtype: %s  norm: %.4f", emb.shape, emb.dtype, float(np.linalg.norm(emb)))
    logger.info("First 8 values: %s", emb[:8])
