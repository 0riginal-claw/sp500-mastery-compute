"""
TimeMixer feature wrapper — kwuking/TimeMixer (ICLR 2024).
Decomposable multiscale mixing for time series forecasting.
Status: needs_pretrained_checkpoint — stub with import-guard.
Set TIMEMIXER_CHECKPOINT env var to a trained .pth file to activate.
"""
import os
import sys
import numpy as np
import pandas as pd

CLONE_DIR = os.path.join(
    os.path.expanduser("~"),
    "Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive",
    "AI-Tools/repos-claude-clones/TimeMixer",
)
CHECKPOINT = os.environ.get("TIMEMIXER_CHECKPOINT", "")


def add_timemixer_features(df: pd.DataFrame, lookback: int = 64) -> pd.DataFrame:
    """
    Add TimeMixer-based 1-step-ahead predicted return.

    Args:
        df: DataFrame with [open, high, low, close, volume] columns.
        lookback: context window (seq_len) for TimeMixer.

    Returns:
        df with added `timemixer_predicted_ret_t1` column.
    """
    df = df.copy()
    df["timemixer_predicted_ret_t1"] = np.nan

    if not CHECKPOINT:
        # TODO: train TimeMixer on OHLCV data and set TIMEMIXER_CHECKPOINT
        return df

    try:
        import torch

        if CLONE_DIR not in sys.path:
            sys.path.insert(0, CLONE_DIR)

        from models.TimeMixer import Model  # type: ignore

        cfg = type(
            "Cfg",
            (),
            {
                "task_name": "short_term_forecast",
                "seq_len": lookback,
                "pred_len": 1,
                "label_len": lookback // 2,
                "enc_in": 5,
                "c_out": 1,
                "d_model": 16,
                "d_ff": 32,
                "e_layers": 2,
                "dropout": 0.1,
                "decomp_method": "moving_avg",
                "moving_avg": 25,
                "down_sampling_layers": 3,
                "down_sampling_method": "avg",
                "down_sampling_window": 2,
                "channel_independence": 0,
                "use_future_temporal_feature": 0,
            },
        )()

        model = Model(cfg)
        state = torch.load(CHECKPOINT, map_location="cpu")
        model.load_state_dict(state)
        model.eval()

        cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        raw = df[cols].values.astype(np.float32)

        preds = []
        with torch.no_grad():
            for i in range(lookback, len(raw)):
                window = raw[i - lookback : i]
                mu = window.mean(0, keepdims=True)
                std = window.std(0, keepdims=True) + 1e-8
                w_norm = (window - mu) / std
                x = torch.tensor(w_norm[np.newaxis]).float()
                out = model(x, None, None, None)
                pred_norm = out[0, 0, 0].item()
                ci = cols.index("close")
                pred_price = pred_norm * std[0, ci] + mu[0, ci]
                cur_close = raw[i - 1, ci]
                preds.append(pred_price / cur_close - 1.0)

        df.iloc[lookback:, df.columns.get_loc("timemixer_predicted_ret_t1")] = preds

    except ImportError:
        pass
    except Exception:
        pass

    return df
