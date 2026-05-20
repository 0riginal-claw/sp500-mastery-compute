"""
Lag-Llama feature wrapper — time-series-foundation-models/lag-llama.
First open-source foundation model for probabilistic time series forecasting.
Weights at: https://huggingface.co/time-series-foundation-models/Lag-Llama
Status: needs_pretrained_checkpoint — set LAG_LLAMA_CHECKPOINT env var.
"""
import os
import sys
import numpy as np
import pandas as pd

CLONE_DIR = os.path.join(
    os.path.expanduser("~"),
    "Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive",
    "AI-Tools/repos-claude-clones/lag-llama",
)
# Download: huggingface_hub.hf_hub_download("time-series-foundation-models/Lag-Llama", "lag-llama.ckpt")
CHECKPOINT = os.environ.get("LAG_LLAMA_CHECKPOINT", "")


def add_lag_llama_features(df: pd.DataFrame, lookback: int = 64) -> pd.DataFrame:
    """
    Add Lag-Llama median predicted return (zero-shot) for next bar.

    Args:
        df: DataFrame with [close] (and optionally open/high/low/volume).
        lookback: context length in bars fed to Lag-Llama.

    Returns:
        df with added `lag_llama_predicted_ret_t1` column.
    """
    df = df.copy()
    df["lag_llama_predicted_ret_t1"] = np.nan

    if not CHECKPOINT:
        # TODO: download ckpt via huggingface_hub and set LAG_LLAMA_CHECKPOINT
        return df

    try:
        import torch
        from gluonts.dataset.pandas import PandasDataset  # type: ignore

        if CLONE_DIR not in sys.path:
            sys.path.insert(0, CLONE_DIR)

        from lag_llama.gluon.estimator import LagLlamaEstimator  # type: ignore

        ckpt = torch.load(CHECKPOINT, map_location="cpu")
        estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

        estimator = LagLlamaEstimator(
            ckpt_path=CHECKPOINT,
            prediction_length=1,
            context_length=lookback,
            **{k: v for k, v in estimator_args.items() if k != "context_length"},
        )
        lightning_module = estimator.create_lightning_module()
        transformation = estimator.create_transformation()
        predictor = estimator.create_predictor(transformation, lightning_module)

        close = df["close"]
        preds = []
        for i in range(lookback, len(df)):
            window = close.iloc[i - lookback : i]
            ds = PandasDataset(
                dict(target=window.reset_index(drop=True)), freq="B"
            )
            forecasts = list(predictor.predict(ds))
            median_pred = float(np.median(forecasts[0].samples))
            cur_close = float(close.iloc[i - 1])
            preds.append(median_pred / cur_close - 1.0)

        df.iloc[lookback:, df.columns.get_loc("lag_llama_predicted_ret_t1")] = preds

    except ImportError:
        # gluonts / lag_llama not installed — return unchanged
        pass
    except Exception:
        pass

    return df
