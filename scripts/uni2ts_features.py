"""
uni2ts / Moirai feature wrapper — SalesforceAIResearch/uni2ts.
Unified Training of Universal Time Series Transformers (Moirai foundation model).
Weights at: https://huggingface.co/Salesforce/moirai-1.0-R-small (and variants)
Status: needs_pretrained_checkpoint — set UNI2TS_MODEL env var.
"""
import os
import sys
import numpy as np
import pandas as pd

CLONE_DIR = os.path.join(
    os.path.expanduser("~"),
    "Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive",
    "AI-Tools/repos-claude-clones/uni2ts",
)
# Options: "Salesforce/moirai-1.0-R-small", "Salesforce/moirai-1.0-R-base", etc.
MODEL_NAME = os.environ.get("UNI2TS_MODEL", "")


def add_uni2ts_features(df: pd.DataFrame, lookback: int = 64) -> pd.DataFrame:
    """
    Add Moirai (uni2ts) 1-step-ahead predicted return — zero-shot inference.

    Args:
        df: DataFrame with [close] (and optionally open/high/low/volume).
        lookback: context length (past_length) fed to Moirai.

    Returns:
        df with added `uni2ts_predicted_ret_t1` column.
    """
    df = df.copy()
    df["uni2ts_predicted_ret_t1"] = np.nan

    if not MODEL_NAME:
        # TODO: set UNI2TS_MODEL=Salesforce/moirai-1.0-R-small
        return df

    try:
        import torch
        from einops import rearrange  # type: ignore

        if CLONE_DIR not in sys.path:
            sys.path.insert(0, CLONE_DIR)

        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule  # type: ignore

        module = MoiraiModule.from_pretrained(MODEL_NAME)
        model = MoiraiForecast(
            module=module,
            prediction_length=1,
            context_length=lookback,
            patch_size="auto",
            num_samples=100,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        model.eval()

        close = df["close"].values.astype(np.float32)
        preds = []

        with torch.no_grad():
            for i in range(lookback, len(close)):
                window = close[i - lookback : i]
                past_target = torch.tensor(window[np.newaxis, np.newaxis]).float()
                past_observed = torch.ones_like(past_target, dtype=torch.bool)
                past_is_pad = torch.zeros(1, lookback, dtype=torch.bool)
                samples = model(
                    past_target=past_target,
                    past_observed_target=past_observed,
                    past_is_pad=past_is_pad,
                )  # (1, num_samples, 1)
                median_pred = float(samples.median().item())
                cur_close = float(close[i - 1])
                preds.append(median_pred / cur_close - 1.0)

        df.iloc[lookback:, df.columns.get_loc("uni2ts_predicted_ret_t1")] = preds

    except ImportError:
        # uni2ts / einops not installed — return unchanged
        pass
    except Exception:
        pass

    return df
