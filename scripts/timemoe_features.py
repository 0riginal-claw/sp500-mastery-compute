"""
Time-MoE wrapper — zero-shot multi-step forecast features.
Downloads TimeMoE-50M from HuggingFace on first use (~200MB).
Adds predicted next-N-steps as features to the input DataFrame.
"""
import os
import sys
import pandas as pd
import numpy as np

_REPO = os.path.join(
    os.path.dirname(__file__),
    os.pardir, os.pardir,
    "repos-claude-clones", "Time-MoE"
)
sys.path.insert(0, os.path.normpath(_REPO))

_MODEL_ID = "Maple728/TimeMoE-50M"


def add_timemoe_features(
    df: pd.DataFrame,
    target_col: str = "close",
    prediction_length: int = 5,
    context_length: int = 64,
    device: str = "cpu",
    model_id: str = _MODEL_ID,
) -> pd.DataFrame:
    """
    Add Time-MoE zero-shot forecast values as features.

    For each row i, uses context_length prior closes to predict the next
    `prediction_length` steps; the row-i feature is the 1-step-ahead forecast.

    Parameters
    ----------
    df : pd.DataFrame  — must contain `target_col`
    target_col : str   — column to forecast (default 'close')
    prediction_length : int — forecast horizon stored in timemoe_forecast_h columns
    context_length : int — context window fed to the model (max 4096)
    device : str       — 'cpu' or 'cuda'
    model_id : str     — HuggingFace model id

    Returns
    -------
    pd.DataFrame with added columns:
        timemoe_forecast_h1 … timemoe_forecast_hN
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError:
        # Return df unchanged if deps missing
        return df

    out = df.copy()
    series = out[target_col].values.astype(float)
    n = len(series)

    forecasts = np.full((n, prediction_length), np.nan)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    with torch.no_grad():
        for i in range(context_length, n):
            ctx = series[max(0, i - context_length): i]
            # normalise
            mean = ctx.mean()
            std = ctx.std() + 1e-8
            ctx_norm = (ctx - mean) / std
            inp = torch.tensor(ctx_norm, dtype=torch.float32).unsqueeze(0)
            out_tensor = model.generate(inp, max_new_tokens=prediction_length)
            pred = out_tensor[0, -prediction_length:].cpu().numpy()
            forecasts[i] = pred * std + mean

    for h in range(1, prediction_length + 1):
        out[f"timemoe_forecast_h{h}"] = forecasts[:, h - 1]

    return out
