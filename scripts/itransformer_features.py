"""
iTransformer feature wrapper — thuml/iTransformer (ICLR 2024 Spotlight).
Inverted Transformer: attention across variates, not timesteps.
Wraps pretrained iTransformer checkpoint for 1-step-ahead return prediction.
Status: needs_pretrained_checkpoint — stub with import-guard.
"""
import os
import numpy as np
import pandas as pd

CLONE_DIR = os.path.join(
    os.path.expanduser("~"),
    "Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive",
    "AI-Tools/repos-claude-clones/iTransformer",
)
CHECKPOINT = os.environ.get("ITRANSFORMER_CHECKPOINT", "")


def add_itransformer_features(df: pd.DataFrame, lookback: int = 64) -> pd.DataFrame:
    """
    Add iTransformer-based 1-step-ahead predicted return to df.

    Args:
        df: DataFrame with at least [open, high, low, close, volume] columns.
        lookback: context window length fed to the model.

    Returns:
        df with added column `itransformer_predicted_ret_t1`.
    """
    df = df.copy()
    df["itransformer_predicted_ret_t1"] = np.nan

    if not CHECKPOINT:
        # TODO: set ITRANSFORMER_CHECKPOINT env var to a trained .pth path
        return df

    try:
        import sys
        import torch

        if CLONE_DIR not in sys.path:
            sys.path.insert(0, CLONE_DIR)

        from model.iTransformer import Model  # type: ignore

        config_class = type(
            "Cfg",
            (),
            {
                "seq_len": lookback,
                "pred_len": 1,
                "d_model": 128,
                "e_layers": 3,
                "n_heads": 8,
                "d_ff": 256,
                "dropout": 0.1,
                "activation": "gelu",
                "output_attention": False,
                "use_norm": True,
                "enc_in": 5,  # OHLCV
                "c_out": 1,
                "class_strategy": "projection",
            },
        )

        model = Model(config_class())
        state = torch.load(CHECKPOINT, map_location="cpu")
        model.load_state_dict(state)
        model.eval()

        cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        raw = df[cols].values.astype(np.float32)

        preds = []
        with torch.no_grad():
            for i in range(lookback, len(raw)):
                window = raw[i - lookback : i]  # (lookback, 5)
                mu = window.mean(0, keepdims=True)
                std = window.std(0, keepdims=True) + 1e-8
                w_norm = (window - mu) / std
                x = torch.tensor(w_norm[np.newaxis]).float()  # (1, lookback, 5)
                out = model(x, None, None, None)  # (1, 1, 1)
                pred_price_norm = out[0, 0, 0].item()
                # Denormalize close
                close_std = std[0, cols.index("close")]
                close_mu = mu[0, cols.index("close")]
                pred_price = pred_price_norm * close_std + close_mu
                cur_close = raw[i - 1, cols.index("close")]
                preds.append(pred_price / cur_close - 1.0)

        df.iloc[lookback:, df.columns.get_loc("itransformer_predicted_ret_t1")] = preds

    except ImportError:
        # iTransformer or torch not installed — return df unchanged
        pass
    except Exception:
        pass

    return df
