"""
TFTS (LongxingTan/Time-series-prediction) wrapper.
Adds seq2seq LSTM 1-step-ahead forecast as a feature using the tfts AutoModel.
"""
import os
import sys
import pandas as pd
import numpy as np

_REPO = os.path.join(
    os.path.dirname(__file__),
    os.pardir, os.pardir,
    "repos-claude-clones", "Time-series-prediction"
)
sys.path.insert(0, os.path.normpath(_REPO))


def add_time_series_prediction_features(
    df: pd.DataFrame,
    target_col: str = "close",
    seq_len: int = 24,
    model_type: str = "seq2seq",
    epochs: int = 5,
) -> pd.DataFrame:
    """
    Fit a lightweight TFTS seq2seq model on the close series and add
    1-step-ahead in-sample predictions as a feature column.

    Parameters
    ----------
    df : pd.DataFrame  — must contain `target_col`
    target_col : str   — target column (default 'close')
    seq_len : int      — input sequence length
    model_type : str   — tfts model key ('seq2seq', 'rnn', 'transformer', etc.)
    epochs : int       — training epochs (keep low for feature engineering use)

    Returns
    -------
    pd.DataFrame with added column: tfts_pred_<model_type>
    """
    try:
        import tensorflow as tf
        from tfts import AutoModel, AutoConfig, KerasTrainer
    except ImportError:
        return df

    out = df.copy()
    series = out[target_col].values.astype(np.float32)
    n = len(series)
    if n < seq_len + 2:
        return out

    # normalise
    mean, std = series.mean(), series.std() + 1e-8
    norm = (series - mean) / std

    # build (X, y) windows
    X = np.array([norm[i: i + seq_len] for i in range(n - seq_len - 1)])
    y = norm[seq_len: n - 1].reshape(-1, 1)
    X = X.reshape(-1, seq_len, 1)

    config = AutoConfig.for_model(model_type)
    model_obj = AutoModel.for_model(model_type, config)
    trainer = KerasTrainer(model_obj)
    trainer.train(
        (X, y), valid_dataset=None, n_epochs=epochs, batch_size=32, learning_rate=1e-3
    )

    preds_norm = trainer.predict(X).flatten()
    preds = preds_norm * std + mean

    col = f"tfts_pred_{model_type}"
    out[col] = np.nan
    out.iloc[seq_len: seq_len + len(preds), out.columns.get_loc(col)] = preds

    return out
