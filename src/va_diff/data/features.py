import numpy as np
import pandas as pd


def add_return_and_volatility(df: pd.DataFrame, window: int = 24) -> pd.DataFrame:
    out = df.copy()
    out["log_return"] = np.log(out["close"]).diff()
    out["realized_vol"] = out["log_return"].rolling(window).std()
    out["hl_range"] = (out["high"] - out["low"]) / out["close"]
    return out.dropna()
