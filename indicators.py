import pandas as pd


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    for window in [20, 50, 200]:
        df[f"ma{window}"] = df["close"].rolling(window).mean()
    return df


def add_bollinger_bands(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    ma  = df["close"].rolling(window).mean()
    std = df["close"].rolling(window).std()
    df["bb_mid"]   = ma
    df["bb_upper"] = ma + 2 * std
    df["bb_lower"] = ma - 2 * std
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    delta    = df["close"].diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, float("nan"))
    df["rsi"] = 100 - (100 / (1 + rs))
    return df
