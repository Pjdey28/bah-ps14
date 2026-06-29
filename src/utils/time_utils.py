import pandas as pd

def ensure_datetime(series):
    return pd.to_datetime(series)

def sort_time(df):
    return df.sort_values("time").reset_index(drop=True)