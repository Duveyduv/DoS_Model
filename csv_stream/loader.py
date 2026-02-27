import pandas as pd
import re

def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        re.sub(r"[^0-9a-zA-Z_]+", "_", c).strip("_").lower()
        for c in df.columns
    ]
    return df


def stream_csv(path: str):
    for chunk in pd.read_csv(path, chunksize=1):
        chunk = clean_cols(chunk)
        yield chunk
