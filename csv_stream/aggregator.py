import pandas as pd
import numpy as np

STATS = ["mean", "std", "min", "max"]

def temporal_aggregate(buffer: pd.DataFrame, features: list[str]):
    rolled = (
        buffer[features]
        .rolling(window=len(buffer), min_periods=len(buffer))
        .agg(STATS)
    )

    rolled.columns = [
        f"{feat}_{stat}"
        for feat, stat in rolled.columns
    ]

    rolled = rolled.replace([np.inf, -np.inf], np.nan)
    return rolled.dropna()
