import pandas as pd
from datetime import datetime

def generate_datetime_index(start: datetime, end: datetime, frequency: str) -> pd.DatetimeIndex:
    return pd.date_range(start=start, end=end, freq=frequency)
