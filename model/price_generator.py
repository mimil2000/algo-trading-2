import pandas as pd
import numpy as np
from datetime import datetime
from .utils import generate_datetime_index

def generate_price(symbol: str, start: datetime, end: datetime, frequency: str = "1min", method: str = "gbm_jump") -> pd.DataFrame:
    index = generate_datetime_index(start, end, frequency)
    n = len(index)

    from .assets import ASSET_CONFIG
    config = ASSET_CONFIG.get(symbol)
    if not config:
        raise ValueError(f"Asset config not found for symbol: {symbol}")

    mu = config["mu"]
    sigma = config["sigma"]
    s0 = config["start_price"]
    dt = 1 / (252 * 24 * 60)  # fréquence 1 min

    # Paramètres des sauts
    lambda_jump = 0.0005     # proba de saut par minute (~1 par jour)
    mu_jump = 0              # saut moyen (log-normal centré)
    sigma_jump = 0.03        # volatilité des sauts

    prices = [s0]
    for _ in range(1, n):
        z = np.random.normal()
        jump = 0
        if np.random.rand() < lambda_jump:
            jump = np.random.normal(mu_jump, sigma_jump)
        drift = (mu - 0.5 * sigma ** 2) * dt
        diffusion = sigma * np.sqrt(dt) * z
        st = prices[-1] * np.exp(drift + diffusion + jump)
        prices.append(st)

    df = pd.DataFrame({
        "datetime": index,
        "open": prices,
        "high": [p + np.random.uniform(0, 0.01 * p) for p in prices],
        "low": [p - np.random.uniform(0, 0.01 * p) for p in prices],
        "close": [p + np.random.normal(0, 0.005 * p) for p in prices],
        "event": [False] + [np.random.rand() < lambda_jump for _ in range(n - 1)],
    })
    return df


if __name__ == "__main__" :

    generate_price()