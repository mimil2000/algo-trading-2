from model.price_generator import generate_price
from datetime import datetime, timedelta

from model.visualizer import plot_candlestick

if __name__ == "__main__":

    start = datetime(2024, 1, 1, 9, 0)
    end = start + timedelta(days=1)
    df = generate_price("EURUSD", start, end, frequency="1h")
    plot_candlestick(df, title="Simulation de prix - EURUSD")
