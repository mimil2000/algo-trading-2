import plotly.graph_objects as go
import pandas as pd

def plot_candlestick(df: pd.DataFrame, title: str = "Simulated Price Data"):
    fig = go.Figure(data=[
        go.Candlestick(
            x=df["datetime"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price"
        )
    ])

    # Affiche les événements s'ils existent
    if "event" in df.columns and df["event"].any():
        event_times = df[df["event"]]["datetime"]
        fig.add_trace(go.Scatter(
            x=event_times,
            y=[df.loc[df["datetime"] == t, "high"].values[0] * 1.01 for t in event_times],
            mode="markers",
            marker=dict(color="red", size=8),
            name="Event"
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=600
    )

    fig.show()
