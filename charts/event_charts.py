import plotly.graph_objects as go
from services.porcessors import (
    events_analysis_processor,
    detect_spikes_global,
    detect_spikes_robust,
)


def plot_customer_events(sellin, sellout, selected_customer, threshold=2.0):
    selected_sellin = sellin[sellin["customer_code"] == selected_customer]
    selected_sellout = sellout[sellout["customer_code"] == selected_customer]

    df = events_analysis_processor(selected_sellin, selected_sellout)
    df = detect_spikes_global(df, threshold)
    # df = detect_spikes_robust(df, threshold=threshold)
    spikes = df[df["is_spike"]]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["sellin"],
            name="Sell-in",
            mode="lines+markers",
            line=dict(color="purple", width=2),
            marker=dict(size=5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["sellout"],
            name="Sell-out",
            mode="lines+markers",
            line=dict(color="orangered", width=2),
            marker=dict(size=5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["stock_remaining"],
            name="Stock Remaining",
            mode="lines+markers",
            line=dict(color="black", width=2),
            marker=dict(size=5),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=spikes["date"],
            y=spikes["sellout"],
            name="Spike",
            mode="markers",
            marker=dict(
                color="yellow",
                size=14,
                symbol="circle",
                line=dict(color="black", width=1),
            ),
        )
    )

    fig.update_layout(
        title="Sell-in vs Sell-out + Stock Remaining",
        xaxis=dict(title="Date"),
        yaxis=dict(title="Quantity"),
        hovermode="x unified",
        clickmode="event",
        legend=dict(orientation="v", x=1.01, y=1),
        height=450,
        margin=dict(t=60, b=40),
    )

    return fig
