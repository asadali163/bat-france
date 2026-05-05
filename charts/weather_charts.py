import plotly.graph_objects as go
from plotly.subplots import make_subplots
from services.porcessors import weather_analysis_processor


def plot_customer_weather(df, rain_range, robust=True):
    df = weather_analysis_processor(df, robust)
    rain_precipitation = df["precipitation"].where(
        df["precipitation"].between(rain_range[0], rain_range[1])
    )

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Sales, Trend & Seasonal", "Residual & Precipitation"),
        row_heights=[0.6, 0.4],
        specs=[[{"secondary_y": True}], [{"secondary_y": True}]],
    )

    # Row 1: Sales + Trend (primary y) + Seasonal (secondary y)
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["sales_quantity"],
            name="Sales",
            mode="lines+markers",
            line=dict(color="steelblue", width=1.5),
            # opacity=0.5,
        ),
        row=1,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["trend"],
            name="Trend",
            mode="lines",
            line=dict(color="crimson", width=2.5),
            opacity=0.5,
        ),
        row=1,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["seasonal"],
            name="Seasonal",
            mode="lines",
            line=dict(color="darkorange", width=1.5, dash="dot"),
            fill="tozeroy",
            fillcolor="rgba(255,140,0,0.08)",
            visible="legendonly",
        ),
        row=1,
        col=1,
        secondary_y=True,
    )

    # Row 2: Residual (primary y) + Precipitation (secondary y)
    fig.add_trace(
        go.Bar(
            x=df["date"],
            y=df["residual"],
            name="Residual",
            marker_color=df["residual"].apply(
                lambda v: "rgba(34,139,34,0.6)" if v >= 0 else "rgba(220,20,60,0.6)"
            ),
        ),
        row=2,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(
            x=df["date"],
            y=rain_precipitation,
            name="Precipitation",
            marker_color="lightblue",
            opacity=0.7,
        ),
        row=2,
        col=1,
        secondary_y=True,
    )

    fig.update_layout(
        height=700,
        hovermode="x unified",
        showlegend=True,
        legend=dict(orientation="h", y=1.02, x=0),
        margin=dict(t=60, b=40),
        barmode="overlay",
    )
    fig.update_yaxes(title_text="Sales / Trend", row=1, col=1, secondary_y=False)
    fig.update_yaxes(
        title_text="Seasonal", row=1, col=1, secondary_y=True, showgrid=False
    )
    fig.update_yaxes(title_text="Residual", row=2, col=1, secondary_y=False)
    fig.update_yaxes(
        title_text="Precipitation (mm)", row=2, col=1, secondary_y=True, showgrid=False
    )
    fig.update_xaxes(title_text="Date", row=2, col=1)

    return fig
