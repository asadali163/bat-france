import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from services.porcessors import weather_analysis_processor, rain_band_processor, ols_rain_processor


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


def plot_rain_band_chart(df, band_stats, stats_dict):
    colors = ["#9aa5b1", "#5b9bd5", "#2f6da8", "#c0392b"]
    band_order = ["none", "light", "moderate", "heavy"]

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.42, 0.58],
        subplot_titles=(
            "Mean STL remainder by rainfall band",
            "Remainder vs precipitation (each dot = one day)",
        ),
    )

    # Left: bar chart per band with error bars
    fig.add_trace(
        go.Bar(
            x=band_stats["band"],
            y=band_stats["mean"],
            error_y=dict(type="data", array=band_stats["sem"], visible=True),
            marker_color=[colors[band_order.index(b)] for b in band_stats["band"]],
            customdata=band_stats[["count", "std"]].values,
            hovertemplate=(
                "<b>%{x}</b><br>mean remainder: %{y:.2f} units"
                "<br>days: %{customdata[0]}"
                "<br>std: %{customdata[1]:.1f}<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=1, col=1)

    # Right: scatter colored by band
    for b, c in zip(band_order, colors):
        sub = df[df["band"] == b]
        if sub.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=sub["precipitation"],
                y=sub["remainder"],
                mode="markers",
                name=b,
                marker=dict(color=c, size=6, opacity=0.6),
                hovertemplate=(
                    f"<b>{b}</b><br>precip: %{{x:.1f}} mm"
                    "<br>remainder: %{y:.1f} units<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )
    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=1, col=2)

    corr = stats_dict.get("corr", float("nan"))
    p_anova = stats_dict.get("p_anova", float("nan"))
    p_ttest = stats_dict.get("p_ttest", float("nan"))
    title = (
        f"Rain vs sales — STL remainder  |  "
        f"corr={corr:+.3f}  |  dry vs rainy p={p_ttest:.3f}  |  ANOVA p={p_anova:.3f}"
    )

    fig.update_xaxes(title_text="rainfall band", row=1, col=1)
    fig.update_yaxes(title_text="STL remainder (units vs expected)", row=1, col=1)
    fig.update_xaxes(title_text="precipitation (mm)", row=1, col=2)
    fig.update_layout(
        height=460,
        title_text=title,
        template="plotly_white",
        hovermode="closest",
        margin=dict(t=80, b=40),
    )

    return fig


def plot_ols_rain_chart(coef_df, scalar_df, meta):
    if coef_df is None or coef_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text=meta.get("error", "Not enough data"),
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14),
        )
        fig.update_layout(height=200, template="plotly_white")
        return fig

    band_colors = {"light": "#5b9bd5", "moderate": "#2f6da8", "heavy": "#c0392b"}
    bands = ["light", "moderate", "heavy"]

    def sig_label(p):
        return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""

    same_day = coef_df[coef_df["effect"] == "Same-day"]

    fig = go.Figure()

    for band in bands:
        row = same_day[same_day["band"] == band]
        if row.empty:
            continue
        row = row.iloc[0]
        sig = sig_label(row["p"])
        fig.add_trace(
            go.Bar(
                name=band,
                x=[band],
                y=[row["pct_change"]],
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=[row["ci_high"] - row["pct_change"]],
                    arrayminus=[row["pct_change"] - row["ci_low"]],
                    visible=True,
                ),
                marker_color=band_colors[band],
                hovertemplate=(
                    f"<b>{band}</b><br>"
                    "% change: %{y:.2f}%<br>"
                    f"p={row['p']:.3f} {sig}<extra></extra>"
                ),
                showlegend=False,
            )
        )

    fig.add_hline(y=0, line_dash="dot", line_color="gray")

    r2 = meta.get("r2", float("nan"))
    n = meta.get("n", 0)

    fig.update_xaxes(title_text="Rainfall band")
    fig.update_yaxes(title_text="% change in sales vs dry day")
    fig.update_layout(
        height=400,
        title_text=f"OLS: Same-day rain effect on sales  |  R²={r2:.3f}  |  n={n} days",
        template="plotly_white",
        hovermode="closest",
        margin=dict(t=60, b=40),
    )

    return fig
