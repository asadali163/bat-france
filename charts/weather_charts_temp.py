import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

from charts.weather_charts_wind import _driver_agg


_SEASON_COLORS = {
    "Winter": "#378ADD",
    "Spring": "#1D9E75",
    "Summer": "#D85A30",
    "Autumn": "#BA7517",
}
_SEASON_ORDER = ["Winter", "Spring", "Summer", "Autumn"]

_TEMP_COLOR      = "#D85A30"
_TEMP_FILL_COLOR = "rgba(216,90,48,0.12)"
_TEMP_KNOTS      = [5, 15, 25]


def plot_temp_contribution(
    temp_contrib_avg, season_contrib, n_customers: int, title: str
):
    """Monthly temp contribution bars + seasonal bars. Mirrors forecasting_fbp.ipynb cell 42."""
    if temp_contrib_avg.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Not enough data.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14),
        )
        fig.update_layout(height=300, template="plotly_white")
        return fig

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Avg temperature contribution to sales by month",
            "Avg temperature contribution by season",
        ),
        specs=[[{"secondary_y": True}, {"secondary_y": False}]],
        horizontal_spacing=0.15,
    )

    fig.add_trace(
        go.Bar(
            x=temp_contrib_avg["month_name"],
            y=temp_contrib_avg["avg_contribution"].round(4),
            name="temperature contribution to sales",
            marker_color=[
                "#1D9E75" if v >= 0 else "#D85A30"
                for v in temp_contrib_avg["avg_contribution"]
            ],
            error_y=dict(
                type="data",
                symmetric=True,
                array=temp_contrib_avg["std_contribution"].round(4),
                color="#5F5E5A",
                thickness=1,
                width=4,
            ),
            hovertemplate="%{x}<br>avg contribution: %{y:.2f}%<extra></extra>",
            showlegend=True,
        ),
        row=1,
        col=1,
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=temp_contrib_avg["month_name"],
            y=temp_contrib_avg["avg_temp"].round(1),
            name="actual temperature (°C)",
            mode="lines+markers",
            line=dict(color="#888780", width=2, dash="dot"),
            marker=dict(size=7, color="#888780"),
            hovertemplate="%{x}<br>avg temp: %{y:.1f}°C<extra></extra>",
            showlegend=True,
        ),
        row=1,
        col=1,
        secondary_y=True,
    )

    fig.add_hline(
        y=0, line_dash="dash", line_color="#888780", line_width=1, row=1, col=1
    )

    for _, row_s in season_contrib.iterrows():
        season = row_s["season"]
        if pd.isna(row_s.get("avg_contribution", float("nan"))):
            continue
        fig.add_trace(
            go.Bar(
                x=[season],
                y=[round(row_s["avg_contribution"], 4)],
                name=season,
                marker_color=_SEASON_COLORS.get(season, "#888"),
                error_y=dict(
                    type="data",
                    symmetric=True,
                    array=[round(row_s["std_contribution"], 4)],
                    color="#5F5E5A",
                    thickness=1,
                    width=6,
                ),
                customdata=[round(row_s["avg_temp"], 1)],
                hovertemplate=(
                    f"<b>{season}</b><br>avg contribution: %{{y:.2f}}%<br>"
                    "avg temp: %{customdata[0]:.1f}°C<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    fig.add_hline(
        y=0, line_dash="dash", line_color="#888780", line_width=1, row=1, col=2
    )

    fig.update_layout(
        title=dict(
            text=f"{title}<br><sub>β × actual temperature ÷ avg monthly sales = "
            "% of sales added/removed by temperature each month</sub>",
            font=dict(size=14),
            x=0.5,
        ),
        height=420,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial", size=11, color="#2C2C2A"),
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5
        ),
        margin=dict(l=20, r=20, t=100, b=100),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#F1EFE8",
        secondary_y=False,
        title_text="% of avg sales contributed by temperature",
        ticksuffix="%",
        row=1,
        col=1,
    )
    fig.update_yaxes(
        showgrid=False,
        secondary_y=True,
        title_text="actual temp (°C)",
        title_font=dict(color="#888780"),
        tickfont=dict(color="#888780"),
        row=1,
        col=1,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#F1EFE8",
        title_text="% of avg sales contributed by temperature",
        ticksuffix="%",
        row=1,
        col=2,
    )
    fig.update_xaxes(showgrid=False)
    return fig


def plot_ols_temp_chart(curve_df, meta):
    """Smooth natural spline curve: temperature effect on sales (single customer)."""
    if curve_df is None or curve_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text=meta.get("error", "Not enough data"),
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14),
        )
        fig.update_layout(height=200, template="plotly_white")
        return fig

    ref_temp = float(curve_df["ref_temp"].iloc[0])
    xs   = curve_df["x_celsius"].values
    ys   = curve_df["y_pct"].values
    y_up = curve_df["y_upper"].values
    y_dn = curve_df["y_lower"].values
    r2   = meta.get("r2", float("nan"))
    n    = meta.get("n", 0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.concatenate([xs, xs[::-1]]),
        y=np.concatenate([y_up, y_dn[::-1]]),
        fill="toself", fillcolor=_TEMP_FILL_COLOR,
        line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines",
        line=dict(color=_TEMP_COLOR, width=2.5),
        hovertemplate="<b>%{x:.1f}°C</b><br>effect: %{y:+.2f}%<extra></extra>",
        showlegend=False,
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="gray", line_width=1)
    fig.add_vline(x=ref_temp, line_dash="dash", line_color="#888", line_width=1.5,
                  annotation_text=f"avg {ref_temp:.1f}°C (ref)",
                  annotation_position="top right",
                  annotation_font=dict(size=10, color="#888"))
    for k in _TEMP_KNOTS:
        if float(xs.min()) <= k <= float(xs.max()):
            fig.add_vline(x=k, line_dash="dot", line_color="#ddd", line_width=1,
                          annotation_text=f"{k}°C",
                          annotation_position="bottom",
                          annotation_font=dict(size=9, color="#aaa"))

    fig.update_layout(
        title_text=f"OLS: Temperature effect on sales  |  R²={r2:.3f}  |  n={n} days",
        xaxis=dict(title="Temperature (°C)", showgrid=True, gridcolor="#f0f0f0"),
        yaxis=dict(title=f"% change vs avg temp ({ref_temp:.1f}°C)",
                   showgrid=True, gridcolor="#f0f0f0"),
        template="plotly_white", height=420,
        margin=dict(t=60, b=40, l=70, r=40),
    )
    return fig


def plot_ols_temp_effect(agg_df: pd.DataFrame, title: str, ref_temp: float = None):
    """Smooth natural spline curve: mean temperature effect, averaged across shops."""
    if agg_df is None or agg_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Not enough data for this selection.",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14),
        )
        fig.update_layout(height=300, template="plotly_white")
        return fig

    n_shops = int(agg_df["n_shops"].iloc[0])
    xs   = agg_df["x_celsius"].values
    ys   = agg_df["mean_pct"].values
    y_up = agg_df["ci_upper"].values
    y_dn = agg_df["ci_lower"].values

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.concatenate([xs, xs[::-1]]),
        y=np.concatenate([y_up, y_dn[::-1]]),
        fill="toself", fillcolor=_TEMP_FILL_COLOR,
        line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines",
        line=dict(color=_TEMP_COLOR, width=2.5),
        hovertemplate="<b>%{x:.1f}°C</b><br>avg effect: %{y:+.2f}%<extra></extra>",
        showlegend=False,
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="gray", line_width=1)
    if ref_temp is not None:
        fig.add_vline(x=ref_temp, line_dash="dash", line_color="#888", line_width=1.5,
                      annotation_text=f"ref {ref_temp:.1f}°C",
                      annotation_position="top right",
                      annotation_font=dict(size=10, color="#888"))
    for k in _TEMP_KNOTS:
        if float(xs.min()) <= k <= float(xs.max()):
            fig.add_vline(x=k, line_dash="dot", line_color="#ddd", line_width=1,
                          annotation_text=f"{k}°C", annotation_position="bottom",
                          annotation_font=dict(size=9, color="#aaa"))
    fig.add_annotation(
        x=0.99, y=0.97, xref="paper", yref="paper", xanchor="right",
        showarrow=False,
        text=f"n={n_shops:,} shops",
        font=dict(size=11, color="#555"),
        bgcolor="rgba(255,255,255,0.8)",
    )
    ref_label = f" ({ref_temp:.1f}°C)" if ref_temp is not None else ""

    # Base y-range on the mean curve, not CI tails (spline CIs blow up at extremes)
    pad    = max(3.0, (float(ys.max()) - float(ys.min())) * 0.4 + 1.0)
    y_min  = float(ys.min()) - pad
    y_max  = float(ys.max()) + pad

    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left"),
        xaxis=dict(title="Temperature (°C)", showgrid=True, gridcolor="#f0f0f0"),
        yaxis=dict(title=f"% change vs ref temp{ref_label}",
                   showgrid=True, gridcolor="#f0f0f0",
                   range=[y_min, y_max]),
        template="plotly_white",
        showlegend=False, height=460,
        margin=dict(l=70, r=30, t=90, b=70),
    )
    return fig


def plot_temp_anomaly_effect(
    beta_vals, mean_beta: float, ci: float, n_shops: int, title: str
):
    """
    2-panel chart for temperature anomaly analysis.
    Left:  histogram of β_temp_anomaly across shops.
    Right: sensitivity line — expected sales impact for −8°C to +12°C anomaly.
    """
    if n_shops < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Not enough data for this selection.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14),
        )
        fig.update_layout(height=300, template="plotly_white")
        return fig

    anomaly_range = np.arange(-8, 13, 1)
    impact = mean_beta * anomaly_range
    impact_hi = (mean_beta + ci) * anomaly_range
    impact_lo = (mean_beta - ci) * anomaly_range

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"Distribution of temperature sensitivity across {n_shops} shops",
            "Sales impact by temperature anomaly (°C above/below normal)",
        ),
        horizontal_spacing=0.14,
    )

    # Left: histogram of β
    fig.add_trace(
        go.Histogram(
            x=beta_vals,
            nbinsx=30,
            marker_color="#B5D4F4",
            marker_line=dict(color="#185FA5", width=0.5),
            hovertemplate="β range: %{x:.3f}<br>shops: %{y}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_vline(
        x=mean_beta,
        line_dash="dash",
        line_color="#D85A30",
        line_width=2,
        annotation_text=f"mean β = {mean_beta:+.3f}",
        annotation_position="top right",
        annotation_font=dict(color="#D85A30", size=11),
        row=1,
        col=1,
    )

    # Right: sensitivity — shaded CI band then central line
    # Upper and lower CI bounds (flip for negative anomaly values)
    upper = np.maximum(impact_hi, impact_lo)
    lower = np.minimum(impact_hi, impact_lo)
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([anomaly_range, anomaly_range[::-1]]),
            y=np.concatenate([upper, lower[::-1]]),
            fill="toself",
            fillcolor="rgba(29,158,117,0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=anomaly_range,
            y=impact,
            mode="lines+markers",
            line=dict(color="#1D9E75", width=2.5),
            marker=dict(size=6, color="#1D9E75"),
            hovertemplate="Anomaly: %{x:+.0f}°C<br>Expected impact: %{y:+.3f} units<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_hline(
        y=0, line_dash="dash", line_color="#888780", line_width=1, row=1, col=2
    )
    fig.add_vline(
        x=0, line_dash="dot", line_color="#888780", line_width=1, row=1, col=2
    )

    fig.update_layout(
        title=dict(
            text=f"{title}<br>"
            "<sub>β × temperature anomaly = sales units added/removed vs a normal-temperature day</sub>",
            font=dict(size=14),
            x=0.5,
        ),
        height=430,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial", size=11, color="#2C2C2A"),
        margin=dict(l=20, r=20, t=100, b=60),
    )
    fig.update_xaxes(
        title_text="β (units per °C anomaly)",
        showgrid=True,
        gridcolor="#F1EFE8",
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text="number of shops", showgrid=True, gridcolor="#F1EFE8", row=1, col=1
    )
    fig.update_xaxes(
        title_text="temperature anomaly (°C above/below monthly average)",
        showgrid=True,
        gridcolor="#F1EFE8",
        row=1,
        col=2,
    )
    fig.update_yaxes(
        title_text="expected sales impact (units)",
        showgrid=True,
        gridcolor="#F1EFE8",
        row=1,
        col=2,
    )
    return fig


def plot_temp_swing(df: pd.DataFrame, title: str = "Temperature Swing vs Sales") -> go.Figure:
    """
    3-bar chart: Big Drop / Neutral / Big Rise day-over-day temperature change.
    Left panel: mean STL residual as % of shop mean.
    Right panel: % change vs Neutral baseline.
    """
    order = ["Big Drop (≤−5°C)", "Neutral (−5 to +5°C)", "Big Rise (≥+5°C)"]
    baseline = "Neutral (−5 to +5°C)"
    agg = _driver_agg(df, "swing_cat", order, baseline)

    colors_left = ["#ef5350" if v < 0 else "#66bb6a" for v in agg["pct_resid"]]
    colors_right = ["#ef5350" if v < 0 else "#66bb6a" for v in agg["pct_vs_baseline"]]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["STL Residual (% of shop mean)", "% Change vs Neutral Day"],
        horizontal_spacing=0.12,
    )

    for i, (yval, colors, ycol) in enumerate([
        (agg["pct_resid"], colors_left, "pct_resid"),
        (agg["pct_vs_baseline"], colors_right, "pct_vs_baseline"),
    ], start=1):
        for _, row in agg.iterrows():
            n = int(row["n_days"]) if not pd.isna(row["n_days"]) else 0
            y = row[ycol] if not pd.isna(row[ycol]) else 0
            clr = "#66bb6a" if y >= 0 else "#ef5350"
            fig.add_trace(go.Bar(
                x=[row["swing_cat"]], y=[y],
                name=row["swing_cat"],
                marker_color=clr,
                text=[f"{y:+.1f}%"],
                textposition="outside",
                hovertemplate=(
                    f"<b>{row['swing_cat']}</b><br>Value: {y:+.1f}%<br>"
                    f"n={n:,} unique dates<extra></extra>"
                ),
                showlegend=False,
            ), row=1, col=i)

    fig.update_layout(
        title=dict(text=title, font=dict(size=13), x=0.5),
        height=400, bargap=0.3,
        margin=dict(l=20, r=20, t=80, b=40),
        paper_bgcolor="white", plot_bgcolor="white",
    )
    fig.update_yaxes(zeroline=True, zerolinecolor="#ccc", gridcolor="#eee")
    return fig


def plot_prophet_seasonality(seas_avg: pd.DataFrame, n_shops: int, title: str):
    """Line + fill chart of Prophet yearly seasonality component averaged across shops by month."""
    if seas_avg.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No seasonality data.",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14),
        )
        fig.update_layout(height=300, template="plotly_white")
        return fig

    y = seas_avg["avg_yearly"].round(3).tolist()
    x = seas_avg["month_name"].tolist()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            line=dict(color="#7B4FA0", width=2.5),
            marker=dict(color="#7B4FA0", size=7),
            fill="tozeroy",
            fillcolor="rgba(148,103,189,0.13)",
            customdata=seas_avg["n_shops"].tolist(),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Seasonality: %{y:.2f} units<br>"
                "Shops: %{customdata}<extra></extra>"
            ),
            showlegend=False,
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#aaaaaa", line_width=1)

    fig.update_layout(
        title=dict(text="Prophet yearly seasonality component", x=0.5, font=dict(size=14)),
        xaxis=dict(title="", showgrid=False, type="category"),
        yaxis=dict(
            title="sales quantity",
            showgrid=True,
            gridcolor="#f0f0f0",
            zeroline=False,
        ),
        template="plotly_white",
        height=380,
        margin=dict(l=60, r=40, t=70, b=50),
    )
    return fig
