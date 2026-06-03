import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from services.porcessors import (
    weather_analysis_processor,
    rain_band_processor,
    ols_rain_processor,
)

_OLS_BAND_COLORS = {"light": "#a8d0e6", "moderate": "#5b9bd5", "heavy": "#1f4e79"}
_SEASON_COLORS = {
    "Winter": "#378ADD",
    "Spring": "#1D9E75",
    "Summer": "#D85A30",
    "Autumn": "#BA7517",
}
_SEASON_ORDER = ["Winter", "Spring", "Summer", "Autumn"]


def plot_ols_rain_effect(agg_df: pd.DataFrame, title: str) -> go.Figure:
    """Bar chart: mean % effect on sales by rain band, averaged across shops (95% CI)."""
    BANDS = ["light", "moderate", "heavy"]

    def _stars(p):
        if pd.isna(p):
            return ""
        return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else "ns"

    rows = agg_df.set_index("band").reindex(BANDS).dropna(subset=["mean"])
    if rows.empty:
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

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=rows.index.tolist(),
            y=rows["mean"].tolist(),
            marker_color=[_OLS_BAND_COLORS[b] for b in rows.index],
            error_y=dict(
                type="data",
                symmetric=True,
                array=rows["ci"].tolist(),
                color="#333",
                thickness=1.5,
                width=8,
            ),
            text=[f"{v:+.1%}  {_stars(p)}" for v, p in zip(rows["mean"], rows["pval"])],
            textposition="outside",
            customdata=rows["n_shops"].tolist(),
            hovertemplate="<b>%{x} rain</b><br>avg effect: %{y:+.2%}<br>n shops: %{customdata}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_color="gray", line_width=1)
    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left"),
        yaxis_title="Mean sales effect vs dry day",
        yaxis_tickformat=".0%",
        xaxis_title="Rainfall band  (light: 0.1–2mm  ·  moderate: 2–8mm  ·  heavy: >8mm)",
        template="plotly_white",
        showlegend=False,
        bargap=0.6,
        height=460,
        margin=dict(l=70, r=30, t=90, b=70),
    )
    return fig


_NIGHT_BANDS = {"00-02", "03-05", "21-23"}
_DUR_COLORS = ["#27ae60", "#f1c40f", "#e67e22", "#e74c3c", "#8e44ad"]


def plot_rain_band_overview(bdf):
    """Part 1 Chart 1: % change when it rains in each 3-hour window."""
    bclr = [
        (
            "#5b9bd5"
            if r["band"] in _NIGHT_BANDS
            else ("#e74c3c" if r["pct_change"] < 0 else "#27ae60")
        )
        for _, r in bdf.iterrows()
    ]

    fig = go.Figure()
    for band in _NIGHT_BANDS:
        if band in bdf["band"].values:
            fig.add_vrect(
                x0=band,
                x1=band,
                fillcolor="#5b9bd5",
                opacity=0.12,
                layer="below",
                line_width=30,
            )

    fig.add_trace(
        go.Bar(
            x=bdf["band"],
            y=bdf["pct_change"],
            marker_color=bclr,
            name="% Change vs dry",
            text=[f"<b>{v:+.1f}%</b>" for v in bdf["pct_change"]],
            textposition="outside",
            yaxis="y1",
            customdata=bdf["n_rainy"],
            hovertemplate="<b>%{x}</b><br>% change: %{y:+.1f}%<br>Rainy days: %{customdata}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=bdf["band"],
            y=bdf["n_rainy"],
            mode="lines+markers+text",
            name="Rainy days (count)",
            line=dict(color="#666", dash="dot", width=1.5),
            marker=dict(size=7, color="#666"),
            text=bdf["n_rainy"].astype(str),
            textposition="top center",
            textfont=dict(size=10, color="#666"),
            yaxis="y2",
            hovertemplate="<b>%{x}</b><br>Rainy days: %{y}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_dash="dot", line_color="gray", yref="y1")
    fig.update_layout(
        title="% Change in Sales When It Rains in Each 3-Hour Window<br>"
        "<sub>Blue = nighttime bands. Grey line = number of unique rainy days (right axis).</sub>",
        xaxis=dict(type="category", title="3-Hour Window"),
        yaxis=dict(title="% change vs dry day"),
        yaxis2=dict(
            title="Rainy days",
            overlaying="y",
            side="right",
            showgrid=False,
            rangemode="tozero",
        ),
        template="plotly_white",
        legend=dict(orientation="h", y=-0.18),
        height=460,
        margin=dict(t=90, b=90),
    )
    return fig


def plot_rain_duration_overview(d_sum, dur_order):
    """Part 1 Chart 2: daytime rain duration effect on sales."""
    present = [b for b in dur_order if b in d_sum.index]
    clrs = [_DUR_COLORS[dur_order.index(b)] for b in present]
    pcts = d_sum.loc[present, "pct_vs_dry"]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Avg Daily Sales by Rain Duration", "% Change vs Dry Days"),
    )
    fig.add_trace(
        go.Bar(
            x=present,
            y=d_sum.loc[present, "avg_qty"],
            marker_color=clrs,
            text=d_sum.loc[present, "avg_qty"].round(0).astype(int),
            textposition="outside",
            showlegend=False,
            customdata=d_sum.loc[present, "n_days"].values,
            hovertemplate="<b>%{x}</b><br>Avg: %{y:.1f}<br>Days: %{customdata}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=present,
            y=pcts,
            showlegend=False,
            marker_color=["#e74c3c" if v < 0 else "#27ae60" for v in pcts],
            text=pcts.round(1).astype(str) + "%",
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Change: %{y:.1f}%<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=1, col=2)
    fig.update_layout(
        title="Daytime Rain Duration Effect on Sales (Raw Averages)",
        template="plotly_white",
        height=430,
        margin=dict(t=70, b=50),
    )
    fig.update_yaxes(title_text="Avg Daily Sales Qty", row=1, col=1)
    fig.update_yaxes(title_text="% change vs dry day", row=1, col=2)
    return fig


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
        f"corr={corr:+.3f}  |  dry vs rainy p={p_ttest:.3f}"
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
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14),
        )
        fig.update_layout(height=200, template="plotly_white")
        return fig

    band_colors = {"light": "#5b9bd5", "moderate": "#2f6da8", "heavy": "#c0392b"}
    bands = ["light", "moderate", "heavy"]

    def sig_label(p):
        return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""

    same_day = coef_df[coef_df["effect"] == "Same-day"].set_index("band")

    # Build aligned arrays for a single trace
    xs, ys, err_hi, err_lo, colors, hover_texts, text_labels = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for band in bands:
        if band not in same_day.index:
            continue
        row = same_day.loc[band]
        sig = sig_label(row["p"])
        xs.append(band)
        ys.append(row["pct_change"])
        err_hi.append(row["ci_high"] - row["pct_change"])
        err_lo.append(row["pct_change"] - row["ci_low"])
        colors.append(band_colors[band])
        hover_texts.append(
            f"<b>{band}</b><br>% change: {row['pct_change']:.2f}%<br>p={row['p']:.3f} {sig}"
        )
        text_labels.append(f"{row['pct_change']:+.1f}% {sig}")

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=xs,
            y=ys,
            width=0.5,  # ← controls bar width (0–1, where 1 fills the category slot)
            marker_color=colors,
            error_y=dict(
                type="data",
                symmetric=False,
                array=err_hi,
                arrayminus=err_lo,
                thickness=1.5,
                width=8,
                color="#333",
            ),
            text=text_labels,
            textposition="outside",
            hovertext=hover_texts,
            hovertemplate="%{hovertext}<extra></extra>",
            showlegend=False,
        )
    )

    fig.add_hline(y=0, line_dash="dot", line_color="gray")

    r2 = meta.get("r2", float("nan"))
    n = meta.get("n", 0)

    fig.update_xaxes(title_text="Rainfall band")
    fig.update_yaxes(title_text="% change in sales vs dry day")
    fig.update_layout(
        height=420,
        width=600,  # ← cap the overall width so bars don't get stretched
        title_text=f"OLS: Same-day rain effect on sales  |  R²={r2:.3f}  |  n={n} days",
        template="plotly_white",
        hovermode="closest",
        bargap=0.4,  # ← also helps narrow bars (only matters with multiple traces)
        margin=dict(t=60, b=40, l=60, r=40),
    )

    return fig


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
            hovertemplate="%{x}<br>avg contribution: %{y:.4f} units<extra></extra>",
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
                    f"<b>{season}</b><br>avg contribution: %{{y:.4f}} units<br>"
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
            text=f"{title}<br><sub>β × actual temperature = estimated sales units "
            "added/removed by temperature each month</sub>",
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
        title_text="sales units contributed by temperature",
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
        title_text="sales units contributed by temperature",
        row=1,
        col=2,
    )
    fig.update_xaxes(showgrid=False)
    return fig


def plot_prophet_seasonality(seas_avg: pd.DataFrame, n_shops: int, title: str):
    """Bar chart of Prophet yearly seasonality component averaged across shops by month."""
    if seas_avg.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No seasonality data.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14),
        )
        fig.update_layout(height=300, template="plotly_white")
        return fig

    def _season(m):
        if m in [12, 1, 2]:
            return "Winter"
        if m in [3, 4, 5]:
            return "Spring"
        if m in [6, 7, 8]:
            return "Summer"
        return "Autumn"

    colors = [_SEASON_COLORS[_season(m)] for m in seas_avg["month"]]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=seas_avg["month_name"],
            y=seas_avg["avg_yearly"].round(3),
            marker_color=colors,
            error_y=dict(
                type="data",
                symmetric=True,
                array=seas_avg["std_yearly"].round(3),
                color="#5F5E5A",
                thickness=1,
                width=4,
            ),
            customdata=seas_avg["n_shops"],
            hovertemplate="%{x}<br>avg yearly component: %{y:.3f}<br>shops: %{customdata}<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#888780", line_width=1)

    # Season legend
    for season, color in _SEASON_COLORS.items():
        fig.add_trace(
            go.Bar(x=[None], y=[None], marker_color=color, name=season, showlegend=True)
        )

    fig.update_layout(
        title=dict(
            text=f"{title}<br><sub>Prophet yearly seasonality component averaged across {n_shops} shops. "
            "Positive = above-average sales period; negative = below-average.</sub>",
            font=dict(size=14),
            x=0.5,
        ),
        height=420,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial", size=11, color="#2C2C2A"),
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5
        ),
        margin=dict(l=20, r=20, t=100, b=90),
        yaxis=dict(
            title="Prophet yearly seasonality (additive units)",
            showgrid=True,
            gridcolor="#F1EFE8",
        ),
        xaxis=dict(showgrid=False),
        bargap=0.4,
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
