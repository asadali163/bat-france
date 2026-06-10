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
    """Smooth spline curve: mean rain effect on sales vs mm, averaged across shops (95% CI)."""
    if agg_df is None or agg_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="Not enough data for this selection.",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(size=14))
        fig.update_layout(height=300, template="plotly_white")
        return fig

    n_shops = int(agg_df["n_shops"].iloc[0])
    xs  = agg_df["x_mm"].values
    ys  = agg_df["mean_pct"].values
    y_up = agg_df["ci_upper"].values
    y_dn = agg_df["ci_lower"].values

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.concatenate([xs, xs[::-1]]),
        y=np.concatenate([y_up, y_dn[::-1]]),
        fill="toself", fillcolor="rgba(47,109,168,0.12)",
        line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines",
        line=dict(color="#2f6da8", width=2.5),
        hovertemplate="<b>%{x:.1f} mm</b><br>avg effect: %{y:+.2f}%<extra></extra>",
        showlegend=False,
    ))

    fig.add_hline(y=0, line_color="gray", line_width=1, line_dash="dot")
    for mm in [1.0, 4.0, 8.0]:
        fig.add_vline(x=mm, line_dash="dot", line_color="#ddd", line_width=1,
                      annotation_text=f"{mm}mm", annotation_position="bottom",
                      annotation_font=dict(size=9, color="#aaa"))

    fig.add_annotation(
        x=0.99, y=0.97, xref="paper", yref="paper", xanchor="right",
        showarrow=False,
        text=f"n={n_shops:,} shops",
        font=dict(size=11, color="#555"),
        bgcolor="rgba(255,255,255,0.8)",
    )
    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left"),
        xaxis=dict(title="Precipitation (mm)", showgrid=True, gridcolor="#f0f0f0", range=[0, 20]),
        yaxis=dict(title="% change in sales vs dry day", showgrid=True, gridcolor="#f0f0f0"),
        template="plotly_white",
        showlegend=False, height=460,
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


_EFFECT_COLORS = {
    "Same-day":  "#2f6da8",
    "Yesterday": "#e67e22",
    "Tomorrow":  "#27ae60",
}
_EFFECT_FILL = {
    "Same-day":  "rgba(47,109,168,0.10)",
    "Yesterday": "rgba(230,126,34,0.10)",
    "Tomorrow":  "rgba(39,174,96,0.10)",
}


def plot_ols_rain_chart(curve_df, scalar_df, meta):
    """Smooth spline curves: Same-day / Yesterday / Tomorrow rain effects (single customer)."""
    if curve_df is None or curve_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text=meta.get("error", "Not enough data"),
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14),
        )
        fig.update_layout(height=200, template="plotly_white")
        return fig

    r2 = meta.get("r2", float("nan"))
    n  = meta.get("n", 0)
    fig = go.Figure()

    curve_df = curve_df[curve_df["effect"] == "Same-day"]
    for effect, grp in curve_df.groupby("effect", sort=False):
        color = _EFFECT_COLORS.get(effect, "#888")
        fill  = _EFFECT_FILL.get(effect,  "rgba(0,0,0,0.05)")
        xs    = grp["x_mm"].values
        ys    = grp["y_pct"].values
        y_up  = grp["y_upper"].values
        y_dn  = grp["y_lower"].values

        fig.add_trace(go.Scatter(
            x=np.concatenate([xs, xs[::-1]]),
            y=np.concatenate([y_up, y_dn[::-1]]),
            fill="toself", fillcolor=fill,
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip", showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", name=effect,
            line=dict(color=color, width=2.5),
            hovertemplate=f"<b>{effect}</b>  %{{x:.1f}} mm → %{{y:+.2f}}%<extra></extra>",
        ))

    fig.add_hline(y=0, line_dash="dot", line_color="gray", line_width=1)
    for mm in [1.0, 4.0, 8.0]:
        fig.add_vline(x=mm, line_dash="dot", line_color="#ddd", line_width=1,
                      annotation_text=f"{mm}mm", annotation_position="bottom",
                      annotation_font=dict(size=9, color="#aaa"))

    fig.update_layout(
        title_text=f"OLS: Rain effect on sales  |  R²={r2:.3f}  |  n={n} days",
        xaxis=dict(title="Precipitation (mm)", showgrid=True, gridcolor="#f0f0f0", range=[0, 20]),
        yaxis=dict(title="% change in sales vs dry day", showgrid=True, gridcolor="#f0f0f0"),
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.08, x=0),
        height=420,
        margin=dict(t=70, b=40, l=60, r=40),
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


_TEMP_COLOR      = "#D85A30"
_TEMP_FILL_COLOR = "rgba(216,90,48,0.12)"
_TEMP_KNOTS      = [5, 15, 25]


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


def plot_ols_rain_band_effect(agg_df: pd.DataFrame, title: str) -> go.Figure:
    """Bar chart: mean same-day % effect on sales by rain band, averaged across shops (95% CI)."""
    BANDS = ["light", "moderate", "heavy"]
    COLORS = {"light": "#a8d0e6", "moderate": "#5b9bd5", "heavy": "#1f4e79"}

    def _stars(p):
        if pd.isna(p): return ""
        return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else "ns"

    rows = agg_df.set_index("band").reindex(BANDS).dropna(subset=["mean"])
    if rows.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Not enough data for this selection.",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14),
        )
        fig.update_layout(height=300, template="plotly_white")
        return fig

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=rows.index.tolist(),
        y=rows["mean"].tolist(),
        marker_color=[COLORS[b] for b in rows.index],
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
    ))
    fig.add_hline(y=0, line_color="gray", line_width=1)
    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left"),
        yaxis_title="Mean sales effect vs dry day",
        yaxis_tickformat=".0%",
        xaxis_title="Rainfall band  (light: 0.1–2 mm  ·  moderate: 2–8 mm  ·  heavy: >8 mm)",
        template="plotly_white",
        showlegend=False,
        bargap=0.6,
        height=460,
        margin=dict(l=70, r=30, t=90, b=70),
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


_WIND_COLOR      = "#1a7a4a"
_WIND_FILL_COLOR = "rgba(26,122,74,0.12)"

_WC_OLS_COLOR      = "#6B3FA0"
_WC_OLS_FILL_COLOR = "rgba(107,63,160,0.12)"
_WC_OLS_KNOTS      = [0, 10, 20]


def plot_ols_wc_effect(agg_df: pd.DataFrame, title: str, ref_wc: float = None, show_knots: bool = True):
    """Smooth natural spline curve: mean wind chill (apparent temperature) effect on sales."""
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
        fill="toself", fillcolor=_WC_OLS_FILL_COLOR,
        line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines",
        line=dict(color=_WC_OLS_COLOR, width=2.5),
        hovertemplate="<b>%{x:.1f}°C feels-like</b><br>avg effect: %{y:+.2f}%<extra></extra>",
        showlegend=False,
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="gray", line_width=1)
    if ref_wc is not None:
        fig.add_vline(x=ref_wc, line_dash="dash", line_color="#888", line_width=1.5,
                      annotation_text=f"ref {ref_wc:.1f}°C",
                      annotation_position="top right",
                      annotation_font=dict(size=10, color="#888"))
    if show_knots:
        for k in _WC_OLS_KNOTS:
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
    ref_label = f" ({ref_wc:.1f}°C)" if ref_wc is not None else ""

    pad   = max(3.0, (float(ys.max()) - float(ys.min())) * 0.4 + 1.0)
    y_min = float(ys.min()) - pad
    y_max = float(ys.max()) + pad

    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left"),
        xaxis=dict(title="Feels-like temperature (°C)", showgrid=True, gridcolor="#f0f0f0"),
        yaxis=dict(title=f"% change vs ref feels-like temp{ref_label}",
                   showgrid=True, gridcolor="#f0f0f0",
                   range=[y_min, y_max]),
        template="plotly_white",
        showlegend=False, height=460,
        margin=dict(l=70, r=30, t=90, b=70),
    )
    return fig


def plot_ols_wind_effect(agg_df: pd.DataFrame, title: str, ref_wind: float = None):
    """Smooth natural spline curve: mean windspeed effect on sales, averaged across shops."""
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
    xs   = agg_df["x_ms"].values
    ys   = agg_df["mean_pct"].values
    y_up = agg_df["ci_upper"].values
    y_dn = agg_df["ci_lower"].values

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.concatenate([xs, xs[::-1]]),
        y=np.concatenate([y_up, y_dn[::-1]]),
        fill="toself", fillcolor=_WIND_FILL_COLOR,
        line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines",
        line=dict(color=_WIND_COLOR, width=2.5),
        hovertemplate="<b>%{x:.1f} m/s</b><br>avg effect: %{y:+.2f}%<extra></extra>",
        showlegend=False,
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="gray", line_width=1)
    if ref_wind is not None:
        fig.add_vline(x=ref_wind, line_dash="dash", line_color="#888", line_width=1.5,
                      annotation_text=f"ref {ref_wind:.1f}",
                      annotation_position="top right",
                      annotation_font=dict(size=10, color="#888"))
    fig.add_annotation(
        x=0.99, y=0.97, xref="paper", yref="paper", xanchor="right",
        showarrow=False,
        text=f"n={n_shops:,} shops",
        font=dict(size=11, color="#555"),
        bgcolor="rgba(255,255,255,0.8)",
    )
    ref_label = f" ({ref_wind:.1f})" if ref_wind is not None else ""

    # Y-axis clamped to mean curve range (CI tails can blow up at extremes)
    pad   = max(3.0, (float(ys.max()) - float(ys.min())) * 0.4 + 1.0)
    y_min = float(ys.min()) - pad
    y_max = float(ys.max()) + pad

    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left"),
        xaxis=dict(title="Wind Speed", showgrid=True, gridcolor="#f0f0f0"),
        yaxis=dict(title=f"% change vs ref wind{ref_label}",
                   showgrid=True, gridcolor="#f0f0f0",
                   range=[y_min, y_max]),
        template="plotly_white",
        showlegend=False, height=460,
        margin=dict(l=70, r=30, t=90, b=70),
    )
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


_RAIN_BAND_COLORS = {
    "none": "#d0e8f5",
    "light": "#5b9bd5",
    "moderate": "#2f6da8",
    "heavy": "#c0392b",
}


def plot_daily_rainfall(daily_df: pd.DataFrame, year: int, month: int, month_name: str):
    """Bar chart: precipitation per day for a given year+month, coloured by rain band."""
    import calendar as _cal

    def _band(p):
        if p <= 0.1:
            return "none"
        if p <= 2:
            return "light"
        if p <= 8:
            return "moderate"
        return "heavy"

    days_in_month = _cal.monthrange(year, month)[1]
    all_days = pd.DataFrame({"day": range(1, days_in_month + 1)})
    df = all_days.merge(
        daily_df[["day", "precipitation"]].copy(), on="day", how="left"
    )
    df["precipitation"] = df["precipitation"].fillna(0)
    df["band"] = df["precipitation"].apply(_band)
    df["color"] = df["band"].map(_RAIN_BAND_COLORS)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["day"].astype(str),
            y=df["precipitation"],
            marker_color=df["color"].tolist(),
            customdata=df["band"].tolist(),
            hovertemplate=(
                "<b>Day %{x}</b><br>"
                "Precipitation: %{y:.1f} mm<br>"
                "Band: %{customdata}<extra></extra>"
            ),
        )
    )

    for threshold, label in [(0.1, "light"), (2.0, "moderate"), (8.0, "heavy")]:
        fig.add_hline(
            y=threshold,
            line_dash="dot",
            line_color="#aaa",
            line_width=1,
            annotation_text=label,
            annotation_position="top right",
            annotation_font=dict(size=10, color="#888"),
        )

    fig.update_layout(
        title=f"Daily Precipitation: {month_name} {year}",
        xaxis=dict(title="Day of Month", type="category"),
        yaxis=dict(title="Precipitation (mm)", rangemode="tozero", showgrid=True, gridcolor="#f0f0f0"),
        template="plotly_white",
        bargap=0.15,
        height=380,
        margin=dict(t=60, b=50, l=60, r=100),
        annotations=[
            dict(
                x=0.99, y=1.06, xref="paper", yref="paper", xanchor="right",
                showarrow=False,
                text=(
                    "<span style='color:#d0e8f5'>■</span> None (&lt;0.1mm)&nbsp;&nbsp;"
                    "<span style='color:#5b9bd5'>■</span> Light (0.1–2mm)&nbsp;&nbsp;"
                    "<span style='color:#2f6da8'>■</span> Moderate (2–8mm)&nbsp;&nbsp;"
                    "<span style='color:#c0392b'>■</span> Heavy (&gt;8mm)"
                ),
                font=dict(size=11),
            )
        ],
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


# ── Sky Condition Charts ──────────────────────────────────────────────────────

_SKY_COLORS = {"Sunny": "#FFD600", "Overcast": "#90A4AE", "Others": "#455A64"}
_SKY_ORDER = ["Sunny", "Overcast", "Others"]

_LAYOUT_SKY_BASE = dict(
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(family="Arial", size=11, color="#2C2C2A"),
    yaxis=dict(showgrid=True, gridcolor="#F1EFE8", zeroline=True, zerolinecolor="#888780", zerolinewidth=1.5),
    xaxis=dict(showgrid=False),
)


def plot_sky_pct_bars(
    agg_df: pd.DataFrame,
    title: str,
    n_shops: int,
    subtitle: str | None = None,
) -> go.Figure:
    """
    3-bar chart (Sunny / Overcast / Others) showing % change vs Overcast baseline.
    agg_df columns: sky, pct_change, sem_pct, n_shop_days
    Overcast bar is always 0 (shown for reference).
    """
    df = agg_df.set_index("sky").reindex(_SKY_ORDER).reset_index()
    fig = go.Figure()
    for _, row in df.iterrows():
        sky = row["sky"]
        pct = float(row["pct_change"]) if pd.notna(row.get("pct_change")) else 0.0
        sem = float(row["sem_pct"]) if pd.notna(row.get("sem_pct")) else None
        n = int(row["n_shop_days"]) if pd.notna(row.get("n_shop_days")) else 0
        fig.add_trace(
            go.Bar(
                name=sky,
                x=[sky],
                y=[pct],
                error_y=dict(type="data", array=[sem], visible=True) if sem else None,
                marker_color=_SKY_COLORS[sky],
                hovertemplate=(
                    f"<b>{sky}</b><br>% change vs Overcast: %{{y:.2f}}%"
                    f"<br>n={n:,} shop-days<extra></extra>"
                ),
            )
        )
    if subtitle is None:
        if n_shops == 1:
            subtitle = "% change vs Overcast (Overcast = 0)"
        else:
            subtitle = f"% change vs Overcast — {n_shops} shops  ·  error bars = SE across shops"
    fig.update_layout(
        title=dict(
            text=f"{title}<br><sub>{subtitle}</sub>",
            font=dict(size=13),
            x=0.5,
        ),
        showlegend=False,
        height=380,
        bargap=0.35,
        yaxis_title="% change vs Overcast",
        margin=dict(l=20, r=20, t=80, b=40),
        **_LAYOUT_SKY_BASE,
    )
    return fig


def plot_sky_territory_bars(
    territory_df: pd.DataFrame,
    title: str,
) -> go.Figure:
    """
    Grouped bar: x=territory label (T1…), two bars per territory (Sunny / Others).
    Overcast = 0 shown as reference line.
    territory_df columns: route_label, pct_sunny, pct_others, n_shops
    """
    labels = territory_df["route_label"].tolist()
    fig = go.Figure()
    for sky, col in [("Sunny", "pct_sunny"), ("Others", "pct_others")]:
        vals = territory_df[col].tolist()
        n_shops = territory_df["n_shops"].tolist()
        fig.add_trace(
            go.Bar(
                name=sky,
                x=labels,
                y=vals,
                marker_color=_SKY_COLORS[sky],
                hovertemplate=(
                    f"<b>{sky}</b><br>%{{x}}<br>% vs Overcast: %{{y:.2f}}%"
                    "<br>n=%{customdata} shops<extra></extra>"
                ),
                customdata=n_shops,
            )
        )
    fig.update_layout(
        title=dict(text=title, font=dict(size=13), x=0.5),
        barmode="group",
        height=380,
        yaxis_title="% change vs Overcast",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=80, b=40),
        **_LAYOUT_SKY_BASE,
    )
    return fig


def plot_sky_shop_bars(
    shop_df: pd.DataFrame,
    title: str,
) -> go.Figure:
    """
    Horizontal grouped bar: y=shop (sorted by Sunny % change desc),
    two bars per shop: Sunny and Others. Overcast = 0 reference.
    shop_df columns: customer_code, pct_sunny, pct_others, n_sunny, n_overcast, n_others
    Sorted descending by pct_sunny.
    """
    df = shop_df.dropna(subset=["pct_sunny"]).sort_values("pct_sunny", ascending=True)
    shops = df["customer_code"].tolist()
    n = len(shops)
    height = max(400, n * 22 + 100)

    fig = go.Figure()
    for sky, col, n_col in [
        ("Others", "pct_others", "n_others"),
        ("Sunny", "pct_sunny", "n_sunny"),
    ]:
        fig.add_trace(
            go.Bar(
                name=sky,
                y=shops,
                x=df[col].tolist(),
                orientation="h",
                marker_color=_SKY_COLORS[sky],
                hovertemplate=(
                    f"<b>{sky}</b><br>%{{y}}<br>% vs Overcast: %{{x:.2f}}%"
                    "<br>n=%{customdata} days<extra></extra>"
                ),
                customdata=df[n_col].tolist(),
            )
        )
    fig.update_layout(
        title=dict(text=title, font=dict(size=13), x=0.5),
        barmode="group",
        height=height,
        xaxis_title="% change vs Overcast",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        margin=dict(l=160, r=20, t=80, b=40),
        **_LAYOUT_SKY_BASE,
    )
    fig.update_yaxes(tickfont=dict(size=9))
    return fig


# ── Wind Chill Charts ─────────────────────────────────────────────────────────

_WC_COLORS = {
    "Cold": "#1565C0",
    "Cool": "#64B5F6",
    "Mild": "#66BB6A",
    "Warm": "#FFA726",
}
_WC_ORDER = ["Cold", "Cool", "Mild", "Warm"]

_LAYOUT_WC_BASE = dict(
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(family="Arial", size=11, color="#2C2C2A"),
    yaxis=dict(showgrid=True, gridcolor="#F1EFE8", zeroline=True, zerolinecolor="#888780", zerolinewidth=1.5),
    xaxis=dict(showgrid=False),
)


def plot_wc_pct_bars(
    agg_df: pd.DataFrame,
    title: str,
    n_shops: int,
    subtitle: str | None = None,
) -> go.Figure:
    """
    4-bar chart (Cold / Cool / Mild / Warm) showing % change vs Mild baseline.
    agg_df columns: wc_cat, pct_change, sem_pct, n_shop_days
    Mild bar is always 0 (reference).
    """
    df = agg_df.set_index("wc_cat").reindex(_WC_ORDER).reset_index()
    fig = go.Figure()
    for _, row in df.iterrows():
        cat = row["wc_cat"]
        pct = float(row["pct_change"]) if pd.notna(row.get("pct_change")) else 0.0
        sem = float(row["sem_pct"]) if pd.notna(row.get("sem_pct")) else None
        n = int(row["n_shop_days"]) if pd.notna(row.get("n_shop_days")) else 0
        fig.add_trace(
            go.Bar(
                name=cat,
                x=[cat],
                y=[pct],
                error_y=dict(type="data", array=[sem], visible=True) if sem else None,
                marker_color=_WC_COLORS[cat],
                hovertemplate=(
                    f"<b>{cat}</b><br>% change vs Mild: %{{y:.2f}}%"
                    f"<br>n={n:,} shop-days<extra></extra>"
                ),
            )
        )
    if subtitle is None:
        if n_shops == 1:
            subtitle = "% change vs Mild (Mild = 0)"
        else:
            subtitle = f"% change vs Mild — {n_shops} shops  ·  error bars = SE across shops"
    fig.update_layout(
        title=dict(
            text=f"{title}<br><sub>{subtitle}</sub>",
            font=dict(size=13),
            x=0.5,
        ),
        showlegend=False,
        height=380,
        bargap=0.35,
        yaxis_title="% change vs Mild",
        margin=dict(l=20, r=20, t=80, b=40),
        **_LAYOUT_WC_BASE,
    )
    return fig


# ── Temperature Gap Charts ────────────────────────────────────────────────────

_GAP_COLORS = {
    "Cold Feel":    "#1565C0",
    "Slight Chill": "#90CAF9",
    "Similar":      "#9E9E9E",
    "Warm Feel":    "#EF6C00",
}
_GAP_ORDER    = ["Cold Feel", "Slight Chill", "Similar", "Warm Feel"]
_GAP_BASELINE = "Similar"

_LAYOUT_GAP_BASE = dict(
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(family="Arial", size=11, color="#2C2C2A"),
    yaxis=dict(showgrid=True, gridcolor="#F1EFE8", zeroline=True,
               zerolinecolor="#888780", zerolinewidth=1.5),
    xaxis=dict(showgrid=False),
)

_MONTH_NAMES = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}


def plot_gap_pct_bars(
    agg_df: pd.DataFrame,
    title: str,
    n_shops: int,
    subtitle: str | None = None,
) -> go.Figure:
    """
    4-bar chart (Cold Feel / Slight Chill / Similar / Warm Feel).
    % change vs Similar baseline.
    agg_df columns: gap_cat, pct_change, sem_pct, n_shop_days
    """
    df = agg_df.set_index("gap_cat").reindex(_GAP_ORDER).reset_index()
    fig = go.Figure()
    for _, row in df.iterrows():
        cat = row["gap_cat"]
        pct = float(row["pct_change"]) if pd.notna(row.get("pct_change")) else 0.0
        sem = float(row["sem_pct"]) if pd.notna(row.get("sem_pct")) else None
        n   = int(row["n_shop_days"]) if pd.notna(row.get("n_shop_days")) else 0
        fig.add_trace(go.Bar(
            name=cat, x=[cat], y=[pct],
            error_y=dict(type="data", array=[sem], visible=True) if sem else None,
            marker_color=_GAP_COLORS[cat],
            hovertemplate=(
                f"<b>{cat}</b><br>% change vs Similar: %{{y:.2f}}%"
                f"<br>n={n:,} unique dates<extra></extra>"
            ),
        ))
    if subtitle is None:
        if n_shops == 1:
            subtitle = "% change vs Similar feels-like (Similar = 0)"
        else:
            subtitle = f"% change vs Similar — {n_shops} shops  ·  error bars = SE across shops"
    fig.update_layout(
        title=dict(text=f"{title}<br><sub>{subtitle}</sub>", font=dict(size=13), x=0.5),
        showlegend=False, height=380, bargap=0.35,
        yaxis_title="% change vs Similar",
        margin=dict(l=20, r=20, t=80, b=40),
        **_LAYOUT_GAP_BASE,
    )
    return fig


def plot_gap_monthly(monthly_df: pd.DataFrame, title: str) -> go.Figure:
    """
    Grouped bar: x = month, bars = gap categories (excl. Similar baseline).
    Shows how the gap effect on sales varies month by month.
    monthly_df columns: month, gap_cat, pct_change, n_shop_days
    """
    months = sorted(monthly_df["month"].unique())
    month_labels = [_MONTH_NAMES.get(m, str(m)) for m in months]

    fig = go.Figure()
    for cat in _GAP_ORDER:
        sub = monthly_df[monthly_df["gap_cat"] == cat].set_index("month")
        ys = [float(sub.loc[m, "pct_change"]) if m in sub.index else float("nan") for m in months]
        ns = [int(sub.loc[m, "n_shop_days"]) if m in sub.index else 0 for m in months]
        fig.add_trace(go.Bar(
            name=cat,
            x=month_labels,
            y=ys,
            marker_color=_GAP_COLORS[cat],
            hovertemplate=(
                f"<b>{cat}</b><br>Month: %{{x}}<br>% vs Similar: %{{y:.2f}}%"
                "<br>n=%{customdata:,} unique dates<extra></extra>"
            ),
            customdata=ns,
        ))
    fig.update_layout(
        title=dict(text=f"{title}<br><sub>% change vs Similar, by month — error bars omitted for clarity</sub>",
                   font=dict(size=13), x=0.5),
        barmode="group", height=420,
        yaxis_title="% change vs Similar",
        xaxis_title="Month",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=90, b=50),
        **_LAYOUT_GAP_BASE,
    )
    return fig


# ── Storm (Wind Gust) Charts ──────────────────────────────────────────────────

_STORM_COLORS = {
    "Calm":     "#A5D6A7",
    "Moderate": "#9E9E9E",
    "Windy":    "#FF8F00",
    "Storm":    "#B71C1C",
}
_STORM_ORDER = ["Calm", "Moderate", "Windy", "Storm"]

_LAYOUT_STORM_BASE = dict(
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(family="Arial", size=11, color="#2C2C2A"),
    yaxis=dict(showgrid=True, gridcolor="#F1EFE8", zeroline=True,
               zerolinecolor="#888780", zerolinewidth=1.5),
    xaxis=dict(showgrid=False),
)


def plot_storm_pct_bars(
    agg_df: pd.DataFrame,
    title: str,
    n_shops: int,
    subtitle: str | None = None,
    baseline: str = "Moderate",
) -> go.Figure:
    """
    4-bar chart (Calm / Moderate / Windy / Storm) showing % change vs baseline.
    agg_df columns: storm_cat, pct_change, sem_pct, n_dates
    """
    df = agg_df.set_index("storm_cat").reindex(_STORM_ORDER).reset_index()
    fig = go.Figure()
    for _, row in df.iterrows():
        cat = row["storm_cat"]
        pct = float(row["pct_change"]) if pd.notna(row.get("pct_change")) else 0.0
        sem = float(row["sem_pct"]) if pd.notna(row.get("sem_pct")) else None
        n   = int(row["n_dates"]) if pd.notna(row.get("n_dates")) else 0
        fig.add_trace(go.Bar(
            name=cat, x=[cat], y=[pct],
            error_y=dict(type="data", array=[sem], visible=True) if sem else None,
            marker_color=_STORM_COLORS[cat],
            hovertemplate=(
                f"<b>{cat}</b><br>% change vs {baseline}: %{{y:.2f}}%"
                f"<br>n={n:,} unique dates<extra></extra>"
            ),
        ))
    if subtitle is None:
        if n_shops == 1:
            subtitle = f"% change vs {baseline} wind ({baseline} = 0)"
        else:
            subtitle = f"% change vs {baseline} — {n_shops} shops  ·  error bars = SE across shops"
    fig.update_layout(
        title=dict(text=f"{title}<br><sub>{subtitle}</sub>", font=dict(size=13), x=0.5),
        showlegend=False, height=380, bargap=0.35,
        yaxis_title=f"% change vs {baseline}",
        margin=dict(l=20, r=20, t=80, b=40),
        **_LAYOUT_STORM_BASE,
    )
    return fig


def plot_storm_monthly(monthly_df: pd.DataFrame, title: str) -> go.Figure:
    """
    Grouped bar: x=month, bars=storm categories.
    monthly_df columns: month, storm_cat, pct_change, n_dates
    """
    months = sorted(monthly_df["month"].unique())
    month_labels = [_MONTH_NAMES.get(m, str(m)) for m in months]
    fig = go.Figure()
    for cat in _STORM_ORDER:
        sub = monthly_df[monthly_df["storm_cat"] == cat].set_index("month")
        ys = [float(sub.loc[m, "pct_change"]) if m in sub.index and pd.notna(sub.loc[m, "pct_change"]) else float("nan") for m in months]
        ns = [int(sub.loc[m, "n_dates"]) if m in sub.index else 0 for m in months]
        fig.add_trace(go.Bar(
            name=cat, x=month_labels, y=ys,
            marker_color=_STORM_COLORS[cat],
            hovertemplate=(
                f"<b>{cat}</b><br>Month: %{{x}}<br>% vs Calm: %{{y:.2f}}%"
                "<br>n=%{customdata:,} unique dates<extra></extra>"
            ),
            customdata=ns,
        ))
    fig.update_layout(
        title=dict(
            text=f"{title}<br><sub>% change vs Calm wind, by month</sub>",
            font=dict(size=13), x=0.5,
        ),
        barmode="group", height=420,
        yaxis_title="% change vs Calm",
        xaxis_title="Month",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=90, b=50),
        **_LAYOUT_STORM_BASE,
    )
    return fig


# ── Sunny Transition Charts ───────────────────────────────────────────────────

_TRANS_COLORS = {
    "Day Before Sunny":  "#90CAF9",   # soft blue — overcast approaching sun
    "Sunny Day":         "#FFD54F",   # gold
    "Day After Sunny":   "#FFAB40",   # amber — sun fading
    "Cloudy":            "#B0BEC5",
    "Sunny":             "#FFD600",
    "Day After Bright":  "#FFAB40",
}
_TRANS_ORDER = ["Day Before Sunny", "Sunny Day", "Day After Sunny"]

_LAYOUT_TRANS_BASE = dict(
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(family="Arial", size=11, color="#2C2C2A"),
    yaxis=dict(showgrid=True, gridcolor="#F1EFE8", zeroline=True,
               zerolinecolor="#888780", zerolinewidth=1.5),
    xaxis=dict(showgrid=False),
)


def plot_transition_pair(
    pair_df: pd.DataFrame,
    title: str,
    subtitle: str | None = None,
    shop_mean: float | None = None,
) -> go.Figure:
    """
    2-bar chart showing absolute average sales (or STL residuals) for two transition categories.
    Annotates % change prominently at the top.

    pair_df columns: transition_cat, mean_val, n_dates  (exactly 2 rows, in display order)
    shop_mean: if provided, % change is (val2-val1)/shop_mean*100 (use for STL residuals).
               if None, % change is (val2-val1)/abs(val1)*100 (raw sales).
    """
    cats = pair_df["transition_cat"].tolist()
    vals = [float(v) if pd.notna(v) else float("nan") for v in pair_df["mean_val"].tolist()]
    ns   = [int(n) if pd.notna(n) else 0 for n in pair_df["n_dates"].tolist()]

    fig = go.Figure()
    y_label = "Avg STL residual (units)" if shop_mean is not None else "Avg daily sales (units)"

    # Pre-compute % change text using original unit values
    v1, v2 = vals[0], vals[1]
    bar_texts   = ["", ""]
    bar_tcolors = ["#333", "#333"]
    if pd.notna(v1) and pd.notna(v2):
        denom = shop_mean if (shop_mean is not None and shop_mean != 0) else (abs(v1) if v1 != 0 else None)
        if denom:
            pct = (v2 - v1) / denom * 100
            sign = "+" if pct >= 0 else ""
            bar_texts[1] = f"<b>{sign}{pct:.1f}%</b>"
            bar_tcolors[1] = "#2E7D32" if pct >= 0 else "#C62828"

    for i, (cat, val, n) in enumerate(zip(cats, vals, ns)):
        fig.add_trace(go.Bar(
            x=[cat], y=[val if pd.notna(val) else 0],
            marker_color=_TRANS_COLORS.get(cat, "#BDBDBD"),
            name=cat,
            text=[bar_texts[i]],
            textposition="outside",
            textfont=dict(size=15, color=bar_tcolors[i], family="Arial"),
            hovertemplate=(
                f"<b>{cat}</b><br>{y_label}: %{{y:.1f}}<br>"
                f"n={n:,} unique dates<extra></extra>"
            ),
        ))

    valid = [v for v in vals if pd.notna(v)]
    y_max = max(valid + [0]) * 1.3 if valid else 1
    y_min = min(valid + [0]) * 1.15 if valid else -1

    fig.update_layout(
        title=dict(
            text=f"{title}<br><sub>{subtitle}</sub>" if subtitle else title,
            font=dict(size=13), x=0.5,
        ),
        showlegend=False, height=420, bargap=0.45,
        yaxis=dict(
            title=y_label,
            showgrid=True, gridcolor="#F1EFE8",
            zeroline=True, zerolinecolor="#888780", zerolinewidth=1.5,
            range=[y_min, y_max],
        ),
        xaxis=dict(showgrid=False),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Arial", size=11, color="#2C2C2A"),
        margin=dict(l=20, r=20, t=80, b=40),
    )
    return fig


_MONTH_LABELS = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May",  6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}


def plot_sunny_temp_combined(df: pd.DataFrame, title: str = "Sunny Day Sales by Temperature") -> go.Figure:
    """
    Two-panel chart:
      Left  — Grouped bars: Day Before Sunny vs Sunny Day STL residuals by temperature bin
      Right — % change (STL) from Day Before to Sunny Day by temperature bin
    df columns: temp_bin, before_resid, sunny_resid, before_sales, sunny_sales,
                pct_stl, pct_raw, n_pairs
    """
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="Not enough data.", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=13, color="#888"))
        return fig

    bins    = df["temp_bin"].tolist()
    n_pairs = df["n_dates"].tolist() if "n_dates" in df.columns else df.get("n_pairs", pd.Series([0]*len(df))).tolist()

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Day Before → Sunny Day  (STL residual)", "% Change  (Sunny vs Before)"],
        horizontal_spacing=0.12,
        column_widths=[0.55, 0.45],
    )

    # ── Left: grouped bars ─────────────────────────────────────────────────────
    fig.add_trace(go.Bar(
        name="Day Before Sunny",
        x=bins, y=df["before_resid"].tolist(),
        marker_color="#90CAF9",
        customdata=list(zip(df["before_sales"].round(1), n_pairs)),
        hovertemplate=(
            "<b>%{x}</b> — Day Before Sunny<br>"
            "STL residual: %{y:.2f}<br>"
            "Raw sales: %{customdata[0]:.1f}<br>"
            "n=%{customdata[1]:,} unique sunny dates<extra></extra>"
        ),
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        name="Sunny Day",
        x=bins, y=df["sunny_resid"].tolist(),
        marker_color="#FFD54F",
        customdata=list(zip(df["sunny_sales"].round(1), n_pairs)),
        hovertemplate=(
            "<b>%{x}</b> — Sunny Day<br>"
            "STL residual: %{y:.2f}<br>"
            "Raw sales: %{customdata[0]:.1f}<br>"
            "n=%{customdata[1]:,} unique sunny dates<extra></extra>"
        ),
    ), row=1, col=1)

    # ── Right: % change bars ───────────────────────────────────────────────────
    pcts   = df["pct_stl"].tolist()
    colors = ["#2E7D32" if (pd.notna(v) and v >= 0) else "#C62828" for v in pcts]
    fig.add_trace(go.Bar(
        name="% Change (STL)",
        x=bins, y=pcts,
        marker_color=colors,
        customdata=list(zip(df["pct_raw"].round(2), n_pairs)),
        text=[f"{v:+.1f}%" if pd.notna(v) else "" for v in pcts],
        textposition="outside",
        textfont=dict(size=10),
        hovertemplate=(
            "<b>%{x}</b><br>"
            "STL % change: %{y:+.2f}%<br>"
            "Raw % change: %{customdata[0]:+.2f}%<br>"
            "n=%{customdata[1]:,} unique sunny dates<extra></extra>"
        ),
        showlegend=False,
    ), row=1, col=2)

    fig.update_yaxes(title_text="STL residual (units)", zeroline=True,
                     zerolinecolor="#888780", zerolinewidth=1.5,
                     showgrid=True, gridcolor="#F1EFE8", row=1, col=1)
    fig.update_yaxes(title_text="% change", zeroline=True,
                     zerolinecolor="#888780", zerolinewidth=1.5,
                     showgrid=True, gridcolor="#F1EFE8", row=1, col=2)
    fig.update_xaxes(showgrid=False)

    fig.update_layout(
        title=dict(text=title, font=dict(size=13), x=0.5),
        barmode="group",
        height=420,
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Arial", size=11, color="#2C2C2A"),
        margin=dict(l=20, r=20, t=80, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="center", x=0.28),
    )
    return fig


def plot_transition_monthly(
    monthly_df: pd.DataFrame,
    cat_a: str,
    cat_b: str,
    title: str,
    show_n_axis: bool = False,
) -> go.Figure:
    """
    Side-by-side (Raw | STL) monthly % change chart for one transition pair.
    monthly_df columns: month, raw_pct, stl_pct, n_dates
    show_n_axis: if True, overlays n_dates as a dotted line on a right y-axis.
    """
    if monthly_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="Not enough data for monthly breakdown.",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(size=13, color="#888"))
        return fig

    month_labels = [_MONTH_LABELS.get(m, str(m)) for m in monthly_df["month"]]
    n_dates      = monthly_df["n_dates"].fillna(0).astype(int).tolist()

    specs = [[{"secondary_y": show_n_axis}, {"secondary_y": show_n_axis}]]
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["% Change", "% Change"],
        horizontal_spacing=0.12,
        specs=specs,
    )

    for col, pct_col in [(1, "raw_pct"), (2, "stl_pct")]:
        vals   = monthly_df[pct_col].tolist()
        colors = ["#2E7D32" if (pd.notna(v) and v >= 0) else "#C62828" for v in vals]
        fig.add_trace(
            go.Bar(
                x=month_labels,
                y=vals,
                marker_color=colors,
                customdata=list(zip(n_dates, vals)),
                hovertemplate=(
                    f"<b>%{{x}}</b><br>{cat_a} → {cat_b}<br>"
                    "% change: %{customdata[1]:+.2f}%<br>"
                    "n days: %{customdata[0]:,}<extra></extra>"
                ),
                showlegend=False,
                text=[f"{v:+.1f}%" if pd.notna(v) else "" for v in vals],
                textposition="outside",
                textfont=dict(size=9),
            ),
            row=1, col=col,
            **({"secondary_y": False} if show_n_axis else {}),
        )
        fig.update_yaxes(
            zeroline=True, zerolinecolor="#888780", zerolinewidth=1.5,
            showgrid=True, gridcolor="#F1EFE8",
            title_text="% change", row=1, col=col,
            **({"secondary_y": False} if show_n_axis else {}),
        )
        fig.update_xaxes(showgrid=False, row=1, col=col)

        if show_n_axis:
            fig.add_trace(
                go.Scatter(
                    x=month_labels,
                    y=n_dates,
                    mode="lines+markers",
                    line=dict(color="#78909C", width=1.5, dash="dot"),
                    marker=dict(size=5, color="#78909C"),
                    name="n days" if col == 1 else None,
                    showlegend=col == 1,
                    hovertemplate="<b>%{x}</b><br>n days: %{y:,}<extra></extra>",
                ),
                row=1, col=col, secondary_y=True,
            )
            fig.update_yaxes(
                title_text="n days",
                secondary_y=True,
                showgrid=False,
                row=1, col=col,
            )

    fig.update_layout(
        title=dict(text=title, font=dict(size=13), x=0.5),
        height=400 if show_n_axis else 380,
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Arial", size=11, color="#2C2C2A"),
        margin=dict(l=20, r=20, t=70, b=40),
        legend=dict(x=1.08, y=1) if show_n_axis else {},
    )
    return fig


def plot_sunny_transition_bars(
    agg_df: pd.DataFrame,
    title: str,
    n_shops: int,
    subtitle: str | None = None,
) -> go.Figure:
    """
    3-bar chart: Day Before Sunny | Sunny Day | Day After Sunny.
    % change vs Sunny Day (Sunny Day = 0).
    agg_df columns: transition_cat, pct_change, sem_pct, n_dates
    """
    df = agg_df.set_index("transition_cat").reindex(_TRANS_ORDER).reset_index()
    fig = go.Figure()
    for _, row in df.iterrows():
        cat = row["transition_cat"]
        pct = float(row["pct_change"]) if pd.notna(row["pct_change"]) else 0.0
        sem = float(row["sem_pct"])    if pd.notna(row["sem_pct"])    else None
        n   = int(row["n_dates"])      if pd.notna(row["n_dates"])    else 0
        fig.add_trace(go.Bar(
            name=cat, x=[cat], y=[pct],
            error_y=dict(type="data", array=[sem], visible=True) if sem else None,
            marker_color=_TRANS_COLORS.get(cat, "#BDBDBD"),
            hovertemplate=(
                f"<b>{cat}</b><br>% change: %{{y:.2f}}%"
                f"<br>n={n:,} unique dates<extra></extra>"
            ),
        ))
    if subtitle is None:
        if n_shops == 1:
            subtitle = "Before Sunny = 0; Sunny Day vs Before Sunny; After Sunny vs Sunny Day"
        else:
            subtitle = (f"Before Sunny = 0  ·  Sunny Day vs Before Sunny  ·  "
                        f"After Sunny vs Sunny Day  ·  {n_shops} shops  ·  error bars = SE")
    fig.update_layout(
        title=dict(text=f"{title}<br><sub>{subtitle}</sub>", font=dict(size=13), x=0.5),
        showlegend=False, height=380, bargap=0.35,
        yaxis_title="% change (normalised by shop mean)",
        margin=dict(l=20, r=20, t=80, b=40),
        **_LAYOUT_TRANS_BASE,
    )
    return fig


# ── Weather Driver Analysis Charts ────────────────────────────────────────────

def _driver_agg(df, cat_col, order, baseline):
    """Aggregate STL residuals per category and compute % vs shop mean."""
    # Use global mean sales as the normalizer so all categories share the same denominator.
    global_mean_sales = df["mean_sales"].mean()
    agg = (
        df.groupby(cat_col)
        .agg(
            mean_resid=("residual", "mean"),
            n_days=("date", "nunique"),
        )
        .reset_index()
    )
    agg["pct_resid"] = agg["mean_resid"] / global_mean_sales * 100
    base_val = agg.loc[agg[cat_col] == baseline, "pct_resid"].values
    agg["pct_vs_baseline"] = agg["pct_resid"] - (base_val[0] if len(base_val) else 0)
    return agg.set_index(cat_col).reindex(order).reset_index()


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


_RAIN_ORDER = [
    "Normal Dry Day", "Rain Day 1", "Rain Day 2",
    "Rain Day 3", "Rain Day 4+", "First Dry Day (after 3+ rain)",
]
_RAIN_COLORS = {
    "Normal Dry Day":              "#90CAF9",
    "Rain Day 1":                  "#64B5F6",
    "Rain Day 2":                  "#42A5F5",
    "Rain Day 3":                  "#1E88E5",
    "Rain Day 4+":                 "#1565C0",
    "First Dry Day (after 3+ rain)": "#66BB6A",
}


def plot_rain_streak(df: pd.DataFrame, title: str = "Rain Streak & First Dry Day") -> go.Figure:
    """
    Two-panel chart:
    Left: mean STL residual per rain category (% of shop mean)
    Right: % change vs Normal Dry Day
    """
    baseline = "Normal Dry Day"
    agg = _driver_agg(df, "rain_cat", _RAIN_ORDER, baseline)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["STL Residual (% of shop mean)", "% Change vs Normal Dry Day"],
        horizontal_spacing=0.12,
    )

    for i, ycol in enumerate(["pct_resid", "pct_vs_baseline"], start=1):
        for _, row in agg.iterrows():
            n = int(row["n_days"]) if not pd.isna(row["n_days"]) else 0
            y = row[ycol] if not pd.isna(row[ycol]) else 0
            clr = _RAIN_COLORS.get(row["rain_cat"], "#90CAF9")
            if i == 2:
                clr = "#66bb6a" if y >= 0 else "#ef5350"
            fig.add_trace(go.Bar(
                x=[row["rain_cat"]], y=[y],
                name=row["rain_cat"],
                marker_color=clr,
                text=[f"{y:+.1f}%"],
                textposition="outside",
                hovertemplate=(
                    f"<b>{row['rain_cat']}</b><br>Value: {y:+.1f}%<br>"
                    f"n={n:,} unique dates<extra></extra>"
                ),
                showlegend=False,
            ), row=1, col=i)

    fig.update_layout(
        title=dict(text=title, font=dict(size=13), x=0.5),
        height=420, bargap=0.25,
        margin=dict(l=20, r=20, t=80, b=80),
        paper_bgcolor="white", plot_bgcolor="white",
    )
    fig.update_xaxes(tickangle=-30)
    fig.update_yaxes(zeroline=True, zerolinecolor="#ccc", gridcolor="#eee")
    return fig


def plot_snow_analysis(df: pd.DataFrame, title: str = "Snow Day Effect on Sales") -> go.Figure:
    """
    Two-bar chart: No Snow vs Snow Day — STL residuals and % change.
    """
    order = ["No Snow", "Snow Day (>0 cm)"]
    baseline = "No Snow"
    agg = _driver_agg(df, "snow_cat", order, baseline)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["STL Residual (% of shop mean)", "% Change vs No Snow"],
        horizontal_spacing=0.15,
    )

    for i, ycol in enumerate(["pct_resid", "pct_vs_baseline"], start=1):
        for _, row in agg.iterrows():
            n = int(row["n_days"]) if not pd.isna(row["n_days"]) else 0
            y = row[ycol] if not pd.isna(row[ycol]) else 0
            clr = "#90CAF9" if row["snow_cat"] == "No Snow" else "#B3E5FC"
            if i == 2:
                clr = "#66bb6a" if y >= 0 else "#ef5350"
            fig.add_trace(go.Bar(
                x=[row["snow_cat"]], y=[y],
                name=row["snow_cat"],
                marker_color=clr,
                text=[f"{y:+.1f}%"],
                textposition="outside",
                hovertemplate=(
                    f"<b>{row['snow_cat']}</b><br>Value: {y:+.1f}%<br>"
                    f"n={n:,} unique dates<extra></extra>"
                ),
                showlegend=False,
            ), row=1, col=i)

    fig.update_layout(
        title=dict(text=title, font=dict(size=13), x=0.5),
        height=400, bargap=0.4,
        margin=dict(l=20, r=20, t=80, b=40),
        paper_bgcolor="white", plot_bgcolor="white",
    )
    fig.update_yaxes(zeroline=True, zerolinecolor="#ccc", gridcolor="#eee")
    return fig


_GUST_ORDER  = ["Calm (<20 km/h)", "Moderate (20–40 km/h)", "Gusty (>40 km/h)"]
_GUST_COLORS = {"Calm (<20 km/h)": "#A5D6A7", "Moderate (20–40 km/h)": "#FFF176", "Gusty (>40 km/h)": "#EF9A9A"}


def plot_wind_gust(df: pd.DataFrame, title: str = "Wind Gust Effect on Sales") -> go.Figure:
    """
    3-bar chart: Calm / Moderate / Gusty wind days — STL residuals and % change.
    """
    baseline = "Calm (<20 km/h)"
    agg = _driver_agg(df, "gust_cat", _GUST_ORDER, baseline)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["STL Residual (% of shop mean)", "% Change vs Calm Days"],
        horizontal_spacing=0.12,
    )

    for i, ycol in enumerate(["pct_resid", "pct_vs_baseline"], start=1):
        for _, row in agg.iterrows():
            n = int(row["n_days"]) if not pd.isna(row["n_days"]) else 0
            y = row[ycol] if not pd.isna(row[ycol]) else 0
            clr = _GUST_COLORS.get(row["gust_cat"], "#BDBDBD")
            if i == 2:
                clr = "#66bb6a" if y >= 0 else "#ef5350"
            fig.add_trace(go.Bar(
                x=[row["gust_cat"]], y=[y],
                name=row["gust_cat"],
                marker_color=clr,
                text=[f"{y:+.1f}%"],
                textposition="outside",
                hovertemplate=(
                    f"<b>{row['gust_cat']}</b><br>Value: {y:+.1f}%<br>"
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


def plot_ols_category_effect(
    ols_df: pd.DataFrame,
    order: list,
    baseline: str,
    title: str = "OLS % Effect vs Baseline",
    height: int = 420,
    bargap: float = 0.35,
) -> go.Figure:
    """
    Horizontal forest-plot for pooled OLS category effects.
    ols_df columns: category, pct_change, ci_low, ci_high, n_shops
    Baseline row is drawn at 0 as a grey reference.
    """
    if ols_df.empty or "category" not in ols_df.columns:
        fig = go.Figure()
        fig.update_layout(title=f"{title}<br><sub>No OLS results (insufficient data per shop)</sub>",
                          height=300, paper_bgcolor="white")
        return fig

    disp_cats = [c for c in order if c == baseline or c in ols_df["category"].values]
    rows = []
    for cat in disp_cats:
        if cat == baseline:
            n = int(ols_df["n_shops"].max()) if not ols_df.empty else 0
            rows.append({"category": cat, "pct_change": 0.0, "ci_low": 0.0,
                         "ci_high": 0.0, "n_shops": n, "is_baseline": True})
        else:
            r = ols_df[ols_df["category"] == cat]
            if not r.empty:
                rows.append({**r.iloc[0].to_dict(), "is_baseline": False})

    if not rows:
        fig = go.Figure()
        fig.update_layout(title=title, height=300)
        return fig

    plot_df = pd.DataFrame(rows)
    colors = [
        "#9E9E9E" if r["is_baseline"]
        else ("#66BB6A" if r["pct_change"] >= 0 else "#EF5350")
        for _, r in plot_df.iterrows()
    ]

    ci_up   = (plot_df["ci_high"] - plot_df["pct_change"]).clip(lower=0).tolist()
    ci_down = (plot_df["pct_change"] - plot_df["ci_low"]).clip(lower=0).tolist()
    hover_text = [
        f"Baseline (0%)" if r["is_baseline"]
        else (
            f"<b>{r['category']}</b><br>"
            f"Effect: {r['pct_change']:+.2f}%<br>"
            f"95% CI: [{r['ci_low']:+.1f}%, {r['ci_high']:+.1f}%]<br>"
            f"n={int(r['n_shops'])} shops"
        )
        for _, r in plot_df.iterrows()
    ]

    fig = go.Figure(go.Bar(
        x=plot_df["category"],
        y=plot_df["pct_change"],
        marker_color=colors,
        error_y=dict(
            type="data", symmetric=False,
            array=ci_up, arrayminus=ci_down,
            color="#555", thickness=1.5, width=6,
        ),
        text=[
            "Baseline" if r["is_baseline"] else f"{r['pct_change']:+.1f}%"
            for _, r in plot_df.iterrows()
        ],
        textposition="outside",
        hovertext=hover_text,
        hoverinfo="text",
        showlegend=False,
    ))
    fig.add_hline(y=0, line_color="#333", line_width=1.2)

    fig.update_layout(
        title=dict(
            text=(
                f"{title}<br>"
                f"<sub>OLS · log(sales) ~ category + DOW + month + trend · "
                f"inverse-variance pooled across shops · 95% CI · baseline = {baseline}</sub>"
            ),
            font=dict(size=13), x=0.5,
        ),
        height=height,
        yaxis_title="% change",
        xaxis=dict(tickangle=-25),
        bargap=bargap,
        margin=dict(l=20, r=20, t=110, b=100),
        paper_bgcolor="white", plot_bgcolor="white",
    )
    fig.update_yaxes(zeroline=False, gridcolor="#eee")
    return fig


# ── Rain Intensity Chart ──────────────────────────────────────────────────────

_INTENSITY_ORDER = ["No Rain", "Drizzle (<1 mm/h)", "Moderate (1–4 mm/h)", "Heavy (>4 mm/h)"]
_INTENSITY_COLORS = {
    "No Rain":              "#90CAF9",
    "Drizzle (<1 mm/h)":   "#64B5F6",
    "Moderate (1–4 mm/h)": "#1E88E5",
    "Heavy (>4 mm/h)":     "#0D47A1",
}


def plot_rain_intensity(df: pd.DataFrame, title: str = "Rain Intensity vs Sales") -> go.Figure:
    """
    Two-panel chart: STL residual and % change vs No Rain, split by rain intensity (mm/h).
    """
    agg = _driver_agg(df, "intensity_cat", _INTENSITY_ORDER, "No Rain")

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["STL Residual (% of shop mean)", "% Change vs No Rain"],
        horizontal_spacing=0.12,
    )

    for i, ycol in enumerate(["pct_resid", "pct_vs_baseline"], start=1):
        for _, row in agg.iterrows():
            n = int(row["n_days"]) if not pd.isna(row["n_days"]) else 0
            y = row[ycol] if not pd.isna(row[ycol]) else 0
            clr = (
                _INTENSITY_COLORS.get(row["intensity_cat"], "#90CAF9")
                if i == 1 else
                ("#66bb6a" if y >= 0 else "#ef5350")
            )
            fig.add_trace(go.Bar(
                x=[row["intensity_cat"]], y=[y],
                name=row["intensity_cat"],
                marker_color=clr,
                text=[f"{y:+.1f}%"],
                textposition="outside",
                hovertemplate=(
                    f"<b>{row['intensity_cat']}</b><br>Value: {y:+.1f}%<br>"
                    f"n={n:,} unique dates<extra></extra>"
                ),
                showlegend=False,
            ), row=1, col=i)

    fig.update_layout(
        title=dict(text=title, font=dict(size=13), x=0.5),
        height=420, bargap=0.3,
        margin=dict(l=20, r=20, t=80, b=90),
        paper_bgcolor="white", plot_bgcolor="white",
    )
    fig.update_xaxes(tickangle=-20)
    fig.update_yaxes(zeroline=True, zerolinecolor="#ccc", gridcolor="#eee")
    return fig


# ── Sunshine Fraction Chart ───────────────────────────────────────────────────

_SUNSHINE_ORDER = [
    "Overcast (<25%)", "Partly Cloudy (25–50%)", "Mostly Sunny (50–75%)", "Clear (>75%)",
]
_SUNSHINE_COLORS = {
    "Overcast (<25%)":        "#90A4AE",
    "Partly Cloudy (25–50%)": "#FFD54F",
    "Mostly Sunny (50–75%)":  "#FFB300",
    "Clear (>75%)":           "#E65100",
}


def plot_sunshine_fraction(df: pd.DataFrame, title: str = "Sunshine Fraction vs Sales") -> go.Figure:
    """
    Two-panel chart: STL residual and % change vs Overcast, split by sunshine fraction.
    """
    agg = _driver_agg(df, "sunshine_cat", _SUNSHINE_ORDER, "Overcast (<25%)")

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["STL Residual (% of shop mean)", "% Change vs Overcast Day"],
        horizontal_spacing=0.12,
    )

    for i, ycol in enumerate(["pct_resid", "pct_vs_baseline"], start=1):
        for _, row in agg.iterrows():
            n = int(row["n_days"]) if not pd.isna(row["n_days"]) else 0
            y = row[ycol] if not pd.isna(row[ycol]) else 0
            clr = (
                _SUNSHINE_COLORS.get(row["sunshine_cat"], "#90A4AE")
                if i == 1 else
                ("#66bb6a" if y >= 0 else "#ef5350")
            )
            fig.add_trace(go.Bar(
                x=[row["sunshine_cat"]], y=[y],
                name=row["sunshine_cat"],
                marker_color=clr,
                text=[f"{y:+.1f}%"],
                textposition="outside",
                hovertemplate=(
                    f"<b>{row['sunshine_cat']}</b><br>Value: {y:+.1f}%<br>"
                    f"n={n:,} unique dates<extra></extra>"
                ),
                showlegend=False,
            ), row=1, col=i)

    fig.update_layout(
        title=dict(text=title, font=dict(size=13), x=0.5),
        height=420, bargap=0.3,
        margin=dict(l=20, r=20, t=80, b=90),
        paper_bgcolor="white", plot_bgcolor="white",
    )
    fig.update_xaxes(tickangle=-20)
    fig.update_yaxes(zeroline=True, zerolinecolor="#ccc", gridcolor="#eee")
    return fig
