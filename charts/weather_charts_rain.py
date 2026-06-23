import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from services.porcessors import weather_analysis_processor


_OLS_BAND_COLORS = {"light": "#a8d0e6", "moderate": "#5b9bd5", "heavy": "#1f4e79"}

_RAIN_BAND_COLORS = {
    "none": "#d0e8f5",
    "light": "#5b9bd5",
    "moderate": "#2f6da8",
    "heavy": "#c0392b",
}

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

_NIGHT_BANDS = {"00-02", "03-05", "21-23"}
_DUR_COLORS = ["#27ae60", "#f1c40f", "#e67e22", "#e74c3c", "#8e44ad"]

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

_INTENSITY_ORDER = ["No Rain", "Drizzle (<1 mm/h)", "Moderate (1–4 mm/h)", "Heavy (>4 mm/h)"]
_INTENSITY_COLORS = {
    "No Rain":              "#90CAF9",
    "Drizzle (<1 mm/h)":   "#64B5F6",
    "Moderate (1–4 mm/h)": "#1E88E5",
    "Heavy (>4 mm/h)":     "#0D47A1",
}


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


def plot_rain_streak(df: pd.DataFrame, title: str = "Rain Streak & First Dry Day") -> go.Figure:
    """
    Two-panel chart:
    Left: mean STL residual per rain category (% of shop mean)
    Right: % change vs Normal Dry Day
    """
    from charts.weather_charts_wind import _driver_agg

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


def plot_rain_intensity(df: pd.DataFrame, title: str = "Rain Intensity vs Sales") -> go.Figure:
    """
    Two-panel chart: STL residual and % change vs No Rain, split by rain intensity (mm/h).
    """
    from charts.weather_charts_wind import _driver_agg

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
