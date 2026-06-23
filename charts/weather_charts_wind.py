import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


# ── Constants ─────────────────────────────────────────────────────────────────

_WIND_COLOR      = "#1a7a4a"
_WIND_FILL_COLOR = "rgba(26,122,74,0.12)"

_WC_OLS_COLOR      = "#6B3FA0"
_WC_OLS_FILL_COLOR = "rgba(107,63,160,0.12)"
_WC_OLS_KNOTS      = [0, 10, 20]

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

_GUST_ORDER  = ["Calm (<20 km/h)", "Moderate (20–40 km/h)", "Gusty (>40 km/h)"]
_GUST_COLORS = {"Calm (<20 km/h)": "#A5D6A7", "Moderate (20–40 km/h)": "#FFF176", "Gusty (>40 km/h)": "#EF9A9A"}

_MONTH_NAMES = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}


# ── Shared driver aggregation helper ─────────────────────────────────────────

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


# ── Rain Band Effect ──────────────────────────────────────────────────────────

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


# ── Wind Chill OLS Effect ─────────────────────────────────────────────────────

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


# ── Wind Speed OLS Effect ─────────────────────────────────────────────────────

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


# ── Wind Chill % Bars ─────────────────────────────────────────────────────────

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


# ── Storm Charts ──────────────────────────────────────────────────────────────

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


# ── Snow Analysis ─────────────────────────────────────────────────────────────

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


# ── Wind Gust Analysis ────────────────────────────────────────────────────────

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


# ── OLS Category Effect (Forest Plot) ────────────────────────────────────────

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
