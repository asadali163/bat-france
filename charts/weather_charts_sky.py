import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


# ── Shared constants used by transition charts ────────────────────────────────

_TRANS_COLORS = {
    "Day Before Sunny":  "#90CAF9",   # soft blue — overcast approaching sun
    "Sunny Day":         "#FFD54F",   # gold
    "Day After Sunny":   "#FFAB40",   # amber — sun fading
    "Cloudy":            "#B0BEC5",
    "Sunny":             "#FFD600",
    "Day After Bright":  "#FFAB40",
}
_TRANS_ORDER = ["Day Before Sunny", "Sunny Day", "Day After Sunny"]

_MONTH_LABELS = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May",  6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}

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

_LAYOUT_TRANS_BASE = dict(
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(family="Arial", size=11, color="#2C2C2A"),
    yaxis=dict(showgrid=True, gridcolor="#F1EFE8", zeroline=True,
               zerolinecolor="#888780", zerolinewidth=1.5),
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
    from charts.weather_charts_wind import _driver_agg

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
