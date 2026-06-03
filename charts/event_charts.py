import plotly.graph_objects as go
import plotly.express as px


def plot_customer_events(df):
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


def plot_customer_events_simple(df):
    """Sell-in / Sell-out + spike markers — no stock remaining trace."""
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
        title="Sell-in vs Sell-out with Spikes",
        xaxis=dict(title="Date"),
        yaxis=dict(title="Quantity"),
        hovermode="x unified",
        legend=dict(orientation="v", x=1.01, y=1),
        height=400,
        margin=dict(t=60, b=40),
    )

    return fig


_CAUSE_COLORS = {
    "event_same_day": "#2ecc71",
    "event_day_before": "#3498db",
    "event_day_after": "#9b59b6",
    "no_event": "#95a5a6",
}
_CAUSE_LABELS = {
    "event_same_day": "Event Same Day",
    "event_day_before": "Event Day Before",
    "event_day_after": "Event Day After",
    "no_event": "No Event",
}


def plot_spike_cause_distribution(cause_df):
    """Bar chart: spike count per cause category across all customers."""
    order = ["event_same_day", "event_day_before", "event_day_after", "no_event"]
    present = set(cause_df["spike_cause"].tolist())
    rows = []
    for cause in order:
        if cause not in present:
            continue
        match = cause_df[cause_df["spike_cause"] == cause]
        count = int(match["count"].iloc[0]) if not match.empty else 0
        rows.append({"cause": cause, "label": _CAUSE_LABELS[cause], "count": count})

    total = sum(r["count"] for r in rows)

    fig = go.Figure()
    for r in rows:
        pct = 100 * r["count"] / total if total else 0
        fig.add_trace(
            go.Bar(
                x=[r["label"]],
                y=[r["count"]],
                name=r["label"],
                marker_color=_CAUSE_COLORS[r["cause"]],
                text=f"<b>{r['count']:,}</b><br><b>({pct:.1f}%)</b>",
                textposition="outside",
                textfont=dict(color="black", size=13),
                hovertemplate=f"<b>{r['label']}</b><br>Count: {r['count']:,}<br>Share: {pct:.1f}%<extra></extra>",
            )
        )

    # More bars → tighter gap; fewer bars → wider gap so they don't stretch
    bargap = max(0.2, 0.75 - 0.1 * len(rows))

    fig.update_layout(
        title="Overall Spike Cause Distribution (All Customers, All Years)",
        xaxis=dict(title=""),
        yaxis=dict(title="Number of Spikes"),
        showlegend=False,
        bargap=bargap,
        height=380,
        margin=dict(t=60, b=40),
        plot_bgcolor="white",
        yaxis_gridcolor="#e8e8e8",
    )
    return fig


def plot_no_event_monthly(monthly_df, year: int, route=None, n_spikes: int = None):
    """Bar chart: no-event spike count per month for a given year (and optional route)."""
    _MONTH_COLORS = {
        1: "#5b9bd5",
        2: "#5b9bd5",
        3: "#27ae60",
        4: "#27ae60",
        5: "#27ae60",
        6: "#e74c3c",
        7: "#e74c3c",
        8: "#e74c3c",
        9: "#e67e22",
        10: "#e67e22",
        11: "#e67e22",
        12: "#5b9bd5",
    }
    colors = [_MONTH_COLORS[m] for m in monthly_df["month"]]

    total = int(monthly_df["count"].sum()) if n_spikes is None else n_spikes
    route_label = f" — Territory {route}" if route else " — All Territories"
    title = (
        f"No-Event Spikes by Month: {year}{route_label}  "
        f"<sub>(total: {total:,})</sub>"
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=monthly_df["month_name"],
            y=monthly_df["count"],
            marker_color=colors,
            text=monthly_df["count"].apply(lambda v: str(v) if v > 0 else ""),
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>No-event spikes: %{y}<extra></extra>",
        )
    )

    # Average line
    avg = monthly_df["count"].mean()
    fig.add_hline(
        y=avg,
        line_dash="dot",
        line_color="#888",
        annotation_text=f"avg {avg:.1f}",
        annotation_position="top right",
        annotation_font=dict(size=11, color="#888"),
    )

    fig.update_layout(
        title=dict(text=title, x=0.0, xanchor="left"),
        xaxis=dict(title="Month", type="category"),
        yaxis=dict(
            title="No-Event Spike Count ",
            rangemode="tozero",
            showgrid=True,
            gridcolor="#f0f0f0",
        ),
        template="plotly_white",
        bargap=0.3,
        height=400,
        margin=dict(t=70, b=50, l=60, r=30),
        annotations=[
            dict(
                x=0.01,
                y=1.08,
                xref="paper",
                yref="paper",
                showarrow=False,
                text=(
                    "<span style='color:#5b9bd5'>■ Winter</span>  "
                    "<span style='color:#27ae60'>■ Spring</span>  "
                    "<span style='color:#e74c3c'>■ Summer</span>  "
                    "<span style='color:#e67e22'>■ Autumn</span>"
                ),
                font=dict(size=11),
            )
        ],
    )
    return fig


def plot_route_spike_bands(band_df, title):
    """Bar chart: no-event spikes binned by % of route shops also spiking."""
    labels = [str(b) for b in band_df["band"]]
    # Red = ≥50% (likely trade promotion), blue = <50%
    colors = ["#e74c3c" if b >= "50-60%" else "#2196F3" for b in labels]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=band_df["count"],
            marker_color=colors,
            text=[
                f"<b>{int(c):,}</b><br>({p:.1f}%)" if c > 0 else ""
                for c, p in zip(band_df["count"], band_df["pct_of_group"])
            ],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Count: %{y:,}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis=dict(title="% of Route Shops That Also Had a Spike", type="category"),
        yaxis=dict(title="# No-Event Spikes"),
        template="plotly_white",
        height=430,
        margin=dict(t=80, b=60),
        plot_bgcolor="white",
        yaxis_gridcolor="#e8e8e8",
        annotations=[
            dict(
                text="<b>Red = ≥50%</b> → likely trade promotion",
                xref="paper",
                yref="paper",
                x=0.99,
                y=0.97,
                showarrow=False,
                align="right",
                bgcolor="lightyellow",
                bordercolor="#ccc",
                borderwidth=1,
                font=dict(size=11),
            )
        ],
    )
    return fig


def plot_no_event_daily(daily_df, year: int, month: int, month_name: str, route=None):
    """Bar chart: no-event spike count per day for a given year+month (and optional route)."""
    import calendar

    route_label = f" — Territory {route}" if route else ""
    n_total = int(daily_df["count"].sum())
    title = (
        f"No-Event Spikes per Day: {month_name} {year}{route_label}  "
        f"<sub>(total: {n_total:,})</sub>"
    )

    # Keep only days that exist in that month (drop trailing zeros beyond month end)
    days_in_month = calendar.monthrange(year, month)[1]
    daily_df = daily_df[daily_df["day"] <= days_in_month].copy()

    # Colour weekends differently
    import datetime

    def _is_weekend(day):
        try:
            return datetime.date(year, month, day).weekday() >= 5
        except ValueError:
            return False

    colors = ["#e67e22" if _is_weekend(d) else "#5b9bd5" for d in daily_df["day"]]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=daily_df["day"].astype(str),
            y=daily_df["count"],
            marker_color=colors,
            text=daily_df["count"].apply(lambda v: str(v) if v > 0 else ""),
            textposition="outside",
            hovertemplate="<b>Day %{x}</b><br>No-event spikes: %{y}<extra></extra>",
        )
    )

    avg = daily_df["count"].mean()
    fig.add_hline(
        y=avg,
        line_dash="dot",
        line_color="#888",
        annotation_text=f"avg {avg:.1f}",
        annotation_position="top right",
        annotation_font=dict(size=11, color="#888"),
    )

    fig.update_layout(
        title=dict(text=title, x=0.0, xanchor="left"),
        xaxis=dict(title="Day of Month", type="category"),
        yaxis=dict(
            title="No-Event Spike Count",
            rangemode="tozero",
            showgrid=True,
            gridcolor="#f0f0f0",
        ),
        template="plotly_white",
        bargap=0.2,
        height=400,
        margin=dict(t=70, b=50, l=60, r=30),
        annotations=[
            dict(
                x=0.01,
                y=1.08,
                xref="paper",
                yref="paper",
                showarrow=False,
                text="<span style='color:#5b9bd5'>■ Weekday</span>  "
                "<span style='color:#e67e22'>■ Weekend</span>",
                font=dict(size=11),
            )
        ],
    )
    return fig
