import pandas as pd
import streamlit as st
from services.porcessors import compute_spike_cause_distribution, compute_route_spike_bands
from charts.event_charts import (
    plot_spike_cause_distribution,
    plot_route_spike_bands,
    plot_no_event_monthly,
    plot_no_event_daily,
)


_SUMMER_MONTHS = {6, 7, 8}


@st.cache_data(show_spinner=False)
def _cached_figures(sellout, df_events):
    """Compute both chart variants once; cache together so checkbox toggle is instant."""
    cause_df, spike_df = compute_spike_cause_distribution(sellout, df_events, threshold=2.0)

    fig_full = plot_spike_cause_distribution(cause_df)

    same_day_count = int(
        cause_df.loc[cause_df["spike_cause"] == "event_same_day", "count"].sum()
    )
    no_event_count = int(cause_df["count"].sum()) - same_day_count
    cause_df_2 = pd.DataFrame([
        {"spike_cause": "event_same_day", "count": same_day_count},
        {"spike_cause": "no_event", "count": no_event_count},
    ])
    fig_simple = plot_spike_cause_distribution(cause_df_2)

    # Summer no-event breakdown
    no_event_df = spike_df[spike_df["spike_cause"] == "no_event"].copy()
    no_event_df["month"] = pd.to_datetime(no_event_df["date"]).dt.month
    no_event_df["year"]  = pd.to_datetime(no_event_df["date"]).dt.year
    summer_no_event = int(no_event_df["month"].isin(_SUMMER_MONTHS).sum())

    # Non-summer no-event spikes for route analysis
    no_event_ns = no_event_df[~no_event_df["month"].isin(_SUMMER_MONTHS)].copy()

    # Trade promotion = ≥50% of route shops also spiked
    trade_promo_count = int((no_event_ns["pct_route_spiking"] >= 50).sum())

    # Truly unexplained = non-summer + not trade promotion
    truly_unexplained_df = no_event_ns[no_event_ns["pct_route_spiking"] < 50].copy()

    # Pre-compute overall band chart
    overall_bands = compute_route_spike_bands(no_event_ns)
    fig_route_overall = plot_route_spike_bands(
        overall_bands,
        f"Route Spike % for Non-Summer No-Event Spikes (n={len(no_event_ns):,})",
    )

    return fig_full, fig_simple, cause_df, summer_no_event, no_event_ns, fig_route_overall, trade_promo_count, no_event_df, truly_unexplained_df


def _pct(n, total):
    return f"{100 * n / total:.1f}%" if total else "0.0%"


@st.cache_data(show_spinner=False)
def _cached_route_bands(no_event_ns, route):
    return compute_route_spike_bands(no_event_ns, route)


_MONTH_NAMES = {
    0: "All Months", 1: "January", 2: "February", 3: "March",
    4: "April", 5: "May", 6: "June", 7: "July",
    8: "August", 9: "September", 10: "October", 11: "November", 12: "December",
}


@st.cache_data(show_spinner=False)
def _cached_monthly_counts(no_event_df, year, route=None):
    """Count no-event spikes per month for a given year and optional route."""
    df = no_event_df[no_event_df["year"] == year]
    if route is not None:
        df = df[df["route"] == route]
    counts = (
        df.groupby("month").size().reindex(range(1, 13), fill_value=0).reset_index()
    )
    counts.columns = ["month", "count"]
    counts["month_name"] = pd.to_datetime(counts["month"], format="%m").dt.strftime("%b")
    return counts


@st.cache_data(show_spinner=False)
def _cached_daily_counts(no_event_df, year, month, route=None):
    """Count no-event spikes per day for a given year+month and optional route."""
    df = no_event_df[(no_event_df["year"] == year) & (no_event_df["month"] == month)]
    if route is not None:
        df = df[df["route"] == route]
    df = df.copy()
    df["day"] = pd.to_datetime(df["date"]).dt.day
    counts = (
        df.groupby("day").size().reindex(range(1, 32), fill_value=0).reset_index()
    )
    counts.columns = ["day", "count"]
    return counts


def render(sellout, df_events, key="spike_cause_chart"):
    with st.spinner("Computing overall spike cause distribution…"):
        fig_full, fig_simple, cause_df, summer_no_event, no_event_ns, fig_route_overall, trade_promo_count, no_event_df, truly_unexplained_df = _cached_figures(sellout, df_events)

    include_adjacent = st.checkbox(
        "Include day-before / day-after events",
        value=True,
        key=f"{key}_adjacent",
    )

    st.plotly_chart(
        fig_full if include_adjacent else fig_simple,
        use_container_width=True,
        key=key,
    )

    def _get(cause):
        return int(cause_df.loc[cause_df["spike_cause"] == cause, "count"].sum())

    total = int(cause_df["count"].sum())
    same_day = _get("event_same_day")
    day_before = _get("event_day_before")
    day_after = _get("event_day_after")
    adjacent = day_before + day_after
    no_event = _get("no_event")

    if include_adjacent:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Spikes", f"{total:,}")
        c2.metric("Same-Day Event", f"{same_day:,}", delta=_pct(same_day, total), delta_color="off")
        c3.metric("Day-Before Event", f"{day_before:,}", delta=_pct(day_before, total), delta_color="off")
        c4.metric("Day-After Event", f"{day_after:,}", delta=_pct(day_after, total), delta_color="off")
        c5.metric("No-Event", f"{no_event:,}", delta=_pct(no_event, total), delta_color="off")
    else:
        no_event_simple = total - same_day
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Spikes", f"{total:,}")
        c2.metric("Same-Day Event", f"{same_day:,}", delta=_pct(same_day, total), delta_color="off")
        c3.metric("No-Event", f"{no_event_simple:,}", delta=_pct(no_event_simple, total), delta_color="off")

    # ── Summer seasonality breakdown ──────────────────────────────────────────
    st.markdown("---")
    st.caption("**Summer Seasonality Adjustment (Jun – Aug)**")
    non_summer_no_event = no_event - summer_no_event
    adj_total           = total - summer_no_event
    event_explained     = total - no_event          # same-day + before + after
    adj_event_pct       = 100 * event_explained / adj_total if adj_total else 0
    adj_no_event_pct    = 100 * non_summer_no_event / adj_total if adj_total else 0

    s1, s2, s3, s4 = st.columns(4)
    s1.metric(
        "No-Event in Summer",
        f"{summer_no_event:,}",
        delta=_pct(summer_no_event, no_event) + " of no-event",
        delta_color="off",
    )
    s2.metric(
        "Non-Summer No-Event",
        f"{non_summer_no_event:,}",
        delta=_pct(non_summer_no_event, no_event) + " of no-event",
        delta_color="off",
    )
    orig_event_pct    = 100 * event_explained / total if total else 0
    orig_no_event_pct = 100 * no_event / total if total else 0

    s3.metric(
        "Adj. Event-Explained %",
        f"{adj_event_pct:.1f}%",
        delta=f"{adj_event_pct - orig_event_pct:+.1f}% (was {orig_event_pct:.1f}%)",
        delta_color="normal",
    )
    s4.metric(
        "Adj. No-Event %",
        f"{adj_no_event_pct:.1f}%",
        delta=f"{adj_no_event_pct - orig_no_event_pct:+.1f}% (was {orig_no_event_pct:.1f}%)",
        delta_color="inverse",
    )

    # ── Route-level trade promotion analysis ──────────────────────────────────
    st.markdown("---")
    st.markdown("**Route-Level Trade Promotion Analysis**")
    st.caption(
        "For the remaining non-summer no-event spikes, what % of shops in the same "
        "territory also spiked on the same day? A high % suggests a coordinated "
        "channel push (trade promotion) rather than an isolated spike."
    )

    # Overall chart (pre-computed)
    st.plotly_chart(fig_route_overall, use_container_width=True, key=f"{key}_route_overall")

    # Route-level drill-down
    routes = sorted(no_event_ns["route"].dropna().unique().tolist())
    if routes:
        selected_route = st.selectbox(
            "Select Territory (Route)", routes, key=f"{key}_route_select"
        )
        route_bands = _cached_route_bands(no_event_ns, selected_route)
        n_route = int((no_event_ns["route"] == selected_route).sum())
        fig_route = plot_route_spike_bands(
            route_bands,
            f"Territory {selected_route}: Route Spike % (n={n_route:,} non-summer no-event spikes)",
        )
        st.plotly_chart(fig_route, use_container_width=True, key=f"{key}_route_chart")

    # ── Trade promotion adjustment ────────────────────────────────────────────
    st.markdown("---")
    st.caption("**Trade Promotion Adjustment (≥50% of Route Spiking)**")
    non_summer_no_event = no_event - summer_no_event
    truly_unexplained   = non_summer_no_event - trade_promo_count
    orig_no_event_pct   = 100 * no_event / total if total else 0
    final_pct           = 100 * truly_unexplained / total if total else 0

    t1, t2, t3, t4 = st.columns(4)
    t1.metric(
        "Non-Summer No-Event",
        f"{non_summer_no_event:,}",
        delta=_pct(non_summer_no_event, total) + " of total",
        delta_color="off",
    )
    t2.metric(
        "Trade Promotion Spikes",
        f"{trade_promo_count:,}",
        delta=_pct(trade_promo_count, non_summer_no_event) + " of non-summer no-event",
        delta_color="off",
    )
    t3.metric(
        "Truly Unexplained",
        f"{truly_unexplained:,}",
        delta=_pct(truly_unexplained, total) + " of total",
        delta_color="off",
    )
    t4.metric(
        "Final No-Event %",
        f"{final_pct:.1f}%",
        delta=f"{final_pct - orig_no_event_pct:+.1f}% (was {orig_no_event_pct:.1f}%)",
        delta_color="inverse",
    )

    # ── No-event spike monthly distribution ──────────────────────────────────
    st.markdown("---")
    st.markdown("**No-Event Spikes — Monthly Distribution**")

    cb1, cb2 = st.columns(2)
    with cb1:
        excl_summer = st.checkbox(
            "Exclude summer spikes (Jun–Aug)",
            value=False,
            key=f"{key}_excl_summer",
        )
    with cb2:
        excl_tp = st.checkbox(
            "Exclude trade promotion spikes (≥50% of route spiking)",
            value=False,
            key=f"{key}_excl_tp",
        )

    active_df = no_event_df.copy()
    if excl_summer:
        active_df = active_df[~active_df["month"].isin(_SUMMER_MONTHS)]
    if excl_tp:
        active_df = active_df[active_df["pct_route_spiking"].fillna(0) < 50]

    labels = []
    if excl_summer:  labels.append("summer")
    if excl_tp:      labels.append("trade promotion")
    if labels:
        st.caption(f"Excluding: **{' & '.join(labels)}** spikes.")
    else:
        st.caption("Showing all no-event spikes.")

    years = sorted(active_df["year"].dropna().unique().tolist(), reverse=True)
    if not years:
        st.info("No spikes in this selection.")
        return

    # ── Overall: year + month dropdowns ──────────────────────────────────────
    col_y1, col_m1, _ = st.columns([1, 1, 2])
    with col_y1:
        sel_year = st.selectbox("Select Year", years, key=f"{key}_month_year")
    with col_m1:
        sel_month = st.selectbox(
            "Select Month", list(_MONTH_NAMES.keys()),
            format_func=lambda x: _MONTH_NAMES[x],
            key=f"{key}_month_sel",
        )

    if sel_month == 0:
        monthly = _cached_monthly_counts(active_df, sel_year)
        st.plotly_chart(
            plot_no_event_monthly(monthly, sel_year),
            use_container_width=True, key=f"{key}_month_chart",
        )
    else:
        daily = _cached_daily_counts(active_df, sel_year, sel_month)
        st.plotly_chart(
            plot_no_event_daily(daily, sel_year, sel_month, _MONTH_NAMES[sel_month]),
            use_container_width=True, key=f"{key}_month_chart",
        )

    # ── By Territory: year + month + route dropdowns ──────────────────────────
    st.markdown("**By Territory**")
    routes_ne = sorted(active_df["route"].dropna().unique().tolist())
    if routes_ne:
        col_y2, col_m2, col_r2 = st.columns([1, 1, 1])
        with col_y2:
            sel_year_r = st.selectbox("Year", years, key=f"{key}_month_year_r")
        with col_m2:
            sel_month_r = st.selectbox(
                "Month", list(_MONTH_NAMES.keys()),
                format_func=lambda x: _MONTH_NAMES[x],
                key=f"{key}_month_sel_r",
            )
        with col_r2:
            sel_route_r = st.selectbox("Territory (Route)", routes_ne, key=f"{key}_month_route")

        if sel_month_r == 0:
            monthly_r = _cached_monthly_counts(active_df, sel_year_r, route=sel_route_r)
            n_spikes = int(monthly_r["count"].sum())
            st.plotly_chart(
                plot_no_event_monthly(monthly_r, sel_year_r, route=sel_route_r,
                                      n_spikes=n_spikes),
                use_container_width=True, key=f"{key}_month_route_chart",
            )
        else:
            daily_r = _cached_daily_counts(active_df, sel_year_r, sel_month_r, route=sel_route_r)
            st.plotly_chart(
                plot_no_event_daily(daily_r, sel_year_r, sel_month_r,
                                    _MONTH_NAMES[sel_month_r], route=sel_route_r),
                use_container_width=True, key=f"{key}_month_route_chart",
            )
