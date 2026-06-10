import pandas as pd
import streamlit as st
from services.porcessors import (
    compute_spike_cause_distribution,
    compute_route_spike_bands,
)
from charts.event_charts import (
    plot_spike_cause_distribution,
    plot_route_spike_bands,
    plot_no_event_daily,
)

_SUMMER_MONTHS = {6, 7, 8}


@st.cache_data(show_spinner=False)
def _cached_figures(sellout, df_events, threshold: float = 2.0):
    """Compute both chart variants once; cache together so checkbox toggle is instant."""
    cause_df, spike_df = compute_spike_cause_distribution(
        sellout, df_events, threshold=threshold
    )

    fig_full = plot_spike_cause_distribution(cause_df)

    same_day_count = int(
        cause_df.loc[cause_df["spike_cause"] == "event_same_day", "count"].sum()
    )
    no_event_count = int(cause_df["count"].sum()) - same_day_count
    cause_df_2 = pd.DataFrame(
        [
            {"spike_cause": "event_same_day", "count": same_day_count},
            {"spike_cause": "no_event", "count": no_event_count},
        ]
    )
    fig_simple = plot_spike_cause_distribution(cause_df_2)

    # Summer no-event breakdown (include_adjacent=True pool)
    no_event_df = spike_df[spike_df["spike_cause"] == "no_event"].copy()
    no_event_df["month"] = pd.to_datetime(no_event_df["date"]).dt.month
    no_event_df["year"] = pd.to_datetime(no_event_df["date"]).dt.year
    summer_no_event = int(no_event_df["month"].isin(_SUMMER_MONTHS).sum())

    # Non-summer no-event spikes for route analysis
    no_event_ns = no_event_df[~no_event_df["month"].isin(_SUMMER_MONTHS)].copy()

    # Trade promotion = ≥50% of route shops also spiked
    trade_promo_count = int((no_event_ns["pct_route_spiking"] >= 50).sum())

    # Truly unexplained = non-summer + not trade promotion
    truly_unexplained_df = no_event_ns[no_event_ns["pct_route_spiking"] < 50].copy()

    # include_adjacent=False pool: day-before + day-after reclassified as no-event
    adj_no_event_df = spike_df[
        spike_df["spike_cause"].isin(["no_event", "event_day_before", "event_day_after"])
    ].copy()
    adj_no_event_df["month"] = pd.to_datetime(adj_no_event_df["date"]).dt.month
    adj_no_event_df["year"] = pd.to_datetime(adj_no_event_df["date"]).dt.year

    # Pre-compute overall band chart
    overall_bands = compute_route_spike_bands(no_event_ns)
    fig_route_overall = plot_route_spike_bands(
        overall_bands,
        f"Route Spike % for Non-Summer No-Event Spikes (n={len(no_event_ns):,})",
    )

    return (
        fig_full,
        fig_simple,
        cause_df,
        summer_no_event,
        no_event_ns,
        fig_route_overall,
        trade_promo_count,
        no_event_df,
        truly_unexplained_df,
        adj_no_event_df,
    )


def _pct(n, total):
    return f"{100 * n / total:.1f}%" if total else "0.0%"


@st.cache_data(show_spinner=False)
def _cached_route_bands(no_event_ns, route):
    return compute_route_spike_bands(no_event_ns, route)


_MONTH_NAMES = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
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
    counts["month_name"] = pd.to_datetime(counts["month"], format="%m").dt.strftime(
        "%b"
    )
    return counts


# @st.cache_data(show_spinner=False)
# def _cached_daily_counts(no_event_df, year, month, route=None):
#     """Count no-event spikes per day for a given year+month and optional route.
#     Also returns unique customer count for the filtered selection."""
#     df = no_event_df[(no_event_df["year"] == year) & (no_event_df["month"] == month)]
#     if route is not None:
#         df = df[df["route"] == route]
#     df = df.copy()
#     df["day"] = pd.to_datetime(df["date"]).dt.day
#     counts = df.groupby("day").size().reindex(range(1, 32), fill_value=0).reset_index()
#     counts.columns = ["day", "count"]
#     unique_customers = int(df["customer_code"].nunique()) if not df.empty else 0
#     return counts, unique_customers


@st.cache_data(show_spinner=False)
def _cached_daily_counts(no_event_df, year, months, routes=None):
    """Count no-event spikes per day for a given year + tuple of months (and optional routes tuple)."""
    df = no_event_df[no_event_df["year"] == year]
    if months:
        df = df[df["month"].isin(months)]
    if routes is not None:
        df = df[df["route"].isin(routes)]
    df = df.copy()
    df["day"] = pd.to_datetime(df["date"]).dt.day
    counts = df.groupby("day").size().reindex(range(1, 32), fill_value=0).reset_index()
    counts.columns = ["day", "count"]
    return counts


@st.cache_data(show_spinner=False)
def _cached_shop_count(cust_route_df, routes=None):
    """Total unique shops from sellout. cust_route_df: deduplicated [customer_code, route] slice."""
    if routes:
        return int(
            cust_route_df[cust_route_df["route"].isin(routes)]["customer_code"].nunique()
        )
    return int(cust_route_df["customer_code"].nunique())


def render(sellout, df_events, sellin=None, key="spike_cause_chart"):
    threshold = st.slider(
        "Spike detection threshold (z-score)",
        min_value=0.5,
        max_value=5.0,
        value=2.0,
        step=0.1,
        key=f"{key}_threshold",
        help="A day is flagged as a spike when its sell-out z-score exceeds this value. "
             "Lower = more spikes detected; higher = only the strongest spikes.",
    )

    with st.spinner("Computing overall spike cause distribution…"):
        (
            fig_full,
            fig_simple,
            cause_df,
            summer_no_event,
            no_event_ns,
            fig_route_overall,
            trade_promo_count,
            no_event_df,
            truly_unexplained_df,
            adj_no_event_df,
        ) = _cached_figures(sellout, df_events, threshold=threshold)

    # Build shop lookup early — needed for route chart titles and monthly section
    if sellin is not None:
        common_codes = set(sellin["customer_code"].unique()) & set(sellout["customer_code"].unique())
        cust_route_df = (
            sellout[sellout["customer_code"].isin(common_codes)][["customer_code", "route"]]
            .drop_duplicates()
        )
    else:
        cust_route_df = sellout[["customer_code", "route"]].drop_duplicates()

    n_shops_total = _cached_shop_count(cust_route_df)

    # Patch the overall route chart title to include total shop count
    fig_route_overall.update_layout(
        title=f"Route Spike % for Non-Summer No-Event Spikes "
              f"(n={len(no_event_ns):,} spikes, {n_shops_total:,} shops)"
    )

    cb_col1, cb_col2, cb_col3 = st.columns(3)
    with cb_col1:
        include_day_before = st.checkbox(
            "Include day-before events",
            value=False,
            key=f"{key}_day_before",
        )
    with cb_col2:
        include_day_after = st.checkbox(
            "Include day-after events",
            value=False,
            key=f"{key}_day_after",
        )
    with cb_col3:
        excl_isolated = st.checkbox(
            "Exclude isolated spikes (only 1 shop spiked in territory that day)",
            value=False,
            key=f"{key}_excl_isolated",
        )

    def _get(cause):
        return int(cause_df.loc[cause_df["spike_cause"] == cause, "count"].sum())

    total      = int(cause_df["count"].sum())
    same_day   = _get("event_same_day")
    day_before = _get("event_day_before")
    day_after  = _get("event_day_after")
    no_event   = _get("no_event")

    # Build chart cause_df: fold excluded adjacent causes into no_event
    _chart_no_event = no_event
    _chart_rows = [{"spike_cause": "event_same_day", "count": same_day}]
    if include_day_before:
        _chart_rows.append({"spike_cause": "event_day_before", "count": day_before})
    else:
        _chart_no_event += day_before
    if include_day_after:
        _chart_rows.append({"spike_cause": "event_day_after", "count": day_after})
    else:
        _chart_no_event += day_after
    _chart_rows.append({"spike_cause": "no_event", "count": _chart_no_event})
    st.plotly_chart(
        plot_spike_cause_distribution(pd.DataFrame(_chart_rows)),
        use_container_width=True,
        key=key,
    )

    # Metrics: one column per included cause
    _metric_data = [("Total Spikes", total, None),
                    ("Same-Day Event", same_day, _pct(same_day, total))]
    if include_day_before:
        _metric_data.append(("Day-Before Event", day_before, _pct(day_before, total)))
    if include_day_after:
        _metric_data.append(("Day-After Event", day_after, _pct(day_after, total)))
    _metric_data.append(("No-Event", _chart_no_event, _pct(_chart_no_event, total)))
    for _mc, (_ml, _mv, _md) in zip(st.columns(len(_metric_data)), _metric_data):
        _mc.metric(_ml, f"{_mv:,}", delta=_md, delta_color="off")

    # ── Effective no-event pool ───────────────────────────────────────────────
    # Unincluded adjacent causes fold into the no-event analysis pool
    day_before_df = adj_no_event_df[adj_no_event_df["spike_cause"] == "event_day_before"]
    day_after_df  = adj_no_event_df[adj_no_event_df["spike_cause"] == "event_day_after"]

    eff_no_event_df_base = no_event_df.copy()
    eff_no_event_count   = no_event
    if not include_day_before:
        eff_no_event_df_base = pd.concat([eff_no_event_df_base, day_before_df], ignore_index=True)
        eff_no_event_count  += day_before
    if not include_day_after:
        eff_no_event_df_base = pd.concat([eff_no_event_df_base, day_after_df], ignore_index=True)
        eff_no_event_count  += day_after

    eff_base_summer      = int(eff_no_event_df_base["month"].isin(_SUMMER_MONTHS).sum())
    _eff_ns              = eff_no_event_df_base[~eff_no_event_df_base["month"].isin(_SUMMER_MONTHS)]
    eff_base_trade_promo = int((_eff_ns["pct_route_spiking"] >= 50).sum())
    eff_no_event_ns      = _eff_ns

    # ── Isolated spike computation ────────────────────────────────────────────
    event_explained = total - eff_no_event_count
    orig_event_pct = 100 * event_explained / total if total else 0

    isolated_mask = eff_no_event_df_base["spiking_shops_in_route"] == 1
    isolated_count = int(isolated_mask.sum())

    if excl_isolated:
        iso_filtered_df = eff_no_event_df_base[~isolated_mask]
        eff_no_event = eff_no_event_count - isolated_count
        eff_summer_no_event = int(iso_filtered_df["month"].isin(_SUMMER_MONTHS).sum())
        eff_non_summer_no_event = eff_no_event - eff_summer_no_event
        eff_ns_df = iso_filtered_df[~iso_filtered_df["month"].isin(_SUMMER_MONTHS)]
        eff_trade_promo_count = int((eff_ns_df["pct_route_spiking"] >= 50).sum())
        total_explained_base = event_explained + isolated_count
    else:
        eff_no_event = eff_no_event_count
        eff_summer_no_event = eff_base_summer
        eff_non_summer_no_event = eff_no_event_count - eff_base_summer
        eff_trade_promo_count = eff_base_trade_promo
        total_explained_base = event_explained

    if excl_isolated:
        st.markdown("---")
        st.caption("**Isolated Spike Adjustment (Only 1 Shop Spiked in Territory That Day)**")
        isolated_pct = 100 * isolated_count / total if total else 0
        total_after_isolated_pct = 100 * total_explained_base / total if total else 0
        eff_no_event_pct = 100 * eff_no_event / total if total else 0

        i1, i2, i3, i4 = st.columns(4)
        i1.metric(
            "Isolated No-Event Spikes",
            f"{isolated_count:,}",
            delta=f"{_pct(isolated_count, no_event)} of no-event  |  {isolated_pct:.1f}% of total",
            delta_color="off",
        )
        i2.metric(
            "Adjusted No-Event",
            f"{eff_no_event:,}",
            delta=f"−{isolated_count:,} removed",
            delta_color="off",
        )
        i3.metric(
            "Total Explained (Event + Isolated)",
            f"{total_after_isolated_pct:.1f}%",
            delta=f"{isolated_pct:+.1f}% (was {orig_event_pct:.1f}%)",
            delta_color="normal",
        )
        i4.metric(
            "Unexplained After Isolated Adj.",
            f"{eff_no_event_pct:.1f}%",
            delta=f"{eff_no_event_pct - 100 * no_event / total:+.1f}% (was {100 * no_event / total:.1f}%)",
            delta_color="inverse",
        )

    # ── Summer seasonality breakdown ──────────────────────────────────────────
    st.markdown("---")
    st.caption("**Summer Seasonality Adjustment (Jun – Aug)**")

    # Waterfall step: summer no-event spikes reclassified as seasonality-explained
    total_explained_after_summer = total_explained_base + eff_summer_no_event
    total_explained_after_summer_pct = 100 * total_explained_after_summer / total if total else 0
    eff_non_summer_no_event_pct = 100 * eff_non_summer_no_event / total if total else 0
    summer_add_pct = 100 * eff_summer_no_event / total if total else 0

    s1, s2, s3, s4 = st.columns(4)
    s1.metric(
        "No-Event in Summer",
        f"{eff_summer_no_event:,}",
        delta=_pct(eff_summer_no_event, eff_no_event) + " of no-event",
        delta_color="off",
    )
    s2.metric(
        "Non-Summer No-Event",
        f"{eff_non_summer_no_event:,}",
        delta=_pct(eff_non_summer_no_event, eff_no_event) + " of no-event",
        delta_color="off",
    )
    s3.metric(
        "Total Explained (Event + Summer)" if not excl_isolated else "Total Explained (Event + Isolated + Summer)",
        f"{total_explained_after_summer_pct:.1f}%",
        delta=f"{summer_add_pct:+.1f}% (was {100 * total_explained_base / total if total else 0:.1f}%)",
        delta_color="normal",
    )
    s4.metric(
        "Unexplained After Summer Adj.",
        f"{eff_non_summer_no_event_pct:.1f}%",
        delta=f"{eff_non_summer_no_event_pct - 100 * eff_no_event / total:+.1f}% (was {100 * eff_no_event / total:.1f}%)",
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
    st.plotly_chart(
        fig_route_overall, use_container_width=True, key=f"{key}_route_overall"
    )

    # Route-level drill-down
    routes = sorted(eff_no_event_ns["route"].dropna().unique().tolist())
    if routes:
        selected_route = st.selectbox(
            "Select Territory (Route)", routes, key=f"{key}_route_select"
        )
        route_bands = _cached_route_bands(eff_no_event_ns, selected_route)
        n_route = int((eff_no_event_ns["route"] == selected_route).sum())
        n_shops_route = _cached_shop_count(cust_route_df, routes=(selected_route,))
        fig_route = plot_route_spike_bands(
            route_bands,
            f"Territory {selected_route}: Route Spike % "
            f"(n={n_route:,} spikes, {n_shops_route:,} shops)",
        )
        st.plotly_chart(fig_route, use_container_width=True, key=f"{key}_route_chart")

    # ── Trade promotion adjustment ────────────────────────────────────────────
    st.markdown("---")
    st.caption("**Trade Promotion Adjustment (≥50% of Route Spiking)**")

    # Waterfall step: trade promo spikes reclassified as channel-push-explained
    truly_unexplained = eff_non_summer_no_event - eff_trade_promo_count
    trade_promo_pct = 100 * eff_trade_promo_count / total if total else 0
    total_explained_final = total_explained_after_summer + eff_trade_promo_count
    total_explained_final_pct = 100 * total_explained_final / total if total else 0
    truly_unexplained_pct = 100 * truly_unexplained / total if total else 0

    t1, t2, t3, t4 = st.columns(4)
    t1.metric(
        "Trade Promotion Spikes",
        f"{eff_trade_promo_count:,}",
        delta=f"{trade_promo_pct:.1f}% of total  |  "
              f"{_pct(eff_trade_promo_count, eff_non_summer_no_event)} of non-summer no-event",
        delta_color="off",
    )
    t2.metric(
        "Total Explained (incl. Trade Promo)",
        f"{total_explained_final_pct:.1f}%",
        delta=f"{trade_promo_pct:+.1f}% (was {total_explained_after_summer_pct:.1f}%)",
        delta_color="normal",
    )
    t3.metric(
        "Truly Unexplained",
        f"{truly_unexplained:,}",
        delta=_pct(truly_unexplained, total) + " of total",
        delta_color="off",
    )
    t4.metric(
        "Final Unexplained %",
        f"{truly_unexplained_pct:.1f}%",
        delta=f"{truly_unexplained_pct - eff_non_summer_no_event_pct:+.1f}% (was {eff_non_summer_no_event_pct:.1f}%)",
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

    active_df = eff_no_event_df_base.copy()
    if excl_summer:
        active_df = active_df[~active_df["month"].isin(_SUMMER_MONTHS)]
    if excl_tp:
        active_df = active_df[active_df["pct_route_spiking"].fillna(0) < 50]

    labels = []
    if excl_summer:
        labels.append("summer")
    if excl_tp:
        labels.append("trade promotion")
    if labels:
        st.caption(f"Excluding: **{' & '.join(labels)}** spikes.")
    else:
        st.caption("Showing all no-event spikes.")

    years = sorted(active_df["year"].dropna().unique().tolist(), reverse=True)
    if not years:
        st.info("No spikes in this selection.")
        return

    # ── Overall: year + month(s) dropdowns ───────────────────────────────────
    col_y1, col_m1, _ = st.columns([1, 2, 1])
    with col_y1:
        sel_year = st.selectbox("Select Year", years, key=f"{key}_month_year")
    with col_m1:
        sel_months = st.multiselect(
            "Select Month(s)",
            list(range(1, 13)),
            default=[1],
            format_func=lambda x: _MONTH_NAMES[x],
            key=f"{key}_month_sel",
        )

    if not sel_months:
        st.info("Please select at least one month.")
    else:
        months_tuple = tuple(sorted(sel_months))
        month_label = ", ".join(_MONTH_NAMES[m][:3] for m in months_tuple)
        n_shops = _cached_shop_count(cust_route_df)
        daily = _cached_daily_counts(active_df, sel_year, months_tuple)
        st.plotly_chart(
            plot_no_event_daily(
                daily, sel_year, month_label, n_shops=n_shops, months=months_tuple
            ),
            use_container_width=True,
            key=f"{key}_month_chart",
        )

    # ── By Territory: year + month + route dropdowns ──────────────────────────
    st.markdown("**By Territory**")
    routes_ne = sorted(active_df["route"].dropna().unique().tolist())
    if routes_ne:
        col_y2, col_m2, col_r2 = st.columns([1, 2, 2])
        with col_y2:
            sel_year_r = st.selectbox("Year", years, key=f"{key}_month_year_r")
        with col_m2:
            sel_months_r = st.multiselect(
                "Month(s)",
                list(range(1, 13)),
                default=[1],
                format_func=lambda x: _MONTH_NAMES[x],
                key=f"{key}_month_sel_r",
            )
        with col_r2:
            sel_routes_r = st.multiselect(
                "Territory (Route)", routes_ne, key=f"{key}_month_route"
            )

        if not sel_months_r or not sel_routes_r:
            st.info("Please select at least one month and one territory.")
        else:
            months_r_tuple = tuple(sorted(sel_months_r))
            routes_r_tuple = tuple(sorted(sel_routes_r))
            month_label_r = ", ".join(_MONTH_NAMES[m][:3] for m in months_r_tuple)
            if len(routes_r_tuple) == 1:
                route_label_r = f" — Territory {routes_r_tuple[0]}"
            else:
                route_label_r = f" — Territories {', '.join(map(str, routes_r_tuple))}"
            n_shops_r = _cached_shop_count(cust_route_df, routes=routes_r_tuple)
            daily_r = _cached_daily_counts(
                active_df, sel_year_r, months_r_tuple, routes=routes_r_tuple
            )
            st.plotly_chart(
                plot_no_event_daily(
                    daily_r,
                    sel_year_r,
                    month_label_r,
                    route_label=route_label_r,
                    n_shops=n_shops_r,
                    months=months_r_tuple,
                ),
                use_container_width=True,
                key=f"{key}_month_route_chart",
            )
