import streamlit as st
import pandas as pd
import calendar as _cal
from decimal import Decimal, ROUND_HALF_UP

from services.filters import get_customer_list_events
from services.data_loader import load_events_data
from services.porcessors import (
    events_analysis_processor,
    detect_spikes_global,
    get_all_shop_events,
)
from charts.event_charts import (
    plot_customer_events_simple,
    plot_monthly_sales_bars,
    plot_monthly_sales_line,
)
from charts.event_map import plot_event_map_v2
from views import spike_summary



def _round_half_up(value: float) -> float:
    return float(
        Decimal(str(value)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
    )


@st.cache_data(show_spinner=False)
def _prepare_df(sellin_cust, sellout_cust, threshold, sellout_full_cust):
    df = events_analysis_processor(sellin_cust, sellout_cust)
    # Build weekday baseline from full history so the z-score isn't affected by date filter
    ref = (
        sellout_full_cust.groupby("date")["sales_quantity"]
        .sum()
        .reset_index()
        .rename(columns={"sales_quantity": "sellout"})
    )
    return detect_spikes_global(df, threshold, reference_df=ref)


@st.cache_data(show_spinner=False)
def _cached_chart(df):
    return plot_customer_events_simple(df)


@st.cache_data(show_spinner=False)
def _cached_all_shop_events(df_events, shop_lat, shop_lon, full_from, full_to):
    """One scan of the 268k-row events DataFrame for this shop + full window."""
    return get_all_shop_events(df_events, shop_lat, shop_lon, full_from, full_to)


def _events_for_spike(all_events_df, spike_dt, days_before=1, days_after=1):
    """In-memory date slice — no DataFrame scan, microseconds."""
    lo = spike_dt - pd.Timedelta(days=days_before)
    hi = spike_dt + pd.Timedelta(days=days_after)
    return all_events_df[(all_events_df["date"] >= lo) & (all_events_df["date"] <= hi)]


def _build_spike_infos(spikes, all_events_df, days_before=1, days_after=1):
    infos = []
    for _, row in spikes.iterrows():
        spike_dt = pd.Timestamp(row["date"])
        df_ev = _events_for_spike(all_events_df, spike_dt, days_before, days_after)
        # Prefer in_range/district for top_event label, but fall back to any event
        # (too_far events are still real events visible on the map — card should be purple)
        preferred = df_ev[df_ev["event_type"].isin(["in_range", "district"])]
        top_source = preferred if not preferred.empty else df_ev
        infos.append(
            {
                "date": spike_dt,
                "sellout": int(row["sellout"]),
                "z_score": float(row["z_score"]),
                "weekday_mean": float(row.get("weekday_mean", 0) or 0),
                "top_event": top_source["name"].iloc[0] if not top_source.empty else None,
            }
        )
    return infos


@st.cache_data(show_spinner=False)
def _get_sorted_customers(sellout, df_events, threshold):
    """Customer list sorted by same-day event spike count (desc)."""
    from services.porcessors import compute_spike_cause_distribution
    _, spike_df = compute_spike_cause_distribution(sellout, df_events, threshold=threshold)
    event_counts = (
        spike_df[spike_df["spike_cause"] == "event_same_day"]
        .groupby("customer_code").size()
        .to_dict()
    )
    customers = get_customer_list_events(sellout, Top_100=False)
    return sorted(customers, key=lambda c: -event_counts.get(c, 0))


@st.cache_data(show_spinner=False)
def _compute_monthly_stats(df_full, all_events_df):
    """Per-month spike counts + same-day event correlation, sorted by total spikes DESC."""
    ev_dates = set(
        all_events_df[all_events_df["event_type"].isin(["in_range", "district"])]["date"]
    )
    spikes = df_full[df_full["is_spike"]].copy()
    if spikes.empty:
        return []
    spikes["_year"]  = pd.to_datetime(spikes["date"]).dt.year
    spikes["_month"] = pd.to_datetime(spikes["date"]).dt.month
    records = []
    for (year, month), grp in spikes.groupby(["_year", "_month"]):
        total       = len(grp)
        event_count = sum(1 for d in grp["date"] if d in ev_dates)
        records.append({"year": int(year), "month": int(month),
                        "total_spikes": total, "event_spikes": event_count})
    records.sort(key=lambda x: (-x["total_spikes"], -x["event_spikes"]))
    return records


# ── Card rendering ────────────────────────────────────────────────────────────

CARDS_PER_ROW = 6
_PURPLE = "#6c3fc7"
_DARK = "#2d3748"


def _truncate(text, n=22):
    return text[:n] + "…" if len(text) > n else text


def _card_html(info, is_selected):
    has_event = info["top_event"] is not None
    bg = _PURPLE if has_event else _DARK
    if is_selected:
        border = "3px solid #ffd700"
    elif has_event:
        border = "2px solid #a78bfa"
    else:
        border = "1px solid #4a5568"

    event_part = (
        f"<br><span style='font-size:10px;opacity:0.85'>{_truncate(info['top_event'])}</span>"
        if has_event
        else ""
    )
    return (
        f"<div style='background:{bg};border:{border};border-radius:8px;"
        f"padding:10px 6px;text-align:center;color:white;min-height:108px'>"
        f"<span style='font-size:11px;opacity:0.8'>{info['date'].strftime('%d %b %a')}</span><br>"
        f"<strong style='font-size:20px'>{info['sellout']:,}</strong><br>"
        f"<span style='font-size:11px;opacity:0.75'>z={info['z_score']:.1f}</span>"
        f"{event_part}"
        f"</div>"
    )


def _set_selected_date(date_key):
    st.session_state["ea2_selected_date"] = date_key


def _render_spike_cards(spike_infos, selected_date_str):
    for row_start in range(0, len(spike_infos), CARDS_PER_ROW):
        row = spike_infos[row_start : row_start + CARDS_PER_ROW]
        cols = st.columns(len(row))
        for col, info in zip(cols, row):
            date_key = str(info["date"].date())
            with col:
                st.html(_card_html(info, selected_date_str == date_key))
                st.button(
                    "Select",
                    key=f"ea2_spike_{date_key}",
                    on_click=_set_selected_date,
                    args=(date_key,),
                )


# ── Main render ───────────────────────────────────────────────────────────────


def render(sellout, sellin):
    df_events = load_events_data()

    st.markdown("### High-Level Overview")
    st.caption("Spike cause distribution across all customers and all years.")
    spike_summary.render(sellout, df_events, sellin=sellin, key="spike_cause_ea2")

    st.markdown("---")
    st.markdown("### Customer-Level Analysis")
    st.caption(
        "Drill down into a specific customer to explore individual spike dates and nearby events."
    )

    cb_col1, cb_col2 = st.columns(2)
    with cb_col1:
        include_day_before = st.checkbox("Include day before spike", value=True, key="ea2_day_before")
    with cb_col2:
        include_day_after = st.checkbox("Include day after spike", value=True, key="ea2_day_after")
    days_before = 1 if include_day_before else 0
    days_after  = 1 if include_day_after  else 0

    col1, col2 = st.columns(2)
    with col1:
        customer_list = get_customer_list_events(sellout, Top_100=False)
        selected_customer = st.selectbox(
            "Select Customer", customer_list, key="ea2_customer"
        )
        if st.session_state.get("ea2_prev_customer") != selected_customer:
            st.session_state["ea2_selected_date"] = None
            st.session_state["ea2_prev_customer"] = selected_customer

    with col2:
        threshold = st.slider(
            "Spike Threshold (z-score)", 0.0, 5.0, 2.0, step=0.1, key="ea2_threshold"
        )
        st.caption(
            "A **spike** is a day where sell-out is abnormally high vs the same "
            "weekday's historical average. The z-score measures how many standard "
            "deviations above that average the day's sales are."
        )

    sellin_cust = sellin[sellin["customer_code"] == selected_customer]
    sellout_cust = sellout[sellout["customer_code"] == selected_customer]
    sellout_cust_full = sellout_cust.copy()  # full history — used for weekday baseline

    min_date = sellout_cust["date"].min().date()
    max_date = sellout_cust["date"].max().date()

    date_range = st.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        key="ea2_date_range",
    )
    if len(date_range) < 2:
        st.info("Please select an end date to continue.")
        st.stop()

    from_date_filter, to_date_filter = date_range
    sellin_cust = sellin_cust[
        (sellin_cust["date"].dt.date >= from_date_filter)
        & (sellin_cust["date"].dt.date <= to_date_filter)
    ]
    sellout_cust = sellout_cust[
        (sellout_cust["date"].dt.date >= from_date_filter)
        & (sellout_cust["date"].dt.date <= to_date_filter)
    ]

    df = _prepare_df(sellin_cust, sellout_cust, threshold, sellout_cust_full)
    fig = _cached_chart(df)
    st.plotly_chart(fig, use_container_width=True)

    spikes = df[df["is_spike"]].copy()
    if spikes.empty:
        st.info("No spikes detected with current threshold.")
        return

    # ── Shop coordinates ──────────────────────────────────────────────────────
    shop_row = sellout[sellout["customer_code"] == selected_customer][
        ["latitude", "longitude"]
    ].iloc[0]
    shop_lat_raw = float(shop_row["latitude"])
    shop_lon_raw = float(shop_row["longitude"])
    shop_lat = _round_half_up(shop_lat_raw)
    shop_lon = _round_half_up(shop_lon_raw)

    # ── Load spike event data once; maps built lazily per clicked date ───────
    map_cache_key = (
        selected_customer,
        threshold,
        from_date_filter,
        to_date_filter,
        days_before,
        days_after,
    )

    if st.session_state.get("ea2_map_cache_key") != map_cache_key:
        full_from = pd.Timestamp(spikes["date"].min()) - pd.Timedelta(days=days_before)
        full_to   = pd.Timestamp(spikes["date"].max()) + pd.Timedelta(days=days_after)

        with st.spinner("Loading spike event data…"):
            all_shop_events = _cached_all_shop_events(
                df_events, shop_lat, shop_lon, full_from, full_to
            )
            spike_infos = _build_spike_infos(spikes, all_shop_events, days_before, days_after)

        st.session_state["ea2_map_cache_key"] = map_cache_key
        st.session_state["ea2_spike_infos"] = spike_infos
        st.session_state["ea2_maps"] = {}  # filled lazily on click
        st.session_state["ea2_all_events"] = all_shop_events
        st.session_state["ea2_selected_date"] = None  # reset selection on new settings

    spike_infos = st.session_state["ea2_spike_infos"]
    maps = st.session_state["ea2_maps"]
    all_shop_events = st.session_state["ea2_all_events"]

    # ── Spike cards ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**Jump to spike date:**")
    _render_spike_cards(spike_infos, st.session_state.get("ea2_selected_date"))

    # ── Map + insight (only when a date is selected) ─────────────────────────
    selected_date_str = st.session_state.get("ea2_selected_date")
    if selected_date_str:
        spike_dt  = pd.Timestamp(selected_date_str)
        spike_row = df[df["date"] == spike_dt]

        if not spike_row.empty:
            sr          = spike_row.iloc[0]
            weekday_avg = int(sr.get("weekday_mean", 0) or 0)
            excess      = int(sr["sellout"]) - weekday_avg
            z           = float(sr["z_score"])
            df_ev       = _events_for_spike(all_shop_events, spike_dt, days_before, days_after)
            close_names = df_ev[df_ev["event_type"].isin(["in_range", "district"])]["name"].unique()
            far_names   = df_ev[df_ev["event_type"] == "too_far"]["name"].unique()
            parts = [f"<b>{n}</b>" for n in close_names] + [
                f"<b>{n}</b> <span style='font-size:11px;opacity:0.7'>(too far)</span>"
                for n in far_names
            ]
            events_html = ", ".join(parts) if parts else "<i>None detected</i>"

            st.markdown("---")
            st.markdown(
                f"""<div style='background:#e8f8e8;border:1px solid #27ae60;border-radius:8px;
                    padding:14px 18px;margin-bottom:12px'>
                    <div style='font-weight:bold;color:#1a7a1a;font-size:15px'>
                        Spike Insight: {spike_dt.strftime('%A %d %b %Y')}
                    </div>
                    <div style='margin-top:8px;color:#1a7a1a'>
                        Volume: <b>{int(sr['sellout']):,} units</b>
                        &nbsp;(weekday avg: {weekday_avg:,}, +{excess:,} excess)
                        &nbsp;|&nbsp; Z-score: <b>{z:.2f}</b>
                    </div>
                    <div style='margin-top:6px;color:#1a7a1a'>
                        Active events: {events_html}
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )

        # Build map for this date on demand; cache so revisiting is instant
        if selected_date_str not in maps:
            spike_dt_map = pd.Timestamp(selected_date_str)
            df_ev_map    = _events_for_spike(all_shop_events, spike_dt_map, days_before, days_after)
            maps[selected_date_str] = plot_event_map_v2(
                shop_lat_raw, shop_lon_raw, selected_customer, df_ev_map
            )._repr_html_()
            st.session_state["ea2_maps"] = maps

        st.components.v1.html(maps[selected_date_str], height=520)

    # ── Monthly Sales — Daily Spike & Event Breakdown ─────────────────────────
    st.markdown("---")
    st.markdown("### Monthly Sales — Daily Spike & Event Breakdown")
    st.caption(
        "Daily sell-out bars coloured by spike type. "
        "**Purple** = spike with a same-day event nearby · "
        "**Red** = unexplained spike · "
        "**Gray** = normal day. "
        "Customers sorted by event-correlated spike count. Months sorted by total spike count."
    )

    c_mb1, c_mb2, c_mb3 = st.columns([1, 2, 2])
    with c_mb1:
        threshold_mb = st.slider(
            "Spike threshold (z)", 0.0, 5.0, 2.0, step=0.1, key="ea2_mb_threshold"
        )
    with c_mb2:
        sorted_custs_mb = _get_sorted_customers(sellout, df_events, threshold_mb)
        sel_cust_mb = st.selectbox("Customer", sorted_custs_mb, key="ea2_mb_customer")

    so_mb = sellout[sellout["customer_code"] == sel_cust_mb]
    si_mb = sellin[sellin["customer_code"] == sel_cust_mb]

    _shop_row_mb = so_mb[["latitude", "longitude"]].iloc[0]
    _lat_mb      = _round_half_up(float(_shop_row_mb["latitude"]))
    _lon_mb      = _round_half_up(float(_shop_row_mb["longitude"]))

    df_mb_full   = _prepare_df(si_mb, so_mb, threshold_mb, so_mb)
    _from_mb     = pd.Timestamp(so_mb["date"].min())
    _to_mb       = pd.Timestamp(so_mb["date"].max())
    all_ev_mb    = _cached_all_shop_events(df_events, _lat_mb, _lon_mb, _from_mb, _to_mb)
    monthly_stats = _compute_monthly_stats(df_mb_full, all_ev_mb)

    if not monthly_stats:
        with c_mb3:
            st.info("No spikes detected with current threshold.")
    else:
        month_labels = [
            f"{_cal.month_abbr[r['month']]} {r['year']} "
            f"— {r['total_spikes']} spike{'s' if r['total_spikes'] != 1 else ''} "
            f"({r['event_spikes']} with event)"
            for r in monthly_stats
        ]
        with c_mb3:
            month_idx = st.selectbox(
                "Month (sorted by spike count)",
                range(len(month_labels)),
                format_func=lambda i: month_labels[i],
                key="ea2_mb_month",
            )

        rec      = monthly_stats[month_idx]
        year_sel = rec["year"]
        mon_sel  = rec["month"]

        month_df = df_mb_full[
            (df_mb_full["date"].dt.year  == year_sel) &
            (df_mb_full["date"].dt.month == mon_sel)
        ].copy()

        # Same-day event lookup (in_range / district only — no day before/after)
        ev_close    = all_ev_mb[all_ev_mb["event_type"].isin(["in_range", "district"])]
        ev_date_set = set(ev_close["date"])
        ev_name_map = ev_close.groupby("date")["name"].first().to_dict()

        def _classify_day(row):
            if not row["is_spike"]:
                return "normal", None
            if row["date"] in ev_date_set:
                return "spike_event", ev_name_map.get(row["date"])
            return "spike_no_event", None

        classified = month_df.apply(_classify_day, axis=1, result_type="expand")
        classified.columns = ["bar_type", "event_name"]
        month_df = pd.concat(
            [month_df.reset_index(drop=True), classified.reset_index(drop=True)], axis=1
        )

        n_sp     = rec["total_spikes"]
        n_ev     = rec["event_spikes"]
        title_mb = (
            f"<b>{_cal.month_name[mon_sel]} {year_sel} — {sel_cust_mb}</b><br>"
            f"<sub>{n_sp} spike{'s' if n_sp != 1 else ''} · "
            f"{n_ev} with same-day event · {n_sp - n_ev} unexplained"
            f"</sub>"
        )

        # ── Chart 1: Bar chart ────────────────────────────────────────────────
        fig_bars = plot_monthly_sales_bars(month_df, title_mb)
        st.plotly_chart(fig_bars, use_container_width=True, key="ea2_mb_bars")

        # ── Chart 2: Line chart ───────────────────────────────────────────────
        fig_line = plot_monthly_sales_line(
            month_df,
            f"Sell-out trend — {_cal.month_name[mon_sel]} {year_sel}",
        )
        st.plotly_chart(fig_line, use_container_width=True, key="ea2_mb_line")
