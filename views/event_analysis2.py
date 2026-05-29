import streamlit as st
import pandas as pd
from decimal import Decimal, ROUND_HALF_UP

from services.filters import get_customer_list_events
from services.data_loader import load_events_data
from services.porcessors import (
    events_analysis_processor,
    detect_spikes_global,
    get_all_shop_events,
)
from charts.event_charts import plot_customer_events_simple
from charts.event_map import plot_event_map_v2


def _round_half_up(value: float) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))


@st.cache_data(show_spinner=False)
def _prepare_df(sellin_cust, sellout_cust, threshold):
    df = events_analysis_processor(sellin_cust, sellout_cust)
    return detect_spikes_global(df, threshold)


@st.cache_data(show_spinner=False)
def _cached_chart(df):
    return plot_customer_events_simple(df)


@st.cache_data(show_spinner=False)
def _cached_all_shop_events(df_events, shop_lat, shop_lon, full_from, full_to):
    """One scan of the 268k-row events DataFrame for this shop + full window."""
    return get_all_shop_events(df_events, shop_lat, shop_lon, full_from, full_to)


def _events_for_spike(all_events_df, spike_dt):
    """In-memory date slice — no DataFrame scan, microseconds."""
    lo = spike_dt - pd.Timedelta(days=1)
    hi = spike_dt + pd.Timedelta(days=1)
    return all_events_df[
        (all_events_df["date"] >= lo) & (all_events_df["date"] <= hi)
    ]


def _build_spike_infos(spikes, all_events_df):
    infos = []
    for _, row in spikes.iterrows():
        spike_dt = pd.Timestamp(row["date"])
        df_ev = _events_for_spike(all_events_df, spike_dt)
        active = df_ev[df_ev["event_type"].isin(["in_range", "district"])]
        infos.append({
            "date": spike_dt,
            "sellout": int(row["sellout"]),
            "z_score": float(row["z_score"]),
            "weekday_mean": float(row.get("weekday_mean", 0) or 0),
            "top_event": active["name"].iloc[0] if not active.empty else None,
        })
    return infos


def _build_all_maps(spike_infos, all_events_df, shop_lat_raw, shop_lon_raw, customer):
    """Generate Folium HTML for every spike in one pass — stored in session_state."""
    maps = {}
    for info in spike_infos:
        date_key = str(info["date"].date())
        df_ev = _events_for_spike(all_events_df, info["date"])
        maps[date_key] = plot_event_map_v2(
            shop_lat_raw, shop_lon_raw, customer, df_ev
        )._repr_html_()
    return maps


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
        if has_event else ""
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


def _render_spike_cards(spike_infos, selected_date_str):
    for row_start in range(0, len(spike_infos), CARDS_PER_ROW):
        row = spike_infos[row_start: row_start + CARDS_PER_ROW]
        cols = st.columns(len(row))
        for col, info in zip(cols, row):
            date_key = str(info["date"].date())
            with col:
                # st.html renders raw HTML without markdown sanitisation — no stray </div>
                st.html(_card_html(info, selected_date_str == date_key))
                if st.button("Select", key=f"ea2_spike_{date_key}"):
                    st.session_state["ea2_selected_date"] = date_key
                    st.rerun()


# ── Main render ───────────────────────────────────────────────────────────────

def render(sellout, sellin):
    df_events = load_events_data()

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
        (sellin_cust["date"].dt.date >= from_date_filter) &
        (sellin_cust["date"].dt.date <= to_date_filter)
    ]
    sellout_cust = sellout_cust[
        (sellout_cust["date"].dt.date >= from_date_filter) &
        (sellout_cust["date"].dt.date <= to_date_filter)
    ]

    df = _prepare_df(sellin_cust, sellout_cust, threshold)
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

    # ── Pre-compute all spike maps once; rebuild only when settings change ────
    map_cache_key = (selected_customer, threshold, from_date_filter, to_date_filter)

    if st.session_state.get("ea2_map_cache_key") != map_cache_key:
        full_from = pd.Timestamp(spikes["date"].min()) - pd.Timedelta(days=1)
        full_to = pd.Timestamp(spikes["date"].max()) + pd.Timedelta(days=1)

        with st.spinner("Loading spike events and building maps…"):
            all_shop_events = _cached_all_shop_events(
                df_events, shop_lat, shop_lon, full_from, full_to
            )
            spike_infos = _build_spike_infos(spikes, all_shop_events)
            maps = _build_all_maps(
                spike_infos, all_shop_events, shop_lat_raw, shop_lon_raw, selected_customer
            )

        st.session_state["ea2_map_cache_key"] = map_cache_key
        st.session_state["ea2_spike_infos"] = spike_infos
        st.session_state["ea2_maps"] = maps
        st.session_state["ea2_all_events"] = all_shop_events

    spike_infos = st.session_state["ea2_spike_infos"]
    maps = st.session_state["ea2_maps"]
    all_shop_events = st.session_state["ea2_all_events"]

    # ── Spike cards ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**Jump to spike date:**")
    _render_spike_cards(spike_infos, st.session_state.get("ea2_selected_date"))

    # ── Map + insight ─────────────────────────────────────────────────────────
    selected_date_str = st.session_state.get("ea2_selected_date")
    if not selected_date_str:
        return

    spike_dt = pd.Timestamp(selected_date_str)
    spike_row = df[df["date"] == spike_dt]

    if not spike_row.empty:
        sr = spike_row.iloc[0]
        weekday_avg = int(sr.get("weekday_mean", 0) or 0)
        excess = int(sr["sellout"]) - weekday_avg
        z = float(sr["z_score"])
        df_ev = _events_for_spike(all_shop_events, spike_dt)
        active_names = df_ev[df_ev["event_type"].isin(["in_range", "district"])]["name"].unique()
        events_html = (
            ", ".join(f"<b>{n}</b>" for n in active_names)
            if len(active_names) else "<i>None detected</i>"
        )

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

    # Map HTML is already built — instant render
    map_html = maps.get(selected_date_str, "")
    if map_html:
        st.components.v1.html(map_html, height=520)
    else:
        st.info("No map available for this date.")
