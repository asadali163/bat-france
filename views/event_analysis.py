import streamlit as st
import pandas as pd
from services.filters import get_fmc_only, get_customer_list_events
from charts.event_charts import plot_customer_events
from charts.event_map import plot_event_map
from services.data_loader import load_events_data
from services.porcessors import (
    get_events_for_shop,
    events_analysis_processor,
    detect_spikes_global,
)
from views import spike_summary


@st.cache_data(show_spinner=False)
def _cached_events_for_shop(
    df_events: pd.DataFrame,
    shop_lat: float,
    shop_lon: float,
    from_date,
    to_date,
    distance: int,
) -> pd.DataFrame:
    return get_events_for_shop(
        df_events, shop_lat, shop_lon, from_date, to_date, distance
    )


@st.cache_data(show_spinner=False)
def _cached_map_html(
    shop_lat: float,
    shop_lon: float,
    customer: str,
    df_filtered: pd.DataFrame,
    distance: int,
) -> str:
    m = plot_event_map(shop_lat, shop_lon, customer, df_filtered, distance)
    return m._repr_html_()


@st.cache_data(show_spinner=False)
def _prepare_events_df(
    sellin_cust: pd.DataFrame, sellout_cust: pd.DataFrame, threshold: float
) -> pd.DataFrame:
    df = events_analysis_processor(sellin_cust, sellout_cust)
    return detect_spikes_global(df, threshold)


@st.cache_data(show_spinner=False)
def _cached_events_chart(df: pd.DataFrame):
    return plot_customer_events(df)


def render(sellout, sellin):
    df_events = load_events_data()

    st.markdown("### High-Level Overview")
    st.caption("Spike cause distribution across all customers and all years.")
    spike_summary.render(sellout, df_events, key="spike_cause_ea1")

    st.markdown("---")
    st.markdown("### Customer-Level Analysis")
    st.caption("Drill down into a specific customer to explore individual spike dates and nearby events.")

    # sellin = get_fmc_only(sellin)
    # sellout = get_fmc_only(sellout)

    # Get Customer List
    col1, col2 = st.columns(2)
    with col1:
        customer_list = get_customer_list_events(sellout, Top_100=False)
        _cust_name_map = (
            sellout[["customer_code", "customer_name"]]
            .drop_duplicates("customer_code")
            .set_index("customer_code")["customer_name"]
            .to_dict()
        )
        selected_customer = st.selectbox(
            "Select Customer",
            customer_list,
            format_func=lambda code: f"{_cust_name_map.get(code, code)} ({code})",
            key="customer_event",
        )
        if st.session_state.get("prev_customer_event") != selected_customer:
            st.session_state["selected_date"] = None
            st.session_state["selected_event_name"] = None
            st.session_state["selected_event_lat"] = None
            st.session_state["selected_event_lon"] = None
            st.session_state["prev_event_name"] = None
            st.session_state["prev_selected_date"] = None
            st.session_state["prev_customer_event"] = selected_customer
    with col2:
        threshold = st.slider("Spike Threshold (z-score)", 0.0, 5.0, 2.0, step=0.1)
        st.caption(
            "A **spike** is a day where sell-out is abnormally high vs the same "
            "weekday's historical average. The z-score measures how many standard "
            "deviations above that weekday average the day's sales are. "
            "Yellow circles on the chart mark spike days."
        )

    sellin_cust = sellin[sellin["customer_code"] == selected_customer]
    sellout_cust = sellout[sellout["customer_code"] == selected_customer]

    # ── Date range filter (single widget, one rerun) ──────────────────────────
    min_date = sellout_cust["date"].min().date()
    max_date = sellout_cust["date"].max().date()

    date_range = st.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        key="event_date_range",
    )

    # Wait until user has picked both dates before filtering
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

    df_events_chart = _prepare_events_df(sellin_cust, sellout_cust, threshold)
    fig = _cached_events_chart(df_events_chart)

    event = st.plotly_chart(fig, on_select="rerun", width="stretch")

    if event.selection.points:
        st.session_state["selected_date"] = event.selection.points[0]["x"]

    selected_date = st.session_state.get("selected_date")

    if selected_date != st.session_state.get("prev_selected_date"):
        st.session_state["selected_event_name"] = None
        st.session_state["selected_event_lat"] = None
        st.session_state["selected_event_lon"] = None
        st.session_state["prev_event_name"] = None
        st.session_state["prev_selected_date"] = selected_date

    if selected_date:
        st.markdown("-----")

        # ── Show spike details for this date ─────────────────────────────────
        selected_date_ts = pd.to_datetime(selected_date)
        row = df_events_chart[df_events_chart["date"] == selected_date_ts]

        if not row.empty:
            r = row.iloc[0]
            weekday_name = selected_date_ts.day_name()
            is_spike = bool(r["is_spike"])

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Sell-out", f"{r['sellout']:,.0f}")
            c2.metric(
                f"Avg {weekday_name}",
                f"{r['weekday_mean']:,.0f}",
                delta=f"{(r['sellout'] - r['weekday_mean']):+,.0f}",
            )
            c3.metric(
                "z-score", f"{r['z_score']:.2f}", delta=f"threshold: {threshold:.1f}"
            )
            c4.metric("Spike?", "🔴 Yes" if is_spike else "⚪ No")

            if is_spike:
                multiplier = (
                    r["sellout"] / r["weekday_mean"] if r["weekday_mean"] else 0
                )
                st.success(
                    f"**This {weekday_name} ({selected_date_ts.date()}) is flagged as a spike.** "
                    f"Sell-out was **{r['sellout']:,.0f}**, which is "
                    f"**{r['z_score']:.1f} standard deviations** above the typical {weekday_name} "
                    f"average of **{r['weekday_mean']:,.0f}** "
                    f"({multiplier:.1f}× the weekday average)."
                )
            else:
                st.info(
                    f"This day's z-score ({r['z_score']:.2f}) is below the spike threshold ({threshold:.1f}), "
                    f"so it isn't flagged as a spike — but you can still explore nearby events."
                )

        # ── Distance slider + map ────────────────────────────────────────────
        col1, col2 = st.columns(2)

        with col1:
            distance = st.slider("Distance (m)", 0, 3000, 1000)
            from_date = selected_date_ts - pd.Timedelta(days=1)
            to_date = selected_date_ts + pd.Timedelta(days=1)

        with col2:
            st.write(f"Selected Customer: {selected_customer}")

        shop = sellout[sellout["customer_code"] == selected_customer][
            ["latitude", "longitude"]
        ].iloc[0]
        shop_lat, shop_lon = shop["latitude"], shop["longitude"]

        df_filtered_events = _cached_events_for_shop(
            df_events, shop_lat, shop_lon, from_date, to_date, distance
        )

        map_html = _cached_map_html(
            shop_lat, shop_lon, selected_customer, df_filtered_events, distance
        )
        st.components.v1.html(map_html, height=500)

    # if selected_date:
    #     st.markdown("-----")
    #     col1, col2 = st.columns(2)

    #     with col1:
    #         distance = st.slider("Distance (m)", 0, 3000, 1000)
    #         selected_date = pd.to_datetime(selected_date)
    #         from_date = selected_date - pd.Timedelta(days=1)
    #         to_date = selected_date + pd.Timedelta(days=1)

    #     with col2:
    #         st.write(f"Selected Customer: {selected_customer}")

    #     shop = sellout[sellout["customer_code"] == selected_customer][
    #         ["latitude", "longitude"]
    #     ].iloc[0]
    #     shop_lat, shop_lon = shop["latitude"], shop["longitude"]

    #     df_filtered_events = _cached_events_for_shop(
    #         df_events, shop_lat, shop_lon, from_date, to_date, distance
    #     )

    #     map_html = _cached_map_html(
    #         shop_lat, shop_lon, selected_customer, df_filtered_events, distance
    #     )
    #     st.components.v1.html(map_html, height=500)

    # if select_event.get("last_object_clicked_tooltip"):
    #     event_name = select_event["last_object_clicked_tooltip"]
    #     clicked = select_event["last_object_clicked"]
    #     if event_name != st.session_state.get("prev_event_name"):
    #         st.session_state["selected_event_name"] = event_name
    #         st.session_state["selected_event_lat"] = clicked["lat"]
    #         st.session_state["selected_event_lon"] = clicked["lng"]
    #         st.session_state["prev_event_name"] = event_name

    # selected_event_name = st.session_state.get("selected_event_name")

    # if selected_event_name:
    #     st.markdown("-----")
    #     c1, c2 = st.columns(2)

    #     with c1:
    #         dist_from_event = st.slider("Distance from Event (m)", 0, 1000, 100)
    #     with c2:
    #         st.write(f"Selected Event: {selected_event_name}")

    #     event_lat = st.session_state["selected_event_lat"]
    #     event_lon = st.session_state["selected_event_lon"]

    #     df_shops = get_shops_for_event(
    #         df_events, selected_event_name, max_distance_m=dist_from_event
    #     )

    #     m2 = plot_shops_for_event(
    #         selected_event_name,
    #         event_lat,
    #         event_lon,
    #         df_shops,
    #         max_distance_m=dist_from_event,
    #     )
    #     st_folium(m2, use_container_width=True, height=500, key="event_map")
