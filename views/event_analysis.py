import streamlit as st
import pandas as pd
from streamlit_folium import st_folium
from services.filters import get_fmc_only, get_customer_list_events
from charts.event_charts import plot_customer_events
from charts.event_map import plot_event_map, plot_shops_for_event
from services.geo import compute_customer_distances
from services.data_loader import load_events_data
from services.porcessors import get_events_for_shop, get_shops_for_event


def render(sellout, sellin):
    df_events = load_events_data()

    sellin = get_fmc_only(sellin)
    sellout = get_fmc_only(sellout)

    # df_distances = compute_customer_distances(sellout)
    all_cust_with_lat_long = compute_customer_distances(sellout)

    # Get Customer List
    col1, col2 = st.columns(2)
    with col1:
        customer_list = get_customer_list_events(sellout)
        selected_customer = st.selectbox(
            "Select Customer", customer_list, key="customer_event"
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

    fig = plot_customer_events(sellin, sellout, selected_customer, threshold)

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
        col1, col2 = st.columns(2)

        with col1:
            distance = st.slider("Distance (m)", 0, 3000, 100)
            selected_date = pd.to_datetime(selected_date)
            from_date = selected_date - pd.Timedelta(days=19)
            to_date = selected_date + pd.Timedelta(days=19)

            print("Selected Date is: ", selected_date)
            # print("Distance: ", distance, "From: ", from_date, "To: ", to_date)

        with col2:
            st.write(f"Selected Customer: {selected_customer}")

        shop = sellout[sellout["customer_code"] == selected_customer][
            ["latitude", "longitude"]
        ].iloc[0]
        shop_lat, shop_lon = shop["latitude"], shop["longitude"]

        df_filtered_events = get_events_for_shop(
            df_events, shop_lat, shop_lon, from_date, to_date, distance
        )

        m = plot_event_map(
            shop_lat, shop_lon, selected_customer, df_filtered_events, distance
        )
        select_event = st_folium(
            m, use_container_width=True, height=500, key="shop_map"
        )

        if select_event.get("last_object_clicked_tooltip"):
            event_name = select_event["last_object_clicked_tooltip"]
            clicked = select_event["last_object_clicked"]
            if event_name != st.session_state.get("prev_event_name"):
                st.session_state["selected_event_name"] = event_name
                st.session_state["selected_event_lat"] = clicked["lat"]
                st.session_state["selected_event_lon"] = clicked["lng"]
                st.session_state["prev_event_name"] = event_name

        selected_event_name = st.session_state.get("selected_event_name")

        if selected_event_name:
            st.markdown("-----")
            c1, c2 = st.columns(2)

            with c1:
                dist_from_event = st.slider("Distance from Event (m)", 0, 1000, 100)
            with c2:
                st.write(f"Selected Event: {selected_event_name}")

            event_lat = st.session_state["selected_event_lat"]
            event_lon = st.session_state["selected_event_lon"]

            df_shops = get_shops_for_event(
                df_events, selected_event_name, max_distance_m=dist_from_event
            )

            m2 = plot_shops_for_event(
                selected_event_name,
                event_lat,
                event_lon,
                df_shops,
                max_distance_m=dist_from_event,
            )
            st_folium(m2, use_container_width=True, height=500, key="event_map")
