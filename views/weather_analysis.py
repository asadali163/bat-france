import streamlit as st
from services.data_loader import load_weather_data
from services.filters import get_customer_list_weather, get_fmc_only
from charts.weather_charts import plot_customer_weather


def render(sellout, sellin):
    df_weather = load_weather_data()

    # Get FMC Category only
    sellin = get_fmc_only(sellin)
    sellout = get_fmc_only(sellout)

    sellout = sellout.merge(
        df_weather, on=["date", "latitude", "longitude"], how="left"
    )
    rain_threshold = 4
    sellout["is_rain"] = sellout["precipitation"] > rain_threshold

    col1, col2 = st.columns(2)
    with col1:
        # Get Customer List
        customer_list = get_customer_list_weather(sellout)
        selected_customer = st.selectbox(
            "Select Customer", customer_list, key="customer_weather"
        )
    with col2:
        customer_max_rain = float(
            sellout[sellout["customer_code"] == selected_customer][
                "precipitation"
            ].max()
        )
        rain_range = st.slider(
            "Rain Range (mm)",
            min_value=0.0,
            max_value=customer_max_rain,
            value=(0.0, customer_max_rain),
            step=0.1,
        )
        robust = st.checkbox("Robust STL", value=True)

    ## Now Display the chart.
    selected_customer_df = sellout[sellout["customer_code"] == selected_customer]
    fig = plot_customer_weather(selected_customer_df, rain_range, robust)

    st.plotly_chart(fig, width="stretch")
