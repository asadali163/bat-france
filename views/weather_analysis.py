import streamlit as st
import pandas as pd
from services.data_loader import load_weather_data
from services.filters import get_customer_list_weather, get_fmc_only
from charts.weather_charts import (
    plot_customer_weather,
    plot_rain_band_chart,
    plot_ols_rain_chart,
)
from services.porcessors import rain_band_processor, ols_rain_processor


@st.cache_data(show_spinner=False)
def _merge_weather(sellout_fmc: pd.DataFrame, df_weather: pd.DataFrame) -> pd.DataFrame:
    merged = sellout_fmc.merge(df_weather, on=["date", "latitude", "longitude"], how="left")
    merged["is_rain"] = merged["precipitation"] > 4
    return merged


@st.cache_data(show_spinner=False)
def _cached_weather_chart(customer_df: pd.DataFrame, rain_range: tuple, robust: bool):
    return plot_customer_weather(customer_df, rain_range, robust)


@st.cache_data(show_spinner=False)
def _cached_rain_band(customer_df: pd.DataFrame):
    return rain_band_processor(customer_df)


@st.cache_data(show_spinner=False)
def _cached_ols(customer_df: pd.DataFrame):
    return ols_rain_processor(customer_df)


def render(sellout, sellin):
    df_weather = load_weather_data()

    sellin = get_fmc_only(sellin)
    sellout = get_fmc_only(sellout)

    # Merge once and cache — not every rerun
    sellout_merged = _merge_weather(sellout, df_weather)

    col1, col2 = st.columns(2)
    with col1:
        customer_list = get_customer_list_weather(sellout_merged)
        selected_customer = st.selectbox(
            "Select Customer", customer_list, key="customer_weather"
        )

    selected_customer_df = sellout_merged[
        sellout_merged["customer_code"] == selected_customer
    ]

    with col2:
        customer_max_rain = float(selected_customer_df["precipitation"].max())
        rain_range = st.slider(
            "Rain Range (mm)",
            min_value=0.0,
            max_value=customer_max_rain,
            value=(0.0, customer_max_rain),
            step=0.1,
        )
        robust = st.checkbox("Robust STL", value=True)

    fig = _cached_weather_chart(selected_customer_df, rain_range, robust)
    st.plotly_chart(fig, width="stretch")

    st.markdown("---")
    dec, band_stats, stats_dict = _cached_rain_band(selected_customer_df)
    fig2 = plot_rain_band_chart(dec, band_stats, stats_dict)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    col1, _ = st.columns(2)
    with col1:
        coef_df, scalar_df, meta = _cached_ols(selected_customer_df)
        fig3 = plot_ols_rain_chart(coef_df, scalar_df, meta)
        st.plotly_chart(fig3, use_container_width=True)
