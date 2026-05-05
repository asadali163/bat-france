import streamlit as st
import pandas as pd
from config import SELL_DATA, WEATHER_CSV, EVENTS_DATA


def wrangle(df: pd.DataFrame) -> pd.DataFrame:
    df.rename(
        columns={
            "data_type": "data_type",
            "source": "provider",
            "Sales Date": "date",
            "Outlet SF ID": "customer_code",
            "Store Participant Code": "customer_name",
            "SKU SF ID": "sku_code",
            "SKU Name": "sku_name",
            "Brand Variant": "brand_variant",
            "Brand Family": "brand_name",
            "Category": "category",
            "Volume in Unit": "sales_amount",
            "Volume in Packs": "sales_quantity",
            "Ownership Type": "channel_name",
            "Latitude": "latitude",
            "Longitude": "longitude",
            "Territory Id": "route",
            # Extra
            "Brand": "brand",
            "SKU Clean": "sku_clean",
            "Month": "month",
        },
        inplace=True,
    )

    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")

    return df


@st.cache_data  # Load sell data locally.
def load_sell_data():
    try:
        df = pd.read_csv(SELL_DATA, low_memory=False)
        df = wrangle(df)
        df_sellin = df[df["data_type"] == "sell_in"].copy()
        df_sellout = df[
            (df["data_type"] == "sell_out")
            & (df["sku_code"].astype(str).str.strip() != "0")
        ].copy()
        # Replace missing category with FMC
        df_sellout["category"] = df_sellout["category"].fillna("FMC")
        return df_sellin, df_sellout
    except FileNotFoundError:
        st.error(
            f"❌ Could not find `{SELL_DATA}`. Make sure it is in the same folder as `app.py`."
        )
        return pd.DataFrame()


@st.cache_data
def load_weather_data() -> pd.DataFrame:
    try:
        df = pd.read_csv(WEATHER_CSV, low_memory=False)
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
        return df
    except FileNotFoundError:
        st.error(
            f"❌ Could not find `{WEATHER_CSV}`. Make sure it is in the same folder as `app.py`."
        )
        return pd.DataFrame()


@st.cache_data
def load_events_data():
    try:
        df = pd.read_csv(EVENTS_DATA, low_memory=False)
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
        return df
    except FileNotFoundError:
        st.error(
            f"❌ Could not find `{EVENTS_DATA}`. Make sure it is in the same folder as `app.py`."
        )
        return pd.DataFrame()
