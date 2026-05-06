import pandas as pd
import streamlit as st
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


@st.cache_data(show_spinner=False)
def load_sell_data():
    df = pd.read_parquet(SELL_DATA)
    df = wrangle(df)
    df_sellin = df[df["data_type"] == "sell_in"].copy()
    df_sellout = df[
        (df["data_type"] == "sell_out")
        & (df["sku_code"].astype(str).str.strip() != "0")
    ].copy()
    df_sellout["category"] = df_sellout["category"].fillna("FMC")
    return df_sellin, df_sellout


@st.cache_data(show_spinner=False)
def load_weather_data() -> pd.DataFrame:
    df = pd.read_parquet(WEATHER_CSV)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    return df


@st.cache_data(show_spinner=False)
def load_events_data():
    df = pd.read_parquet(EVENTS_DATA)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    return df
