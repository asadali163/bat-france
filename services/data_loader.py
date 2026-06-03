import glob
import pandas as pd
import streamlit as st
from config import SELL_DATA, WEATHER_CSV, EVENTS_DATA

HOURLY_WEATHER_DIR = "./data/raw/hourly_weather"


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
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y-%m-%d")
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
    df_sellout = df_sellout[df_sellout["customer_code"] != 0].copy()
    # df_sellout["category"] = df_sellout["category"].fillna("FMC")
    # df_sellin = df_sellin[df_sellin["category"] == "FMC"].copy()
    # df_sellout = df_sellout[df_sellout["category"] == "FMC"].copy()
    return df_sellin, df_sellout


@st.cache_data(show_spinner=False)
def load_weather_data() -> pd.DataFrame:
    df = pd.read_parquet(WEATHER_CSV)
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y-%m-%d")
    return df


@st.cache_data(show_spinner=False)
def load_hourly_weather_data() -> pd.DataFrame:
    files = glob.glob(f"{HOURLY_WEATHER_DIR}/*.parquet")
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["date"] = df["time"].dt.date
    df["latitude"] = df["latitude"].round(4)
    df["longitude"] = df["longitude"].round(4)
    return df


@st.cache_data(show_spinner=False)
def load_events_data():
    df = pd.read_parquet(EVENTS_DATA)
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y-%m-%d")
    # float32 → float64 so .round(4) comparisons with sellout lat/lon (float64) work correctly
    df["shop_lat"] = df["shop_lat"].astype("float64")
    df["shop_lon"] = df["shop_lon"].astype("float64")
    return df
