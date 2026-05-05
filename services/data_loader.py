import os
import gdown
import streamlit as st
import pandas as pd
from config import GDRIVE_SELL_DATA, GDRIVE_WEATHER_CSV, GDRIVE_EVENTS_DATA, DATA_CACHE_DIR


def _is_valid_csv(path: str) -> bool:
    """Return False if the file looks like an HTML page (failed gdown download)."""
    try:
        with open(path, "rb") as f:
            header = f.read(512).lstrip()
        return not header.startswith(b"<")
    except OSError:
        return False


def _download(file_id: str, filename: str) -> str:
    """Download a file from Google Drive if not already cached. Returns local path."""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    local_path = os.path.join(DATA_CACHE_DIR, filename)

    # Remove corrupted/HTML files from a previous bad download
    if os.path.exists(local_path) and not _is_valid_csv(local_path):
        os.remove(local_path)

    if not os.path.exists(local_path):
        with st.spinner(f"Downloading {filename} from Google Drive…"):
            # fuzzy=True handles the virus-scan confirmation page for large files (>100 MB)
            gdown.download(id=file_id, output=local_path, quiet=False)

        if not _is_valid_csv(local_path):
            os.remove(local_path)
            st.error(
                f"❌ Download of **{filename}** failed (got an HTML page instead of CSV). "
                "Make sure the Google Drive file is shared as 'Anyone with the link'."
            )
            st.stop()

    return local_path


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
    path = _download(GDRIVE_SELL_DATA, "combined_df.csv")
    df = pd.read_csv(path, low_memory=False)
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
    path = _download(GDRIVE_WEATHER_CSV, "weather_till_march2.csv")
    df = pd.read_csv(path, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    return df


@st.cache_data(show_spinner=False)
def load_events_data():
    path = _download(GDRIVE_EVENTS_DATA, "events_batch2.csv")
    df = pd.read_csv(path, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    return df
