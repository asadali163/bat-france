import pandas as pd


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


def load_sell_data():
    try:
        df = pd.read_csv("http://192.168.99.122:8000/combined_df.csv", low_memory=False)
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
        print(
            "Could not find `combined_df.csv`. Make sure it is in the same folder as `app.py`."
        )
        return pd.DataFrame()


def get_only_fmc(df):
    return df[df["category"] == "FMC"].copy()


def get_selected_customer_data(sellin, sellout, selected_customer, selected_sku=None):
    if selected_sku != None:
        df_sellin_selected = sellin[sellin["customer_code"] == selected_customer].copy()
        df_sellout_selected = sellout[
            sellout["customer_code"] == selected_customer
        ].copy()
    else:
        df_sellin_selected = sellin[sellin["customer_code"] == selected_customer].copy()
        df_sellout_selected = sellout[
            (sellout["customer_code"] == selected_customer)
            & (sellout["sku_code"] == selected_sku)
        ].copy()

    df_sellin_selected = (
        df_sellin_selected.groupby("date")["sales_quantity"].sum().reset_index()
    )
    df_sellout_selected = (
        df_sellout_selected.groupby("date")["sales_quantity"].sum().reset_index()
    )
    df_sellin_min_date = df_sellin_selected["date"].min()
    df_sellin_max_date = df_sellin_selected["date"].max()

    df_sellin_selected = (
        pd.DataFrame(
            {"date": pd.date_range(df_sellin_min_date, df_sellin_max_date, freq="D")}
        )
        .merge(df_sellin_selected, on="date", how="left")
        .fillna(0)
    )

    df_sellout_min_date = df_sellout_selected["date"].min()
    df_sellout_max_date = df_sellout_selected["date"].max()

    df_sellout_selected = (
        pd.DataFrame(
            {"date": pd.date_range(df_sellout_min_date, df_sellout_max_date, freq="D")}
        )
        .merge(df_sellout_selected, on="date", how="left")
        .fillna(0)
    )

    return df_sellin_selected, df_sellout_selected
