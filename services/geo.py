import math
import pandas as pd
import streamlit as st


def _haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.asin(math.sqrt(a)) * 1000


@st.cache_data
def compute_customer_distances(df: pd.DataFrame):
    customers = (
        df.groupby("customer_code")[["latitude", "longitude"]]
        .first()
        .reset_index()
        .dropna(subset=["latitude", "longitude"])
    )

    rows = []
    customers = [
        {
            "customer_code": row["customer_code"],
            "latitude": row["latitude"],
            "longitude": row["longitude"],
        }
        for _, row in customers.iterrows()
    ]
    return customers
    # for _, origin in customers.iterrows():
    #     row = {}
    #     for _, target in customers.iterrows():
    #         if origin["customer_code"] != target["customer_code"]:
    #             row[target["customer_code"]] = round(
    #                 _haversine(
    #                     origin["latitude"], origin["longitude"],
    #                     target["latitude"], target["longitude"],
    #                 )
    #             )
    #     rows.append(row)

    # return pd.DataFrame(rows, index=customers["customer_code"])
