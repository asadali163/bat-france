import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL


def get_fmc_only(df):
    return df[df["category"] == "FMC"].copy()


@st.cache_data
def get_customer_list_weather(df: pd.DataFrame, Top_100: bool = True) -> list:
    """
    df: A sellout Dataframe and list the customers on basis of the highest impact will sell data.
    Top_100: If true, will return top 100 customers by sales_quantity.
    """
    if Top_100:
        customer_list = (
            df.groupby("customer_code")["sales_quantity"]
            .sum()
            .sort_values(ascending=False)
            .head(100)
            .index.tolist()
        )
    else:
        customer_list = df["customer_code"].unique().tolist()

    # Now run the STL

    results = []
    for customer in customer_list:
        df_selected = df[df["customer_code"] == customer]
        df_daily = (
            df_selected.groupby("date")
            .agg(
                {
                    "sales_quantity": "sum",
                    "precipitation": "mean",
                    "is_rain": "max",
                }
            )
            .reset_index()
            .sort_values("date")
        )

        df_daily = df_daily.fillna(0)

        if len(df_daily) < 30:
            continue

        stl = STL(
            df_daily["sales_quantity"], period=7, robust=True
        )  # weekly seasonality
        result = stl.fit()

        df_daily["trend"] = result.trend
        df_daily["seasonal"] = result.seasonal
        df_daily["residual"] = result.resid

        cor = df_daily[["residual", "precipitation"]].corr()
        rain_mean = df_daily.groupby("is_rain")["residual"].mean()

        result = {
            "customer_code": customer,
            "correlation": cor["residual"]["precipitation"],
            "rain_mean": rain_mean.get(True, 0),
            "no_rain_mean": rain_mean.get(False, 0),
            "rain_days": sum(df_daily["is_rain"]),
            "no_rain_days": len(df_daily) - sum(df_daily["is_rain"]),
        }

        results.append(result)

    if not results:
        return []

    df_results = pd.DataFrame(results)
    df_results["abs_correlation"] = df_results["correlation"].abs()
    df_results = df_results.sort_values("abs_correlation", ascending=False)

    return df_results["customer_code"].tolist()


st.cache_data


def get_customer_list_events(df: pd.DataFrame, Top_100: bool = True) -> list:
    """
    Get customer list for events by sales quantity
    """
    if Top_100:
        customer_list = (
            df.groupby("customer_code")["sales_quantity"]
            .sum()
            .sort_values(ascending=False)
            .head(100)
            .index.tolist()
        )
    else:
        customer_list = (
            df.groupby("customer_code")["sales_quantity"]
            .sum()
            .sort_values(ascending=False)
            .index.tolist()
        )
    return customer_list
