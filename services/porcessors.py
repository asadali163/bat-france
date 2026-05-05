import pandas as pd
import numpy as np
import streamlit as st
from decimal import Decimal, ROUND_HALF_UP
from statsmodels.tsa.seasonal import STL


def weather_analysis_processor(df: pd.DataFrame, robust=True) -> pd.DataFrame:
    df = df.copy()

    df = (
        df.groupby("date")
        .agg(
            sales_quantity=("sales_quantity", "sum"),
            precipitation=("precipitation", "mean"),
            temperature=("temperature", "mean"),
            is_rain=("is_rain", "max"),
        )
        .reset_index()
        .sort_values("date")
    )

    stl = STL(df["sales_quantity"], period=7, robust=robust)
    result = stl.fit()
    df["trend"] = result.trend
    df["seasonal"] = result.seasonal
    df["residual"] = result.resid

    return df


def events_analysis_processor(
    sellin: pd.DataFrame, sellout: pd.DataFrame
) -> pd.DataFrame:
    sellin_daily = (
        sellin.groupby("date")["sales_quantity"]
        .sum()
        .reset_index()
        .rename(columns={"sales_quantity": "sellin"})
    )
    sellout_daily = (
        sellout.groupby("date")["sales_quantity"]
        .sum()
        .reset_index()
        .rename(columns={"sales_quantity": "sellout"})
    )

    sellin_dates = pd.date_range(
        sellin_daily["date"].min(), sellin_daily["date"].max(), freq="D"
    )
    sellout_dates = pd.date_range(
        sellout_daily["date"].min(), sellout_daily["date"].max(), freq="D"
    )

    sellin_daily = (
        pd.DataFrame({"date": sellin_dates})
        .merge(sellin_daily, on="date", how="left")
        .fillna(0)
    )
    sellout_daily = (
        pd.DataFrame({"date": sellout_dates})
        .merge(sellout_daily, on="date", how="left")
        .fillna(0)
    )

    df = (
        sellin_daily.merge(sellout_daily, on="date", how="outer")
        .fillna(0)
        .sort_values("date")
    )

    # Stock starts accumulating from first non-zero sellin date
    first_sellin_date = df.loc[df["sellin"] > 0, "date"].min()
    mask = df["date"] >= first_sellin_date
    df["stock_remaining"] = np.nan
    df.loc[mask, "stock_remaining"] = (
        df.loc[mask, "sellin"] - df.loc[mask, "sellout"]
    ).cumsum()

    return df


def detect_spikes_global(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    df = df.copy()
    df["weekday"] = pd.to_datetime(df["date"]).dt.dayofweek

    weekday_stats = df.groupby("weekday")["sellout"].agg(["mean", "std"])
    df["weekday_mean"] = df["weekday"].map(weekday_stats["mean"])
    df["weekday_std"] = df["weekday"].map(weekday_stats["std"])

    df["weekday_std"] = df["weekday_std"].replace(0, np.nan)
    df["z_score"] = (df["sellout"] - df["weekday_mean"]) / df["weekday_std"]
    df["is_spike"] = df["z_score"].fillna(0) > threshold

    return df


def detect_spikes_robust(
    df: pd.DataFrame, threshold: float = 2.5, window: int = 28, use_robust: bool = True
) -> pd.DataFrame:
    df = df.copy()

    # Ensure datetime and sort
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # Extract weekday
    df["weekday"] = df["date"].dt.dayofweek

    # Function for rolling stats per weekday
    def compute_stats(group):
        group = group.sort_values("date")

        if use_robust:
            # Median & MAD (robust to outliers)
            rolling_median = group["sellout"].rolling(window, min_periods=5).median()
            mad = (
                group["sellout"]
                .rolling(window, min_periods=5)
                .apply(lambda x: np.median(np.abs(x - np.median(x))), raw=True)
            )
            # Convert MAD to std equivalent
            rolling_std = 1.4826 * mad
            return pd.DataFrame({"center": rolling_median, "spread": rolling_std})
        else:
            # Mean & Std
            rolling_mean = group["sellout"].rolling(window, min_periods=5).mean()
            rolling_std = group["sellout"].rolling(window, min_periods=5).std()
            return pd.DataFrame({"center": rolling_mean, "spread": rolling_std})

    # Apply per weekday
    stats = df.groupby("weekday", group_keys=False).apply(compute_stats)

    df["center"] = stats["center"]
    df["spread"] = stats["spread"]

    # Avoid division issues
    df["spread"] = df["spread"].replace(0, np.nan)

    # Z-score (robust or standard)
    df["z_score"] = (df["sellout"] - df["center"]) / df["spread"]

    # Detect both spikes and drops
    df["is_spike"] = df["z_score"].abs() > threshold

    return df


def get_events_for_shop(
    df_events: pd.DataFrame,
    shop_lat: float,
    shop_lon: float,
    from_date,
    to_date,
    max_distance_m: int,
) -> pd.DataFrame:

    print("#### Shop Lat and Long: ", shop_lat, shop_lon)

    def _round_half_up(value):
        return float(
            Decimal(str(value)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
        )

    shop_lat = _round_half_up(shop_lat)
    shop_lon = _round_half_up(shop_lon)
    print("#### Shop Lat and Long: ", shop_lat, shop_lon)
    df = df_events[
        (df_events["shop_lat"].round(4) == shop_lat)
        & (df_events["shop_lon"].round(4) == shop_lon)
        & (df_events["date"] >= from_date)
        & (df_events["date"] <= to_date)
        & (df_events["distance_m"] <= max_distance_m)
    ].copy()

    print("Shop lat and long from csv is: ", df[["shop_lat", "shop_lon"]].head(1))
    return df


def get_shops_for_event(
    df_events: pd.DataFrame,
    event_name: str,
    max_distance_m: int,
) -> pd.DataFrame:
    # print("Event name is : ", event_name)
    return df_events[
        (df_events["name"] == event_name) & (df_events["distance_m"] <= max_distance_m)
    ].copy()
