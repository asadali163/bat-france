import pandas as pd
import numpy as np
import streamlit as st
from decimal import Decimal, ROUND_HALF_UP
from statsmodels.tsa.seasonal import STL
from scipy import stats as scipy_stats
import statsmodels.formula.api as smf


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

    if not sellin_daily.empty and pd.notna(sellin_daily["date"].min()):
        sellin_dates = pd.date_range(
            sellin_daily["date"].min(), sellin_daily["date"].max(), freq="D"
        )
        sellin_daily = (
            pd.DataFrame({"date": sellin_dates})
            .merge(sellin_daily, on="date", how="left")
            .fillna(0)
        )

    if not sellout_daily.empty and pd.notna(sellout_daily["date"].min()):
        sellout_dates = pd.date_range(
            sellout_daily["date"].min(), sellout_daily["date"].max(), freq="D"
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


def rain_band_processor(
    df: pd.DataFrame, rain_mm: float = 3.0
) -> tuple:
    """
    Returns (dec_df, band_stats_df, stats_dict) for the rainfall-band chart.
    df must already have precipitation merged in (as in the weather analysis view).
    """
    df = df.copy()

    # Daily aggregation — multiple SKU rows per date collapse here
    m = (
        df.groupby("date")
        .agg(
            sales_quantity=("sales_quantity", "sum"),
            precipitation=("precipitation", "mean"),
        )
        .reset_index()
        .sort_values("date")
    )
    m["rained"] = m["precipitation"] > rain_mm

    # STL requires a gap-free series; interpolate closed-day gaps
    s = m.set_index("date")["sales_quantity"].asfreq("D")
    s = s.interpolate("time", limit_direction="both")

    stl = STL(s, period=7, robust=True).fit()

    dec = pd.DataFrame(
        {"trend": stl.trend, "seasonal": stl.seasonal, "remainder": stl.resid}
    )
    dec.index.name = "date"
    dec = dec.reset_index()
    dec = dec.merge(
        m[["date", "precipitation", "rained", "sales_quantity"]], on="date", how="inner"
    )

    # Keep only real selling days
    real_days = m.loc[m["sales_quantity"] > 0, "date"]
    dec = dec[dec["date"].isin(real_days)].copy()

    dec["band"] = pd.cut(
        dec["precipitation"],
        [-0.01, 0.1, 2, 8, 1e9],
        labels=["none", "light", "moderate", "heavy"],
    )

    # Correlation & t-test (dry vs rainy)
    corr = dec[["remainder", "precipitation"]].corr().iloc[0, 1]
    dry = dec.loc[~dec["rained"], "remainder"]
    rainy = dec.loc[dec["rained"], "remainder"]

    stats_dict: dict = {"corr": corr, "p_ttest": float("nan"), "F_anova": float("nan"), "p_anova": float("nan")}
    if len(dry) > 1 and len(rainy) > 1:
        _, p_ttest = scipy_stats.ttest_ind(dry, rainy, equal_var=False)
        stats_dict["p_ttest"] = p_ttest

    groups = [g["remainder"].values for _, g in dec.groupby("band", observed=True) if len(g) > 1]
    if len(groups) >= 2:
        F, p_anova = scipy_stats.f_oneway(*groups)
        stats_dict.update({"F_anova": F, "p_anova": p_anova})

    band_stats = (
        dec.groupby("band", observed=True)["remainder"]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    band_stats["sem"] = band_stats["std"] / band_stats["count"] ** 0.5

    return dec, band_stats, stats_dict


_RAIN_BINS = [-0.01, 0.1, 2, 8, 1e9]
_RAIN_LABS = ["none", "light", "moderate", "heavy"]


def ols_rain_processor(df: pd.DataFrame) -> tuple:
    """
    Fits OLS: log(sales) ~ rain band (same-day, lag-1, lead-1) + temperature + windspeed + DOW + month + trend.
    Returns (coef_df, scalar_df, meta_dict).
    coef_df  — % change vs dry day per band × effect, with 95 % CI and p-value.
    scalar_df — temperature & windspeed % change per unit, with CI and p-value.
    meta_dict — r2, n, or error key if fitting failed.
    """
    df = df.copy()

    m = (
        df.groupby("date")
        .agg(
            sales_quantity=("sales_quantity", "sum"),
            precipitation=("precipitation", "mean"),
            temperature=("temperature", "mean"),
            windspeed=("windspeed", "mean"),
        )
        .reset_index()
        .sort_values("date")
        .reset_index(drop=True)
    )

    m = m.dropna(subset=["precipitation", "temperature", "windspeed"])
    m = m[m["sales_quantity"] > 0].copy()

    if len(m) < 30:
        return None, None, {"error": "Not enough data (< 30 selling days)"}

    m["log_q"] = np.log(m["sales_quantity"])
    m["dow"] = m["date"].dt.dayofweek.astype("category")
    m["month"] = m["date"].dt.month.astype("category")
    m["trend"] = (m["date"] - m["date"].min()).dt.days
    m["band"] = pd.Categorical(
        pd.cut(m["precipitation"], _RAIN_BINS, labels=_RAIN_LABS), categories=_RAIN_LABS
    )
    m["band_lag1"] = m["band"].shift(1)
    m["band_lead1"] = m["band"].shift(-1)
    m["lag_ok"] = m["date"].diff().dt.days == 1
    m["lead_ok"] = (-m["date"].diff(-1).dt.days) == 1

    model_df = m.dropna(subset=["band_lag1", "band_lead1"])
    model_df = model_df[model_df["lag_ok"] & model_df["lead_ok"]]

    if len(model_df) < 20:
        return None, None, {"error": "Not enough consecutive days for lag/lead model"}

    formula = (
        "log_q ~ C(band) + C(band_lag1) + C(band_lead1)"
        " + temperature + windspeed"
        " + C(dow) + C(month) + trend"
    )

    try:
        model = smf.ols(formula, data=model_df).fit(cov_type="HC3")
    except Exception as exc:
        return None, None, {"error": str(exc)}

    ci = model.conf_int()

    coef_rows = []
    for effect_label, prefix in [
        ("Same-day", "C(band)["),
        ("Yesterday", "C(band_lag1)["),
        ("Tomorrow", "C(band_lead1)["),
    ]:
        for term in model.params.index:
            if term.startswith(prefix) and "T." in term:
                band = term.split("T.")[-1].rstrip("]")
                coef = model.params[term]
                lo, hi = ci.loc[term]
                coef_rows.append(
                    {
                        "effect": effect_label,
                        "band": band,
                        "pct_change": np.expm1(coef) * 100,
                        "ci_low": np.expm1(lo) * 100,
                        "ci_high": np.expm1(hi) * 100,
                        "p": model.pvalues[term],
                    }
                )

    scalar_rows = []
    for var in ["temperature", "windspeed"]:
        if var in model.params:
            coef = model.params[var]
            lo, hi = ci.loc[var]
            scalar_rows.append(
                {
                    "variable": var,
                    "pct_change": np.expm1(coef) * 100,
                    "ci_low": np.expm1(lo) * 100,
                    "ci_high": np.expm1(hi) * 100,
                    "p": model.pvalues[var],
                }
            )

    return (
        pd.DataFrame(coef_rows),
        pd.DataFrame(scalar_rows),
        {"r2": model.rsquared, "n": len(model_df)},
    )
