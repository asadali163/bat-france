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


def detect_spikes_global(
    df: pd.DataFrame, threshold: float, reference_df: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Detect spikes using weekday z-score.
    reference_df: if provided, weekday mean/std are computed from this (full history)
                  rather than from df (filtered view). Keeps baseline stable regardless
                  of the date range selected in the UI.
    """
    df = df.copy()
    df["weekday"] = pd.to_datetime(df["date"]).dt.dayofweek

    ref = reference_df if reference_df is not None else df
    if "weekday" not in ref.columns:
        ref = ref.copy()
        ref["weekday"] = pd.to_datetime(ref["date"]).dt.dayofweek

    weekday_stats = ref.groupby("weekday")["sellout"].agg(["mean", "std"])
    df["weekday_mean"] = df["weekday"].map(weekday_stats["mean"])
    df["weekday_std"]  = df["weekday"].map(weekday_stats["std"])

    df["weekday_std"] = df["weekday_std"].replace(0, np.nan)
    df["z_score"]     = (df["sellout"] - df["weekday_mean"]) / df["weekday_std"]
    df["is_spike"]    = df["z_score"].fillna(0) > threshold

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


SHOP_CATCHMENT_M = 500


def _capacity_to_radius(cap) -> int:
    """Derive event impact radius (metres) from estimated audience capacity."""
    try:
        cap = float(cap)
    except (TypeError, ValueError):
        return 200
    if cap != cap:  # NaN
        return 200
    if cap < 500:
        return 200
    if cap < 2000:
        return 400
    if cap < 10_000:
        return 800
    return 1500  # large / district-scale events


def get_spike_events(
    df_events: pd.DataFrame,
    shop_lat: float,
    shop_lon: float,
    from_date,
    to_date,
) -> pd.DataFrame:
    """
    Return events near a shop for the given date window, tagged with:
      impact_radius_m — derived from estimated_capacity
      event_type      — "district" | "in_range" | "too_far"
    """
    df = df_events[
        (df_events["shop_lat"].round(4) == shop_lat)
        & (df_events["shop_lon"].round(4) == shop_lon)
        & (df_events["date"] >= from_date)
        & (df_events["date"] <= to_date)
    ].copy()

    df["impact_radius_m"] = df["estimated_capacity"].apply(_capacity_to_radius)

    def _classify(row):
        try:
            cap = float(row["estimated_capacity"])
        except (TypeError, ValueError):
            cap = float("nan")
        if cap == cap and cap >= 10_000:  # not NaN and >= 10k
            return "district"
        if row["distance_m"] <= SHOP_CATCHMENT_M + row["impact_radius_m"]:
            return "in_range"
        return "too_far"

    df["event_type"] = df.apply(_classify, axis=1)
    return df


def _classify_events(df: pd.DataFrame) -> pd.DataFrame:
    """Add impact_radius_m and event_type to an already-filtered events df."""
    if df.empty:
        df["impact_radius_m"] = pd.Series(dtype=int)
        df["event_type"] = pd.Series(dtype=str)
        return df

    df["impact_radius_m"] = df["estimated_capacity"].apply(_capacity_to_radius)

    def _classify(row):
        try:
            cap = float(row["estimated_capacity"])
        except (TypeError, ValueError):
            cap = float("nan")
        if cap == cap and cap >= 10_000:
            return "district"
        if row["distance_m"] <= SHOP_CATCHMENT_M + row["impact_radius_m"]:
            return "in_range"
        return "too_far"

    df["event_type"] = df.apply(_classify, axis=1)
    return df


def get_all_shop_events(
    df_events: pd.DataFrame,
    shop_lat: float,
    shop_lon: float,
    from_date,
    to_date,
) -> pd.DataFrame:
    """
    Single scan of df_events for this shop across the full date range.
    Returns a classified DataFrame; per-spike slicing is done in memory.
    """
    df = df_events[
        (df_events["shop_lat"].round(4) == shop_lat)
        & (df_events["shop_lon"].round(4) == shop_lon)
        & (df_events["date"] >= from_date)
        & (df_events["date"] <= to_date)
    ].copy()
    return _classify_events(df)


def compute_spike_cause_distribution(
    sellout: pd.DataFrame,
    df_events: pd.DataFrame,
    threshold: float = 2.0,
) -> pd.DataFrame:
    """
    Vectorized spike cause distribution across all customers and all years.
    Returns a DataFrame with columns [spike_cause, count].
    """
    from decimal import Decimal, ROUND_HALF_UP

    def _round4(v):
        return float(Decimal(str(v)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))

    # Daily sell-out per customer
    df = sellout.copy()
    df["date"] = pd.to_datetime(df["date"])
    daily = (
        df.groupby(["customer_code", "date"], as_index=False)["sales_quantity"]
        .sum()
        .rename(columns={"sales_quantity": "sellout"})
    )
    daily["weekday"] = daily["date"].dt.dayofweek

    # Per-customer weekday stats
    wstats = (
        daily.groupby(["customer_code", "weekday"])["sellout"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "weekday_mean", "std": "weekday_std"})
    )
    daily = daily.merge(wstats, on=["customer_code", "weekday"], how="left")
    daily["weekday_std"] = daily["weekday_std"].replace(0, np.nan)
    daily["z_score"] = (daily["sellout"] - daily["weekday_mean"]) / daily["weekday_std"]
    daily["is_spike"] = daily["z_score"].fillna(0) > threshold

    # Add route to daily for route-date summary
    route_meta = (
        df[["customer_code", "route"]]
        .drop_duplicates("customer_code")
        .dropna(subset=["route"])
    )
    daily = daily.merge(route_meta, on="customer_code", how="left")

    # Route-date summary: total & spiking shops per (date, route)
    route_date_summary = (
        daily.dropna(subset=["route"])
        .groupby(["date", "route"])
        .agg(
            total_shops_in_route=("customer_code", "nunique"),
            spiking_shops_in_route=("is_spike", "sum"),
        )
        .reset_index()
    )
    route_date_summary["spiking_shops_in_route"] = route_date_summary["spiking_shops_in_route"].astype(int)
    route_date_summary["pct_route_spiking"] = (
        route_date_summary["spiking_shops_in_route"]
        / route_date_summary["total_shops_in_route"] * 100
    ).round(2)

    spike_df = daily[daily["is_spike"]].copy()
    if spike_df.empty:
        return pd.DataFrame({"spike_cause": [], "count": []}), spike_df

    # Attach shop coordinates
    coords = (
        sellout[["customer_code", "latitude", "longitude"]]
        .drop_duplicates("customer_code")
        .assign(
            lat_r=lambda x: x["latitude"].apply(_round4),
            lon_r=lambda x: x["longitude"].apply(_round4),
        )
    )
    spike_df = spike_df.merge(coords, on="customer_code", how="left")

    # Build event lookup: (lat_r, lon_r) -> set of event dates
    # Use ALL events associated with a shop (no distance/type filtering) to match
    # the notebook's enriched-events approach for the high-level distribution.
    df_ev = df_events.copy()
    df_ev["date"] = pd.to_datetime(df_ev["date"]).dt.normalize()
    df_ev["lat_r"] = df_ev["shop_lat"].apply(_round4)
    df_ev["lon_r"] = df_ev["shop_lon"].apply(_round4)

    event_lookup: dict = {}
    for (slat, slon), grp in df_ev.groupby(["lat_r", "lon_r"]):
        event_lookup[(slat, slon)] = set(grp["date"])

    # Flag each spike
    spike_df["date_norm"] = spike_df["date"].dt.normalize()

    def _add_flags(group):
        key = (group["lat_r"].iloc[0], group["lon_r"].iloc[0])
        event_dates = event_lookup.get(key, set())
        dates = group["date_norm"]
        g = group.copy()
        g["event_same_day"] = dates.isin(event_dates)
        g["event_day_before"] = (dates - pd.Timedelta(days=1)).isin(event_dates)
        g["event_day_after"] = (dates + pd.Timedelta(days=1)).isin(event_dates)
        return g

    spike_df = spike_df.groupby(["lat_r", "lon_r"], group_keys=False).apply(_add_flags)

    # Priority assignment: same_day > day_before > day_after > no_event
    spike_df["spike_cause"] = "no_event"
    spike_df.loc[spike_df["event_day_after"], "spike_cause"] = "event_day_after"
    spike_df.loc[spike_df["event_day_before"], "spike_cause"] = "event_day_before"
    spike_df.loc[spike_df["event_same_day"], "spike_cause"] = "event_same_day"

    # Join route-spike stats onto spike_df (route already present via daily)
    spike_df = spike_df.merge(
        route_date_summary[["date", "route", "total_shops_in_route",
                            "spiking_shops_in_route", "pct_route_spiking"]],
        on=["date", "route"], how="left",
    )
    spike_df["is_solo_in_route"] = spike_df["total_shops_in_route"] == 1

    cause_counts = spike_df["spike_cause"].value_counts().reset_index()
    cause_counts.columns = ["spike_cause", "count"]
    return cause_counts, spike_df


def compute_route_spike_bands(no_event_df, route=None):
    """
    Band non-summer no-event spikes by % of route shops also spiking.
    If route is given, filter to that route only.
    """
    BINS   = list(range(0, 101, 10))
    LABELS = [f"{i}-{i+10}%" for i in range(0, 100, 10)]

    df = no_event_df.copy()
    if route is not None:
        df = df[df["route"] == route]

    if df.empty or df["pct_route_spiking"].isna().all():
        return pd.DataFrame({"band": LABELS, "count": [0] * 10, "pct_of_group": [0.0] * 10})

    df["pct_band"] = pd.cut(
        df["pct_route_spiking"].clip(0, 100),
        bins=BINS, labels=LABELS, right=True, include_lowest=True,
    )
    tbl = (
        df["pct_band"].value_counts()
        .reindex(LABELS, fill_value=0)
        .reset_index()
    )
    tbl.columns = ["band", "count"]
    total = tbl["count"].sum()
    tbl["pct_of_group"] = (tbl["count"] / total * 100).round(1) if total else 0.0
    return tbl


def get_shops_for_event(
    df_events: pd.DataFrame,
    event_name: str,
    max_distance_m: int,
) -> pd.DataFrame:
    # print("Event name is : ", event_name)
    return df_events[
        (df_events["name"] == event_name) & (df_events["distance_m"] <= max_distance_m)
    ].copy()


def rain_band_processor(df: pd.DataFrame, rain_mm: float = 3.0) -> tuple:
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

    stats_dict: dict = {
        "corr": corr,
        "p_ttest": float("nan"),
        "F_anova": float("nan"),
        "p_anova": float("nan"),
    }
    if len(dry) > 1 and len(rainy) > 1:
        _, p_ttest = scipy_stats.ttest_ind(dry, rainy, equal_var=False)
        stats_dict["p_ttest"] = p_ttest

    groups = [
        g["remainder"].values
        for _, g in dec.groupby("band", observed=True)
        if len(g) > 1
    ]
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


def compute_part1_overview(sellout_fmc: pd.DataFrame, hourly: pd.DataFrame):
    """
    Compute Part 1 raw analysis across all FMC customers.
    Returns (bdf, d_sum) for the 3-hour band and duration charts.
    """
    RAIN_MM = 1.0
    BAND_MAP = {0: "00_02", 3: "03_05", 6: "06_08", 9: "09_11",
                12: "12_14", 15: "15_17", 18: "18_20", 21: "21_23"}
    DUR_ORDER = ["0h", "1-2h", "3-4h", "5-6h", "7h+"]

    hourly = hourly.copy()
    hourly["hour"] = hourly["time"].dt.hour
    hourly["time_band"] = hourly["hour"].apply(lambda h: BAND_MAP.get((h // 3) * 3, "21_23"))
    hourly["is_day"] = ((hourly["hour"] >= 6) & (hourly["hour"] <= 20)).astype(int)
    hourly["is_rainy"] = (hourly["precipitation"] >= RAIN_MM).astype(int)

    # 3-hour band precipitation totals per location-date
    band_wide = (
        hourly.groupby(["latitude", "longitude", "date", "time_band"])["precipitation"]
        .sum().unstack(fill_value=0).reset_index()
    )
    band_wide.columns.name = None
    band_wide.columns = (
        ["latitude", "longitude", "date"]
        + [f"rain_{c}" for c in band_wide.columns[3:]]
    )

    # Daytime rainy hours per location-date
    day_stats = (
        hourly[hourly["is_day"] == 1]
        .groupby(["latitude", "longitude", "date"])["is_rainy"].sum()
        .reset_index().rename(columns={"is_rainy": "day_rain_hours"})
    )

    wx_daily = band_wide.merge(day_stats, on=["latitude", "longitude", "date"], how="left")
    wx_daily["day_rain_hours"] = wx_daily["day_rain_hours"].fillna(0)

    # Daily customer sales
    cust_daily = (
        sellout_fmc.groupby(["customer_code", "date", "latitude", "longitude"])["sales_quantity"]
        .sum().reset_index()
    )
    cust_daily["date"] = pd.to_datetime(cust_daily["date"]).dt.date
    cust_daily["latitude"] = cust_daily["latitude"].round(4)
    cust_daily["longitude"] = cust_daily["longitude"].round(4)

    cust_panel = cust_daily.merge(wx_daily, on=["latitude", "longitude", "date"], how="inner")

    def _dur_band(h):
        if h == 0:   return "0h"
        if h <= 2:   return "1-2h"
        if h <= 4:   return "3-4h"
        if h <= 6:   return "5-6h"
        return "7h+"

    cust_panel["day_dur_band"] = cust_panel["day_rain_hours"].apply(_dur_band)
    BAND_COLS = sorted([c for c in cust_panel.columns if c.startswith("rain_")])

    # Chart 1 data: 3-hour band % change
    band_res = []
    for col in BAND_COLS:
        band_lbl = col.replace("rain_", "").replace("_", "-")
        rainy = cust_panel[cust_panel[col] >= RAIN_MM]["sales_quantity"]
        dry   = cust_panel[cust_panel[col] <  RAIN_MM]["sales_quantity"]
        if len(rainy) < 5 or dry.mean() == 0:
            continue
        pct = (rainy.mean() / dry.mean() - 1) * 100
        n_days = int(cust_panel[cust_panel[col] >= RAIN_MM]["date"].nunique())
        band_res.append({"band": band_lbl, "pct_change": round(pct, 1), "n_rainy": n_days})
    bdf = pd.DataFrame(band_res)

    # Chart 2 data: duration % change
    d_sum = (
        cust_panel.groupby("day_dur_band")
        .agg(avg_qty=("sales_quantity", "mean"), n_days=("date", "nunique"))
        .reindex([b for b in DUR_ORDER if b in cust_panel["day_dur_band"].values])
        .dropna()
    )
    base = d_sum.loc["0h", "avg_qty"] if "0h" in d_sum.index else 1.0
    d_sum["pct_vs_dry"] = ((d_sum["avg_qty"] / base) - 1) * 100

    return bdf, d_sum, DUR_ORDER


def run_ols_rain_all_shops(sellout: pd.DataFrame, df_weather: pd.DataFrame) -> pd.DataFrame:
    """
    Per-shop OLS: log(sales) ~ C(rain_band) + temperature + windspeed + C(dow) + C(month) + trend.
    Returns shop-level coefficient rows for bands light/moderate/heavy vs none (baseline).
    Mirrors notebook new_data.ipynb cells 35-37.
    """
    RAIN_BINS = [-0.01, 0.1, 2, 8, 1e9]
    RAIN_LABS = ["none", "light", "moderate", "heavy"]
    MIN_ROWS  = 30

    route_map = sellout.groupby("customer_code")["route"].first().to_dict()
    shops     = sellout["customer_code"].unique().tolist()
    wx        = df_weather[["date", "latitude", "longitude",
                             "precipitation", "temperature", "windspeed"]].copy()

    results = []
    for shop in shops:
        sp = (
            sellout[sellout["customer_code"] == shop]
            .drop_duplicates()
            .groupby("date")
            .agg(
                sales_quantity=("sales_quantity", "sum"),
                latitude=("latitude", "first"),
                longitude=("longitude", "first"),
            )
            .reset_index()
        )
        m = (
            sp.merge(wx, on=["date", "latitude", "longitude"], how="left")
            .dropna(subset=["precipitation", "temperature", "windspeed"])
        )
        m = m[m["sales_quantity"] > 0].copy()
        if len(m) < MIN_ROWS:
            continue

        m = m.sort_values("date").reset_index(drop=True)
        m["log_q"] = np.log(m["sales_quantity"])
        m["dow"]   = m["date"].dt.dayofweek.astype("category")
        m["month"] = m["date"].dt.month.astype("category")
        m["trend"] = (m["date"] - m["date"].min()).dt.days
        m["band"]  = pd.Categorical(
            pd.cut(m["precipitation"], RAIN_BINS, labels=RAIN_LABS),
            categories=RAIN_LABS,
        )
        if m["band"].nunique(dropna=True) < 2:
            continue

        formula = "log_q ~ C(band) + temperature + windspeed + C(dow) + C(month) + trend"
        try:
            model = smf.ols(formula, data=m).fit(cov_type="HC3")
        except Exception:
            continue

        for term in model.params.index:
            if term.startswith("C(band)[") and "T." in term:
                band = term.split("T.")[-1].rstrip("]")
                coef = model.params[term]
                lo, hi = model.conf_int().loc[term]
                results.append({
                    "customer_code": shop,
                    "route":         route_map.get(shop),
                    "band":          band,
                    "effect_pct":    np.expm1(coef),
                    "ci_low_pct":    np.expm1(lo),
                    "ci_high_pct":   np.expm1(hi),
                    "p_value":       model.pvalues[term],
                    "n_rows":        len(m),
                    "r_squared":     model.rsquared,
                })

    return pd.DataFrame(results)


def aggregate_ols_rain(results_df: pd.DataFrame, route=None) -> pd.DataFrame:
    """Aggregate per-shop OLS results to per-band mean ± 95% CI + t-test."""
    from scipy.stats import ttest_1samp, t as t_dist

    df = results_df if route is None else results_df[results_df["route"] == route]

    BANDS = ["light", "moderate", "heavy"]
    rows  = []
    for band in BANDS:
        vals = df[df["band"] == band]["effect_pct"].dropna()
        if len(vals) < 2:
            continue
        mean = vals.mean()
        sem  = vals.sem()
        ci   = t_dist.ppf(0.975, df=len(vals) - 1) * sem
        _, pval = ttest_1samp(vals, 0)
        rows.append({"band": band, "mean": mean, "ci": ci, "pval": pval, "n_shops": len(vals)})
    return pd.DataFrame(rows)


def run_prophet_ols_all_shops(sellout: pd.DataFrame, df_weather: pd.DataFrame,
                              progress_callback=None) -> pd.DataFrame:
    """
    Per-shop Prophet + OLS. Returns (results_df, seasonality_df).
    progress_callback(current, total, n_fitted, n_skipped) called each iteration.
    """
    from prophet import Prophet
    import statsmodels.api as sm

    def _set_rain(p):
        if p == 0.0:   return "No Rain"
        if p <= 2:     return "Light"
        if p <= 8:     return "Moderate"
        return "Heavy"

    route_map = sellout.groupby("customer_code")["route"].first().to_dict()
    wx = df_weather[["date", "latitude", "longitude",
                     "precipitation", "temperature", "windspeed"]].copy()
    rain_map = {"No Rain": 0, "Light": 1, "Moderate": 2, "Heavy": 3}
    ols_vars  = ["temperature", "precipitation", "windspeed",
                 "rain_Heavy", "rain_Light", "rain_Moderate"]

    rows        = []
    season_rows = []
    n_skipped   = 0
    shops       = sellout["customer_code"].unique().tolist()
    n_total     = len(shops)

    for idx, code in enumerate(shops):
        try:
            cust = (
                sellout[sellout["customer_code"] == code]
                .groupby("date")
                .agg(sales_quantity=("sales_quantity", "sum"),
                     latitude=("latitude", "first"),
                     longitude=("longitude", "first"))
                .reset_index()
            )
            cust = cust.merge(wx, on=["date", "latitude", "longitude"], how="left")
            cust["rain"] = cust["precipitation"].apply(_set_rain)
            cust = cust.dropna(subset=["temperature", "precipitation", "windspeed"])
            if len(cust) < 60:
                continue

            # OLS
            df_model = pd.get_dummies(cust, columns=["rain"], drop_first=False)
            for col in ["rain_No Rain", "rain_No_Rain"]:
                if col in df_model.columns:
                    df_model.drop(columns=[col], inplace=True)
            present_rain = [c for c in ols_vars if c in df_model.columns]
            feat_cols = ["temperature", "precipitation", "windspeed"] + [
                c for c in present_rain if c not in ["temperature", "precipitation", "windspeed"]
            ]
            X = sm.add_constant(df_model[feat_cols].astype(float))
            ols = sm.OLS(df_model["sales_quantity"].astype(float), X).fit()
            ols_r2 = ols.rsquared

            row = {
                "customer_code": code,
                "route": route_map.get(code),
                "n_days": len(cust),
                "ols_r2": ols_r2,
            }
            for var in ols_vars:
                row[f"ols_coef_{var}"] = ols.params.get(var, np.nan)
                row[f"ols_pval_{var}"] = ols.pvalues.get(var, np.nan)

            # Prophet — raw temperature (yearly seasonality captures seasonal pattern)
            df_p = cust[["date", "sales_quantity", "temperature",
                          "precipitation", "windspeed"]].copy()
            df_p["ds"] = pd.to_datetime(df_p["date"])
            df_p["y"]  = df_p["sales_quantity"]
            df_p["rain_encoded"] = cust["rain"].map(rain_map)
            df_p = df_p[["ds", "y", "temperature", "precipitation",
                          "windspeed", "rain_encoded"]].dropna()

            m = Prophet(yearly_seasonality=True, weekly_seasonality=True,
                        daily_seasonality=False, seasonality_mode="additive")
            prophet_vars = ["temperature", "precipitation", "windspeed", "rain_encoded"]
            for v in prophet_vars:
                m.add_regressor(v)
            m.fit(df_p)

            beta_raw = m.params["beta"].mean(axis=0)
            betas = dict(zip(prophet_vars, beta_raw[-len(prophet_vars):]))
            for v in prophet_vars:
                row[f"prophet_{v}"] = betas.get(v, np.nan)

            rows.append(row)

            # Extract yearly seasonality component per month
            fc = m.predict(df_p[["ds", "temperature", "precipitation",
                                  "windspeed", "rain_encoded"]])
            fc["month"] = fc["ds"].dt.month
            pm = fc.groupby("month")["yearly"].mean().reset_index()
            pm["customer_code"] = code
            pm["route"] = route_map.get(code)
            season_rows.append(pm)

        except Exception:
            n_skipped += 1

        if progress_callback:
            progress_callback(idx + 1, n_total, len(rows), n_skipped)

    results_df    = pd.DataFrame(rows)
    seasonality_df = pd.concat(season_rows, ignore_index=True) if season_rows else pd.DataFrame()
    return results_df, seasonality_df


def compute_temp_contribution(results_df: pd.DataFrame,
                              sellout: pd.DataFrame,
                              df_weather: pd.DataFrame,
                              route=None) -> tuple:
    """Compute monthly + seasonal temperature contribution (β × actual temp). Fast — no refit."""
    def _get_season(m):
        if m in [12, 1, 2]: return "Winter"
        if m in [3, 4, 5]:  return "Spring"
        if m in [6, 7, 8]:  return "Summer"
        return "Autumn"

    SEASON_ORDER = ["Winter", "Spring", "Summer", "Autumn"]
    df = results_df.dropna(subset=["prophet_temperature"])
    if route is not None:
        df = df[df["route"] == route]
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), 0

    beta_map  = df.set_index("customer_code")["prophet_temperature"].to_dict()
    wx        = df_weather[["date", "latitude", "longitude", "temperature"]].copy()
    cust_daily = (
        sellout[sellout["customer_code"].isin(beta_map)]
        .groupby(["customer_code", "date"])
        .agg(latitude=("latitude", "first"), longitude=("longitude", "first"))
        .reset_index()
    )
    cust_daily = cust_daily.merge(wx, on=["date", "latitude", "longitude"], how="left")
    cust_daily = cust_daily.dropna(subset=["temperature"])
    cust_daily["beta_temp"]        = cust_daily["customer_code"].map(beta_map)
    cust_daily["temp_contribution"] = cust_daily["beta_temp"] * cust_daily["temperature"]
    cust_daily["month"]            = pd.to_datetime(cust_daily["date"]).dt.month
    cust_daily["season"]           = cust_daily["month"].apply(_get_season)

    contrib_avg = (
        cust_daily.groupby("month")
        .agg(avg_contribution=("temp_contribution", "mean"),
             std_contribution=("temp_contribution", "std"),
             avg_temp=("temperature", "mean"))
        .reset_index()
    )
    contrib_avg["month_name"] = pd.to_datetime(contrib_avg["month"], format="%m").dt.strftime("%b")

    season_contrib = (
        cust_daily.groupby("season")
        .agg(avg_contribution=("temp_contribution", "mean"),
             std_contribution=("temp_contribution", "std"),
             avg_temp=("temperature", "mean"))
        .reindex(SEASON_ORDER).reset_index()
    )
    return contrib_avg, season_contrib, cust_daily["customer_code"].nunique()


def compute_prophet_seasonality(seasonality_df: pd.DataFrame, route=None) -> pd.DataFrame:
    """Aggregate Prophet yearly seasonality component by month across shops."""
    df = seasonality_df if route is None else seasonality_df[seasonality_df["route"] == route]
    if df.empty:
        return pd.DataFrame()
    avg = (
        df.groupby("month")
        .agg(avg_yearly=("yearly", "mean"),
             std_yearly=("yearly", "std"),
             n_shops=("customer_code", "nunique"))
        .reset_index()
    )
    avg["month_name"] = pd.to_datetime(avg["month"], format="%m").dt.strftime("%b")
    return avg
