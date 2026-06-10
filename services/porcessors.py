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


SHOP_CATCHMENT_M = 500


def _capacity_to_radius(cap) -> int:
    """Derive event impact radius (metres) from estimated audience capacity."""
    try:
        cap = float(cap)
    except (TypeError, ValueError):
        return 800
    if cap != cap:  # NaN
        return 800
    if cap < 500:
        return 800
    if cap < 2000:
        return 1200
    if cap < 10_000:
        return 1500
    return 2200  # large / district-scale events


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
        return float(
            Decimal(str(v)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
        )

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
    route_date_summary["spiking_shops_in_route"] = route_date_summary[
        "spiking_shops_in_route"
    ].astype(int)
    route_date_summary["pct_route_spiking"] = (
        route_date_summary["spiking_shops_in_route"]
        / route_date_summary["total_shops_in_route"]
        * 100
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
        route_date_summary[
            [
                "date",
                "route",
                "total_shops_in_route",
                "spiking_shops_in_route",
                "pct_route_spiking",
            ]
        ],
        on=["date", "route"],
        how="left",
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
    BINS = list(range(0, 101, 10))
    LABELS = [f"{i}-{i+10}%" for i in range(0, 100, 10)]

    df = no_event_df.copy()
    if route is not None:
        df = df[df["route"] == route]

    if df.empty or df["pct_route_spiking"].isna().all():
        return pd.DataFrame(
            {"band": LABELS, "count": [0] * 10, "pct_of_group": [0.0] * 10}
        )

    df["pct_band"] = pd.cut(
        df["pct_route_spiking"].clip(0, 100),
        bins=BINS,
        labels=LABELS,
        right=True,
        include_lowest=True,
    )
    tbl = df["pct_band"].value_counts().reindex(LABELS, fill_value=0).reset_index()
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


_SPLINE_KNOTS = (1, 4, 8)   # mm — light/moderate/heavy transition points
_MM_GRID_FINE = np.linspace(0, 20, 200)
_MM_GRID_COARSE = np.linspace(0, 20, 50)  # for all-shops (speed vs. resolution)


def _make_pred_base(model_df, n, precip_cols):
    """Build a prediction DataFrame with all controls at mean/mode and precip columns zeroed."""
    mode_dow   = int(model_df["dow"].mode().iloc[0])
    mode_month = int(model_df["month"].mode().iloc[0])
    base = {
        "temperature": np.full(n, model_df["temperature"].mean()),
        "windspeed":   np.full(n, model_df["windspeed"].mean()),
        "dow":   pd.Categorical(np.full(n, mode_dow,   dtype=int), categories=model_df["dow"].cat.categories),
        "month": pd.Categorical(np.full(n, mode_month, dtype=int), categories=model_df["month"].cat.categories),
        "trend": np.full(n, int(model_df["trend"].median())),
    }
    for col in precip_cols:
        base[col] = np.zeros(n)
    return pd.DataFrame(base)


def _spline_curve(model, model_df, mm_grid, vary_col, all_precip_cols):
    """Partial effect of `vary_col` from 0→max mm, holding others at 0 and controls at mean."""
    base = _make_pred_base(model_df, len(mm_grid), all_precip_cols)
    pred_0 = float(model.predict(base.iloc[[0]]).iloc[0])

    data = base.copy()
    data[vary_col] = mm_grid
    pf = model.get_prediction(data).summary_frame(alpha=0.05)

    y    = (np.exp(pf["mean"].values          - pred_0) - 1) * 100
    y_up = (np.exp(pf["mean_ci_upper"].values  - pred_0) - 1) * 100
    y_dn = (np.exp(pf["mean_ci_lower"].values  - pred_0) - 1) * 100
    return y, y_up, y_dn


def ols_rain_processor(df: pd.DataFrame) -> tuple:
    """
    Fits OLS with natural cubic splines for precipitation (same-day, lag, lead).
    Knots at 1, 4, 8 mm — captures non-monotonic rain effects.
    Returns (curve_df, scalar_df, meta_dict).
    curve_df  — partial effect curve per timing effect: x_mm, y_pct, y_upper, y_lower.
    scalar_df — temperature & windspeed coefficients.
    meta_dict — r2, n, or error key.
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

    m["log_q"]              = np.log(m["sales_quantity"])
    m["precipitation_lag1"] = m["precipitation"].shift(1)
    m["precipitation_lead1"]= m["precipitation"].shift(-1)
    m["lag_ok"]             = m["date"].diff().dt.days == 1
    m["lead_ok"]            = (-m["date"].diff(-1).dt.days) == 1
    m["dow"]   = m["date"].dt.dayofweek.astype("category")
    m["month"] = m["date"].dt.month.astype("category")
    m["trend"] = (m["date"] - m["date"].min()).dt.days

    model_df = m.dropna(subset=["precipitation_lag1", "precipitation_lead1"])
    model_df = model_df[model_df["lag_ok"] & model_df["lead_ok"]]

    if len(model_df) < 20:
        return None, None, {"error": "Not enough consecutive days for lag/lead model"}

    knots = _SPLINE_KNOTS
    formula = (
        f"log_q ~ cr(precipitation, knots={knots})"
        f" + cr(precipitation_lag1, knots={knots})"
        f" + cr(precipitation_lead1, knots={knots})"
        " + temperature + windspeed + C(dow) + C(month) + trend"
    )

    try:
        model = smf.ols(formula, data=model_df).fit(cov_type="HC3")
    except Exception as exc:
        return None, None, {"error": str(exc)}

    all_precip_cols = ["precipitation", "precipitation_lag1", "precipitation_lead1"]
    curve_rows = []
    for effect_label, col in [
        ("Same-day",  "precipitation"),
        ("Yesterday", "precipitation_lag1"),
        ("Tomorrow",  "precipitation_lead1"),
    ]:
        y, y_up, y_dn = _spline_curve(model, model_df, _MM_GRID_FINE, col, all_precip_cols)
        for i, mm in enumerate(_MM_GRID_FINE):
            curve_rows.append({
                "effect":  effect_label,
                "x_mm":    round(float(mm), 4),
                "y_pct":   float(y[i]),
                "y_upper": float(y_up[i]),
                "y_lower": float(y_dn[i]),
            })

    ci = model.conf_int()
    scalar_rows = []
    for var in ["temperature", "windspeed"]:
        if var in model.params:
            coef = model.params[var]
            lo, hi = ci.loc[var]
            scalar_rows.append({
                "variable":   var,
                "pct_change": np.expm1(coef) * 100,
                "ci_low":     np.expm1(lo) * 100,
                "ci_high":    np.expm1(hi) * 100,
                "p":          model.pvalues[var],
            })

    return (
        pd.DataFrame(curve_rows),
        pd.DataFrame(scalar_rows),
        {"r2": model.rsquared, "n": len(model_df)},
    )


# ── Temperature natural spline ────────────────────────────────────────────────

_TEMP_SPLINE_KNOTS = (5, 15, 25)   # °C — cold/mild/warm/hot transitions


def _spline_temp_curve(model, model_df, temp_grid, ref_temp):
    """Partial effect of temperature vs ref_temp, holding other controls at mean/mode."""
    mode_dow   = int(model_df["dow"].mode().iloc[0])
    mode_month = int(model_df["month"].mode().iloc[0])
    n = len(temp_grid)
    base = pd.DataFrame({
        "temperature":   np.full(n, ref_temp),
        "precipitation": np.full(n, float(model_df["precipitation"].mean())),
        "windspeed":     np.full(n, float(model_df["windspeed"].mean())),
        "dow":   pd.Categorical(np.full(n, mode_dow,   dtype=int), categories=model_df["dow"].cat.categories),
        "month": pd.Categorical(np.full(n, mode_month, dtype=int), categories=model_df["month"].cat.categories),
        "trend": np.full(n, int(model_df["trend"].median())),
    })
    pred_ref = float(model.predict(base.iloc[[0]]).iloc[0])
    data = base.copy()
    data["temperature"] = temp_grid
    pf   = model.get_prediction(data).summary_frame(alpha=0.05)
    y    = (np.exp(pf["mean"].values          - pred_ref) - 1) * 100
    y_up = (np.exp(pf["mean_ci_upper"].values  - pred_ref) - 1) * 100
    y_dn = (np.exp(pf["mean_ci_lower"].values  - pred_ref) - 1) * 100
    return y, y_up, y_dn


def ols_temp_processor(df: pd.DataFrame) -> tuple:
    """
    Fits OLS with natural cubic spline for temperature (knots at 5, 15, 25°C).
    Reference = customer's mean temperature.
    Returns (curve_df, meta_dict).
    curve_df: x_celsius, y_pct, y_upper, y_lower, ref_temp.
    """
    df = df.copy()
    m = (
        df.groupby("date")
        .agg(
            sales_quantity=("sales_quantity", "sum"),
            temperature=("temperature", "mean"),
            precipitation=("precipitation", "mean"),
            windspeed=("windspeed", "mean"),
        )
        .reset_index()
        .sort_values("date")
        .reset_index(drop=True)
    )
    m = m.dropna(subset=["temperature", "precipitation", "windspeed"])
    m = m[m["sales_quantity"] > 0].copy()
    if len(m) < 30:
        return None, {"error": "Not enough data (< 30 selling days)"}

    m["log_q"] = np.log(m["sales_quantity"])
    m["dow"]   = m["date"].dt.dayofweek.astype("category")
    m["month"] = m["date"].dt.month.astype("category")
    m["trend"] = (m["date"] - m["date"].min()).dt.days

    knots = _TEMP_SPLINE_KNOTS
    formula = (
        f"log_q ~ cr(temperature, knots={knots})"
        " + precipitation + windspeed + C(dow) + C(month) + trend"
    )
    try:
        model = smf.ols(formula, data=m).fit(cov_type="HC3")
    except Exception as exc:
        return None, {"error": str(exc)}

    t_min     = max(-10.0, float(m["temperature"].min()))
    t_max     = min(45.0,  float(m["temperature"].max()))
    temp_grid = np.linspace(t_min, t_max, 200)
    ref_temp  = float(m["temperature"].mean())

    y, y_up, y_dn = _spline_temp_curve(model, m, temp_grid, ref_temp)

    curve_df = pd.DataFrame({
        "x_celsius": temp_grid.round(3),
        "y_pct":     y,
        "y_upper":   y_up,
        "y_lower":   y_dn,
        "ref_temp":  ref_temp,
    })
    return curve_df, {"r2": model.rsquared, "n": len(m), "ref_temp": ref_temp}


def run_ols_temp_all_shops(
    sellout: pd.DataFrame, df_weather: pd.DataFrame,
    sellin: pd.DataFrame = None,
) -> pd.DataFrame:
    """Per-shop OLS temperature spline. Returns per-shop curve rows (x_celsius, y_pct)."""
    if sellin is not None:
        common = set(sellin["customer_code"].unique()) & set(sellout["customer_code"].unique())
        shops  = [c for c in sellout["customer_code"].unique() if c in common]
    else:
        shops = sellout["customer_code"].unique().tolist()

    route_map   = sellout.groupby("customer_code")["route"].first().to_dict()
    max_date    = pd.Timestamp(sellout["date"].max())
    wx          = df_weather[
        df_weather["date"] <= max_date
    ][["date", "latitude", "longitude", "temperature", "precipitation", "windspeed"]].copy()
    global_ref  = float(wx["temperature"].mean())
    knots       = _TEMP_SPLINE_KNOTS
    results     = []

    for shop in shops:
        try:
            sp = (
                sellout[sellout["customer_code"] == shop]
                .groupby("date")
                .agg(sales_quantity=("sales_quantity","sum"),
                     latitude=("latitude","first"),
                     longitude=("longitude","first"))
                .reset_index()
            )
            m = sp.merge(wx, on=["date","latitude","longitude"], how="left")
            m = m.dropna(subset=["temperature","precipitation","windspeed"])
            m = m[m["sales_quantity"] > 0].copy()
            if len(m) < 30:
                continue

            m = m.sort_values("date").reset_index(drop=True)
            m["log_q"] = np.log(m["sales_quantity"])
            m["dow"]   = m["date"].dt.dayofweek.astype("category")
            m["month"] = m["date"].dt.month.astype("category")
            m["trend"] = (m["date"] - m["date"].min()).dt.days

            formula = (
                f"log_q ~ cr(temperature, knots={knots})"
                " + precipitation + windspeed + C(dow) + C(month) + trend"
            )
            model = smf.ols(formula, data=m).fit(cov_type="HC3")

            t_min     = max(-10.0, float(m["temperature"].min()))
            t_max     = min(45.0,  float(m["temperature"].max()))
            temp_grid = np.linspace(t_min, t_max, 50)

            y, _, _ = _spline_temp_curve(model, m, temp_grid, global_ref)

            for i, t in enumerate(temp_grid):
                results.append({
                    "customer_code": shop,
                    "route":         route_map.get(shop),
                    "x_celsius":     round(float(t), 2),
                    "y_pct":         float(y[i]),
                    "r_squared":     model.rsquared,
                })
        except Exception:
            continue

    return pd.DataFrame(results)


def aggregate_ols_temp(results_df: pd.DataFrame, route=None) -> pd.DataFrame:
    """Aggregate per-shop temperature spline curves to mean ± 95% CI across shops."""
    from scipy.stats import t as t_dist

    df = results_df if route is None else results_df[results_df["route"] == route]
    if df.empty:
        return pd.DataFrame(columns=["x_celsius","mean_pct","ci_upper","ci_lower","n_shops"])

    agg = (
        df.groupby("x_celsius")["y_pct"]
        .agg(["mean","std","count"])
        .reset_index()
        .rename(columns={"mean":"mean_pct","std":"std_pct","count":"n_shops"})
    )
    agg["sem"]      = agg["std_pct"] / np.sqrt(agg["n_shops"])
    agg["ci"]       = t_dist.ppf(0.975, df=agg["n_shops"] - 1) * agg["sem"]
    agg["ci_upper"] = agg["mean_pct"] + agg["ci"]
    agg["ci_lower"] = agg["mean_pct"] - agg["ci"]
    return agg[["x_celsius","mean_pct","ci_upper","ci_lower","n_shops"]]


# ── Feels-Like Temperature (same structure as temp, uses apparent_temperature_mean) ──

def _spline_fl_curve(model, model_df, fl_grid, ref_fl):
    """Partial effect of apparent_temperature_mean vs ref_fl, same controls as temp spline."""
    mode_dow   = int(model_df["dow"].mode().iloc[0])
    mode_month = int(model_df["month"].mode().iloc[0])
    n = len(fl_grid)
    base = pd.DataFrame({
        "apparent_temperature_mean": np.full(n, ref_fl),
        "precipitation": np.full(n, float(model_df["precipitation"].mean())),
        "windspeed":     np.full(n, float(model_df["windspeed"].mean())),
        "dow":   pd.Categorical(np.full(n, mode_dow,   dtype=int), categories=model_df["dow"].cat.categories),
        "month": pd.Categorical(np.full(n, mode_month, dtype=int), categories=model_df["month"].cat.categories),
        "trend": np.full(n, int(model_df["trend"].median())),
    })
    pred_ref = float(model.predict(base.iloc[[0]]).iloc[0])
    data = base.copy()
    data["apparent_temperature_mean"] = fl_grid
    pf   = model.get_prediction(data).summary_frame(alpha=0.05)
    y    = (np.exp(pf["mean"].values          - pred_ref) - 1) * 100
    y_up = (np.exp(pf["mean_ci_upper"].values  - pred_ref) - 1) * 100
    y_dn = (np.exp(pf["mean_ci_lower"].values  - pred_ref) - 1) * 100
    return y, y_up, y_dn


def run_ols_fl_all_shops(
    sellout: pd.DataFrame, df_weather: pd.DataFrame,
    sellin: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Per-shop OLS feels-like temperature spline.
    Same structure as run_ols_temp_all_shops but uses apparent_temperature_mean
    with the same knots (5, 15, 25°C) and same controls (windspeed, precipitation, dow, month, trend).
    """
    if sellin is not None:
        common = set(sellin["customer_code"].unique()) & set(sellout["customer_code"].unique())
        shops  = [c for c in sellout["customer_code"].unique() if c in common]
    else:
        shops = sellout["customer_code"].unique().tolist()

    route_map  = sellout.groupby("customer_code")["route"].first().to_dict()
    max_date   = pd.Timestamp(sellout["date"].max())
    wx         = df_weather[df_weather["date"] <= max_date][
        ["date", "latitude", "longitude", "apparent_temperature_mean", "precipitation", "windspeed"]
    ].copy()
    global_ref = float(wx["apparent_temperature_mean"].mean())
    knots      = _TEMP_SPLINE_KNOTS   # same knots as temperature: (5, 15, 25)
    results    = []

    for shop in shops:
        try:
            sp = (
                sellout[sellout["customer_code"] == shop]
                .groupby("date")
                .agg(sales_quantity=("sales_quantity", "sum"),
                     latitude=("latitude", "first"),
                     longitude=("longitude", "first"))
                .reset_index()
            )
            m = sp.merge(wx, on=["date", "latitude", "longitude"], how="left")
            m = m.dropna(subset=["apparent_temperature_mean", "precipitation", "windspeed"])
            m = m[m["sales_quantity"] > 0].copy()
            if len(m) < 30:
                continue

            m = m.sort_values("date").reset_index(drop=True)
            m["log_q"] = np.log(m["sales_quantity"])
            m["dow"]   = m["date"].dt.dayofweek.astype("category")
            m["month"] = m["date"].dt.month.astype("category")
            m["trend"] = (m["date"] - m["date"].min()).dt.days

            formula = (
                f"log_q ~ cr(apparent_temperature_mean, knots={knots})"
                " + precipitation + windspeed + C(dow) + C(month) + trend"
            )
            model = smf.ols(formula, data=m).fit(cov_type="HC3")

            fl_min  = max(-15.0, float(m["apparent_temperature_mean"].min()))
            fl_max  = min(45.0,  float(m["apparent_temperature_mean"].max()))
            fl_grid = np.linspace(fl_min, fl_max, 50)

            y, _, _ = _spline_fl_curve(model, m, fl_grid, global_ref)

            for i, t in enumerate(fl_grid):
                results.append({
                    "customer_code": shop,
                    "route":         route_map.get(shop),
                    "x_celsius":     round(float(t), 2),
                    "y_pct":         float(y[i]),
                    "r_squared":     model.rsquared,
                })
        except Exception:
            continue

    return pd.DataFrame(results)


# aggregate_ols_fl reuses the same x_celsius/y_pct aggregation as temperature
aggregate_ols_fl = aggregate_ols_temp


def run_prophet_fl_all_shops(
    sellout: pd.DataFrame, df_weather: pd.DataFrame,
    sellin: pd.DataFrame = None, progress_callback=None,
) -> pd.DataFrame:
    """
    Per-shop Prophet with apparent_temperature_mean as linear regressor.
    Same regressors as run_prophet_temp_all_shops but temperature replaced with
    apparent_temperature_mean. Produces prophet_apparent_temperature_mean beta column.
    """
    from prophet import Prophet

    def _set_rain(p):
        if p == 0.0: return "No Rain"
        if p <= 2:   return "Light"
        if p <= 8:   return "Moderate"
        return "Heavy"

    if sellin is not None:
        common = set(sellin["customer_code"].unique()) & set(sellout["customer_code"].unique())
        shops = [c for c in sellout["customer_code"].unique() if c in common]
    else:
        shops = sellout["customer_code"].unique().tolist()

    route_map = sellout.groupby("customer_code")["route"].first().to_dict()
    rain_map  = {"No Rain": 0, "Light": 1, "Moderate": 2, "Heavy": 3}
    wx = df_weather[
        ["date", "latitude", "longitude",
         "apparent_temperature_mean", "precipitation", "windspeed"]
    ].copy()
    prophet_vars = ["apparent_temperature_mean", "precipitation", "windspeed", "rain_encoded"]

    rows = []
    n_skipped = 0
    n_total   = len(shops)

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
            cust = cust.dropna(subset=["apparent_temperature_mean", "precipitation", "windspeed"])
            if len(cust) < 60:
                continue

            df_p = cust[["date", "sales_quantity",
                          "apparent_temperature_mean", "precipitation", "windspeed"]].copy()
            df_p["ds"] = pd.to_datetime(df_p["date"])
            df_p["y"]  = df_p["sales_quantity"]
            df_p["rain_encoded"] = cust["precipitation"].apply(_set_rain).map(rain_map)
            df_p = df_p[["ds", "y", "apparent_temperature_mean",
                          "precipitation", "windspeed", "rain_encoded"]].dropna()

            m = Prophet(yearly_seasonality=True, weekly_seasonality=True,
                        daily_seasonality=False, seasonality_mode="additive")
            for v in prophet_vars:
                m.add_regressor(v)
            m.fit(df_p)

            beta_raw = m.params["beta"].mean(axis=0)
            betas    = dict(zip(prophet_vars, beta_raw[-len(prophet_vars):]))
            rows.append({
                "customer_code": code,
                "route": route_map.get(code),
                "n_days": len(cust),
                **{f"prophet_{v}": betas.get(v, np.nan) for v in prophet_vars},
            })
        except Exception:
            n_skipped += 1

        if progress_callback:
            progress_callback(idx + 1, n_total, len(rows), n_skipped)

    return pd.DataFrame(rows)


def compute_prophet_fl_curve(
    results_df: pd.DataFrame,
    sellout: pd.DataFrame,
    df_weather: pd.DataFrame,
    route=None,
) -> pd.DataFrame:
    """
    Build a feels-like temperature % change curve from Prophet β coefficients.
    Same logic as compute_prophet_temp_curve but uses prophet_apparent_temperature_mean
    and apparent_temperature_mean as the reference weather column.
    Returns x_celsius, mean_pct, ci_upper, ci_lower, n_shops.
    """
    from scipy.stats import t as t_dist

    col = "prophet_apparent_temperature_mean"
    df  = results_df.dropna(subset=[col])
    if route is not None:
        df = df[df["route"] == route]
    if df.empty:
        return pd.DataFrame(columns=["x_celsius", "mean_pct", "ci_upper", "ci_lower", "n_shops"])

    daily = (
        sellout[sellout["customer_code"].isin(df["customer_code"])]
        .groupby(["customer_code", "date"])["sales_quantity"].sum()
        .reset_index()
    )
    median_sales = daily.groupby("customer_code")["sales_quantity"].median()

    norm_betas = []
    for _, row in df.iterrows():
        baseline = median_sales.get(row["customer_code"], np.nan)
        if baseline > 0 and not np.isnan(row[col]):
            norm_betas.append(row[col] / baseline)

    if len(norm_betas) < 2:
        return pd.DataFrame(columns=["x_celsius", "mean_pct", "ci_upper", "ci_lower", "n_shops"])

    norm_betas = np.array(norm_betas)
    n          = len(norm_betas)
    mean_beta  = norm_betas.mean()
    sem_beta   = norm_betas.std(ddof=1) / np.sqrt(n)
    t_val      = t_dist.ppf(0.975, df=n - 1)

    max_date   = pd.Timestamp(sellout["date"].max())
    global_ref = float(
        df_weather[df_weather["date"] <= max_date]["apparent_temperature_mean"].mean()
    )
    fl_grid = np.linspace(-15.0, 40.0, 200)

    rows = []
    for fl in fl_grid:
        d        = fl - global_ref
        mean_pct = mean_beta * d * 100
        ci       = t_val * sem_beta * abs(d) * 100
        rows.append({
            "x_celsius": round(float(fl), 2),
            "mean_pct":  float(mean_pct),
            "ci_upper":  float(mean_pct + ci),
            "ci_lower":  float(mean_pct - ci),
            "n_shops":   n,
        })
    return pd.DataFrame(rows)


# ── Windspeed natural spline ──────────────────────────────────────────────────

_WIND_SPLINE_KNOTS = (3, 8, 15)   # m/s — calm/breeze/strong transitions


def _spline_wind_curve(model, model_df, wind_grid, ref_wind):
    """Partial effect of windspeed vs ref_wind, holding other controls at mean/mode."""
    mode_dow   = int(model_df["dow"].mode().iloc[0])
    mode_month = int(model_df["month"].mode().iloc[0])
    n = len(wind_grid)
    base = pd.DataFrame({
        "windspeed":     np.full(n, ref_wind),
        "precipitation": np.full(n, float(model_df["precipitation"].mean())),
        "temperature":   np.full(n, float(model_df["temperature"].mean())),
        "dow":   pd.Categorical(np.full(n, mode_dow,   dtype=int), categories=model_df["dow"].cat.categories),
        "month": pd.Categorical(np.full(n, mode_month, dtype=int), categories=model_df["month"].cat.categories),
        "trend": np.full(n, int(model_df["trend"].median())),
    })
    pred_ref = float(model.predict(base.iloc[[0]]).iloc[0])
    data = base.copy()
    data["windspeed"] = wind_grid
    pf   = model.get_prediction(data).summary_frame(alpha=0.05)
    y    = (np.exp(pf["mean"].values          - pred_ref) - 1) * 100
    y_up = (np.exp(pf["mean_ci_upper"].values  - pred_ref) - 1) * 100
    y_dn = (np.exp(pf["mean_ci_lower"].values  - pred_ref) - 1) * 100
    return y, y_up, y_dn


def run_ols_wind_all_shops(
    sellout: pd.DataFrame, df_weather: pd.DataFrame,
    sellin: pd.DataFrame = None,
) -> pd.DataFrame:
    """Per-shop OLS windspeed spline. Returns per-shop curve rows (x_ms, y_pct)."""
    if sellin is not None:
        common = set(sellin["customer_code"].unique()) & set(sellout["customer_code"].unique())
        shops  = [c for c in sellout["customer_code"].unique() if c in common]
    else:
        shops = sellout["customer_code"].unique().tolist()

    route_map  = sellout.groupby("customer_code")["route"].first().to_dict()
    max_date   = pd.Timestamp(sellout["date"].max())
    wx         = df_weather[
        df_weather["date"] <= max_date
    ][["date", "latitude", "longitude", "temperature", "precipitation", "windspeed"]].copy()
    global_ref = float(wx["windspeed"].mean())
    results    = []

    for shop in shops:
        try:
            sp = (
                sellout[sellout["customer_code"] == shop]
                .groupby("date")
                .agg(sales_quantity=("sales_quantity", "sum"),
                     latitude=("latitude", "first"),
                     longitude=("longitude", "first"))
                .reset_index()
            )
            m = sp.merge(wx, on=["date", "latitude", "longitude"], how="left")
            m = m.dropna(subset=["temperature", "precipitation", "windspeed"])
            m = m[m["sales_quantity"] > 0].copy()
            if len(m) < 30:
                continue

            m = m.sort_values("date").reset_index(drop=True)
            m["log_q"] = np.log(m["sales_quantity"])
            m["dow"]   = m["date"].dt.dayofweek.astype("category")
            m["month"] = m["date"].dt.month.astype("category")
            m["trend"] = (m["date"] - m["date"].min()).dt.days

            # df=5 places 3 interior knots at data quantiles — avoids out-of-range knot errors
            formula = (
                "log_q ~ cr(windspeed, df=5)"
                " + precipitation + temperature + C(dow) + C(month) + trend"
            )
            model = smf.ols(formula, data=m).fit(cov_type="HC3")

            w_min     = max(0.0,  float(m["windspeed"].min()))
            w_max     = min(40.0, float(m["windspeed"].max()))
            wind_grid = np.linspace(w_min, w_max, 50)

            y, _, _ = _spline_wind_curve(model, m, wind_grid, global_ref)

            for i, w in enumerate(wind_grid):
                results.append({
                    "customer_code": shop,
                    "route":         route_map.get(shop),
                    "x_ms":          round(float(w), 1),
                    "y_pct":         float(y[i]),
                    "r_squared":     model.rsquared,
                })
        except Exception:
            continue

    return pd.DataFrame(results)


def aggregate_ols_wind(results_df: pd.DataFrame, route=None) -> pd.DataFrame:
    """Aggregate per-shop windspeed spline curves to mean ± 95% CI across shops."""
    from scipy.stats import t as t_dist

    df = results_df if route is None else results_df[results_df["route"] == route]
    if df.empty:
        return pd.DataFrame(columns=["x_ms", "mean_pct", "ci_upper", "ci_lower", "n_shops"])

    agg = (
        df.groupby("x_ms")["y_pct"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_pct", "std": "std_pct", "count": "n_shops"})
    )
    agg["sem"]      = agg["std_pct"] / np.sqrt(agg["n_shops"])
    agg["ci"]       = t_dist.ppf(0.975, df=agg["n_shops"] - 1) * agg["sem"]
    agg["ci_upper"] = agg["mean_pct"] + agg["ci"]
    agg["ci_lower"] = agg["mean_pct"] - agg["ci"]
    return agg[["x_ms", "mean_pct", "ci_upper", "ci_lower", "n_shops"]]


# ── Wind Chill OLS natural spline ─────────────────────────────────────────────

_WC_SPLINE_KNOTS = (0, 10, 20)   # °C apparent temp — Cold/Cool/Mild/Warm boundaries


def _spline_wc_curve(model, model_df, wc_grid, ref_wc):
    """Partial effect of apparent_temperature_mean vs ref_wc, holding controls at mean/mode."""
    mode_dow   = int(model_df["dow"].mode().iloc[0])
    mode_month = int(model_df["month"].mode().iloc[0])
    n = len(wc_grid)
    base = pd.DataFrame({
        "apparent_temperature_mean": np.full(n, ref_wc),
        "precipitation": np.full(n, float(model_df["precipitation"].mean())),
        "dow":   pd.Categorical(np.full(n, mode_dow,   dtype=int), categories=model_df["dow"].cat.categories),
        "month": pd.Categorical(np.full(n, mode_month, dtype=int), categories=model_df["month"].cat.categories),
        "trend": np.full(n, int(model_df["trend"].median())),
    })
    pred_ref = float(model.predict(base.iloc[[0]]).iloc[0])
    data = base.copy()
    data["apparent_temperature_mean"] = wc_grid
    pf   = model.get_prediction(data).summary_frame(alpha=0.05)
    y    = (np.exp(pf["mean"].values          - pred_ref) - 1) * 100
    y_up = (np.exp(pf["mean_ci_upper"].values  - pred_ref) - 1) * 100
    y_dn = (np.exp(pf["mean_ci_lower"].values  - pred_ref) - 1) * 100
    return y, y_up, y_dn


def run_ols_wc_all_shops(
    sellout: pd.DataFrame, df_weather: pd.DataFrame,
    sellin: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Per-shop OLS wind chill spline.
    Model: log_q ~ cr(apparent_temperature_mean, knots=(0,10,20)) + precipitation + C(dow) + C(month) + trend
    Returns per-shop curve rows (x_celsius, y_pct).
    """
    if sellin is not None:
        common = set(sellin["customer_code"].unique()) & set(sellout["customer_code"].unique())
        shops  = [c for c in sellout["customer_code"].unique() if c in common]
    else:
        shops = sellout["customer_code"].unique().tolist()

    route_map  = sellout.groupby("customer_code")["route"].first().to_dict()
    max_date   = pd.Timestamp(sellout["date"].max())
    wx = df_weather[df_weather["date"] <= max_date][
        ["date", "latitude", "longitude", "apparent_temperature_mean", "precipitation"]
    ].copy()
    global_ref = float(wx["apparent_temperature_mean"].mean())
    knots      = _WC_SPLINE_KNOTS
    results    = []

    for shop in shops:
        try:
            sp = (
                sellout[sellout["customer_code"] == shop]
                .groupby("date")
                .agg(sales_quantity=("sales_quantity", "sum"),
                     latitude=("latitude", "first"),
                     longitude=("longitude", "first"))
                .reset_index()
            )
            m = sp.merge(wx, on=["date", "latitude", "longitude"], how="left")
            m = m.dropna(subset=["apparent_temperature_mean", "precipitation"])
            m = m[m["sales_quantity"] > 0].copy()
            if len(m) < 30:
                continue

            m = m.sort_values("date").reset_index(drop=True)
            m["log_q"] = np.log(m["sales_quantity"])
            m["dow"]   = m["date"].dt.dayofweek.astype("category")
            m["month"] = m["date"].dt.month.astype("category")
            m["trend"] = (m["date"] - m["date"].min()).dt.days

            formula = (
                f"log_q ~ cr(apparent_temperature_mean, knots={knots})"
                " + precipitation + C(dow) + C(month) + trend"
            )
            model = smf.ols(formula, data=m).fit(cov_type="HC3")

            wc_min  = max(-15.0, float(m["apparent_temperature_mean"].min()))
            wc_max  = min(45.0,  float(m["apparent_temperature_mean"].max()))
            wc_grid = np.linspace(wc_min, wc_max, 50)

            y, _, _ = _spline_wc_curve(model, m, wc_grid, global_ref)

            for i, wc in enumerate(wc_grid):
                results.append({
                    "customer_code": shop,
                    "route":         route_map.get(shop),
                    "x_celsius":     round(float(wc), 2),
                    "y_pct":         float(y[i]),
                    "r_squared":     model.rsquared,
                })
        except Exception:
            continue

    return pd.DataFrame(results)


def aggregate_ols_wc(results_df: pd.DataFrame, route=None) -> pd.DataFrame:
    """Aggregate per-shop wind chill spline curves to mean ± 95% CI across shops."""
    from scipy.stats import t as t_dist

    df = results_df if route is None else results_df[results_df["route"] == route]
    if df.empty:
        return pd.DataFrame(columns=["x_celsius", "mean_pct", "ci_upper", "ci_lower", "n_shops"])

    agg = (
        df.groupby("x_celsius")["y_pct"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_pct", "std": "std_pct", "count": "n_shops"})
    )
    agg["sem"]      = agg["std_pct"] / np.sqrt(agg["n_shops"])
    agg["ci"]       = t_dist.ppf(0.975, df=agg["n_shops"] - 1) * agg["sem"]
    agg["ci_upper"] = agg["mean_pct"] + agg["ci"]
    agg["ci_lower"] = agg["mean_pct"] - agg["ci"]
    return agg[["x_celsius", "mean_pct", "ci_upper", "ci_lower", "n_shops"]]


def compute_part1_overview(sellout_fmc: pd.DataFrame, hourly: pd.DataFrame):
    """
    Compute Part 1 raw analysis across all FMC customers.
    Returns (bdf, d_sum) for the 3-hour band and duration charts.
    """
    RAIN_MM = 1.0
    BAND_MAP = {
        0: "00_02",
        3: "03_05",
        6: "06_08",
        9: "09_11",
        12: "12_14",
        15: "15_17",
        18: "18_20",
        21: "21_23",
    }
    DUR_ORDER = ["0h", "1-2h", "3-4h", "5-6h", "7h+"]

    hourly = hourly.copy()
    hourly["hour"] = hourly["time"].dt.hour
    hourly["time_band"] = hourly["hour"].apply(
        lambda h: BAND_MAP.get((h // 3) * 3, "21_23")
    )
    hourly["is_day"] = ((hourly["hour"] >= 6) & (hourly["hour"] <= 20)).astype(int)
    hourly["is_rainy"] = (hourly["precipitation"] >= RAIN_MM).astype(int)

    # 3-hour band precipitation totals per location-date
    band_wide = (
        hourly.groupby(["latitude", "longitude", "date", "time_band"])["precipitation"]
        .sum()
        .unstack(fill_value=0)
        .reset_index()
    )
    band_wide.columns.name = None
    band_wide.columns = ["latitude", "longitude", "date"] + [
        f"rain_{c}" for c in band_wide.columns[3:]
    ]

    # Daytime rainy hours per location-date
    day_stats = (
        hourly[hourly["is_day"] == 1]
        .groupby(["latitude", "longitude", "date"])["is_rainy"]
        .sum()
        .reset_index()
        .rename(columns={"is_rainy": "day_rain_hours"})
    )

    wx_daily = band_wide.merge(
        day_stats, on=["latitude", "longitude", "date"], how="left"
    )
    wx_daily["day_rain_hours"] = wx_daily["day_rain_hours"].fillna(0)

    # Daily customer sales
    cust_daily = (
        sellout_fmc.groupby(["customer_code", "date", "latitude", "longitude"])[
            "sales_quantity"
        ]
        .sum()
        .reset_index()
    )
    cust_daily["date"] = pd.to_datetime(cust_daily["date"]).dt.date
    cust_daily["latitude"] = cust_daily["latitude"].round(4)
    cust_daily["longitude"] = cust_daily["longitude"].round(4)

    cust_panel = cust_daily.merge(
        wx_daily, on=["latitude", "longitude", "date"], how="inner"
    )

    def _dur_band(h):
        if h == 0:
            return "0h"
        if h <= 2:
            return "1-2h"
        if h <= 4:
            return "3-4h"
        if h <= 6:
            return "5-6h"
        return "7h+"

    cust_panel["day_dur_band"] = cust_panel["day_rain_hours"].apply(_dur_band)
    BAND_COLS = sorted([c for c in cust_panel.columns if c.startswith("rain_")])

    # Chart 1 data: 3-hour band % change
    band_res = []
    for col in BAND_COLS:
        band_lbl = col.replace("rain_", "").replace("_", "-")
        rainy = cust_panel[cust_panel[col] >= RAIN_MM]["sales_quantity"]
        dry = cust_panel[cust_panel[col] < RAIN_MM]["sales_quantity"]
        if len(rainy) < 5 or dry.mean() == 0:
            continue
        pct = (rainy.mean() / dry.mean() - 1) * 100
        n_days = int(cust_panel[cust_panel[col] >= RAIN_MM]["date"].nunique())
        band_res.append(
            {"band": band_lbl, "pct_change": round(pct, 1), "n_rainy": n_days}
        )
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


def run_ols_rain_all_shops(
    sellout: pd.DataFrame, df_weather: pd.DataFrame, sellin: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Per-shop OLS: log(sales) ~ C(rain_band) + temperature + windspeed + C(dow) + C(month) + trend.
    Returns shop-level coefficient rows for bands light/moderate/heavy vs none (baseline).
    Mirrors notebook new_data.ipynb cells 35-37.
    """
    RAIN_BINS = [-0.01, 0.1, 2, 8, 1e9]
    RAIN_LABS = ["none", "light", "moderate", "heavy"]
    MIN_ROWS = 30

    route_map = sellout.groupby("customer_code")["route"].first().to_dict()
    if sellin is not None:
        common = set(sellin["customer_code"].unique()) & set(sellout["customer_code"].unique())
        shops = [c for c in sellout["customer_code"].unique() if c in common]
    else:
        shops = sellout["customer_code"].unique().tolist()
    wx = df_weather[
        ["date", "latitude", "longitude", "precipitation", "temperature", "windspeed"]
    ].copy()

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
        m = sp.merge(wx, on=["date", "latitude", "longitude"], how="left").dropna(
            subset=["precipitation", "temperature", "windspeed"]
        )
        m = m[m["sales_quantity"] > 0].copy()
        if len(m) < MIN_ROWS:
            continue

        m = m.sort_values("date").reset_index(drop=True)
        m["log_q"] = np.log(m["sales_quantity"])
        m["dow"]   = m["date"].dt.dayofweek.astype("category")
        m["month"] = m["date"].dt.month.astype("category")
        m["trend"] = (m["date"] - m["date"].min()).dt.days

        knots = _SPLINE_KNOTS
        formula = (
            f"log_q ~ cr(precipitation, knots={knots})"
            " + temperature + windspeed + C(dow) + C(month) + trend"
        )
        try:
            model = smf.ols(formula, data=m).fit(cov_type="HC3")
        except Exception:
            continue

        # Predict partial effect curve for this shop
        try:
            base = _make_pred_base(m, len(_MM_GRID_COARSE), ["precipitation"])
            pred_0 = float(model.predict(base.iloc[[0]]).iloc[0])
            data = base.copy()
            data["precipitation"] = _MM_GRID_COARSE
            preds = model.predict(data).values
            y = (np.exp(preds - pred_0) - 1) * 100
        except Exception:
            continue

        for i, mm in enumerate(_MM_GRID_COARSE):
            results.append({
                "customer_code": shop,
                "route":         route_map.get(shop),
                "x_mm":          round(float(mm), 4),
                "y_pct":         float(y[i]),
                "r_squared":     model.rsquared,
            })

    return pd.DataFrame(results)


def aggregate_ols_rain(results_df: pd.DataFrame, route=None) -> pd.DataFrame:
    """Aggregate per-shop spline curves to mean ± 95% CI across shops at each mm value."""
    from scipy.stats import t as t_dist

    df = results_df if route is None else results_df[results_df["route"] == route]
    if df.empty:
        return pd.DataFrame(columns=["x_mm", "mean_pct", "ci_upper", "ci_lower", "n_shops"])

    agg = (
        df.groupby("x_mm")["y_pct"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_pct", "std": "std_pct", "count": "n_shops"})
    )
    agg["sem"]      = agg["std_pct"] / np.sqrt(agg["n_shops"])
    agg["ci"]       = t_dist.ppf(0.975, df=agg["n_shops"] - 1) * agg["sem"]
    agg["ci_upper"] = agg["mean_pct"] + agg["ci"]
    agg["ci_lower"] = agg["mean_pct"] - agg["ci"]
    return agg[["x_mm", "mean_pct", "ci_upper", "ci_lower", "n_shops"]]


_BAND_BINS = [-0.01, 0.1, 2, 8, 1e9]
_BAND_LABS = ["none", "light", "moderate", "heavy"]


def run_ols_rain_band_all_shops(
    sellout: pd.DataFrame, df_weather: pd.DataFrame, sellin: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Per-shop OLS with categorical rain bands — same-day effect only.
    log_q ~ C(band) + temperature + windspeed + C(dow) + C(month) + trend
    Bands: none <0.1 mm, light 0.1–2 mm, moderate 2–8 mm, heavy >8 mm.
    Returns per-shop rows: customer_code, route, band, effect_pct, ci_low_pct, ci_high_pct, p_value, r_squared.
    """
    MIN_ROWS = 30

    route_map = sellout.groupby("customer_code")["route"].first().to_dict()
    if sellin is not None:
        common = set(sellin["customer_code"].unique()) & set(sellout["customer_code"].unique())
        shops = [c for c in sellout["customer_code"].unique() if c in common]
    else:
        shops = sellout["customer_code"].unique().tolist()
    wx = df_weather[
        ["date", "latitude", "longitude", "precipitation", "temperature", "windspeed"]
    ].copy()

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
        m = sp.merge(wx, on=["date", "latitude", "longitude"], how="left").dropna(
            subset=["precipitation", "temperature", "windspeed"]
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
            pd.cut(m["precipitation"], _BAND_BINS, labels=_BAND_LABS),
            categories=_BAND_LABS,
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
                    "r_squared":     model.rsquared,
                })

    return pd.DataFrame(results)


def aggregate_ols_rain_band(results_df: pd.DataFrame, route=None) -> pd.DataFrame:
    """Aggregate per-shop band OLS results to per-band mean ± 95% CI."""
    from scipy.stats import ttest_1samp, t as t_dist

    df = results_df if route is None else results_df[results_df["route"] == route]
    BANDS = ["light", "moderate", "heavy"]
    rows = []
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


def run_prophet_ols_all_shops(
    sellout: pd.DataFrame, df_weather: pd.DataFrame,
    sellin: pd.DataFrame = None, progress_callback=None
) -> pd.DataFrame:
    """
    Per-shop Prophet + OLS. Returns (results_df, seasonality_df).
    progress_callback(current, total, n_fitted, n_skipped) called each iteration.
    """
    from prophet import Prophet
    import statsmodels.api as sm

    def _set_rain(p):
        if p == 0.0:
            return "No Rain"
        if p <= 2:
            return "Light"
        if p <= 8:
            return "Moderate"
        return "Heavy"

    route_map = sellout.groupby("customer_code")["route"].first().to_dict()
    wx = df_weather[
        ["date", "latitude", "longitude", "precipitation", "temperature", "windspeed"]
    ].copy()
    rain_map = {"No Rain": 0, "Light": 1, "Moderate": 2, "Heavy": 3}
    ols_vars = [
        "temperature",
        "precipitation",
        "windspeed",
        "rain_Heavy",
        "rain_Light",
        "rain_Moderate",
    ]

    rows = []
    season_rows = []
    n_skipped = 0
    if sellin is not None:
        common = set(sellin["customer_code"].unique()) & set(sellout["customer_code"].unique())
        shops = [c for c in sellout["customer_code"].unique() if c in common]
    else:
        shops = sellout["customer_code"].unique().tolist()
    n_total = len(shops)

    for idx, code in enumerate(shops):
        try:
            cust = (
                sellout[sellout["customer_code"] == code]
                .groupby("date")
                .agg(
                    sales_quantity=("sales_quantity", "sum"),
                    latitude=("latitude", "first"),
                    longitude=("longitude", "first"),
                )
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
                c
                for c in present_rain
                if c not in ["temperature", "precipitation", "windspeed"]
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

            # Prophet — no temperature regressor so yearly captures pure seasonality
            df_p = cust[
                ["date", "sales_quantity", "precipitation", "windspeed"]
            ].copy()
            df_p["ds"] = pd.to_datetime(df_p["date"])
            df_p["y"] = df_p["sales_quantity"]
            df_p["rain_encoded"] = cust["rain"].map(rain_map)
            df_p = df_p[
                ["ds", "y", "precipitation", "windspeed", "rain_encoded"]
            ].dropna()

            m = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                seasonality_mode="additive",
            )
            prophet_vars = ["precipitation", "windspeed", "rain_encoded"]
            for v in prophet_vars:
                m.add_regressor(v)
            m.fit(df_p)

            beta_raw = m.params["beta"].mean(axis=0)
            betas = dict(zip(prophet_vars, beta_raw[-len(prophet_vars) :]))
            for v in prophet_vars:
                row[f"prophet_{v}"] = betas.get(v, np.nan)

            rows.append(row)

            # Extract yearly seasonality component per month
            fc = m.predict(
                df_p[["ds", "precipitation", "windspeed", "rain_encoded"]]
            )
            fc["month"] = fc["ds"].dt.month
            pm = fc.groupby("month")["yearly"].mean().reset_index()
            pm["customer_code"] = code
            pm["route"] = route_map.get(code)
            season_rows.append(pm)

        except Exception:
            n_skipped += 1

        if progress_callback:
            progress_callback(idx + 1, n_total, len(rows), n_skipped)

    results_df = pd.DataFrame(rows)
    seasonality_df = (
        pd.concat(season_rows, ignore_index=True) if season_rows else pd.DataFrame()
    )
    return results_df, seasonality_df


def run_prophet_temp_all_shops(
    sellout: pd.DataFrame, df_weather: pd.DataFrame,
    sellin: pd.DataFrame = None, progress_callback=None
) -> pd.DataFrame:
    """
    Per-shop Prophet WITH temperature regressor.
    Returns results_df with prophet_temperature (and other betas) for the
    temperature contribution analysis. Separate from the seasonality model.
    """
    from prophet import Prophet

    def _set_rain(p):
        if p == 0.0: return "No Rain"
        if p <= 2:   return "Light"
        if p <= 8:   return "Moderate"
        return "Heavy"

    if sellin is not None:
        common = set(sellin["customer_code"].unique()) & set(sellout["customer_code"].unique())
        shops = [c for c in sellout["customer_code"].unique() if c in common]
    else:
        shops = sellout["customer_code"].unique().tolist()

    route_map = sellout.groupby("customer_code")["route"].first().to_dict()
    wx = df_weather[
        ["date", "latitude", "longitude", "precipitation", "temperature", "windspeed"]
    ].copy()
    rain_map = {"No Rain": 0, "Light": 1, "Moderate": 2, "Heavy": 3}
    prophet_vars = ["temperature", "precipitation", "windspeed", "rain_encoded"]

    rows = []
    n_skipped = 0
    n_total = len(shops)

    for idx, code in enumerate(shops):
        try:
            cust = (
                sellout[sellout["customer_code"] == code]
                .groupby("date")
                .agg(
                    sales_quantity=("sales_quantity", "sum"),
                    latitude=("latitude", "first"),
                    longitude=("longitude", "first"),
                )
                .reset_index()
            )
            cust = cust.merge(wx, on=["date", "latitude", "longitude"], how="left")
            cust["rain"] = cust["precipitation"].apply(_set_rain)
            cust = cust.dropna(subset=["temperature", "precipitation", "windspeed"])
            if len(cust) < 60:
                continue

            df_p = cust[
                ["date", "sales_quantity", "temperature", "precipitation", "windspeed"]
            ].copy()
            df_p["ds"] = pd.to_datetime(df_p["date"])
            df_p["y"] = df_p["sales_quantity"]
            df_p["rain_encoded"] = cust["rain"].map(rain_map)
            df_p = df_p[
                ["ds", "y", "temperature", "precipitation", "windspeed", "rain_encoded"]
            ].dropna()

            m = Prophet(
                yearly_seasonality=True, weekly_seasonality=True,
                daily_seasonality=False, seasonality_mode="additive",
            )
            for v in prophet_vars:
                m.add_regressor(v)
            m.fit(df_p)

            beta_raw = m.params["beta"].mean(axis=0)
            betas = dict(zip(prophet_vars, beta_raw[-len(prophet_vars):]))

            rows.append({
                "customer_code": code,
                "route": route_map.get(code),
                "n_days": len(cust),
                **{f"prophet_{v}": betas.get(v, np.nan) for v in prophet_vars},
            })
        except Exception:
            n_skipped += 1

        if progress_callback:
            progress_callback(idx + 1, n_total, len(rows), n_skipped)

    return pd.DataFrame(rows)


def compute_prophet_temp_curve(
    results_df: pd.DataFrame,
    sellout: pd.DataFrame,
    df_weather: pd.DataFrame,
    route=None,
) -> pd.DataFrame:
    """
    Build a temperature % change curve from Prophet β coefficients.
    Prophet's temperature regressor is linear, so per-shop effect = β × (T - T_ref).
    Normalised by each shop's median daily sales to get % change.
    Returns DataFrame: x_celsius, mean_pct, ci_upper, ci_lower, n_shops.
    """
    from scipy.stats import t as t_dist

    df = results_df.dropna(subset=["prophet_temperature"])
    if route is not None:
        df = df[df["route"] == route]
    if df.empty:
        return pd.DataFrame(columns=["x_celsius", "mean_pct", "ci_upper", "ci_lower", "n_shops"])

    daily = (
        sellout[sellout["customer_code"].isin(df["customer_code"])]
        .groupby(["customer_code", "date"])["sales_quantity"].sum()
        .reset_index()
    )
    median_sales = daily.groupby("customer_code")["sales_quantity"].median()

    norm_betas = []
    for _, row in df.iterrows():
        baseline = median_sales.get(row["customer_code"], np.nan)
        if baseline > 0 and not np.isnan(row["prophet_temperature"]):
            norm_betas.append(row["prophet_temperature"] / baseline)

    if len(norm_betas) < 2:
        return pd.DataFrame(columns=["x_celsius", "mean_pct", "ci_upper", "ci_lower", "n_shops"])

    norm_betas = np.array(norm_betas)
    n         = len(norm_betas)
    mean_beta = norm_betas.mean()
    sem_beta  = norm_betas.std(ddof=1) / np.sqrt(n)
    t_val     = t_dist.ppf(0.975, df=n - 1)

    max_date   = pd.Timestamp(sellout["date"].max())
    global_ref = float(df_weather[df_weather["date"] <= max_date]["temperature"].mean())
    temp_grid  = np.linspace(-10.0, 40.0, 200)

    rows = []
    for t in temp_grid:
        d        = t - global_ref
        mean_pct = mean_beta * d * 100
        ci       = t_val * sem_beta * abs(d) * 100
        rows.append({
            "x_celsius": round(float(t), 2),
            "mean_pct":  float(mean_pct),
            "ci_upper":  float(mean_pct + ci),
            "ci_lower":  float(mean_pct - ci),
            "n_shops":   n,
        })
    return pd.DataFrame(rows)


def compute_prophet_rain_curve(
    results_df: pd.DataFrame,
    sellout: pd.DataFrame,
    route=None,
) -> pd.DataFrame:
    """
    Build a rain % change curve from Prophet β (prophet_precipitation already in results_df).
    Reference = 0 mm (dry day), so % change = β_norm × mm × 100.
    Returns x_mm, mean_pct, ci_upper, ci_lower, n_shops.
    """
    from scipy.stats import t as t_dist

    df = results_df.dropna(subset=["prophet_precipitation"])
    if route is not None:
        df = df[df["route"] == route]
    if df.empty:
        return pd.DataFrame(columns=["x_mm", "mean_pct", "ci_upper", "ci_lower", "n_shops"])

    daily = (
        sellout[sellout["customer_code"].isin(df["customer_code"])]
        .groupby(["customer_code", "date"])["sales_quantity"].sum()
        .reset_index()
    )
    median_sales = daily.groupby("customer_code")["sales_quantity"].median()

    norm_betas = []
    for _, row in df.iterrows():
        baseline = median_sales.get(row["customer_code"], np.nan)
        if baseline > 0 and not np.isnan(row["prophet_precipitation"]):
            norm_betas.append(row["prophet_precipitation"] / baseline)

    if len(norm_betas) < 2:
        return pd.DataFrame(columns=["x_mm", "mean_pct", "ci_upper", "ci_lower", "n_shops"])

    norm_betas = np.array(norm_betas)
    n         = len(norm_betas)
    mean_beta = norm_betas.mean()
    sem_beta  = norm_betas.std(ddof=1) / np.sqrt(n)
    t_val     = t_dist.ppf(0.975, df=n - 1)

    # Reference = 0 mm (dry day) so effect at mm rainfall = β_norm × mm × 100
    mm_grid = np.linspace(0.0, 20.0, 200)
    rows = []
    for mm in mm_grid:
        mean_pct = mean_beta * mm * 100
        ci       = t_val * sem_beta * mm * 100
        rows.append({
            "x_mm":     round(float(mm), 2),
            "mean_pct": float(mean_pct),
            "ci_upper": float(mean_pct + ci),
            "ci_lower": float(mean_pct - ci),
            "n_shops":  n,
        })
    return pd.DataFrame(rows)


def compute_prophet_wind_curve(
    results_df: pd.DataFrame,
    sellout: pd.DataFrame,
    df_weather: pd.DataFrame,
    route=None,
) -> pd.DataFrame:
    """
    Build a windspeed % change curve from Prophet β coefficients already stored in results_df.
    prophet_windspeed is fitted as a linear regressor alongside temperature, so this is a
    straight line through (ref_wind, 0%). Returns x_ms, mean_pct, ci_upper, ci_lower, n_shops.
    """
    from scipy.stats import t as t_dist

    df = results_df.dropna(subset=["prophet_windspeed"])
    if route is not None:
        df = df[df["route"] == route]
    if df.empty:
        return pd.DataFrame(columns=["x_ms", "mean_pct", "ci_upper", "ci_lower", "n_shops"])

    daily = (
        sellout[sellout["customer_code"].isin(df["customer_code"])]
        .groupby(["customer_code", "date"])["sales_quantity"].sum()
        .reset_index()
    )
    median_sales = daily.groupby("customer_code")["sales_quantity"].median()

    norm_betas = []
    for _, row in df.iterrows():
        baseline = median_sales.get(row["customer_code"], np.nan)
        if baseline > 0 and not np.isnan(row["prophet_windspeed"]):
            norm_betas.append(row["prophet_windspeed"] / baseline)

    if len(norm_betas) < 2:
        return pd.DataFrame(columns=["x_ms", "mean_pct", "ci_upper", "ci_lower", "n_shops"])

    norm_betas = np.array(norm_betas)
    n         = len(norm_betas)
    mean_beta = norm_betas.mean()
    sem_beta  = norm_betas.std(ddof=1) / np.sqrt(n)
    t_val     = t_dist.ppf(0.975, df=n - 1)

    max_date   = pd.Timestamp(sellout["date"].max())
    global_ref = float(df_weather[df_weather["date"] <= max_date]["windspeed"].mean())
    wind_grid  = np.linspace(0.0, 45.0, 200)

    rows = []
    for w in wind_grid:
        d        = w - global_ref
        mean_pct = mean_beta * d * 100
        ci       = t_val * sem_beta * abs(d) * 100
        rows.append({
            "x_ms":     round(float(w), 2),
            "mean_pct": float(mean_pct),
            "ci_upper": float(mean_pct + ci),
            "ci_lower": float(mean_pct - ci),
            "n_shops":  n,
        })
    return pd.DataFrame(rows)


def run_prophet_wc_all_shops(
    sellout: pd.DataFrame, df_weather: pd.DataFrame,
    sellin: pd.DataFrame = None, progress_callback=None,
) -> pd.DataFrame:
    """
    Per-shop Prophet with apparent_temperature_mean (feels-like) as linear regressor.
    Returns results_df with prophet_apparent_temperature_mean for wind chill Prophet curve.
    """
    from prophet import Prophet

    if sellin is not None:
        common = set(sellin["customer_code"].unique()) & set(sellout["customer_code"].unique())
        shops = [c for c in sellout["customer_code"].unique() if c in common]
    else:
        shops = sellout["customer_code"].unique().tolist()

    route_map = sellout.groupby("customer_code")["route"].first().to_dict()
    wx = df_weather[
        ["date", "latitude", "longitude", "apparent_temperature_mean", "precipitation"]
    ].copy()
    prophet_vars = ["apparent_temperature_mean", "precipitation"]

    rows = []
    n_skipped = 0
    n_total = len(shops)

    for idx, code in enumerate(shops):
        try:
            cust = (
                sellout[sellout["customer_code"] == code]
                .groupby("date")
                .agg(
                    sales_quantity=("sales_quantity", "sum"),
                    latitude=("latitude", "first"),
                    longitude=("longitude", "first"),
                )
                .reset_index()
            )
            cust = cust.merge(wx, on=["date", "latitude", "longitude"], how="left")
            cust = cust.dropna(subset=["apparent_temperature_mean", "precipitation"])
            if len(cust) < 60:
                continue

            df_p = cust[
                ["date", "sales_quantity", "apparent_temperature_mean", "precipitation"]
            ].copy()
            df_p["ds"] = pd.to_datetime(df_p["date"])
            df_p["y"]  = df_p["sales_quantity"]
            df_p = df_p[["ds", "y", "apparent_temperature_mean", "precipitation"]].dropna()

            m = Prophet(
                yearly_seasonality=True, weekly_seasonality=True,
                daily_seasonality=False, seasonality_mode="additive",
            )
            for v in prophet_vars:
                m.add_regressor(v)
            m.fit(df_p)

            beta_raw = m.params["beta"].mean(axis=0)
            betas = dict(zip(prophet_vars, beta_raw[-len(prophet_vars):]))
            rows.append({
                "customer_code": code,
                "route": route_map.get(code),
                "n_days": len(cust),
                **{f"prophet_{v}": betas.get(v, np.nan) for v in prophet_vars},
            })
        except Exception:
            n_skipped += 1

        if progress_callback:
            progress_callback(idx + 1, n_total, len(rows), n_skipped)

    return pd.DataFrame(rows)


def compute_prophet_wc_curve(
    results_df: pd.DataFrame,
    sellout: pd.DataFrame,
    df_weather: pd.DataFrame,
    route=None,
) -> pd.DataFrame:
    """
    Build a feels-like temperature % change curve from Prophet β coefficients.
    Linear regressor → straight line through (global_ref_wc, 0%).
    Returns x_celsius, mean_pct, ci_upper, ci_lower, n_shops.
    """
    from scipy.stats import t as t_dist

    col = "prophet_apparent_temperature_mean"
    df = results_df.dropna(subset=[col])
    if route is not None:
        df = df[df["route"] == route]
    if df.empty:
        return pd.DataFrame(columns=["x_celsius", "mean_pct", "ci_upper", "ci_lower", "n_shops"])

    daily = (
        sellout[sellout["customer_code"].isin(df["customer_code"])]
        .groupby(["customer_code", "date"])["sales_quantity"].sum()
        .reset_index()
    )
    median_sales = daily.groupby("customer_code")["sales_quantity"].median()

    norm_betas = []
    for _, row in df.iterrows():
        baseline = median_sales.get(row["customer_code"], np.nan)
        if baseline > 0 and not np.isnan(row[col]):
            norm_betas.append(row[col] / baseline)

    if len(norm_betas) < 2:
        return pd.DataFrame(columns=["x_celsius", "mean_pct", "ci_upper", "ci_lower", "n_shops"])

    norm_betas = np.array(norm_betas)
    n          = len(norm_betas)
    mean_beta  = norm_betas.mean()
    sem_beta   = norm_betas.std(ddof=1) / np.sqrt(n)
    t_val      = t_dist.ppf(0.975, df=n - 1)

    max_date   = pd.Timestamp(sellout["date"].max())
    global_ref = float(
        df_weather[df_weather["date"] <= max_date]["apparent_temperature_mean"].mean()
    )
    wc_grid = np.linspace(-15.0, 40.0, 200)

    rows = []
    for wc in wc_grid:
        d        = wc - global_ref
        mean_pct = mean_beta * d * 100
        ci       = t_val * sem_beta * abs(d) * 100
        rows.append({
            "x_celsius": round(float(wc), 2),
            "mean_pct":  float(mean_pct),
            "ci_upper":  float(mean_pct + ci),
            "ci_lower":  float(mean_pct - ci),
            "n_shops":   n,
        })
    return pd.DataFrame(rows)


def compute_temp_contribution(
    results_df: pd.DataFrame,
    sellout: pd.DataFrame,
    df_weather: pd.DataFrame,
    route=None,
    beta_col: str = "prophet_temperature",
    temp_col: str = "temperature",
) -> tuple:
    """Compute monthly + seasonal temperature contribution (β × actual temp). Fast — no refit.

    beta_col / temp_col can be overridden to use apparent_temperature_mean (feels-like).
    """

    def _get_season(m):
        if m in [12, 1, 2]:
            return "Winter"
        if m in [3, 4, 5]:
            return "Spring"
        if m in [6, 7, 8]:
            return "Summer"
        return "Autumn"

    SEASON_ORDER = ["Winter", "Spring", "Summer", "Autumn"]
    df = results_df.dropna(subset=[beta_col])
    if route is not None:
        df = df[df["route"] == route]
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), 0

    beta_map = df.set_index("customer_code")[beta_col].to_dict()
    wx = df_weather[["date", "latitude", "longitude", temp_col]].copy()
    cust_daily = (
        sellout[sellout["customer_code"].isin(beta_map)]
        .groupby(["customer_code", "date"])
        .agg(
            latitude=("latitude", "first"),
            longitude=("longitude", "first"),
            sales_quantity=("sales_quantity", "sum"),
        )
        .reset_index()
    )
    cust_daily = cust_daily.merge(wx, on=["date", "latitude", "longitude"], how="left")
    cust_daily = cust_daily.dropna(subset=[temp_col])
    cust_daily["beta_temp"]         = cust_daily["customer_code"].map(beta_map)
    cust_daily["temp_contribution"]  = cust_daily["beta_temp"] * cust_daily[temp_col]
    cust_daily["month"]             = pd.to_datetime(cust_daily["date"]).dt.month
    cust_daily["season"]            = cust_daily["month"].apply(_get_season)

    monthly_avg_sales = (
        cust_daily[cust_daily["sales_quantity"] > 0]
        .groupby(["customer_code", "month"])["sales_quantity"]
        .mean()
        .reset_index()
        .rename(columns={"sales_quantity": "avg_monthly_sales"})
    )
    cust_daily = cust_daily.merge(monthly_avg_sales, on=["customer_code", "month"], how="left")
    cust_daily["temp_contribution_pct"] = (
        cust_daily["temp_contribution"] / cust_daily["avg_monthly_sales"] * 100
    )

    contrib_avg = (
        cust_daily.groupby("month")
        .agg(
            avg_contribution=("temp_contribution_pct", "mean"),
            std_contribution=("temp_contribution_pct", "std"),
            avg_temp=(temp_col, "mean"),
        )
        .reset_index()
    )
    contrib_avg["month_name"] = pd.to_datetime(
        contrib_avg["month"], format="%m"
    ).dt.strftime("%b")

    season_contrib = (
        cust_daily.groupby("season")
        .agg(
            avg_contribution=("temp_contribution_pct", "mean"),
            std_contribution=("temp_contribution_pct", "std"),
            avg_temp=(temp_col, "mean"),
        )
        .reindex(SEASON_ORDER)
        .reset_index()
    )
    return contrib_avg, season_contrib, cust_daily["customer_code"].nunique()


def compute_prophet_seasonality(
    seasonality_df: pd.DataFrame, route=None
) -> pd.DataFrame:
    """Aggregate Prophet yearly seasonality component by month across shops."""
    df = (
        seasonality_df
        if route is None
        else seasonality_df[seasonality_df["route"] == route]
    )
    if df.empty:
        return pd.DataFrame()
    avg = (
        df.groupby("month")
        .agg(
            avg_yearly=("yearly", "mean"),
            std_yearly=("yearly", "std"),
            n_shops=("customer_code", "nunique"),
        )
        .reset_index()
    )
    avg["month_name"] = pd.to_datetime(avg["month"], format="%m").dt.strftime("%b")
    return avg


# ── Sky Condition Analysis ────────────────────────────────────────────────────

_SKY_SUNNY_CODES = frozenset([0, 1, 2])
_SKY_OVERCAST_CODE = 3


def _sky_label(code) -> str | None:
    if pd.isna(code):
        return None
    c = int(code)
    if c in _SKY_SUNNY_CODES:
        return "Sunny"
    if c == _SKY_OVERCAST_CODE:
        return "Overcast"
    return "Others"


def compute_sky_analysis(sellout: pd.DataFrame, df_weather: pd.DataFrame) -> pd.DataFrame:
    """
    For each shop:
      1. Aggregate daily sales (sum by date — one row per shop × date)
      2. Run STL(period=7, robust=True) to extract the residual component
      3. Merge weather codes → sky condition label

    Returns shop-day DataFrame with columns:
      customer_code, route, date, sky, residual, mean_sales
    where residual is the STL residual (trend + seasonality removed).
    Use residual to compare sky conditions free of weekly and trend effects.
    """
    from statsmodels.tsa.seasonal import STL

    # Step 1: one row per shop × date (sum in case of multiple SKUs)
    daily = (
        sellout[["customer_code", "route", "date", "sales_quantity", "latitude", "longitude"]]
        .groupby(["customer_code", "route", "latitude", "longitude", "date"])
        .agg(sales_quantity=("sales_quantity", "sum"))
        .reset_index()
    )

    # Step 2: per-shop STL decomposition
    records = []
    for (cust, route, lat, lon), grp in daily.groupby(
        ["customer_code", "route", "latitude", "longitude"]
    ):
        grp = grp.sort_values("date").reset_index(drop=True)
        if len(grp) < 14:  # need ≥ 2 full periods
            continue
        try:
            result = STL(grp["sales_quantity"], period=7, robust=True).fit()
            grp = grp.copy()
            grp["residual"] = result.resid
            grp["mean_sales"] = grp["sales_quantity"].mean()
            records.append(
                grp[["customer_code", "route", "latitude", "longitude",
                      "date", "sales_quantity", "residual", "mean_sales"]]
            )
        except Exception:
            continue

    if not records:
        return pd.DataFrame()

    shop_resid = pd.concat(records, ignore_index=True)

    # Step 3: merge sky condition
    merged = shop_resid.merge(
        df_weather[["date", "latitude", "longitude", "weathercode"]],
        on=["date", "latitude", "longitude"],
        how="left",
    )
    merged["sky"] = merged["weathercode"].apply(_sky_label)
    return merged.dropna(subset=["sky", "residual"])


# ── Sunny Day Transition Analysis ────────────────────────────────────────────

_TRANS_ORDER    = ["Day Before Sunny", "Sunny Day", "Day After Sunny"]
_TRANS_BASELINE = "Sunny Day"


def compute_sunny_transition_analysis(sky_df: pd.DataFrame, max_lookback: int = 7) -> pd.DataFrame:
    """
    For every sunny day, look back up to max_lookback days to find the nearest
    non-sunny day (Day Before Sunny), and forward to find the nearest non-sunny
    day (Day After Sunny). Data gaps are skipped and the search continues.
    Non-sunny days are deduplicated per shop so the same date is emitted only
    once even when it is the lookback target for multiple consecutive sunny days.
    Returns sky_df rows with an added 'transition_cat' column.
    """
    df = sky_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    result_rows = []
    for code, grp in df.groupby("customer_code"):
        grp = grp.sort_values("date").reset_index(drop=True)
        date_to_sky = dict(zip(grp["date"], grp["sky"]))
        date_to_idx = {row["date"]: i for i, row in grp.iterrows()}

        seen_before: set = set()
        seen_after:  set = set()

        for _, row in grp.iterrows():
            d         = row["date"]
            today_sky = row["sky"]

            if today_sky != "Sunny":
                continue

            # Emit the sunny day itself
            r = row.to_dict()
            r["transition_cat"] = "Sunny Day"
            result_rows.append(r)

            # Look BACK for the nearest non-sunny day
            for lag in range(1, max_lookback + 1):
                prev_d   = d - pd.Timedelta(days=lag)
                prev_sky = date_to_sky.get(prev_d)
                if prev_sky is None:
                    continue          # date not in data — skip gap, keep looking
                if prev_sky != "Sunny":
                    if prev_d not in seen_before:
                        rb = grp.iloc[date_to_idx[prev_d]].to_dict()
                        rb["transition_cat"] = "Day Before Sunny"
                        result_rows.append(rb)
                        seen_before.add(prev_d)
                    break             # found nearest non-sunny, stop
                # prev is also sunny → keep looking further back

            # Look FORWARD for the nearest non-sunny day
            for lag in range(1, max_lookback + 1):
                next_d   = d + pd.Timedelta(days=lag)
                next_sky = date_to_sky.get(next_d)
                if next_sky is None:
                    continue
                if next_sky != "Sunny":
                    if next_d not in seen_after:
                        ra = grp.iloc[date_to_idx[next_d]].to_dict()
                        ra["transition_cat"] = "Day After Sunny"
                        result_rows.append(ra)
                        seen_after.add(next_d)
                    break

    if not result_rows:
        return pd.DataFrame()
    return pd.DataFrame(result_rows)


# ── Sunny Day × Temperature Combined Analysis ────────────────────────────────

_SUNNY_TEMP_BINS   = [-20, 10, 15, 20, 25]
_SUNNY_TEMP_LABELS = ["<10°C", "10–15°C", "15–20°C", "20–25°C"]


def compute_sunny_temp_combined(
    trans_df: pd.DataFrame,
    df_weather: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each temperature bin (temperature on the Sunny Day), compute:
      - avg STL residual and raw sales for the paired Day Before Sunny
      - avg STL residual and raw sales for the Sunny Day
      - % change (STL-based, normalised by shop mean)
      - % change (raw, normalised by before_sales)
    Returns one row per temperature bin with aggregated stats.
    """
    df = trans_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    wx = df_weather[["date", "latitude", "longitude", "temperature"]].copy()
    wx["date"] = pd.to_datetime(wx["date"])

    df = df.merge(wx, on=["date", "latitude", "longitude"], how="left")

    records = []
    for shop, grp in df.groupby("customer_code"):
        grp = grp.sort_values("date").reset_index(drop=True)
        date_to_cat  = dict(zip(grp["date"], grp["transition_cat"]))
        date_to_row  = grp.drop_duplicates(subset="date", keep="first").set_index("date").to_dict("index")
        shop_mean    = float(grp["mean_sales"].mean()) if grp["mean_sales"].notna().any() else 1.0

        for _, row in grp.iterrows():
            if row["transition_cat"] != "Sunny Day":
                continue
            d         = row["date"]
            sunny_temp = row.get("temperature", float("nan"))
            if pd.isna(sunny_temp):
                continue

            # Find paired Day Before Sunny (nearest non-sunny within 7 days back)
            before_resid = float("nan")
            before_sales = float("nan")
            for lag in range(1, 8):
                prev_d = d - pd.Timedelta(days=lag)
                if date_to_cat.get(prev_d) == "Day Before Sunny":
                    pr = date_to_row.get(prev_d, {})
                    before_resid = pr.get("residual", float("nan"))
                    before_sales = pr.get("sales_quantity", float("nan"))
                    break

            records.append({
                "customer_code": shop,
                "route":         row.get("route"),
                "sunny_date":    d,
                "sunny_temp":    float(sunny_temp),
                "sunny_resid":   float(row.get("residual", float("nan"))),
                "sunny_sales":   float(row.get("sales_quantity", float("nan"))),
                "before_resid":  before_resid,
                "before_sales":  before_sales,
                "shop_mean":     shop_mean,
            })

    if not records:
        return pd.DataFrame()

    pairs = pd.DataFrame(records)
    pairs["temp_bin"] = pd.cut(
        pairs["sunny_temp"],
        bins=_SUNNY_TEMP_BINS,
        labels=_SUNNY_TEMP_LABELS,
    )
    pairs = pairs.dropna(subset=["temp_bin", "before_resid", "sunny_resid",
                                  "before_sales", "sunny_sales"])

    rows = []
    for tbin, grp in pairs.groupby("temp_bin", observed=True):
        if len(grp) < 3:
            continue
        sm     = float(grp["shop_mean"].mean()) or 1.0
        bs_r   = float(grp["before_resid"].mean())
        su_r   = float(grp["sunny_resid"].mean())
        bs_s   = float(grp["before_sales"].mean())
        su_s   = float(grp["sunny_sales"].mean())
        rows.append({
            "temp_bin":     str(tbin),
            "before_resid": bs_r,
            "sunny_resid":  su_r,
            "before_sales": bs_s,
            "sunny_sales":  su_s,
            "pct_stl":      (su_r - bs_r) / sm * 100,
            "pct_raw":      (su_s - bs_s) / abs(bs_s) * 100 if bs_s != 0 else float("nan"),
            "n_dates":      int(grp["sunny_date"].nunique()),
        })

    return pd.DataFrame(rows)


# ── Temperature Gap Analysis ─────────────────────────────────────────────────

_GAP_BINS   = [-999, -3, -1, 1, 999]
_GAP_LABELS = ["Cold Feel", "Slight Chill", "Similar", "Warm Feel"]
_GAP_BASELINE = "Similar"


def _gap_label(gap) -> str | None:
    if pd.isna(gap):
        return None
    g = float(gap)
    if g <= -3:
        return "Cold Feel"
    if g <= -1:
        return "Slight Chill"
    if g <= 1:
        return "Similar"
    return "Warm Feel"


def compute_gap_analysis(sky_df: pd.DataFrame, df_weather: pd.DataFrame) -> pd.DataFrame:
    """
    Lightweight — reuses STL residuals from sky_df (no new STL run).
    Merges apparent_temperature_mean and temperature from df_weather,
    computes gap = apparent - actual, labels into 4 categories.
    Returns sky_df enriched with: temperature, apparent_temperature_mean, gap, gap_cat, month.
    """
    wx = df_weather[["date", "latitude", "longitude", "temperature", "apparent_temperature_mean"]].copy()
    wx["date"] = pd.to_datetime(wx["date"])
    df = sky_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    merged = df.merge(wx, on=["date", "latitude", "longitude"], how="left")
    merged["gap"] = merged["apparent_temperature_mean"] - merged["temperature"]
    merged["gap_cat"] = merged["gap"].apply(_gap_label)
    merged["month"] = pd.to_datetime(merged["date"]).dt.month
    return merged.dropna(subset=["gap_cat"])


# ── Storm (Wind Gust) Analysis ───────────────────────────────────────────────


def _storm_label(gusts) -> str | None:
    if pd.isna(gusts):
        return None
    g = float(gusts)
    if g < 25:
        return "Calm"
    if g < 40:
        return "Moderate"
    if g < 60:
        return "Windy"
    return "Storm"


def compute_storm_analysis(sky_df: pd.DataFrame, df_weather: pd.DataFrame) -> pd.DataFrame:
    """
    Lightweight — reuses STL residuals from sky_df (no new STL run).
    Merges windgusts_max from df_weather, labels into 4 storm categories.
    Returns sky_df enriched with: windgusts_max, storm_cat, month.
    """
    wx = df_weather[["date", "latitude", "longitude", "windgusts_max"]].copy()
    wx["date"] = pd.to_datetime(wx["date"])
    df = sky_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    merged = df.merge(wx, on=["date", "latitude", "longitude"], how="left")
    merged["storm_cat"] = merged["windgusts_max"].apply(_storm_label)
    merged["month"] = pd.to_datetime(merged["date"]).dt.month
    return merged.dropna(subset=["storm_cat"])


# ── Wind Chill (Apparent Temperature) Analysis ───────────────────────────────

_WC_BASELINE = "Mild"


def _wc_label(apparent_temp) -> str | None:
    if pd.isna(apparent_temp):
        return None
    t = float(apparent_temp)
    if t < 0:
        return "Cold"
    if t < 10:
        return "Cool"
    if t < 20:
        return "Mild"
    return "Warm"


def compute_windchill_analysis(sellout: pd.DataFrame, df_weather: pd.DataFrame) -> pd.DataFrame:
    """
    STL-based apparent-temperature (wind chill) analysis across all shops.
    Returns shop-day DataFrame:
      customer_code, route, date, wc_cat, sales_quantity, residual, mean_sales
    """
    from statsmodels.tsa.seasonal import STL

    daily = (
        sellout[["customer_code", "route", "date", "sales_quantity", "latitude", "longitude"]]
        .groupby(["customer_code", "route", "latitude", "longitude", "date"])
        .agg(sales_quantity=("sales_quantity", "sum"))
        .reset_index()
    )

    records = []
    for (cust, route, lat, lon), grp in daily.groupby(
        ["customer_code", "route", "latitude", "longitude"]
    ):
        grp = grp.sort_values("date").reset_index(drop=True)
        if len(grp) < 14:
            continue
        try:
            result = STL(grp["sales_quantity"], period=7, robust=True).fit()
            grp = grp.copy()
            grp["residual"] = result.resid
            grp["mean_sales"] = grp["sales_quantity"].mean()
            records.append(
                grp[["customer_code", "route", "latitude", "longitude",
                      "date", "sales_quantity", "residual", "mean_sales"]]
            )
        except Exception:
            continue

    if not records:
        return pd.DataFrame()

    shop_resid = pd.concat(records, ignore_index=True)

    merged = shop_resid.merge(
        df_weather[["date", "latitude", "longitude", "apparent_temperature_mean"]],
        on=["date", "latitude", "longitude"],
        how="left",
    )
    merged["wc_cat"] = merged["apparent_temperature_mean"].apply(_wc_label)
    return merged.dropna(subset=["wc_cat", "residual"])


def compute_customer_sku_windchill(
    sellout: pd.DataFrame,
    df_weather: pd.DataFrame,
    customer_code: str,
    sku_name: str | None = None,
) -> pd.DataFrame:
    """
    Fast on-the-fly wind chill analysis for a single customer, optionally filtered to one SKU.
    Returns day-level DataFrame: date, wc_cat, sales_quantity, residual, mean_sales
    """
    from statsmodels.tsa.seasonal import STL

    cust_rows = sellout[sellout["customer_code"] == customer_code]
    if cust_rows.empty:
        return pd.DataFrame()

    lat = float(cust_rows["latitude"].iloc[0])
    lon = float(cust_rows["longitude"].iloc[0])

    mask = sellout["customer_code"] == customer_code
    if sku_name:
        mask &= sellout["sku_name"] == sku_name
    df = sellout[mask]
    if df.empty:
        return pd.DataFrame()

    daily = (
        df.groupby("date")
        .agg(sales_quantity=("sales_quantity", "sum"))
        .reset_index()
        .sort_values("date")
    )

    full_range = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    daily = (
        daily.set_index("date")
        .reindex(full_range, fill_value=0)
        .rename_axis("date")
        .reset_index()
    )

    if len(daily) < 14:
        return pd.DataFrame()

    try:
        result = STL(daily["sales_quantity"], period=7, robust=True).fit()
        daily["residual"] = result.resid
        daily["mean_sales"] = daily["sales_quantity"].mean()
    except Exception:
        return pd.DataFrame()

    daily["latitude"] = lat
    daily["longitude"] = lon

    wx = df_weather[
        (df_weather["latitude"] == lat) & (df_weather["longitude"] == lon)
    ][["date", "apparent_temperature_mean"]]

    daily["date"] = pd.to_datetime(daily["date"])
    wx = wx.copy()
    wx["date"] = pd.to_datetime(wx["date"])

    merged = daily.merge(wx, on="date", how="left")
    merged["wc_cat"] = merged["apparent_temperature_mean"].apply(_wc_label)
    return merged.dropna(subset=["wc_cat", "residual"])


def compute_customer_sku_sky(
    sellout: pd.DataFrame,
    df_weather: pd.DataFrame,
    customer_code: str,
    sku_name: str | None = None,
) -> pd.DataFrame:
    """
    Fast on-the-fly sky condition analysis for a single customer,
    optionally filtered to one SKU.

    Returns day-level DataFrame: date, sky, sales_quantity, residual, mean_sales
    """
    from statsmodels.tsa.seasonal import STL

    cust_rows = sellout[sellout["customer_code"] == customer_code]
    if cust_rows.empty:
        return pd.DataFrame()

    lat = float(cust_rows["latitude"].iloc[0])
    lon = float(cust_rows["longitude"].iloc[0])

    mask = sellout["customer_code"] == customer_code
    if sku_name:
        mask &= sellout["sku_name"] == sku_name
    df = sellout[mask]
    if df.empty:
        return pd.DataFrame()

    daily = (
        df.groupby("date")
        .agg(sales_quantity=("sales_quantity", "sum"))
        .reset_index()
        .sort_values("date")
    )

    # Fill gaps so STL sees a continuous weekly series (missing days = 0 sold)
    full_range = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    daily = (
        daily.set_index("date")
        .reindex(full_range, fill_value=0)
        .rename_axis("date")
        .reset_index()
    )

    if len(daily) < 14:
        return pd.DataFrame()

    try:
        result = STL(daily["sales_quantity"], period=7, robust=True).fit()
        daily["residual"] = result.resid
        daily["mean_sales"] = daily["sales_quantity"].mean()
    except Exception:
        return pd.DataFrame()

    daily["latitude"] = lat
    daily["longitude"] = lon

    wx = df_weather[
        (df_weather["latitude"] == lat) & (df_weather["longitude"] == lon)
    ][["date", "weathercode"]]

    # align date types before merge
    daily["date"] = pd.to_datetime(daily["date"])
    wx = wx.copy()
    wx["date"] = pd.to_datetime(wx["date"])

    merged = daily.merge(wx, on="date", how="left")
    merged["sky"] = merged["weathercode"].apply(_sky_label)
    return merged.dropna(subset=["sky", "residual"])


# ── Weather Driver Analyses ────────────────────────────────────────────────────

def _stl_shop_base(
    sellout: pd.DataFrame,
    df_weather: pd.DataFrame,
    wx_cols: list,
) -> pd.DataFrame:
    """Shared STL-per-shop + weather merge used by driver analyses."""
    from statsmodels.tsa.seasonal import STL

    daily = (
        sellout[["customer_code", "route", "date", "sales_quantity", "latitude", "longitude"]]
        .groupby(["customer_code", "route", "latitude", "longitude", "date"])
        .agg(sales_quantity=("sales_quantity", "sum"))
        .reset_index()
    )
    daily["date"] = pd.to_datetime(daily["date"])

    records = []
    for (cust, route, lat, lon), grp in daily.groupby(
        ["customer_code", "route", "latitude", "longitude"]
    ):
        grp = grp.sort_values("date").reset_index(drop=True)
        if len(grp) < 14:
            continue
        try:
            result = STL(grp["sales_quantity"], period=7, robust=True).fit()
            grp = grp.copy()
            grp["residual"] = result.resid
            grp["mean_sales"] = grp["sales_quantity"].mean()
            records.append(grp)
        except Exception:
            continue

    if not records:
        return pd.DataFrame()

    shop_resid = pd.concat(records, ignore_index=True)

    wx = df_weather[["date", "latitude", "longitude"] + wx_cols].copy()
    wx["date"] = pd.to_datetime(wx["date"])

    return shop_resid.merge(wx, on=["date", "latitude", "longitude"], how="left")


def compute_temp_swing_analysis(
    sellout: pd.DataFrame, df_weather: pd.DataFrame
) -> pd.DataFrame:
    """
    Day-over-day temperature change analysis.
    Returns shop-day df with swing_cat: Big Drop, Neutral, Big Rise.
    """
    merged = _stl_shop_base(sellout, df_weather, ["temperature"])
    if merged.empty:
        return pd.DataFrame()

    merged = merged.sort_values(["customer_code", "date"])
    merged["temp_delta"] = merged.groupby("customer_code")["temperature"].diff()
    merged = merged.dropna(subset=["temp_delta", "residual", "mean_sales"])
    merged = merged[merged["mean_sales"] > 0]

    def _swing_label(delta):
        if delta <= -5:
            return "Big Drop (≤−5°C)"
        if delta >= 5:
            return "Big Rise (≥+5°C)"
        return "Neutral (−5 to +5°C)"

    merged["swing_cat"] = merged["temp_delta"].apply(_swing_label)
    merged["prev_temp"] = merged["temperature"] - merged["temp_delta"]
    return merged[
        ["customer_code", "route", "date", "swing_cat", "temp_delta",
         "temperature", "prev_temp", "residual", "mean_sales", "sales_quantity"]
    ]


def compute_rain_streak_analysis(
    sellout: pd.DataFrame, df_weather: pd.DataFrame, rain_threshold_mm: float = 1.0
) -> pd.DataFrame:
    """
    Consecutive rain streak + first-dry-day bounce analysis.
    Returns shop-day df with rain_cat:
      Normal Dry Day | Rain Day 1 | Rain Day 2 | Rain Day 3 |
      Rain Day 4+ | First Dry Day (after 3+ rain)
    rain_threshold_mm: precipitation > this value counts as a rain day.
    """
    merged = _stl_shop_base(sellout, df_weather, ["precipitation"])
    if merged.empty:
        return pd.DataFrame()

    merged = merged.sort_values(["customer_code", "date"])
    merged = merged.dropna(subset=["precipitation", "residual", "mean_sales"])
    merged = merged[merged["mean_sales"] > 0]

    def _label_streaks(grp):
        grp = grp.reset_index(drop=True)
        streak = 0
        cats = []
        prev_date = None
        for _, row in grp.iterrows():
            cur_date = pd.Timestamp(row["date"])
            if prev_date is not None and (cur_date - prev_date).days > 1:
                streak = 0
            prev_date = cur_date
            is_rainy = row["precipitation"] > rain_threshold_mm
            if is_rainy:
                streak += 1
                if streak == 1:
                    cats.append("Rain Day 1")
                elif streak == 2:
                    cats.append("Rain Day 2")
                elif streak == 3:
                    cats.append("Rain Day 3")
                else:
                    cats.append("Rain Day 4+")
            else:
                prev = streak
                streak = 0
                if prev >= 3:
                    cats.append("First Dry Day (after 3+ rain)")
                else:
                    cats.append("Normal Dry Day")
        grp["rain_cat"] = cats
        return grp

    merged = merged.groupby("customer_code", group_keys=False).apply(_label_streaks)
    return merged[
        ["customer_code", "route", "date", "rain_cat", "precipitation",
         "residual", "mean_sales", "sales_quantity"]
    ]


def compute_snow_analysis(
    sellout: pd.DataFrame, df_weather: pd.DataFrame
) -> pd.DataFrame:
    """
    Snow day vs no-snow sales analysis.
    Returns shop-day df with snow_cat: Snow Day | No Snow.
    """
    merged = _stl_shop_base(sellout, df_weather, ["snowfall_sum", "temperature"])
    if merged.empty:
        return pd.DataFrame()

    merged = merged.dropna(subset=["snowfall_sum", "residual", "mean_sales"])
    merged = merged[merged["mean_sales"] > 0]
    merged["snow_cat"] = merged["snowfall_sum"].apply(
        lambda s: "Snow Day (>0 cm)" if s > 0 else "No Snow"
    )
    return merged[
        ["customer_code", "route", "date", "snow_cat", "snowfall_sum",
         "temperature", "residual", "mean_sales", "sales_quantity"]
    ]


def compute_wind_gust_analysis(
    sellout: pd.DataFrame, df_weather: pd.DataFrame
) -> pd.DataFrame:
    """
    Wind gust analysis bucketed into Calm / Moderate / Gusty.
    Returns shop-day df with gust_cat.
    """
    merged = _stl_shop_base(sellout, df_weather, ["windgusts_max"])
    if merged.empty:
        return pd.DataFrame()

    merged = merged.dropna(subset=["windgusts_max", "residual", "mean_sales"])
    merged = merged[merged["mean_sales"] > 0]

    def _gust_label(g):
        if g < 20:
            return "Calm (<20 km/h)"
        if g < 40:
            return "Moderate (20–40 km/h)"
        return "Gusty (>40 km/h)"

    merged["gust_cat"] = merged["windgusts_max"].apply(_gust_label)
    return merged[
        ["customer_code", "route", "date", "gust_cat", "windgusts_max",
         "residual", "mean_sales", "sales_quantity"]
    ]


def _run_category_ols(base_df: pd.DataFrame, cat_col: str, baseline: str) -> pd.DataFrame:
    """
    Per-shop OLS: log1p(sales) ~ category_dummies + DOW_dummies + month_dummies + trend
    Pools coefficients across shops with inverse-variance weighting.

    Returns df: category | pct_change | ci_low | ci_high | n_shops
    (% change vs baseline, with 95% CI)
    """
    import statsmodels.api as sm

    df = base_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["log_q"] = np.log1p(df["sales_quantity"])
    df["dow"] = df["date"].dt.dayofweek
    df["month_num"] = df["date"].dt.month

    all_cats = sorted(df[cat_col].dropna().unique().tolist())
    non_base_cats = [c for c in all_cats if c != baseline]
    if not non_base_cats:
        return pd.DataFrame()

    shop_coefs: dict = {c: [] for c in non_base_cats}

    for cust, grp in df.groupby("customer_code"):
        grp = grp.dropna(subset=[cat_col, "log_q"]).copy().reset_index(drop=True)
        if len(grp) < 20 or grp[cat_col].nunique() < 2:
            continue

        grp["trend"] = (grp["date"] - grp["date"].min()).dt.days

        # Category dummies — drop baseline column
        cat_dum = pd.get_dummies(grp[cat_col], prefix="cat")
        baseline_col = f"cat_{baseline}"
        cat_dum = cat_dum.drop(columns=[baseline_col], errors="ignore")
        if cat_dum.empty:
            continue

        dow_dum = pd.get_dummies(grp["dow"], prefix="dow", drop_first=True)
        mon_dum = pd.get_dummies(grp["month_num"], prefix="mon", drop_first=True)

        X = pd.concat([cat_dum, dow_dum, mon_dum, grp[["trend"]]], axis=1).astype(float)
        X = sm.add_constant(X, has_constant="add")
        y = grp["log_q"].astype(float)

        try:
            res = sm.OLS(y, X).fit()
        except Exception:
            continue

        for cat in non_base_cats:
            col = f"cat_{cat}"
            if col in res.params.index and res.bse[col] > 0:
                shop_coefs[cat].append((res.params[col], res.bse[col]))

    rows = []
    for cat in non_base_cats:
        entries = shop_coefs[cat]
        if not entries:
            continue
        w = [1 / se ** 2 for _, se in entries]
        total_w = sum(w)
        pc = sum(wi * c for wi, (c, _) in zip(w, entries)) / total_w
        pse = (1 / total_w) ** 0.5
        rows.append({
            "category": cat,
            "coef": pc,
            "se": pse,
            "pct_change": (np.exp(pc) - 1) * 100,
            "ci_low":     (np.exp(pc - 1.96 * pse) - 1) * 100,
            "ci_high":    (np.exp(pc + 1.96 * pse) - 1) * 100,
            "n_shops":    len(entries),
        })

    return pd.DataFrame(rows)


def compute_temp_swing_ols(base_df: pd.DataFrame) -> pd.DataFrame:
    return _run_category_ols(base_df, "swing_cat", "Neutral (−5 to +5°C)")


def compute_rain_streak_ols(base_df: pd.DataFrame) -> pd.DataFrame:
    return _run_category_ols(base_df, "rain_cat", "Normal Dry Day")


def compute_snow_ols(base_df: pd.DataFrame) -> pd.DataFrame:
    return _run_category_ols(base_df, "snow_cat", "No Snow")


def compute_wind_gust_ols(base_df: pd.DataFrame) -> pd.DataFrame:
    return _run_category_ols(base_df, "gust_cat", "Calm (<20 km/h)")


# ── Rain Intensity Analysis ───────────────────────────────────────────────────

def compute_rain_intensity_analysis(
    sellout: pd.DataFrame, df_weather: pd.DataFrame
) -> pd.DataFrame:
    """
    Rain intensity (mm per rainy hour) effect on sales via STL residuals.
    Categories: No Rain | Drizzle (<1 mm/h) | Moderate (1–4 mm/h) | Heavy (>4 mm/h)
    """
    merged = _stl_shop_base(sellout, df_weather, ["precipitation", "precipitation_hours"])
    if merged.empty:
        return pd.DataFrame()

    merged = merged.dropna(subset=["precipitation", "precipitation_hours", "residual", "mean_sales"])
    merged = merged[merged["mean_sales"] > 0].copy()

    # mm per rainy hour; 0 when hours reported as 0 but precipitation > 0 (data edge case → drizzle)
    merged["rain_intensity"] = np.where(
        merged["precipitation_hours"] > 0,
        merged["precipitation"] / merged["precipitation_hours"],
        np.where(merged["precipitation"] > 0, 0.0, float("nan")),
    )

    merged["intensity_cat"] = np.select(
        [
            merged["precipitation"] == 0,
            merged["rain_intensity"] < 1,
            merged["rain_intensity"] < 4,
        ],
        ["No Rain", "Drizzle (<1 mm/h)", "Moderate (1–4 mm/h)"],
        default="Heavy (>4 mm/h)",
    )

    return merged[[
        "customer_code", "route", "date", "intensity_cat", "rain_intensity",
        "precipitation", "precipitation_hours", "residual", "mean_sales", "sales_quantity",
    ]]


def compute_rain_intensity_ols(base_df: pd.DataFrame) -> pd.DataFrame:
    return _run_category_ols(base_df, "intensity_cat", "No Rain")


# ── Sunshine Fraction Analysis ────────────────────────────────────────────────

def compute_sunshine_fraction_analysis(
    sellout: pd.DataFrame, df_weather: pd.DataFrame, rain_threshold_mm: float = 1.0
) -> pd.DataFrame:
    """
    Sunshine fraction (sunshine_duration / daylight_duration) effect on sales.
    Only dry days (precipitation <= rain_threshold_mm) are included so cloud cover
    and rain effects do not contaminate each other.
    Categories: Overcast (<25%) | Partly Cloudy (25–50%) | Mostly Sunny (50–75%) | Clear (>75%)
    """
    merged = _stl_shop_base(
        sellout, df_weather, ["sunshine_duration", "daylight_duration", "precipitation"]
    )
    if merged.empty:
        return pd.DataFrame()

    merged = merged.dropna(subset=["sunshine_duration", "daylight_duration", "precipitation", "residual", "mean_sales"])
    merged = merged[(merged["mean_sales"] > 0) & (merged["daylight_duration"] > 0)].copy()

    # Keep only dry days so sunshine fraction reflects cloud cover, not rain
    merged = merged[merged["precipitation"] <= rain_threshold_mm]
    if merged.empty:
        return pd.DataFrame()

    merged["sunshine_fraction"] = merged["sunshine_duration"] / merged["daylight_duration"]

    merged["sunshine_cat"] = pd.cut(
        merged["sunshine_fraction"],
        bins=[-0.01, 0.25, 0.50, 0.75, 1.01],
        labels=["Overcast (<25%)", "Partly Cloudy (25–50%)", "Mostly Sunny (50–75%)", "Clear (>75%)"],
    ).astype(str)

    return merged[[
        "customer_code", "route", "date", "sunshine_cat", "sunshine_fraction",
        "sunshine_duration", "daylight_duration", "residual", "mean_sales", "sales_quantity",
    ]]


def compute_sunshine_fraction_ols(base_df: pd.DataFrame) -> pd.DataFrame:
    return _run_category_ols(base_df, "sunshine_cat", "Overcast (<25%)")


# ── Sunshine Threshold Transition Analysis ────────────────────────────────────

def compute_sunshine_transition_analysis(
    sellout: pd.DataFrame,
    df_weather: pd.DataFrame,
    threshold: float = 0.8,
    max_lookback: int = 7,
) -> pd.DataFrame:
    """
    For every 'Bright Day' (sunshine_fraction >= threshold), search back and forward
    up to max_lookback days to find the nearest non-bright day in each direction.
    Non-bright days are deduplicated per shop.
    Returns a shop-day df with transition_cat:
      'Day Before Bright' | 'Bright Day' | 'Day After Bright'
    Only bright days that have a found 'Day Before Bright' are emitted;
    'Day After Bright' is additionally searched for those same bright days.
    """
    merged = _stl_shop_base(
        sellout, df_weather, ["sunshine_duration", "daylight_duration", "precipitation"]
    )
    if merged.empty:
        return pd.DataFrame()

    merged = merged.dropna(
        subset=["sunshine_duration", "daylight_duration", "precipitation", "residual", "mean_sales"]
    )
    merged = merged[(merged["mean_sales"] > 0) & (merged["daylight_duration"] > 0)].copy()
    merged["sunshine_fraction"] = merged["sunshine_duration"] / merged["daylight_duration"]
    merged["date"] = pd.to_datetime(merged["date"])

    result_rows = []
    for code, grp in merged.groupby("customer_code"):
        grp = grp.sort_values("date").reset_index(drop=True)
        date_to_frac = dict(zip(grp["date"], grp["sunshine_fraction"]))
        date_to_idx  = {row["date"]: i for i, row in grp.iterrows()}

        seen_before: set = set()
        seen_after:  set = set()

        for _, row in grp.iterrows():
            d    = row["date"]
            frac = row["sunshine_fraction"]

            if frac < threshold:
                continue  # not a bright day

            # Search BACK for the nearest non-bright day FIRST
            found_before = None
            for lag in range(1, max_lookback + 1):
                prev_d    = d - pd.Timedelta(days=lag)
                prev_frac = date_to_frac.get(prev_d)
                if prev_frac is None:
                    continue          # date gap — skip, keep looking
                if prev_frac < threshold:
                    found_before = prev_d
                    break
                # prev is also bright → keep looking further back

            if found_before is None:
                continue  # no non-bright day found within lookback — skip this bright day

            # Emit the bright day
            r = row.to_dict()
            r["transition_cat"] = "Sunny"
            result_rows.append(r)

            # Emit the day before (deduplicated per shop)
            if found_before not in seen_before:
                rb = grp.iloc[date_to_idx[found_before]].to_dict()
                rb["transition_cat"] = "Cloudy"
                result_rows.append(rb)
                seen_before.add(found_before)

            # Search FORWARD for the nearest non-bright day
            for lag in range(1, max_lookback + 1):
                next_d    = d + pd.Timedelta(days=lag)
                next_frac = date_to_frac.get(next_d)
                if next_frac is None:
                    continue          # date gap — skip, keep looking
                if next_frac < threshold:
                    if next_d not in seen_after:
                        ra = grp.iloc[date_to_idx[next_d]].to_dict()
                        ra["transition_cat"] = "Day After Bright"
                        result_rows.append(ra)
                        seen_after.add(next_d)
                    break
                # next is also bright → keep looking further forward

    if not result_rows:
        return pd.DataFrame()
    return pd.DataFrame(result_rows)
