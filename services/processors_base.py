import pandas as pd
import numpy as np
from decimal import Decimal, ROUND_HALF_UP
from statsmodels.tsa.seasonal import STL
import statsmodels.api as sm


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
    spike_df["event_same_day"]   = False
    spike_df["event_day_before"] = False
    spike_df["event_day_after"]  = False

    for (lat_r, lon_r), group in spike_df.groupby(["lat_r", "lon_r"]):
        event_dates = event_lookup.get((lat_r, lon_r), set())
        idx = group.index
        spike_df.loc[idx, "event_same_day"]   = group["date_norm"].isin(event_dates).values
        spike_df.loc[idx, "event_day_before"] = (group["date_norm"] - pd.Timedelta(days=1)).isin(event_dates).values
        spike_df.loc[idx, "event_day_after"]  = (group["date_norm"] + pd.Timedelta(days=1)).isin(event_dates).values

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
