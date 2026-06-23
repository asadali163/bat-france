import pandas as pd
import numpy as np

from services.processors_base import _stl_shop_base, _run_category_ols


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
