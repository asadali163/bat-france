import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

from services.processors_base import _stl_shop_base, _run_category_ols
from services.processors_temp import _TEMP_SPLINE_KNOTS, aggregate_ols_temp


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


def compute_snow_ols(base_df: pd.DataFrame) -> pd.DataFrame:
    return _run_category_ols(base_df, "snow_cat", "No Snow")


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


def compute_wind_gust_ols(base_df: pd.DataFrame) -> pd.DataFrame:
    return _run_category_ols(base_df, "gust_cat", "Calm (<20 km/h)")
