import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

from services.processors_base import _stl_shop_base, _run_category_ols


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


def compute_temp_swing_ols(base_df: pd.DataFrame) -> pd.DataFrame:
    return _run_category_ols(base_df, "swing_cat", "Neutral (−5 to +5°C)")


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
