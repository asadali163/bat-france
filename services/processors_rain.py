import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
from scipy import stats as scipy_stats
import statsmodels.formula.api as smf

from services.processors_base import _stl_shop_base, _run_category_ols


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


def compute_rain_streak_ols(base_df: pd.DataFrame) -> pd.DataFrame:
    return _run_category_ols(base_df, "rain_cat", "Normal Dry Day")


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
