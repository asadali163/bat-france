import pandas as pd
import numpy as np


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
