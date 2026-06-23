import os
import streamlit as st
import pandas as pd
from charts.weather_charts import (
    plot_ols_wind_effect,
    plot_ols_wc_effect,
    plot_wc_pct_bars,
    plot_wind_gust,
)
from services.porcessors import (
    run_ols_wind_all_shops,
    aggregate_ols_wind,
    run_ols_wc_all_shops,
    aggregate_ols_wc,
    compute_windchill_analysis,
    compute_customer_sku_windchill,
    run_prophet_wc_all_shops,
    compute_prophet_wc_curve,
    compute_prophet_wind_curve,
    compute_wind_gust_analysis,
    compute_wind_gust_ols,
    compute_temp_contribution,
)

_TOP10_SKUS = [
    "125 - LUCKY STRIKE RED 20",
    "85548 - VOGUE L'ORIGINALE VERTE ICE 20s",
    "3721 - VOGUE ORIGINALE BLEUE EN 20",
    "85551 - LUCKY STRIKE ICE 20s",
    "6414 - VOGUE ORIGINALE PASTEL EN 20",
    "302 - PETER STUYVESANT ROUGE EN 20",
    "1201 - LUCKY STRIKE BLEU EN 20",
    "10500 - WINFIELD ROUGE 30",
    "61434 - LUCKY STRIKE RED 30",
    "71831 - LUCKY STRIKE RED LONGUES EN 20",
]

OLS_WIND_CACHE_PATH = "./data/cache/ols_wind_all.parquet"
OLS_WC_CACHE_PATH = "./data/cache/ols_wc_all.parquet"
PROPHET_WC_CACHE_PATH = "./data/cache/prophet_wc_all.parquet"
PROPHET_TEMP_CACHE_PATH = "./data/cache/prophet_temp_all.parquet"


# ── Cached helpers ────────────────────────────────────────────────────────────


@st.cache_data(show_spinner=False)
def _load_ols_wind_cache(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def _cached_agg_ols_wind(results_df: pd.DataFrame, route=None, _v=1) -> pd.DataFrame:
    return aggregate_ols_wind(results_df, route)


@st.cache_data(show_spinner=False)
def _load_ols_wc_cache(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def _cached_agg_ols_wc(results_df: pd.DataFrame, route=None, _v=1) -> pd.DataFrame:
    return aggregate_ols_wc(results_df, route)


@st.cache_data(show_spinner=False)
def _cached_wc_analysis(sellout: pd.DataFrame, df_weather: pd.DataFrame) -> pd.DataFrame:
    return compute_windchill_analysis(sellout, df_weather)


@st.cache_data(show_spinner=False)
def _cached_single_wc(
    sellout: pd.DataFrame, df_weather: pd.DataFrame, customer_code: str, sku_name: str | None
) -> pd.DataFrame:
    return compute_customer_sku_windchill(sellout, df_weather, customer_code, sku_name)


@st.cache_data(show_spinner=False)
def _cached_wc_contribution(prophet_df, sellout, df_weather, route=None):
    return compute_temp_contribution(
        prophet_df, sellout, df_weather, route,
        beta_col="prophet_apparent_temperature_mean",
        temp_col="apparent_temperature_mean",
    )


@st.cache_data(show_spinner=False)
def _cached_prophet_wind_curve(prophet_df, sellout, df_weather, route=None):
    return compute_prophet_wind_curve(prophet_df, sellout, df_weather, route)


@st.cache_data(show_spinner=False)
def _cached_prophet_wc_curve(prophet_df, sellout, df_weather, route=None):
    return compute_prophet_wc_curve(prophet_df, sellout, df_weather, route)


@st.cache_data(show_spinner=False)
def _cached_wind_gust(sellout: pd.DataFrame, df_weather: pd.DataFrame) -> pd.DataFrame:
    return compute_wind_gust_analysis(sellout, df_weather)


@st.cache_data(show_spinner=False)
def _cached_wg_ols(base_df: pd.DataFrame) -> pd.DataFrame:
    return compute_wind_gust_ols(base_df)


@st.cache_data(show_spinner=False)
def _load_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def _load_ols_cache(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def render_wind(sellout, df_weather, sellin):
    # ── Windspeed Effect (OLS Natural Spline) ────────────────────────────────
    st.markdown("---")
    show_wind_spline = st.checkbox(
        "Show Windspeed Effect — OLS Natural Spline",
        value=False,
        key="wa_show_wind_spline",
    )
    if show_wind_spline:
        st.markdown("### Windspeed Effect on Sales — OLS Natural Spline")
        st.caption(
            "Per-shop OLS with natural cubic spline for windspeed (df=5, knots at data quantiles). "
            "`log(sales) ~ cr(windspeed, df=5) + precipitation + temperature + C(dow) + C(month) + trend`. "
            "Partial effect vs mean windspeed, averaged across shops. Shaded area = 95% CI."
        )

        ols_wind_results = None
        if os.path.exists(OLS_WIND_CACHE_PATH):
            ols_wind_results = _load_ols_wind_cache(OLS_WIND_CACHE_PATH)

        if ols_wind_results is None:
            st.info(
                "No cached windspeed OLS results found. Running per-shop OLS for all shops — "
                "this takes ~10–15 minutes and will be saved for future visits."
            )
            with st.spinner("Running OLS windspeed spline for all shops…"):
                ols_wind_results = run_ols_wind_all_shops(
                    sellout, df_weather, sellin=sellin
                )
                os.makedirs(os.path.dirname(OLS_WIND_CACHE_PATH), exist_ok=True)
                ols_wind_results.to_parquet(OLS_WIND_CACHE_PATH, index=False)
            st.success("OLS windspeed analysis complete. Results saved to cache.")

        if ols_wind_results is None or ols_wind_results.empty:
            st.warning(
                "No OLS windspeed results — check that weather and sales data overlap."
            )
        else:
            _max_date_wind = pd.Timestamp(sellout["date"].max())
            global_ref_wind = float(
                df_weather[df_weather["date"] <= _max_date_wind]["windspeed"].mean()
            )
            n_wind_shops = ols_wind_results["customer_code"].nunique()
            wind_routes = sorted(ols_wind_results["route"].dropna().unique().tolist())

            agg_wind_all = _cached_agg_ols_wind(ols_wind_results)
            fig_wind_all = plot_ols_wind_effect(
                agg_wind_all,
                f"<b>Windspeed effect on sales (OLS spline) — {n_wind_shops} shops</b><br>"
                "<sub>% change vs mean windspeed, averaged across shops (shaded = 95% CI). "
                "Knots placed at data quantiles per shop (df=5)</sub>",
                ref_wind=global_ref_wind,
            )
            st.plotly_chart(
                fig_wind_all, use_container_width=True, key="wa_ols_wind_all"
            )

            st.markdown("#### Windspeed Effect by Territory")
            if wind_routes:
                sel_wind_route = st.selectbox(
                    "Select Territory (Route)", wind_routes, key="wa_ols_wind_route"
                )
                n_wind_route = int(
                    ols_wind_results[ols_wind_results["route"] == sel_wind_route][
                        "customer_code"
                    ].nunique()
                )
                agg_wind_route = _cached_agg_ols_wind(
                    ols_wind_results, route=sel_wind_route
                )
                fig_wind_route = plot_ols_wind_effect(
                    agg_wind_route,
                    f"<b>Territory {sel_wind_route} — {n_wind_route} shops</b><br>"
                    "<sub>% change vs mean windspeed, OLS spline (shaded = 95% CI)</sub>",
                    ref_wind=global_ref_wind,
                )
                st.plotly_chart(
                    fig_wind_route,
                    use_container_width=True,
                    key="wa_ols_wind_route_chart",
                )

        if st.button("🔄 Regenerate Windspeed OLS Cache", key="wa_ols_wind_regen"):
            if os.path.exists(OLS_WIND_CACHE_PATH):
                os.remove(OLS_WIND_CACHE_PATH)
            _load_ols_wind_cache.clear()
            _cached_agg_ols_wind.clear()
            st.rerun()

    # ── Wind Chill Effect (OLS Natural Spline) ────────────────────────────────
    st.markdown("---")
    show_wc_ols = st.checkbox(
        "Show Wind Chill Effect (OLS Natural Spline)",
        value=False,
        key="wa_show_wc_ols",
    )
    if show_wc_ols:
        st.markdown("### Wind Chill Effect on Sales (OLS Spline)")
        st.caption(
            "Uses **apparent_temperature_mean** (feels-like °C, accounts for wind & humidity) from the weather API. "
            "`log(sales) ~ cr(apparent_temperature_mean, knots=(0,10,20)) + precipitation + C(dow) + C(month) + trend`. "
            "Partial effect vs mean feels-like temperature, averaged across shops. Shaded area = 95% CI. "
            "Dotted lines = spline knots at 0, 10, 20°C (Cold/Cool/Mild/Warm boundaries)."
        )

        ols_wc_results = None
        if os.path.exists(OLS_WC_CACHE_PATH):
            ols_wc_results = _load_ols_wc_cache(OLS_WC_CACHE_PATH)

        if ols_wc_results is None:
            st.info(
                "No cached OLS wind chill results found. Running per-shop OLS for all shops — "
                "this takes ~10–15 minutes and will be saved for future visits."
            )
            with st.spinner("Running OLS wind chill spline for all shops…"):
                ols_wc_results = run_ols_wc_all_shops(sellout, df_weather, sellin=sellin)
                os.makedirs(os.path.dirname(OLS_WC_CACHE_PATH), exist_ok=True)
                ols_wc_results.to_parquet(OLS_WC_CACHE_PATH, index=False)
            st.success("OLS wind chill analysis complete. Results saved to cache.")

        if ols_wc_results is None or ols_wc_results.empty:
            st.warning("No OLS wind chill results — check that weather and sales data overlap.")
        else:
            _max_date_wc = pd.Timestamp(sellout["date"].max())
            global_ref_wc = float(
                df_weather[df_weather["date"] <= _max_date_wc]["apparent_temperature_mean"].mean()
            )
            n_wc_ols_shops = ols_wc_results["customer_code"].nunique()
            wc_ols_routes  = sorted(ols_wc_results["route"].dropna().unique().tolist())

            agg_wc_ols_all = _cached_agg_ols_wc(ols_wc_results)
            fig_wc_ols_all = plot_ols_wc_effect(
                agg_wc_ols_all,
                f"<b>Wind Chill effect on sales (OLS spline) — {n_wc_ols_shops} shops</b><br>"
                "<sub>% change vs mean feels-like temperature, averaged across shops (shaded = 95% CI). "
                "Dotted lines = spline knots at 0, 10, 20°C</sub>",
                ref_wc=global_ref_wc,
            )
            st.plotly_chart(fig_wc_ols_all, use_container_width=True, key="wa_ols_wc_all")

            st.markdown("#### OLS Wind Chill Effect by Territory")
            if wc_ols_routes:
                sel_wc_ols_route = st.selectbox(
                    "Select Territory (Route)", wc_ols_routes, key="wa_ols_wc_route"
                )
                n_wc_ols_route = int(
                    ols_wc_results[ols_wc_results["route"] == sel_wc_ols_route][
                        "customer_code"
                    ].nunique()
                )
                agg_wc_ols_route = _cached_agg_ols_wc(
                    ols_wc_results, route=sel_wc_ols_route
                )
                fig_wc_ols_route = plot_ols_wc_effect(
                    agg_wc_ols_route,
                    f"<b>Territory {sel_wc_ols_route} — {n_wc_ols_route} shops</b><br>"
                    "<sub>% change vs mean feels-like temperature, OLS spline (shaded = 95% CI)</sub>",
                    ref_wc=global_ref_wc,
                )
                st.plotly_chart(
                    fig_wc_ols_route, use_container_width=True, key="wa_ols_wc_route_chart"
                )

        if st.button("🔄 Regenerate Wind Chill OLS Cache", key="wa_ols_wc_regen"):
            if os.path.exists(OLS_WC_CACHE_PATH):
                os.remove(OLS_WC_CACHE_PATH)
            _load_ols_wc_cache.clear()
            _cached_agg_ols_wc.clear()
            st.rerun()

    # ── Wind Chill Effect — Prophet (linear regressor) ───────────────────────
    st.markdown("---")
    show_wc_prophet = st.checkbox(
        "Show Wind Chill Effect — Prophet (linear regressor)",
        value=False,
        key="wa_show_wc_prophet",
    )
    if show_wc_prophet:
        st.markdown("### Wind Chill Effect on Sales — Prophet")
        st.caption(
            "Uses **apparent_temperature_mean** (feels-like °C) as a linear Prophet regressor. "
            "Prophet treats regressors as linear, so this produces a straight line — compare with "
            "the OLS natural spline above to see what non-linearity the spline captures. "
            "Each shop's β is normalised by its median daily sales before averaging."
        )

        wc_prophet_df = None
        if os.path.exists(PROPHET_WC_CACHE_PATH):
            wc_prophet_df = _load_parquet(PROPHET_WC_CACHE_PATH)

        if wc_prophet_df is None or wc_prophet_df.empty:
            st.info(
                "No Wind Chill Prophet cache found. "
                "Click **Run** to fit per-shop Prophet models with apparent_temperature_mean — "
                "this takes ~15–20 minutes and will be saved for future visits."
            )
            if st.button("▶ Run Wind Chill Prophet for All Shops", key="wa_wc_prophet_run"):
                prog_bar_wcp = st.progress(0)
                prog_text_wcp = st.empty()

                def _on_progress_wcp(current, total, fitted, skipped):
                    prog_bar_wcp.progress(current / total)
                    prog_text_wcp.text(
                        f"Shop {current} / {total}  —  fitted: {fitted}  |  skipped: {skipped}"
                    )

                wc_prophet_df = run_prophet_wc_all_shops(
                    sellout, df_weather, sellin=sellin, progress_callback=_on_progress_wcp
                )
                os.makedirs("./data/cache", exist_ok=True)
                wc_prophet_df.to_parquet(PROPHET_WC_CACHE_PATH, index=False)
                _load_parquet.clear()
                st.rerun()
        elif "prophet_apparent_temperature_mean" not in wc_prophet_df.columns:
            st.warning("Cache is missing the wind chill column — click regenerate below.")
        else:
            _max_date_wcp = pd.Timestamp(sellout["date"].max())
            global_ref_wcp = float(
                df_weather[df_weather["date"] <= _max_date_wcp]["apparent_temperature_mean"].mean()
            )
            wcp_routes = sorted(wc_prophet_df["route"].dropna().unique().tolist())
            n_wcp = wc_prophet_df["customer_code"].nunique()

            curve_wcp_all = _cached_prophet_wc_curve(wc_prophet_df, sellout, df_weather)
            fig_wcp_all = plot_ols_wc_effect(
                curve_wcp_all,
                f"<b>Wind Chill effect on sales (Prophet) — {n_wcp} shops</b><br>"
                "<sub>% change vs mean feels-like temperature (straight line = Prophet linear β)</sub>",
                ref_wc=global_ref_wcp,
                show_knots=False,
            )
            st.plotly_chart(fig_wcp_all, use_container_width=True, key="wa_prophet_wc_all")

            st.markdown("#### Prophet Wind Chill Effect by Territory")
            if wcp_routes:
                sel_wcp_route = st.selectbox(
                    "Select Territory (Route)", wcp_routes, key="wa_prophet_wc_route"
                )
                n_wcp_r = int(
                    wc_prophet_df[wc_prophet_df["route"] == sel_wcp_route][
                        "customer_code"
                    ].nunique()
                )
                curve_wcp_route = _cached_prophet_wc_curve(
                    wc_prophet_df, sellout, df_weather, route=sel_wcp_route
                )
                fig_wcp_route = plot_ols_wc_effect(
                    curve_wcp_route,
                    f"<b>Territory {sel_wcp_route} — {n_wcp_r} shops (Prophet)</b><br>"
                    "<sub>% change vs mean feels-like temperature</sub>",
                    ref_wc=global_ref_wcp,
                    show_knots=False,
                )
                st.plotly_chart(
                    fig_wcp_route, use_container_width=True, key="wa_prophet_wc_route_chart"
                )

        if st.button("🔄 Regenerate Wind Chill Prophet Cache", key="wa_wc_prophet_regen"):
            if os.path.exists(PROPHET_WC_CACHE_PATH):
                os.remove(PROPHET_WC_CACHE_PATH)
            _load_parquet.clear()
            _cached_prophet_wc_curve.clear()
            st.rerun()

    # ── Windspeed Effect — Prophet (linear regressor) ────────────────────────
    st.markdown("---")
    show_wind_prophet = st.checkbox(
        "Show Windspeed Effect — Prophet (linear regressor)",
        value=False,
        key="wa_show_wind_prophet",
    )
    if show_wind_prophet:
        st.markdown("### Windspeed Effect on Sales — Prophet")
        st.caption(
            "Windspeed β extracted from per-shop Prophet models (already computed for the "
            "Temperature → Prophet section). Prophet treats regressors as **linear**, so "
            "this curve is always a straight line — compare with the OLS spline above to see "
            "what non-linearity the spline captures."
        )
        prophet_wind_df = None
        if os.path.exists(PROPHET_TEMP_CACHE_PATH):
            prophet_wind_df = _load_parquet(PROPHET_TEMP_CACHE_PATH)

        if prophet_wind_df is None or prophet_wind_df.empty:
            st.info(
                "No Prophet cache found. Run the **Temperature → Prophet** section first — "
                "it fits windspeed as a regressor at the same time, so no extra computation needed."
            )
        elif "prophet_windspeed" not in prophet_wind_df.columns:
            st.warning(
                "prophet_windspeed column not found in cache — regenerate the Prophet temperature cache."
            )
        else:
            _max_date_pw = pd.Timestamp(sellout["date"].max())
            ref_wind_pw = float(
                df_weather[df_weather["date"] <= _max_date_pw]["windspeed"].mean()
            )
            wind_routes_p = sorted(prophet_wind_df["route"].dropna().unique().tolist())

            agg_pw_all = _cached_prophet_wind_curve(
                prophet_wind_df, sellout, df_weather
            )
            n_pw = int(agg_pw_all["n_shops"].iloc[0]) if not agg_pw_all.empty else 0
            fig_pw_all = plot_ols_wind_effect(
                agg_pw_all,
                f"<b>Windspeed effect on sales (Prophet) — {n_pw} shops</b><br>"
                "<sub>% change vs mean windspeed</sub>",
                ref_wind=ref_wind_pw,
            )
            st.plotly_chart(
                fig_pw_all, use_container_width=True, key="wa_prophet_wind_all"
            )

            st.markdown("#### Prophet Windspeed Effect by Territory")
            if wind_routes_p:
                sel_pw_route = st.selectbox(
                    "Select Territory (Route)",
                    wind_routes_p,
                    key="wa_prophet_wind_route",
                )
                agg_pw_route = _cached_prophet_wind_curve(
                    prophet_wind_df, sellout, df_weather, route=sel_pw_route
                )
                n_pw_r = (
                    int(agg_pw_route["n_shops"].iloc[0])
                    if not agg_pw_route.empty
                    else 0
                )
                fig_pw_route = plot_ols_wind_effect(
                    agg_pw_route,
                    f"<b>Territory {sel_pw_route} — {n_pw_r} shops (Prophet)</b><br>"
                    "<sub>% change vs mean windspeed. </sub>",
                    ref_wind=ref_wind_pw,
                )
                st.plotly_chart(
                    fig_pw_route,
                    use_container_width=True,
                    key="wa_prophet_wind_route_chart",
                )

    # ── Wind Chill Analysis ───────────────────────────────────────────────────
    st.markdown("---")
    show_wc = st.checkbox(
        "Show Wind Chill (Feels-Like Temperature) Effect on Sales",
        value=False,
        key="wa_show_wc",
    )
    if show_wc:
        st.markdown("### Wind Chill Effect on Sales")
        st.caption(
            "Uses **apparent_temperature_mean** (feels-like °C, accounting for wind) from the weather API. "
            "**Cold** < 0°C · **Cool** 0–10°C · **Mild** 10–20°C *(baseline = 0%)* · **Warm** > 20°C. "
            "STL decomposition removes trend + weekly seasonality; residual isolates the temperature-feel signal. "
            "**% change = (category residual − Mild residual) / shop mean sales × 100**. "
            "Left = raw sales · Right = STL residual."
        )

        WC_CACHE_PATH = "./data/cache/wc_analysis.parquet"

        wc_df = None
        if os.path.exists(WC_CACHE_PATH):
            wc_df = _load_ols_cache(WC_CACHE_PATH)
            if wc_df is not None and "mean_sales" not in wc_df.columns:
                os.remove(WC_CACHE_PATH)
                _load_ols_cache.clear()
                _cached_wc_analysis.clear()
                wc_df = None

        if wc_df is None:
            st.info(
                "No wind chill cache found. Running STL decomposition for all shops — "
                "this may take a few minutes and will be saved for future visits."
            )
            with st.spinner("Running STL for all shops…"):
                wc_df = _cached_wc_analysis(sellout, df_weather)
                if not wc_df.empty:
                    os.makedirs(os.path.dirname(WC_CACHE_PATH), exist_ok=True)
                    wc_df.to_parquet(WC_CACHE_PATH, index=False)
            st.success("Wind chill analysis complete. Results saved to cache.")

        if wc_df is None or wc_df.empty:
            st.warning(
                "No wind chill data — check that sellout and weather data share "
                "matching (date, latitude, longitude) keys."
            )
        else:
            _WC_CATS = ["Cold", "Cool", "Mild", "Warm"]

            def _shop_wc_pct(df):
                """Per-shop % change vs Mild baseline for all wind chill categories."""
                records = []
                for cust, grp in df.groupby("customer_code"):
                    route = grp["route"].iloc[0]
                    shop_mean = float(grp["mean_sales"].iloc[0])
                    if shop_mean == 0:
                        continue
                    cat_raw = grp.groupby("wc_cat")["sales_quantity"].mean()
                    cat_resid = grp.groupby("wc_cat")["residual"].mean()
                    bl_raw = cat_raw.get("Mild", float("nan"))
                    bl_resid = cat_resid.get("Mild", float("nan"))
                    if pd.isna(bl_raw) or pd.isna(bl_resid):
                        continue
                    for cat in _WC_CATS:
                        rv = cat_raw.get(cat, float("nan"))
                        dv = cat_resid.get(cat, float("nan"))
                        raw_pct = (rv - bl_raw) / shop_mean * 100 if pd.notna(rv) else float("nan")
                        res_pct = (dv - bl_resid) / shop_mean * 100 if pd.notna(dv) else float("nan")
                        n = int((grp["wc_cat"] == cat).sum())
                        records.append({
                            "customer_code": cust, "route": route, "wc_cat": cat,
                            "raw_pct": raw_pct, "resid_pct": res_pct, "n_days": n,
                        })
                return pd.DataFrame(records)

            shop_wc_df = _shop_wc_pct(wc_df)
            if shop_wc_df.empty:
                st.warning("Could not compute % change — too few shops have Mild days as baseline.")
            else:
                n_wc_shops = shop_wc_df["customer_code"].nunique()

                def _agg_wc(df):
                    return (
                        df.groupby("wc_cat")
                        .agg(
                            raw_pct=("raw_pct", "mean"),
                            sem_raw=("raw_pct", "sem"),
                            resid_pct=("resid_pct", "mean"),
                            sem_resid=("resid_pct", "sem"),
                            n_shop_days=("n_days", "sum"),
                        )
                        .reset_index()
                    )

                def _make_wc_bar_df(agg, pct_col, sem_col):
                    return agg.rename(columns={pct_col: "pct_change", sem_col: "sem_pct"})

                # ── 1. All Shops ──────────────────────────────────────────────
                st.markdown(f"#### All Shops — {n_wc_shops} shops")
                agg_wc_all = _agg_wc(shop_wc_df)
                c1, c2 = st.columns(2)
                with c1:
                    st.plotly_chart(
                        plot_wc_pct_bars(
                            _make_wc_bar_df(agg_wc_all, "raw_pct", "sem_raw"),
                            "Wind Chill Effect — Raw Sales",
                            n_wc_shops,
                            subtitle=f"Raw mean sales % vs Mild — {n_wc_shops} shops  ·  error bars = SE across shops",
                        ),
                        use_container_width=True,
                        key="wa_wc_all_raw",
                    )
                with c2:
                    st.plotly_chart(
                        plot_wc_pct_bars(
                            _make_wc_bar_df(agg_wc_all, "resid_pct", "sem_resid"),
                            "Wind Chill Effect — STL Residual",
                            n_wc_shops,
                            subtitle=f"STL residual % vs Mild — {n_wc_shops} shops  ·  error bars = SE across shops",
                        ),
                        use_container_width=True,
                        key="wa_wc_all_resid",
                    )

                # ── 2. By Territory ───────────────────────────────────────────
                st.markdown("#### By Territory")
                wc_route_options = sorted(shop_wc_df["route"].dropna().unique().tolist())
                sel_wc_route = st.selectbox(
                    "Select Territory", wc_route_options, key="wa_wc_terr_sel"
                )
                terr_wc_df = shop_wc_df[shop_wc_df["route"] == sel_wc_route]
                n_wc_terr = terr_wc_df["customer_code"].nunique()
                agg_wc_terr = _agg_wc(terr_wc_df)
                c1, c2 = st.columns(2)
                with c1:
                    st.plotly_chart(
                        plot_wc_pct_bars(
                            _make_wc_bar_df(agg_wc_terr, "raw_pct", "sem_raw"),
                            f"Territory {sel_wc_route} — Raw Sales",
                            n_wc_terr,
                            subtitle=f"Raw mean sales % vs Mild — {n_wc_terr} shops  ·  error bars = SE across shops",
                        ),
                        use_container_width=True,
                        key="wa_wc_terr_raw",
                    )
                with c2:
                    st.plotly_chart(
                        plot_wc_pct_bars(
                            _make_wc_bar_df(agg_wc_terr, "resid_pct", "sem_resid"),
                            f"Territory {sel_wc_route} — STL Residual",
                            n_wc_terr,
                            subtitle=f"STL residual % vs Mild — {n_wc_terr} shops  ·  error bars = SE across shops",
                        ),
                        use_container_width=True,
                        key="wa_wc_terr_resid",
                    )

                # ── 3. By Shop + SKU ──────────────────────────────────────────
                st.markdown("#### By Shop & SKU")
                cold_rank = (
                    shop_wc_df[shop_wc_df["wc_cat"] == "Cold"]
                    .sort_values("resid_pct")["customer_code"]
                    .tolist()
                )
                remaining_wc = [
                    c for c in shop_wc_df["customer_code"].unique()
                    if c not in cold_rank
                ]
                sorted_wc_shops = cold_rank + remaining_wc

                wcs1, wcs2 = st.columns(2)
                with wcs1:
                    sel_wc_shop = st.selectbox(
                        "Select Shop (sorted by Cold impact, most negative first)",
                        sorted_wc_shops,
                        key="wa_wc_shop_sel",
                    )
                with wcs2:
                    sel_wc_sku = st.selectbox(
                        "Select SKU",
                        ["All"] + _TOP10_SKUS,
                        key="wa_wc_shop_sku",
                    )

                sku_wc_filter = sel_wc_sku if sel_wc_sku != "All" else None
                with st.spinner("Loading wind chill data for selected shop & SKU…"):
                    single_wc_df = _cached_single_wc(
                        sellout, df_weather, sel_wc_shop, sku_wc_filter
                    )

                if single_wc_df.empty:
                    st.info(
                        "No data for this shop / SKU combination "
                        "(needs ≥14 days with weather data)."
                    )
                else:
                    wc_raw = single_wc_df.groupby("wc_cat")["sales_quantity"].mean()
                    wc_resid = single_wc_df.groupby("wc_cat")["residual"].mean()
                    shop_wc_mean = float(single_wc_df["mean_sales"].iloc[0])
                    bl_raw = wc_raw.get("Mild", float("nan"))
                    bl_resid = wc_resid.get("Mild", float("nan"))

                    raw_wc_rows, res_wc_rows = [], []
                    for cat in _WC_CATS:
                        n = int((single_wc_df["wc_cat"] == cat).sum())
                        rv = wc_raw.get(cat, float("nan"))
                        dv = wc_resid.get(cat, float("nan"))
                        raw_pct = (rv - bl_raw) / shop_wc_mean * 100 if pd.notna(rv) and pd.notna(bl_raw) else float("nan")
                        res_pct = (dv - bl_resid) / shop_wc_mean * 100 if pd.notna(dv) and pd.notna(bl_resid) else float("nan")
                        raw_wc_rows.append({"wc_cat": cat, "pct_change": raw_pct, "sem_pct": float("nan"), "n_shop_days": n})
                        res_wc_rows.append({"wc_cat": cat, "pct_change": res_pct, "sem_pct": float("nan"), "n_shop_days": n})

                    sku_wc_label = sel_wc_sku
                    c1, c2 = st.columns(2)
                    with c1:
                        st.plotly_chart(
                            plot_wc_pct_bars(
                                pd.DataFrame(raw_wc_rows),
                                f"{sel_wc_shop} · {sku_wc_label} — Raw Sales",
                                1,
                                subtitle="Raw mean daily sales % vs Mild (Mild = 0)",
                            ),
                            use_container_width=True,
                            key="wa_wc_shop_raw",
                        )
                    with c2:
                        st.plotly_chart(
                            plot_wc_pct_bars(
                                pd.DataFrame(res_wc_rows),
                                f"{sel_wc_shop} · {sku_wc_label} — STL Residual",
                                1,
                                subtitle="STL residual % vs Mild — trend & weekly seasonality removed",
                            ),
                            use_container_width=True,
                            key="wa_wc_shop_resid",
                        )

        if st.button("🔄 Regenerate Wind Chill Cache", key="wa_wc_regen"):
            WC_CACHE_PATH = "./data/cache/wc_analysis.parquet"
            if os.path.exists(WC_CACHE_PATH):
                os.remove(WC_CACHE_PATH)
            _load_ols_cache.clear()
            _cached_wc_analysis.clear()
            st.rerun()

