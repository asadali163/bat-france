import os
import streamlit as st
import pandas as pd
from services.data_loader import load_weather_data
from services.filters import get_customer_list_weather, get_fmc_only
import calendar as _cal
from charts.weather_charts import (
    plot_customer_weather,
    plot_rain_band_chart,
    plot_ols_rain_chart,
    plot_ols_rain_effect,
    plot_ols_rain_band_effect,
    plot_ols_temp_effect,
    plot_ols_wind_effect,
    plot_temp_contribution,
    plot_prophet_seasonality,
    plot_daily_rainfall,
    plot_sky_pct_bars,
    plot_sky_territory_bars,
    plot_sky_shop_bars,
    plot_wc_pct_bars,
    plot_ols_wc_effect,
    plot_gap_pct_bars,
    plot_gap_monthly,
    plot_storm_pct_bars,
    plot_storm_monthly,
    plot_transition_pair,
    plot_transition_monthly,
    plot_sunny_temp_combined,
    plot_temp_swing,
    plot_rain_streak,
    plot_snow_analysis,
    plot_wind_gust,
    plot_ols_category_effect,
    plot_rain_intensity,
    plot_sunshine_fraction,
)
from services.porcessors import (
    rain_band_processor,
    ols_rain_processor,
    run_ols_rain_all_shops,
    aggregate_ols_rain,
    run_ols_rain_band_all_shops,
    aggregate_ols_rain_band,
    run_ols_temp_all_shops,
    aggregate_ols_temp,
    run_ols_wind_all_shops,
    aggregate_ols_wind,
    compute_prophet_rain_curve,
    compute_prophet_wind_curve,
    run_prophet_ols_all_shops,
    run_prophet_temp_all_shops,
    compute_prophet_temp_curve,
    compute_temp_contribution,
    compute_prophet_seasonality,
    compute_sky_analysis,
    compute_customer_sku_sky,
    compute_windchill_analysis,
    compute_customer_sku_windchill,
    run_ols_wc_all_shops,
    aggregate_ols_wc,
    run_prophet_wc_all_shops,
    compute_prophet_wc_curve,
    run_ols_fl_all_shops,
    aggregate_ols_fl,
    run_prophet_fl_all_shops,
    compute_prophet_fl_curve,
    compute_gap_analysis,
    compute_storm_analysis,
    compute_sunny_transition_analysis,
    compute_sunny_temp_combined,
    compute_temp_swing_analysis,
    compute_rain_streak_analysis,
    compute_snow_analysis,
    compute_wind_gust_analysis,
    compute_temp_swing_ols,
    compute_rain_streak_ols,
    compute_snow_ols,
    compute_wind_gust_ols,
    compute_rain_intensity_analysis,
    compute_rain_intensity_ols,
    compute_sunshine_fraction_analysis,
    compute_sunshine_fraction_ols,
    compute_sunshine_transition_analysis,
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

OLS_CACHE_PATH = "./data/cache/ols_rain_all.parquet"
BAND_OLS_CACHE_PATH = "./data/cache/ols_rain_band_all.parquet"
OLS_TEMP_CACHE_PATH = "./data/cache/ols_temp_all.parquet"
OLS_WIND_CACHE_PATH = "./data/cache/ols_wind_all.parquet"
OLS_WC_CACHE_PATH      = "./data/cache/ols_wc_all.parquet"
PROPHET_WC_CACHE_PATH  = "./data/cache/prophet_wc_all.parquet"
OLS_FL_CACHE_PATH      = "./data/cache/ols_fl_all.parquet"
PROPHET_FL_CACHE_PATH  = "./data/cache/prophet_fl_all.parquet"
PROPHET_CACHE_PATH = "./data/cache/prophet_ols_all.parquet"
SEASONALITY_CACHE_PATH = "./data/cache/prophet_seasonality.parquet"
PROPHET_TEMP_CACHE_PATH = "./data/cache/prophet_temp_all.parquet"


# ── Cached helpers ────────────────────────────────────────────────────────────


@st.cache_data(show_spinner=False)
def _merge_weather(sellout_fmc: pd.DataFrame, df_weather: pd.DataFrame) -> pd.DataFrame:
    merged = sellout_fmc.merge(
        df_weather, on=["date", "latitude", "longitude"], how="left"
    )
    merged["is_rain"] = merged["precipitation"] > 4
    return merged


@st.cache_data(show_spinner=False)
def _cached_weather_chart(customer_df: pd.DataFrame, rain_range: tuple, robust: bool):
    return plot_customer_weather(customer_df, rain_range, robust)


@st.cache_data(show_spinner=False)
def _cached_rain_band(customer_df: pd.DataFrame):
    return rain_band_processor(customer_df)


@st.cache_data(show_spinner=False)
def _cached_ols(customer_df: pd.DataFrame, _v=3):
    # _v: bump this integer whenever ols_rain_processor logic changes to bust stale cache
    return ols_rain_processor(customer_df)


@st.cache_data(show_spinner=False)
def _load_ols_cache(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def _cached_agg_ols(results_df: pd.DataFrame, route=None, _v=2) -> pd.DataFrame:
    # _v: bump when aggregate_ols_rain return format changes
    return aggregate_ols_rain(results_df, route)


@st.cache_data(show_spinner=False)
def _load_band_ols_cache(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def _cached_agg_band_ols(results_df: pd.DataFrame, route=None, _v=1) -> pd.DataFrame:
    return aggregate_ols_rain_band(results_df, route)


@st.cache_data(show_spinner=False)
def _cached_prophet_temp_curve(prophet_df, sellout, df_weather, route=None):
    return compute_prophet_temp_curve(prophet_df, sellout, df_weather, route)


@st.cache_data(show_spinner=False)
def _load_ols_temp_cache(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def _cached_agg_ols_temp(results_df: pd.DataFrame, route=None, _v=1) -> pd.DataFrame:
    return aggregate_ols_temp(results_df, route)


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
def _cached_prophet_rain_curve(prophet_df, sellout, route=None):
    return compute_prophet_rain_curve(prophet_df, sellout, route)


@st.cache_data(show_spinner=False)
def _cached_prophet_wind_curve(prophet_df, sellout, df_weather, route=None):
    return compute_prophet_wind_curve(prophet_df, sellout, df_weather, route)


@st.cache_data(show_spinner=False)
def _cached_prophet_wc_curve(prophet_df, sellout, df_weather, route=None):
    return compute_prophet_wc_curve(prophet_df, sellout, df_weather, route)


@st.cache_data(show_spinner=False)
def _load_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def _cached_sky_analysis(sellout: pd.DataFrame, df_weather: pd.DataFrame) -> pd.DataFrame:
    return compute_sky_analysis(sellout, df_weather)


@st.cache_data(show_spinner=False)
def _cached_single_sky(
    sellout: pd.DataFrame, df_weather: pd.DataFrame, customer_code: str, sku_name: str | None
) -> pd.DataFrame:
    return compute_customer_sku_sky(sellout, df_weather, customer_code, sku_name)


@st.cache_data(show_spinner=False)
def _cached_gap_analysis(sky_df: pd.DataFrame, df_weather: pd.DataFrame) -> pd.DataFrame:
    return compute_gap_analysis(sky_df, df_weather)


@st.cache_data(show_spinner=False)
def _cached_storm_analysis(sky_df: pd.DataFrame, df_weather: pd.DataFrame) -> pd.DataFrame:
    return compute_storm_analysis(sky_df, df_weather)


@st.cache_data(show_spinner=False)
def _cached_sunny_transition(sky_df: pd.DataFrame) -> pd.DataFrame:
    return compute_sunny_transition_analysis(sky_df)


@st.cache_data(show_spinner=False)
def _cached_wc_analysis(sellout: pd.DataFrame, df_weather: pd.DataFrame) -> pd.DataFrame:
    return compute_windchill_analysis(sellout, df_weather)


@st.cache_data(show_spinner=False)
def _cached_single_wc(
    sellout: pd.DataFrame, df_weather: pd.DataFrame, customer_code: str, sku_name: str | None
) -> pd.DataFrame:
    return compute_customer_sku_windchill(sellout, df_weather, customer_code, sku_name)


@st.cache_data(show_spinner=False)
def _cached_temp_contribution(prophet_df, sellout, df_weather, route=None):
    return compute_temp_contribution(prophet_df, sellout, df_weather, route)


@st.cache_data(show_spinner=False)
def _cached_wc_contribution(prophet_df, sellout, df_weather, route=None):
    return compute_temp_contribution(
        prophet_df, sellout, df_weather, route,
        beta_col="prophet_apparent_temperature_mean",
        temp_col="apparent_temperature_mean",
    )


@st.cache_data(show_spinner=False)
def _load_ols_fl_cache(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def _cached_agg_ols_fl(results_df: pd.DataFrame, route=None, _v=1) -> pd.DataFrame:
    return aggregate_ols_fl(results_df, route)


@st.cache_data(show_spinner=False)
def _cached_prophet_fl_curve(prophet_df, sellout, df_weather, route=None):
    return compute_prophet_fl_curve(prophet_df, sellout, df_weather, route)


@st.cache_data(show_spinner=False)
def _cached_sunny_temp_combined(trans_df: pd.DataFrame, df_weather: pd.DataFrame, _v=3) -> pd.DataFrame:
    return compute_sunny_temp_combined(trans_df, df_weather)


@st.cache_data(show_spinner=False)
def _cached_temp_swing(sellout: pd.DataFrame, df_weather: pd.DataFrame, _v=2) -> pd.DataFrame:
    return compute_temp_swing_analysis(sellout, df_weather)


@st.cache_data(show_spinner=False)
def _cached_rain_streak(sellout: pd.DataFrame, df_weather: pd.DataFrame, threshold: float = 1.0) -> pd.DataFrame:
    return compute_rain_streak_analysis(sellout, df_weather, rain_threshold_mm=threshold)


@st.cache_data(show_spinner=False)
def _cached_snow(sellout: pd.DataFrame, df_weather: pd.DataFrame) -> pd.DataFrame:
    return compute_snow_analysis(sellout, df_weather)


@st.cache_data(show_spinner=False)
def _cached_wind_gust(sellout: pd.DataFrame, df_weather: pd.DataFrame) -> pd.DataFrame:
    return compute_wind_gust_analysis(sellout, df_weather)


@st.cache_data(show_spinner=False)
def _cached_ts_ols(base_df: pd.DataFrame) -> pd.DataFrame:
    return compute_temp_swing_ols(base_df)


@st.cache_data(show_spinner=False)
def _cached_rs_ols(base_df: pd.DataFrame) -> pd.DataFrame:
    return compute_rain_streak_ols(base_df)


@st.cache_data(show_spinner=False)
def _cached_sn_ols(base_df: pd.DataFrame) -> pd.DataFrame:
    return compute_snow_ols(base_df)


@st.cache_data(show_spinner=False)
def _cached_wg_ols(base_df: pd.DataFrame) -> pd.DataFrame:
    return compute_wind_gust_ols(base_df)


@st.cache_data(show_spinner=False)
def _cached_rain_intensity(sellout: pd.DataFrame, df_weather: pd.DataFrame) -> pd.DataFrame:
    return compute_rain_intensity_analysis(sellout, df_weather)


@st.cache_data(show_spinner=False)
def _cached_ri_ols(base_df: pd.DataFrame) -> pd.DataFrame:
    return compute_rain_intensity_ols(base_df)


@st.cache_data(show_spinner=False)
def _cached_sunshine_fraction(sellout: pd.DataFrame, df_weather: pd.DataFrame, threshold: float = 1.0) -> pd.DataFrame:
    return compute_sunshine_fraction_analysis(sellout, df_weather, rain_threshold_mm=threshold)


@st.cache_data(show_spinner=False)
def _cached_sf_ols(base_df: pd.DataFrame) -> pd.DataFrame:
    return compute_sunshine_fraction_ols(base_df)


@st.cache_data(show_spinner=False)
def _cached_sunshine_transition(
    sellout: pd.DataFrame, df_weather: pd.DataFrame, threshold: float = 0.8, _v=3
) -> pd.DataFrame:
    return compute_sunshine_transition_analysis(sellout, df_weather, threshold=threshold)


@st.cache_data(show_spinner=False)
def _cached_fl_contribution(prophet_df, sellout, df_weather, route=None):
    return compute_temp_contribution(
        prophet_df, sellout, df_weather, route,
        beta_col="prophet_apparent_temperature_mean",
        temp_col="apparent_temperature_mean",
    )


@st.cache_data(show_spinner=False)
def _cached_seasonality(seasonality_df, route=None):
    return compute_prophet_seasonality(seasonality_df, route)


# ── Main render ───────────────────────────────────────────────────────────────


def render(sellout, sellin):
    df_weather = load_weather_data()

    sellin = get_fmc_only(sellin)
    sellout = get_fmc_only(sellout)

    # ── Customer-level analysis ───────────────────────────────────────────────
    sellout_merged = _merge_weather(sellout, df_weather)

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        customer_list = get_customer_list_weather(sellout_merged)
        selected_customer = st.selectbox(
            "Select Customer", customer_list, key="customer_weather"
        )
    with col2:
        selected_sku = st.selectbox(
            "Select SKU", ["All"] + _TOP10_SKUS, key="wa_sku_sel"
        )

    selected_customer_df = sellout_merged[
        sellout_merged["customer_code"] == selected_customer
    ]
    if selected_sku != "All":
        selected_customer_df = selected_customer_df[
            selected_customer_df["sku_name"] == selected_sku
        ]

    with col3:
        customer_max_rain = float(selected_customer_df["precipitation"].max()) if not selected_customer_df.empty else 1.0
        rain_range = st.slider(
            "Rain Range (mm)",
            min_value=0.0,
            max_value=max(customer_max_rain, 0.1),
            value=(0.0, max(customer_max_rain, 0.1)),
            step=0.1,
        )
        robust = st.checkbox("Robust STL", value=True)

    fig = _cached_weather_chart(selected_customer_df, rain_range, robust)
    st.plotly_chart(fig, width="stretch")

    st.markdown("---")
    dec, band_stats, stats_dict = _cached_rain_band(selected_customer_df)
    fig2 = plot_rain_band_chart(dec, band_stats, stats_dict)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    col1, _ = st.columns(2)
    with col1:
        coef_df, scalar_df, meta = _cached_ols(selected_customer_df)
        fig3 = plot_ols_rain_chart(coef_df, scalar_df, meta)
        st.plotly_chart(fig3, use_container_width=True)

    # ── OLS Rain Effect — All Shops ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("### OLS Rain Effect on Sales — All Shops")
    st.caption(
        "Per-shop OLS regression: `log(sales) ~ rain_band + temperature + windspeed + "
        "day-of-week + month + trend`. Results averaged across all shops. "
        "Error bars = 95% CI."
    )

    results_df = None
    _ols_stale = False

    if os.path.exists(OLS_CACHE_PATH):
        results_df = _load_ols_cache(OLS_CACHE_PATH)
        if "x_mm" not in results_df.columns:
            os.remove(OLS_CACHE_PATH)
            _load_ols_cache.clear()
            _cached_agg_ols.clear()
            results_df = None
            _ols_stale = True
            st.warning(
                "OLS cache is from the old model (band format). "
                "Click **🔄 Regenerate Rain OLS Cache** below to rebuild with natural splines."
            )

    if results_df is None and not _ols_stale:
        st.info(
            "No cached results found. Running OLS for all shops — "
            "this takes ~15 minutes and will be saved for future visits."
        )
        with st.spinner("Running OLS analysis for all shops…"):
            results_df = run_ols_rain_all_shops(sellout, df_weather, sellin=sellin)
            os.makedirs(os.path.dirname(OLS_CACHE_PATH), exist_ok=True)
            results_df.to_parquet(OLS_CACHE_PATH, index=False)
        st.success("Analysis complete. Results saved to cache.")

    if results_df is None or results_df.empty:
        st.warning("No results — check that weather and sales data overlap.")
    else:
        n_shops = results_df["customer_code"].nunique()
        agg_all = _cached_agg_ols(results_df)
        fig_all = plot_ols_rain_effect(
            agg_all,
            f"<b>Effect of same-day rain on sales — {n_shops} shops</b><br>"
            "<sub>Mean % effect on sales by rainfall band vs dry days, "
            "averaged across shops (error bars = 95% CI)</sub>",
        )
        st.plotly_chart(fig_all, use_container_width=True, key="wa_ols_all")

        # ── OLS Rain Effect — By Territory ────────────────────────────────────────
        st.markdown("---")
        st.markdown("### OLS Rain Effect by Territory")
        st.caption("Filter to shops within a specific route/territory.")

        routes = sorted(results_df["route"].dropna().unique().tolist())
        if routes:
            selected_route = st.selectbox(
                "Select Territory (Route)", routes, key="wa_ols_route"
            )
            n_route = int(
                results_df[results_df["route"] == selected_route][
                    "customer_code"
                ].nunique()
            )
            agg_route = _cached_agg_ols(results_df, route=selected_route)
            fig_route = plot_ols_rain_effect(
                agg_route,
                f"<b>Territory {selected_route} — {n_route} shops</b><br>"
                "<sub>Mean % effect on sales by rainfall band vs dry days (error bars = 95% CI)</sub>",
            )
            st.plotly_chart(
                fig_route, use_container_width=True, key="wa_ols_route_chart"
            )

        if st.button("🔄 Regenerate Rain OLS Cache", key="wa_ols_regen"):
            if os.path.exists(OLS_CACHE_PATH):
                os.remove(OLS_CACHE_PATH)
            _load_ols_cache.clear()
            _cached_agg_ols.clear()
            st.rerun()

    # ── Rain Effect — Prophet (linear regressor) ─────────────────────────────
    st.markdown("---")
    show_rain_prophet = st.checkbox(
        "Show Rain Effect — Prophet (linear regressor)",
        value=False,
        key="wa_show_rain_prophet",
    )
    if show_rain_prophet:
        st.markdown("### Rain Effect on Sales — Prophet")
        st.caption(
            "Rain β extracted from per-shop Prophet models (reuses the **Temperature → Prophet** cache). "
            "Prophet treats precipitation as a **linear** regressor → straight line. "
            "Compare with the OLS natural spline above to see what non-linearity the spline captures."
        )
        prophet_rain_df = None
        if os.path.exists(PROPHET_TEMP_CACHE_PATH):
            prophet_rain_df = _load_parquet(PROPHET_TEMP_CACHE_PATH)

        if prophet_rain_df is None or prophet_rain_df.empty:
            st.info(
                "No Prophet cache found. Run the **Temperature → Prophet** section first — "
                "it fits precipitation as a regressor at the same time, so no extra computation needed."
            )
        elif "prophet_precipitation" not in prophet_rain_df.columns:
            st.warning(
                "prophet_precipitation column not found — regenerate the Prophet temperature cache."
            )
        else:
            rain_routes_p = sorted(prophet_rain_df["route"].dropna().unique().tolist())

            agg_rp_all = _cached_prophet_rain_curve(prophet_rain_df, sellout)
            n_rp = int(agg_rp_all["n_shops"].iloc[0]) if not agg_rp_all.empty else 0
            fig_rp_all = plot_ols_rain_effect(
                agg_rp_all,
                f"<b>Rain effect on sales (Prophet) — {n_rp} shops</b><br>"
                "<sub>% change vs dry day (0 mm). Prophet uses a linear regressor → straight line</sub>",
            )
            st.plotly_chart(
                fig_rp_all, use_container_width=True, key="wa_prophet_rain_all"
            )

            st.markdown("#### Prophet Rain Effect by Territory")
            if rain_routes_p:
                sel_rp_route = st.selectbox(
                    "Select Territory (Route)",
                    rain_routes_p,
                    key="wa_prophet_rain_route",
                )
                agg_rp_route = _cached_prophet_rain_curve(
                    prophet_rain_df, sellout, route=sel_rp_route
                )
                n_rp_r = (
                    int(agg_rp_route["n_shops"].iloc[0])
                    if not agg_rp_route.empty
                    else 0
                )
                fig_rp_route = plot_ols_rain_effect(
                    agg_rp_route,
                    f"<b>Territory {sel_rp_route} — {n_rp_r} shops (Prophet)</b><br>"
                    "<sub>% change vs dry day. Linear regressor → straight line</sub>",
                )
                st.plotly_chart(
                    fig_rp_route,
                    use_container_width=True,
                    key="wa_prophet_rain_route_chart",
                )

    # ── Rain Band OLS (categorical) ───────────────────────────────────────────
    st.markdown("---")
    show_band_ols = st.checkbox(
        "Show Rain Band OLS (categorical bands: none / light / moderate / heavy)",
        value=False,
        key="wa_show_band_ols",
    )
    if show_band_ols:
        st.markdown("### OLS Rain Effect by Band — All Shops")
        st.caption(
            "Per-shop OLS using categorical rainfall bands (same-day effect only). "
            "`log(sales) ~ C(band) + temperature + windspeed + C(dow) + C(month) + trend`. "
            "Bands: none (<0.1 mm) · light (0.1–2 mm) · moderate (2–8 mm) · heavy (>8 mm). "
            "Reference = none (dry day). Results averaged across shops."
        )

        band_results_df = None
        if os.path.exists(BAND_OLS_CACHE_PATH):
            band_results_df = _load_band_ols_cache(BAND_OLS_CACHE_PATH)
            # Invalidate cache if it's the old lag/lead format
            if band_results_df is not None and "pct_change" in band_results_df.columns:
                os.remove(BAND_OLS_CACHE_PATH)
                _load_band_ols_cache.clear()
                _cached_agg_band_ols.clear()
                band_results_df = None

        if band_results_df is None:
            st.info(
                "No cached band OLS results found. Running per-shop band OLS for all shops — "
                "this takes ~5–10 minutes and will be saved for future visits."
            )
            with st.spinner("Running band OLS for all shops…"):
                band_results_df = run_ols_rain_band_all_shops(
                    sellout, df_weather, sellin=sellin
                )
                os.makedirs(os.path.dirname(BAND_OLS_CACHE_PATH), exist_ok=True)
                band_results_df.to_parquet(BAND_OLS_CACHE_PATH, index=False)
            st.success("Band OLS complete. Results saved to cache.")

        if band_results_df is None or band_results_df.empty:
            st.warning(
                "No band OLS results — check that weather and sales data overlap."
            )
        else:
            n_band_shops = band_results_df["customer_code"].nunique()
            band_agg_all = _cached_agg_band_ols(band_results_df)
            fig_band_all = plot_ols_rain_band_effect(
                band_agg_all,
                f"<b>Rain band effect on sales — {n_band_shops} shops</b><br>"
                "<sub>Mean % change vs dry day, averaged across shops (error bars = 95% CI)</sub>",
            )
            st.plotly_chart(
                fig_band_all, use_container_width=True, key="wa_band_ols_all"
            )

            st.markdown("#### Band OLS by Territory")
            band_routes = sorted(band_results_df["route"].dropna().unique().tolist())
            if band_routes:
                sel_band_route = st.selectbox(
                    "Select Territory (Route)", band_routes, key="wa_band_ols_route"
                )
                n_band_route = int(
                    band_results_df[band_results_df["route"] == sel_band_route][
                        "customer_code"
                    ].nunique()
                )
                band_agg_route = _cached_agg_band_ols(
                    band_results_df, route=sel_band_route
                )
                fig_band_route = plot_ols_rain_band_effect(
                    band_agg_route,
                    f"<b>Territory {sel_band_route} — {n_band_route} shops</b><br>"
                    "<sub>Mean % change vs dry day (error bars = 95% CI)</sub>",
                )
                st.plotly_chart(
                    fig_band_route,
                    use_container_width=True,
                    key="wa_band_ols_route_chart",
                )

        if st.button("🔄 Regenerate Band OLS Cache", key="wa_band_ols_regen"):
            if os.path.exists(BAND_OLS_CACHE_PATH):
                os.remove(BAND_OLS_CACHE_PATH)
            _load_band_ols_cache.clear()
            _cached_agg_band_ols.clear()
            st.rerun()

    # ── Temperature Effect Curve (Prophet) ───────────────────────────────────
    st.markdown("---")
    show_temp_curve = st.checkbox(
        "Show Temperature Effect on Sales (% change curve)",
        value=False,
        key="wa_show_temp_curve",
    )
    if show_temp_curve:
        temp_var_curve = st.radio(
            "Temperature variable:",
            ["Temperature", "Feels-Like Temperature"],
            horizontal=True, key="wa_temp_curve_var",
        )
        _use_wc_curve = temp_var_curve == "Feels-Like Temperature"

        if _use_wc_curve:
            st.markdown("### Feels-Like Temperature Effect on Sales — All Shops (Prophet)")
            st.caption(
                "Prophet apparent_temperature_mean (feels-like °C) regressor β normalised by "
                "each shop's median daily sales. Straight line — Prophet uses a linear regressor. "
                "Reference = global mean feels-like temperature. Shaded area = 95% CI across shops."
            )
            _flp_df_c = _load_parquet(PROPHET_FL_CACHE_PATH) if os.path.exists(PROPHET_FL_CACHE_PATH) else None
            if _flp_df_c is None or _flp_df_c.empty:
                st.info(
                    "No Feels-Like Prophet cache found. "
                    "This model will be trained separately from the Wind Chill section. "
                    "Click **Regenerate Feels-Like Curve Cache** below to build it."
                )
                if st.button("🔄 Build Feels-Like Prophet Cache", key="wa_fl_curve_build"):
                    prog_bar_flc = st.progress(0)
                    prog_text_flc = st.empty()
                    def _on_prog_flc(current, total, fitted, skipped):
                        prog_bar_flc.progress(current / total)
                        prog_text_flc.text(f"Shop {current}/{total} — fitted: {fitted} | skipped: {skipped}")
                    _fl_res_c = run_prophet_fl_all_shops(
                        sellout, df_weather, sellin=sellin, progress_callback=_on_prog_flc
                    )
                    prog_bar_flc.empty(); prog_text_flc.empty()
                    os.makedirs(os.path.dirname(PROPHET_FL_CACHE_PATH), exist_ok=True)
                    _fl_res_c.to_parquet(PROPHET_FL_CACHE_PATH, index=False)
                    _load_parquet.clear()
                    st.success("Feels-Like Prophet cache built.")
                    st.rerun()
            else:
                _max_d_c = pd.Timestamp(sellout["date"].max())
                _ref_fl_c = float(df_weather[df_weather["date"] <= _max_d_c]["apparent_temperature_mean"].mean())
                _n_fl_c = _flp_df_c["customer_code"].nunique()
                _routes_fl_c = sorted(_flp_df_c["route"].dropna().unique().tolist())

                _crv_fl_c = _cached_prophet_fl_curve(_flp_df_c, sellout, df_weather)
                st.plotly_chart(
                    plot_ols_temp_effect(
                        _crv_fl_c,
                        f"<b>Feels-Like Temperature effect on sales — {_n_fl_c} shops</b><br>"
                        "<sub>% change vs mean feels-like temp, averaged across shops (shaded = 95% CI)</sub>",
                        ref_temp=_ref_fl_c,
                    ),
                    use_container_width=True, key="wa_temp_curve_fl_all",
                )
                st.markdown("#### Feels-Like Temperature Effect by Territory")
                if _routes_fl_c:
                    _sel_fl_c = st.selectbox("Select Territory (Route)", _routes_fl_c, key="wa_temp_curve_fl_route")
                    _n_fl_c_r = int(_flp_df_c[_flp_df_c["route"] == _sel_fl_c]["customer_code"].nunique())
                    _crv_fl_c_r = _cached_prophet_fl_curve(_flp_df_c, sellout, df_weather, route=_sel_fl_c)
                    st.plotly_chart(
                        plot_ols_temp_effect(
                            _crv_fl_c_r,
                            f"<b>Territory {_sel_fl_c} — {_n_fl_c_r} shops</b><br>"
                            "<sub>% change vs mean feels-like temp (shaded = 95% CI)</sub>",
                            ref_temp=_ref_fl_c,
                        ),
                        use_container_width=True, key="wa_temp_curve_fl_route_chart",
                    )
                if st.button("🔄 Regenerate Feels-Like Curve Cache", key="wa_fl_curve_regen"):
                    if os.path.exists(PROPHET_FL_CACHE_PATH):
                        os.remove(PROPHET_FL_CACHE_PATH)
                    _load_parquet.clear()
                    _cached_prophet_fl_curve.clear()
                    st.rerun()
        else:
            st.markdown("### Temperature Effect on Sales — All Shops")
            st.caption(
                "Prophet temperature regressor β normalised by each shop's median daily sales. "
                "Shows % change in sales vs mean temperature across all shops. "
                "Reference = global mean temperature. Shaded area = 95% CI across shops."
            )

            if os.path.exists(PROPHET_TEMP_CACHE_PATH):
                prophet_temp_df = _load_parquet(PROPHET_TEMP_CACHE_PATH)
            else:
                st.info(
                    "No cached Prophet results found. Running Prophet with temperature for all shops — "
                    "this takes ~15–20 minutes and will be saved for future visits."
                )
                prog_bar_tc = st.progress(0)
                prog_text_tc = st.empty()

                def _on_progress_tc(current, total, fitted, skipped):
                    prog_bar_tc.progress(current / total)
                    prog_text_tc.text(
                        f"Shop {current} / {total}  —  "
                        f"fitted: {fitted}  |  skipped: {skipped}"
                    )

                prophet_temp_df = run_prophet_temp_all_shops(
                    sellout, df_weather, sellin=sellin, progress_callback=_on_progress_tc
                )
                prog_bar_tc.empty()
                prog_text_tc.empty()
                os.makedirs(os.path.dirname(PROPHET_TEMP_CACHE_PATH), exist_ok=True)
                prophet_temp_df.to_parquet(PROPHET_TEMP_CACHE_PATH, index=False)
                st.success("Prophet analysis complete. Results saved to cache.")

            if prophet_temp_df.empty:
                st.warning("No Prophet results — check data covers ≥60 days per shop.")
            else:
                _max_date = pd.Timestamp(sellout["date"].max())
                global_ref = float(
                    df_weather[df_weather["date"] <= _max_date]["temperature"].mean()
                )
                n_tc_shops = prophet_temp_df["customer_code"].nunique()
                tc_routes = sorted(prophet_temp_df["route"].dropna().unique().tolist())

                curve_all = _cached_prophet_temp_curve(prophet_temp_df, sellout, df_weather)
                fig_tc_all = plot_ols_temp_effect(
                    curve_all,
                    f"<b>Temperature effect on sales — {n_tc_shops} shops</b><br>"
                    "<sub>% change vs mean temperature, averaged across shops (shaded = 95% CI)</sub>",
                    ref_temp=global_ref,
                )
                st.plotly_chart(
                    fig_tc_all, use_container_width=True, key="wa_temp_curve_all"
                )

                st.markdown("#### Temperature Effect by Territory")
                if tc_routes:
                    sel_tc_route = st.selectbox(
                        "Select Territory (Route)", tc_routes, key="wa_temp_curve_route"
                    )
                    n_tc_route = int(
                        prophet_temp_df[prophet_temp_df["route"] == sel_tc_route][
                            "customer_code"
                        ].nunique()
                    )
                    curve_route = _cached_prophet_temp_curve(
                        prophet_temp_df, sellout, df_weather, route=sel_tc_route
                    )
                    fig_tc_route = plot_ols_temp_effect(
                        curve_route,
                        f"<b>Territory {sel_tc_route} — {n_tc_route} shops</b><br>"
                        "<sub>% change vs mean temperature (shaded = 95% CI)</sub>",
                        ref_temp=global_ref,
                    )
                    st.plotly_chart(
                        fig_tc_route,
                        use_container_width=True,
                        key="wa_temp_curve_route_chart",
                    )

            if st.button("🔄 Regenerate Temperature Curve Cache", key="wa_temp_curve_regen"):
                if os.path.exists(PROPHET_TEMP_CACHE_PATH):
                    os.remove(PROPHET_TEMP_CACHE_PATH)
                _load_parquet.clear()
                _cached_prophet_temp_curve.clear()
                st.rerun()

    # ── Temperature Effect Curve (OLS Natural Spline) ────────────────────────
    st.markdown("---")
    show_temp_spline = st.checkbox(
        "Show Temperature Effect — OLS Natural Spline",
        value=False,
        key="wa_show_temp_spline",
    )
    if show_temp_spline:
        temp_var_spline = st.radio(
            "Temperature variable:",
            ["Temperature", "Feels-Like Temperature"],
            horizontal=True, key="wa_temp_spline_var",
        )
        _use_wc_spline = temp_var_spline == "Feels-Like Temperature"

        if _use_wc_spline:
            st.markdown("### Feels-Like Temperature Effect on Sales — OLS Natural Spline")
            st.caption(
                "Per-shop OLS with natural cubic spline for apparent_temperature_mean "
                "(feels-like °C, knots at 5, 15, 25°C). "
                "`log(sales) ~ cr(apparent_temperature_mean, knots=(5,15,25)) + precipitation + windspeed + C(dow) + C(month) + trend`. "
                "Partial effect vs mean feels-like temperature. Shaded area = 95% CI."
            )
            _ols_fl_res = _load_ols_fl_cache(OLS_FL_CACHE_PATH) if os.path.exists(OLS_FL_CACHE_PATH) else None
            if _ols_fl_res is None or _ols_fl_res.empty:
                st.info(
                    "No Feels-Like OLS cache found. "
                    "This is a separate model from Wind Chill OLS. "
                    "Click **Build Feels-Like OLS Cache** below to generate it."
                )
                if st.button("🔄 Build Feels-Like OLS Cache", key="wa_fl_spline_build"):
                    with st.spinner("Running OLS feels-like spline for all shops…"):
                        _fl_ols_res = run_ols_fl_all_shops(sellout, df_weather, sellin=sellin)
                        os.makedirs(os.path.dirname(OLS_FL_CACHE_PATH), exist_ok=True)
                        _fl_ols_res.to_parquet(OLS_FL_CACHE_PATH, index=False)
                    _load_ols_fl_cache.clear()
                    st.success("Feels-Like OLS cache built.")
                    st.rerun()
            else:
                _max_d_ws = pd.Timestamp(sellout["date"].max())
                _ref_fl_s = float(df_weather[df_weather["date"] <= _max_d_ws]["apparent_temperature_mean"].mean())
                _n_fl_s = _ols_fl_res["customer_code"].nunique()
                _routes_fl_s = sorted(_ols_fl_res["route"].dropna().unique().tolist())

                _agg_fl_all = _cached_agg_ols_fl(_ols_fl_res)
                st.plotly_chart(
                    plot_ols_temp_effect(
                        _agg_fl_all,
                        f"<b>Feels-Like Temperature effect on sales (OLS spline) — {_n_fl_s} shops</b><br>"
                        "<sub>% change vs mean feels-like temp, averaged across shops (shaded = 95% CI). "
                        "Dotted lines = spline knots at 5, 15, 25°C</sub>",
                        ref_temp=_ref_fl_s,
                    ),
                    use_container_width=True, key="wa_ols_spline_fl_all",
                )
                st.markdown("#### OLS Feels-Like Temperature Effect by Territory")
                if _routes_fl_s:
                    _sel_fl_s = st.selectbox("Select Territory (Route)", _routes_fl_s, key="wa_ols_spline_fl_route")
                    _n_fl_s_r = int(_ols_fl_res[_ols_fl_res["route"] == _sel_fl_s]["customer_code"].nunique())
                    _agg_fl_r = _cached_agg_ols_fl(_ols_fl_res, route=_sel_fl_s)
                    st.plotly_chart(
                        plot_ols_temp_effect(
                            _agg_fl_r,
                            f"<b>Territory {_sel_fl_s} — {_n_fl_s_r} shops</b><br>"
                            "<sub>% change vs mean feels-like temp, OLS spline (shaded = 95% CI)</sub>",
                            ref_temp=_ref_fl_s,
                        ),
                        use_container_width=True, key="wa_ols_spline_fl_route_chart",
                    )
                if st.button("🔄 Regenerate Feels-Like OLS Cache", key="wa_fl_spline_regen"):
                    if os.path.exists(OLS_FL_CACHE_PATH):
                        os.remove(OLS_FL_CACHE_PATH)
                    _load_ols_fl_cache.clear()
                    _cached_agg_ols_fl.clear()
                    st.rerun()
        else:
            st.markdown("### Temperature Effect on Sales — OLS Natural Spline")
            st.caption(
                "Per-shop OLS with natural cubic spline for temperature (knots at 5, 15, 25°C). "
                "`log(sales) ~ cr(temperature, knots=(5,15,25)) + precipitation + windspeed + C(dow) + C(month) + trend`. "
                "Partial effect vs mean temperature, averaged across shops. Shaded area = 95% CI."
            )

            ols_temp_results = None
            if os.path.exists(OLS_TEMP_CACHE_PATH):
                ols_temp_results = _load_ols_temp_cache(OLS_TEMP_CACHE_PATH)

            if ols_temp_results is None:
                st.info(
                    "No cached OLS temperature results found. Running per-shop OLS for all shops — "
                    "this takes ~10–15 minutes and will be saved for future visits."
                )
                with st.spinner("Running OLS temperature spline for all shops…"):
                    ols_temp_results = run_ols_temp_all_shops(
                        sellout, df_weather, sellin=sellin
                    )
                    os.makedirs(os.path.dirname(OLS_TEMP_CACHE_PATH), exist_ok=True)
                    ols_temp_results.to_parquet(OLS_TEMP_CACHE_PATH, index=False)
                st.success("OLS temperature analysis complete. Results saved to cache.")

            if ols_temp_results is None or ols_temp_results.empty:
                st.warning("No OLS temperature results — check that weather and sales data overlap.")
            else:
                _max_date_ols = pd.Timestamp(sellout["date"].max())
                global_ref_ols = float(
                    df_weather[df_weather["date"] <= _max_date_ols]["temperature"].mean()
                )
                n_ols_shops = ols_temp_results["customer_code"].nunique()
                ols_temp_routes = sorted(ols_temp_results["route"].dropna().unique().tolist())

                agg_ols_temp_all = _cached_agg_ols_temp(ols_temp_results)
                st.plotly_chart(
                    plot_ols_temp_effect(
                        agg_ols_temp_all,
                        f"<b>Temperature effect on sales (OLS spline) — {n_ols_shops} shops</b><br>"
                        "<sub>% change vs mean temperature, averaged across shops (shaded = 95% CI). "
                        "Dotted lines = spline knots at 5, 15, 25°C</sub>",
                        ref_temp=global_ref_ols,
                    ),
                    use_container_width=True, key="wa_ols_temp_all",
                )

                st.markdown("#### OLS Temperature Effect by Territory")
                if ols_temp_routes:
                    sel_ols_temp_route = st.selectbox(
                        "Select Territory (Route)", ols_temp_routes, key="wa_ols_temp_route"
                    )
                    n_ols_temp_route = int(
                        ols_temp_results[ols_temp_results["route"] == sel_ols_temp_route][
                            "customer_code"
                        ].nunique()
                    )
                    agg_ols_temp_route = _cached_agg_ols_temp(ols_temp_results, route=sel_ols_temp_route)
                    st.plotly_chart(
                        plot_ols_temp_effect(
                            agg_ols_temp_route,
                            f"<b>Territory {sel_ols_temp_route} — {n_ols_temp_route} shops</b><br>"
                            "<sub>% change vs mean temperature, OLS spline (shaded = 95% CI)</sub>",
                            ref_temp=global_ref_ols,
                        ),
                        use_container_width=True, key="wa_ols_temp_route_chart",
                    )

            if st.button("🔄 Regenerate OLS Temperature Cache", key="wa_ols_temp_regen"):
                if os.path.exists(OLS_TEMP_CACHE_PATH):
                    os.remove(OLS_TEMP_CACHE_PATH)
                _load_ols_temp_cache.clear()
                _cached_agg_ols_temp.clear()
                st.rerun()

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

    # ── Temperature Effect on Sales (optional) ───────────────────────────────
    st.markdown("---")
    show_temp = st.checkbox(
        "Show Temperature Effect on Sales",
        value=False,
        key="wa_show_temp",
    )
    if show_temp:
        temp_var_contrib = st.radio(
            "Temperature variable:",
            ["Temperature", "Feels-Like Temperature"],
            horizontal=True, key="wa_temp_contrib_var",
        )
        _use_wc_contrib = temp_var_contrib == "Feels-Like Temperature"

        if _use_wc_contrib:
            st.markdown("### Feels-Like Temperature Effect on Sales")
            st.caption(
                "Prophet apparent_temperature_mean β × actual feels-like temperature = estimated "
                "sales units added/removed by feels-like temperature each month. "
                "Uses a dedicated Feels-Like Prophet model (same structure as temperature). "
                "Averaged across all FMC shops."
            )
            _flp_df_ct = _load_parquet(PROPHET_FL_CACHE_PATH) if os.path.exists(PROPHET_FL_CACHE_PATH) else None
            if _flp_df_ct is None or _flp_df_ct.empty:
                st.info(
                    "No Feels-Like Prophet cache found. "
                    "Enable **Show Temperature Effect on Sales (% change curve)** and select "
                    "Feels-Like Temperature to build the cache first."
                )
            elif "prophet_apparent_temperature_mean" not in _flp_df_ct.columns:
                st.warning("Cache missing the feels-like column — regenerate the Feels-Like Prophet cache.")
            else:
                _n_flct = _flp_df_ct["customer_code"].nunique()
                _routes_flct = sorted(_flp_df_ct["route"].dropna().unique().tolist())

                _t_flct, _s_flct, _n_flct_c = _cached_fl_contribution(_flp_df_ct, sellout, df_weather)
                st.plotly_chart(
                    plot_temp_contribution(
                        _t_flct, _s_flct, _n_flct_c,
                        f"Feels-Like Temperature Contribution — Avg Across {_n_flct_c} Customers",
                    ),
                    use_container_width=True, key="wa_temp_fl_contrib_all",
                )
                st.markdown("#### Feels-Like Temperature Contribution by Territory")
                if _routes_flct:
                    _sel_flct_r = st.selectbox(
                        "Select Territory (Route)", _routes_flct, key="wa_temp_fl_contrib_route"
                    )
                    _t_flct_r, _s_flct_r, _n_flct_r = _cached_fl_contribution(
                        _flp_df_ct, sellout, df_weather, route=_sel_flct_r
                    )
                    st.plotly_chart(
                        plot_temp_contribution(
                            _t_flct_r, _s_flct_r, _n_flct_r,
                            f"Territory {_sel_flct_r} — {_n_flct_r} Customers",
                        ),
                        use_container_width=True, key="wa_temp_fl_contrib_route_chart",
                    )
        else:
            st.markdown("### Temperature Effect on Sales")
            st.caption(
                "Prophet regressor β × actual temperature = estimated sales units "
                "added/removed by temperature each month. Uses a separate Prophet model "
                "that includes temperature as a regressor. Averaged across all FMC shops."
            )

            if os.path.exists(PROPHET_TEMP_CACHE_PATH):
                prophet_temp_df = _load_parquet(PROPHET_TEMP_CACHE_PATH)
            else:
                st.info(
                    "No temperature cache found. Running Prophet with temperature for all shops — "
                    "this takes ~15–20 minutes and will be saved for future visits."
                )
                prog_bar_t = st.progress(0)
                prog_text_t = st.empty()

                def _on_progress_t(current, total, fitted, skipped):
                    prog_bar_t.progress(current / total)
                    prog_text_t.text(
                        f"Shop {current} / {total}  —  "
                        f"fitted: {fitted}  |  skipped: {skipped}"
                    )

                prophet_temp_df = run_prophet_temp_all_shops(
                    sellout, df_weather, sellin=sellin, progress_callback=_on_progress_t
                )
                prog_bar_t.empty()
                prog_text_t.empty()
                os.makedirs(os.path.dirname(PROPHET_TEMP_CACHE_PATH), exist_ok=True)
                prophet_temp_df.to_parquet(PROPHET_TEMP_CACHE_PATH, index=False)
                st.success("Temperature analysis complete. Results saved to cache.")

            if not prophet_temp_df.empty:
                n_t = prophet_temp_df["customer_code"].nunique()
                temp_routes = sorted(prophet_temp_df["route"].dropna().unique().tolist())

                t_avg, s_avg, n_c = _cached_temp_contribution(
                    prophet_temp_df, sellout, df_weather
                )
                st.plotly_chart(
                    plot_temp_contribution(
                        t_avg, s_avg, n_c,
                        f"Prophet Temperature Regressor Contribution — Avg Across {n_c} Customers",
                    ),
                    use_container_width=True, key="wa_temp_all",
                )

                st.markdown("#### Temperature Contribution by Territory")
                if temp_routes:
                    sel_temp_route = st.selectbox(
                        "Select Territory (Route)", temp_routes, key="wa_temp_route"
                    )
                    t_r, s_r, n_r = _cached_temp_contribution(
                        prophet_temp_df, sellout, df_weather, route=sel_temp_route
                    )
                    st.plotly_chart(
                        plot_temp_contribution(
                            t_r, s_r, n_r,
                            f"Territory {sel_temp_route} — {n_r} Customers",
                        ),
                        use_container_width=True, key="wa_temp_route_chart",
                    )

            if st.button("🔄 Regenerate Temperature Cache", key="wa_temp_regen"):
                if os.path.exists(PROPHET_TEMP_CACHE_PATH):
                    os.remove(PROPHET_TEMP_CACHE_PATH)
                _load_parquet.clear()
                _cached_temp_contribution.clear()
                st.rerun()

    # ── Temperature Effect ────────────────────────────────────────────────────
    # ── Prophet Yearly Seasonality ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Prophet Yearly Seasonality")
    st.caption(
        "Average yearly seasonality component from Prophet across all shops. "
        "Positive = above-average sales period; negative = below-average. "
        "Controlling for rain and weekday effects only — temperature excluded "
        "so the seasonal pattern is not diluted."
    )

    both_cached = os.path.exists(PROPHET_CACHE_PATH) and os.path.exists(
        SEASONALITY_CACHE_PATH
    )
    if both_cached:
        prophet_df = _load_parquet(PROPHET_CACHE_PATH)
        seasonality_df = _load_parquet(SEASONALITY_CACHE_PATH)
    else:
        st.info(
            "No Prophet cache found. Running Prophet for all shops — "
            "this takes ~15–20 minutes and will be saved for future visits."
        )
        prog_bar = st.progress(0)
        prog_text = st.empty()

        def _on_progress(current, total, fitted, skipped):
            prog_bar.progress(current / total)
            prog_text.text(
                f"Shop {current} / {total}  —  "
                f"fitted: {fitted}  |  skipped: {skipped}"
            )

        prophet_df, seasonality_df = run_prophet_ols_all_shops(
            sellout, df_weather, sellin=sellin, progress_callback=_on_progress
        )
        prog_bar.empty()
        prog_text.empty()
        os.makedirs(os.path.dirname(PROPHET_CACHE_PATH), exist_ok=True)
        prophet_df.to_parquet(PROPHET_CACHE_PATH, index=False)
        seasonality_df.to_parquet(SEASONALITY_CACHE_PATH, index=False)
        st.success("Prophet analysis complete. Results saved to cache.")

    if prophet_df.empty:
        st.warning("No Prophet results — check data covers ≥60 days per shop.")
    else:
        n_shops_p = prophet_df["customer_code"].nunique()
        seas_routes = sorted(prophet_df["route"].dropna().unique().tolist())

        seas_avg = _cached_seasonality(seasonality_df)
        if not seas_avg.empty:
            fig_seas = plot_prophet_seasonality(
                seas_avg,
                n_shops_p,
                f"Prophet Yearly Seasonality — Avg Across {n_shops_p} Shops",
            )
            st.plotly_chart(fig_seas, use_container_width=True, key="wa_seas_all")

        st.markdown("#### Seasonality by Territory")
        if seas_routes:
            sel_seas_route = st.selectbox(
                "Select Territory (Route)", seas_routes, key="wa_seas_route"
            )
            seas_r = _cached_seasonality(seasonality_df, route=sel_seas_route)
            n_r_s = int(
                seasonality_df[seasonality_df["route"] == sel_seas_route][
                    "customer_code"
                ].nunique()
            )
            fig_seas_r = plot_prophet_seasonality(
                seas_r,
                n_r_s,
                f"Territory {sel_seas_route} — {n_r_s} Shops",
            )
            st.plotly_chart(
                fig_seas_r, use_container_width=True, key="wa_seas_route_chart"
            )

    st.markdown("---")
    if st.button("🔄 Regenerate Prophet Cache", key="wa_prophet_regen"):
        for p in [PROPHET_CACHE_PATH, SEASONALITY_CACHE_PATH]:
            if os.path.exists(p):
                os.remove(p)
        _load_parquet.clear()
        _cached_seasonality.clear()
        st.rerun()

    # ── Daily Rainfall ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Daily Rainfall")
    st.caption(
        "Precipitation per day at the selected customer's location. "
        "Bands: none (<0.1 mm) · light (0.1–2 mm) · moderate (2–8 mm) · heavy (>8 mm)."
    )

    daily_rain = (
        selected_customer_df.groupby("date")["precipitation"].mean().reset_index()
    )
    daily_rain["year"] = daily_rain["date"].dt.year
    daily_rain["month"] = daily_rain["date"].dt.month
    daily_rain["day"] = daily_rain["date"].dt.day

    years_rain = sorted(
        daily_rain["year"].dropna().unique().astype(int).tolist(), reverse=True
    )

    r_col1, r_col2 = st.columns(2)
    with r_col1:
        sel_rain_year = st.selectbox("Year", years_rain, key="wa_rain_year")
    with r_col2:
        sel_rain_month = st.selectbox(
            "Month",
            list(range(1, 13)),
            format_func=lambda m: _cal.month_name[m],
            key="wa_rain_month",
        )

    filtered_rain = daily_rain[
        (daily_rain["year"] == sel_rain_year) & (daily_rain["month"] == sel_rain_month)
    ]

    if filtered_rain.empty:
        st.info("No rainfall data for this selection.")
    else:
        fig_rain = plot_daily_rainfall(
            filtered_rain,
            sel_rain_year,
            sel_rain_month,
            _cal.month_name[sel_rain_month],
        )
        st.plotly_chart(fig_rain, use_container_width=True, key="wa_daily_rain")

    # ── Sky Condition Analysis ────────────────────────────────────────────────
    st.markdown("---")
    show_sky = st.checkbox(
        "Show Sky Condition Analysis (Sunny / Overcast / Others)",
        value=False,
        key="wa_show_sky",
    )
    if show_sky:
        st.markdown("### Sky Condition Effect on Sales")
        st.caption(
            "**Sunny** = WMO codes 0–2 · **Overcast** = code 3 · **Others** = remaining codes (rain, drizzle, snow…). "
            "STL decomposition per shop removes trend + weekly seasonality; the residual isolates the sky-condition signal. "
            "**% change = (sky residual − Overcast residual) / shop mean sales × 100**.  "
            "Left = % change from raw sales · Right = % change from STL residual."
        )

        SKY_CACHE_PATH = "./data/cache/sky_analysis.parquet"

        sky_df = None
        if os.path.exists(SKY_CACHE_PATH):
            sky_df = _load_ols_cache(SKY_CACHE_PATH)
            if sky_df is not None and "mean_sales" not in sky_df.columns:
                os.remove(SKY_CACHE_PATH)
                _load_ols_cache.clear()
                _cached_sky_analysis.clear()
                sky_df = None

        if sky_df is None:
            st.info(
                "No sky cache found. Running STL decomposition for all shops — "
                "this may take a few minutes and will be saved for future visits."
            )
            with st.spinner("Running STL for all shops…"):
                sky_df = _cached_sky_analysis(sellout, df_weather)
                if not sky_df.empty:
                    os.makedirs(os.path.dirname(SKY_CACHE_PATH), exist_ok=True)
                    sky_df.to_parquet(SKY_CACHE_PATH, index=False)
            st.success("Sky analysis complete. Results saved to cache.")

        if sky_df is None or sky_df.empty:
            st.warning(
                "No sky data — check that sellout and weather data share "
                "matching (date, latitude, longitude) keys."
            )
        else:
            # Build route label map T1..TN sorted alphabetically
            sky_route_ids = sorted(sky_df["route"].dropna().unique())

            # ── Per-shop % change vs Overcast ─────────────────────────────
            # Compute once: per (shop × sky) mean raw sales and mean residual
            # then express both as % vs Overcast, normalised by shop mean sales.
            def _shop_pct(df):
                """
                Returns DataFrame with columns:
                  customer_code, route, sky, raw_pct, resid_pct, n_days
                for every shop that has at least one Overcast day.
                """
                records = []
                for cust, grp in df.groupby("customer_code"):
                    route = grp["route"].iloc[0]
                    shop_mean = float(grp["mean_sales"].iloc[0])
                    if shop_mean == 0:
                        continue
                    sky_raw = grp.groupby("sky")["sales_quantity"].mean()
                    sky_resid = grp.groupby("sky")["residual"].mean()
                    oc_raw = sky_raw.get("Overcast", float("nan"))
                    oc_resid = sky_resid.get("Overcast", float("nan"))
                    if pd.isna(oc_raw) or pd.isna(oc_resid):
                        continue
                    for sky in ["Sunny", "Overcast", "Others"]:
                        raw_val = sky_raw.get(sky, float("nan"))
                        res_val = sky_resid.get(sky, float("nan"))
                        raw_pct = (raw_val - oc_raw) / shop_mean * 100 if pd.notna(raw_val) else float("nan")
                        res_pct = (res_val - oc_resid) / shop_mean * 100 if pd.notna(res_val) else float("nan")
                        n = int((grp["sky"] == sky).sum())
                        records.append({
                            "customer_code": cust, "route": route, "sky": sky,
                            "raw_pct": raw_pct, "resid_pct": res_pct, "n_days": n,
                        })
                return pd.DataFrame(records)

            shop_pct_df = _shop_pct(sky_df)
            if shop_pct_df.empty:
                st.warning("Could not compute % change — too few shops have Overcast days as baseline.")
            else:
                n_sky_shops = shop_pct_df["customer_code"].nunique()

                def _agg_pct(df):
                    """Aggregate shop_pct_df by sky → mean/SE across shops."""
                    return (
                        df.groupby("sky")
                        .agg(
                            raw_pct=("raw_pct", "mean"),
                            sem_raw=("raw_pct", "sem"),
                            resid_pct=("resid_pct", "mean"),
                            sem_resid=("resid_pct", "sem"),
                            n_shop_days=("n_days", "sum"),
                        )
                        .reset_index()
                        .rename(columns={"raw_pct": "raw_pct", "resid_pct": "resid_pct"})
                    )

                def _make_bar_df(agg, pct_col, sem_col):
                    """Reshape _agg_pct output into format expected by plot_sky_pct_bars."""
                    return agg.rename(columns={pct_col: "pct_change", sem_col: "sem_pct"})

                # ── 1. All Shops ──────────────────────────────────────────
                st.markdown(f"#### All Shops — {n_sky_shops} shops")
                agg_all = _agg_pct(shop_pct_df)
                c1, c2 = st.columns(2)
                with c1:
                    st.plotly_chart(
                        plot_sky_pct_bars(
                            _make_bar_df(agg_all, "raw_pct", "sem_raw"),
                            "Sky Effect — Raw Sales",
                            n_sky_shops,
                            subtitle=f"Raw mean sales % vs Overcast — {n_sky_shops} shops  ·  error bars = SE across shops",
                        ),
                        use_container_width=True,
                        key="wa_sky_all_raw",
                    )
                with c2:
                    st.plotly_chart(
                        plot_sky_pct_bars(
                            _make_bar_df(agg_all, "resid_pct", "sem_resid"),
                            "Sky Effect — STL Residual",
                            n_sky_shops,
                            subtitle=f"STL residual % vs Overcast — {n_sky_shops} shops  ·  error bars = SE across shops",
                        ),
                        use_container_width=True,
                        key="wa_sky_all_resid",
                    )

                # ── 2. By Territory ───────────────────────────────────────
                st.markdown("#### By Territory")
                sky_route_options = sorted(shop_pct_df["route"].dropna().unique().tolist())
                sel_sky_route = st.selectbox(
                    "Select Territory",
                    sky_route_options,
                    key="wa_sky_terr_sel",
                )
                terr_pct_df = shop_pct_df[shop_pct_df["route"] == sel_sky_route]
                n_terr_shops = terr_pct_df["customer_code"].nunique()
                agg_terr = _agg_pct(terr_pct_df)
                terr_label = sel_sky_route
                c1, c2 = st.columns(2)
                with c1:
                    st.plotly_chart(
                        plot_sky_pct_bars(
                            _make_bar_df(agg_terr, "raw_pct", "sem_raw"),
                            f"Territory {terr_label} — Raw Sales",
                            n_terr_shops,
                            subtitle=f"Raw mean sales % vs Overcast — {n_terr_shops} shops  ·  error bars = SE across shops",
                        ),
                        use_container_width=True,
                        key="wa_sky_terr_raw",
                    )
                with c2:
                    st.plotly_chart(
                        plot_sky_pct_bars(
                            _make_bar_df(agg_terr, "resid_pct", "sem_resid"),
                            f"Territory {terr_label} — STL Residual",
                            n_terr_shops,
                            subtitle=f"STL residual % vs Overcast — {n_terr_shops} shops  ·  error bars = SE across shops",
                        ),
                        use_container_width=True,
                        key="wa_sky_terr_resid",
                    )

                # ── 3. By Shop + SKU ──────────────────────────────────────
                st.markdown("#### By Shop & SKU")
                # Sort shops by Sunny STL residual % descending
                sunny_rank = (
                    shop_pct_df[shop_pct_df["sky"] == "Sunny"]
                    .sort_values("resid_pct", ascending=False)["customer_code"]
                    .tolist()
                )
                remaining = [
                    c for c in shop_pct_df["customer_code"].unique()
                    if c not in sunny_rank
                ]
                sorted_shops = sunny_rank + remaining

                ss1, ss2 = st.columns(2)
                with ss1:
                    sel_sky_shop = st.selectbox(
                        "Select Shop (sorted by Sunny STL impact, highest first)",
                        sorted_shops,
                        key="wa_sky_shop_sel",
                    )
                with ss2:
                    sel_sky_sku = st.selectbox(
                        "Select SKU",
                        ["All"] + _TOP10_SKUS,
                        key="wa_sky_shop_sku",
                    )

                sku_filter = sel_sky_sku if sel_sky_sku != "All" else None
                with st.spinner("Loading sky data for selected shop & SKU…"):
                    single_df = _cached_single_sky(
                        sellout, df_weather, sel_sky_shop, sku_filter
                    )

                if single_df.empty:
                    st.info(
                        "No data for this shop / SKU combination "
                        "(needs ≥14 days with weather data)."
                    )
                else:
                    sky_raw = single_df.groupby("sky")["sales_quantity"].mean()
                    sky_resid = single_df.groupby("sky")["residual"].mean()
                    shop_mean = float(single_df["mean_sales"].iloc[0])
                    oc_raw = sky_raw.get("Overcast", float("nan"))
                    oc_resid = sky_resid.get("Overcast", float("nan"))

                    raw_rows, res_rows = [], []
                    for sky in ["Sunny", "Overcast", "Others"]:
                        n = int((single_df["sky"] == sky).sum())
                        r = sky_raw.get(sky, float("nan"))
                        d = sky_resid.get(sky, float("nan"))
                        raw_pct = (r - oc_raw) / shop_mean * 100 if pd.notna(r) and pd.notna(oc_raw) else float("nan")
                        res_pct = (d - oc_resid) / shop_mean * 100 if pd.notna(d) and pd.notna(oc_resid) else float("nan")
                        raw_rows.append({"sky": sky, "pct_change": raw_pct, "sem_pct": float("nan"), "n_shop_days": n})
                        res_rows.append({"sky": sky, "pct_change": res_pct, "sem_pct": float("nan"), "n_shop_days": n})

                    sku_label = sel_sky_sku
                    c1, c2 = st.columns(2)
                    with c1:
                        st.plotly_chart(
                            plot_sky_pct_bars(
                                pd.DataFrame(raw_rows),
                                f"{sel_sky_shop} · {sku_label} — Raw Sales",
                                1,
                                subtitle="Raw mean daily sales % vs Overcast (Overcast = 0)",
                            ),
                            use_container_width=True,
                            key="wa_sky_shop_raw",
                        )
                    with c2:
                        st.plotly_chart(
                            plot_sky_pct_bars(
                                pd.DataFrame(res_rows),
                                f"{sel_sky_shop} · {sku_label} — STL Residual",
                                1,
                                subtitle="STL residual % vs Overcast — trend & weekly seasonality removed",
                            ),
                            use_container_width=True,
                            key="wa_sky_shop_resid",
                        )

        if st.button("🔄 Regenerate Sky Analysis Cache", key="wa_sky_regen"):
            SKY_CACHE_PATH = "./data/cache/sky_analysis.parquet"
            if os.path.exists(SKY_CACHE_PATH):
                os.remove(SKY_CACHE_PATH)
            _load_ols_cache.clear()
            _cached_sky_analysis.clear()
            st.rerun()

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

    # ── WMO Code-Level Sunny Analysis ─────────────────────────────────────────
    st.markdown("---")
    show_wmo = st.checkbox(
        "Show WMO Code-Level Sunny Analysis",
        value=False,
        key="wa_show_wmo",
    )
    if show_wmo:
        st.markdown("### WMO Code-Level Sunny Analysis")
        st.caption(
            "Select which WMO codes to treat as **Sunny**. "
            "Overcast (code 3) and Others (all remaining codes) stay fixed. "
            "Analysis is identical to Sky Condition — same STL residual & raw % vs Overcast baseline."
        )

        SKY_CACHE_PATH_WMO = "./data/cache/sky_analysis.parquet"
        wmo_sky_df = None
        if os.path.exists(SKY_CACHE_PATH_WMO):
            wmo_sky_df = _load_ols_cache(SKY_CACHE_PATH_WMO)
            if wmo_sky_df is not None and (
                "mean_sales" not in wmo_sky_df.columns
                or "weathercode" not in wmo_sky_df.columns
            ):
                wmo_sky_df = None

        if wmo_sky_df is None:
            st.warning(
                "Sky analysis cache not found or outdated. "
                "Enable **Show Sky Condition Analysis** above and let it run first."
            )
        else:
            # Unique calendar dates per code (not shop-days)
            _wmo_counts = {
                c: int(wmo_sky_df[wmo_sky_df["weathercode"] == c]["date"].nunique())
                for c in [0, 1, 2]
            }

            st.markdown("**Select Sunny codes:**")
            wc0, wc1, wc2 = st.columns(3)
            with wc0:
                sel_0 = st.checkbox(
                    f"Code 0 — Clear sky  ({_wmo_counts[0]:,} unique calendar days)",
                    value=True, key="wa_wmo_0",
                )
            with wc1:
                sel_1 = st.checkbox(
                    f"Code 1 — Mainly clear  ({_wmo_counts[1]:,} unique calendar days)",
                    value=True, key="wa_wmo_1",
                )
            with wc2:
                sel_2 = st.checkbox(
                    f"Code 2 — Partly cloudy  ({_wmo_counts[2]:,} unique calendar days)",
                    value=True, key="wa_wmo_2",
                )

            selected_sunny_codes = frozenset(
                c for c, sel in [(0, sel_0), (1, sel_1), (2, sel_2)] if sel
            )

            if not selected_sunny_codes:
                st.warning("Select at least one code to define Sunny days.")
            else:
                selected_n = sum(_wmo_counts[c] for c in selected_sunny_codes)
                st.caption(
                    f"Sunny definition: codes {sorted(selected_sunny_codes)} → "
                    f"**{selected_n:,} shop-days** tagged as Sunny"
                )

                # Relabel sky based on selected codes
                def _relabel_sky(code, selected):
                    if pd.isna(code):
                        return None
                    c = int(code)
                    if c in selected:
                        return "Sunny"
                    if c == 3:
                        return "Overcast"
                    return "Others"

                wmo_df = wmo_sky_df.copy()
                wmo_df["sky"] = wmo_df["weathercode"].apply(
                    lambda c: _relabel_sky(c, selected_sunny_codes)
                )
                wmo_df = wmo_df.dropna(subset=["sky"])

                # Per-shop % change vs Overcast
                def _shop_wmo_pct(df):
                    records = []
                    for cust, grp in df.groupby("customer_code"):
                        route = grp["route"].iloc[0]
                        shop_mean = float(grp["mean_sales"].iloc[0])
                        if shop_mean == 0:
                            continue
                        sky_raw = grp.groupby("sky")["sales_quantity"].mean()
                        sky_resid = grp.groupby("sky")["residual"].mean()
                        oc_raw = sky_raw.get("Overcast", float("nan"))
                        oc_resid = sky_resid.get("Overcast", float("nan"))
                        if pd.isna(oc_raw) or pd.isna(oc_resid):
                            continue
                        for sky in ["Sunny", "Overcast", "Others"]:
                            rv = sky_raw.get(sky, float("nan"))
                            dv = sky_resid.get(sky, float("nan"))
                            raw_pct = (rv - oc_raw) / shop_mean * 100 if pd.notna(rv) else float("nan")
                            res_pct = (dv - oc_resid) / shop_mean * 100 if pd.notna(dv) else float("nan")
                            n = int((grp["sky"] == sky).sum())
                            records.append({
                                "customer_code": cust, "route": route, "sky": sky,
                                "raw_pct": raw_pct, "resid_pct": res_pct, "n_days": n,
                            })
                    return pd.DataFrame(records)

                wmo_pct_df = _shop_wmo_pct(wmo_df)
                if wmo_pct_df.empty:
                    st.warning("Not enough shops have both Sunny and Overcast days for the selected codes.")
                else:
                    n_wmo_shops = wmo_pct_df["customer_code"].nunique()

                    def _agg_wmo(df):
                        return (
                            df.groupby("sky")
                            .agg(
                                raw_pct=("raw_pct", "mean"),
                                sem_raw=("raw_pct", "sem"),
                                resid_pct=("resid_pct", "mean"),
                                sem_resid=("resid_pct", "sem"),
                                n_shop_days=("n_days", "sum"),
                            )
                            .reset_index()
                        )

                    def _make_wmo_bar(agg, pct_col, sem_col):
                        return agg.rename(columns={pct_col: "pct_change", sem_col: "sem_pct"})

                    codes_label = "+".join(str(c) for c in sorted(selected_sunny_codes))

                    # ── 1. All Shops ──────────────────────────────────────────
                    st.markdown(f"#### All Shops — {n_wmo_shops} shops")
                    agg_wmo_all = _agg_wmo(wmo_pct_df)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.plotly_chart(
                            plot_sky_pct_bars(
                                _make_wmo_bar(agg_wmo_all, "raw_pct", "sem_raw"),
                                f"Sunny codes [{codes_label}] — Raw Sales",
                                n_wmo_shops,
                                subtitle=f"Raw mean sales % vs Overcast — {n_wmo_shops} shops  ·  error bars = SE across shops",
                            ),
                            use_container_width=True,
                            key="wa_wmo_all_raw",
                        )
                    with c2:
                        st.plotly_chart(
                            plot_sky_pct_bars(
                                _make_wmo_bar(agg_wmo_all, "resid_pct", "sem_resid"),
                                f"Sunny codes [{codes_label}] — STL Residual",
                                n_wmo_shops,
                                subtitle=f"STL residual % vs Overcast — {n_wmo_shops} shops  ·  error bars = SE across shops",
                            ),
                            use_container_width=True,
                            key="wa_wmo_all_resid",
                        )

                    # ── 2. By Territory ───────────────────────────────────────
                    st.markdown("#### By Territory")
                    wmo_routes = sorted(wmo_pct_df["route"].dropna().unique().tolist())
                    sel_wmo_route = st.selectbox(
                        "Select Territory", wmo_routes, key="wa_wmo_terr_sel"
                    )
                    terr_wmo_df = wmo_pct_df[wmo_pct_df["route"] == sel_wmo_route]
                    n_wmo_terr = terr_wmo_df["customer_code"].nunique()
                    agg_wmo_terr = _agg_wmo(terr_wmo_df)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.plotly_chart(
                            plot_sky_pct_bars(
                                _make_wmo_bar(agg_wmo_terr, "raw_pct", "sem_raw"),
                                f"Territory {sel_wmo_route} · codes [{codes_label}] — Raw Sales",
                                n_wmo_terr,
                                subtitle=f"Raw mean sales % vs Overcast — {n_wmo_terr} shops  ·  error bars = SE across shops",
                            ),
                            use_container_width=True,
                            key="wa_wmo_terr_raw",
                        )
                    with c2:
                        st.plotly_chart(
                            plot_sky_pct_bars(
                                _make_wmo_bar(agg_wmo_terr, "resid_pct", "sem_resid"),
                                f"Territory {sel_wmo_route} · codes [{codes_label}] — STL Residual",
                                n_wmo_terr,
                                subtitle=f"STL residual % vs Overcast — {n_wmo_terr} shops  ·  error bars = SE across shops",
                            ),
                            use_container_width=True,
                            key="wa_wmo_terr_resid",
                        )

                    # ── 3. By Shop & SKU ──────────────────────────────────────
                    st.markdown("#### By Shop & SKU")
                    sunny_wmo_rank = (
                        wmo_pct_df[wmo_pct_df["sky"] == "Sunny"]
                        .sort_values("resid_pct", ascending=False)["customer_code"]
                        .tolist()
                    )
                    remaining_wmo = [
                        c for c in wmo_pct_df["customer_code"].unique()
                        if c not in sunny_wmo_rank
                    ]
                    sorted_wmo_shops = sunny_wmo_rank + remaining_wmo

                    ws1, ws2 = st.columns(2)
                    with ws1:
                        sel_wmo_shop = st.selectbox(
                            "Select Shop (sorted by Sunny STL impact, highest first)",
                            sorted_wmo_shops,
                            key="wa_wmo_shop_sel",
                        )
                    with ws2:
                        sel_wmo_sku = st.selectbox(
                            "Select SKU",
                            ["All"] + _TOP10_SKUS,
                            key="wa_wmo_shop_sku",
                        )

                    sku_wmo_filter = sel_wmo_sku if sel_wmo_sku != "All" else None
                    with st.spinner("Loading data for selected shop & SKU…"):
                        single_wmo_df = _cached_single_sky(
                            sellout, df_weather, sel_wmo_shop, sku_wmo_filter
                        )

                    if single_wmo_df.empty:
                        st.info("No data for this shop / SKU combination (needs ≥14 days).")
                    else:
                        # Relabel using selected codes
                        single_wmo_df = single_wmo_df.copy()
                        single_wmo_df["sky"] = single_wmo_df["weathercode"].apply(
                            lambda c: _relabel_sky(c, selected_sunny_codes)
                        )
                        single_wmo_df = single_wmo_df.dropna(subset=["sky"])

                        wmo_raw = single_wmo_df.groupby("sky")["sales_quantity"].mean()
                        wmo_resid = single_wmo_df.groupby("sky")["residual"].mean()
                        shop_wmo_mean = float(single_wmo_df["mean_sales"].iloc[0])
                        oc_raw = wmo_raw.get("Overcast", float("nan"))
                        oc_resid = wmo_resid.get("Overcast", float("nan"))

                        raw_wmo_rows, res_wmo_rows = [], []
                        for sky in ["Sunny", "Overcast", "Others"]:
                            n = int((single_wmo_df["sky"] == sky).sum())
                            rv = wmo_raw.get(sky, float("nan"))
                            dv = wmo_resid.get(sky, float("nan"))
                            raw_pct = (rv - oc_raw) / shop_wmo_mean * 100 if pd.notna(rv) and pd.notna(oc_raw) else float("nan")
                            res_pct = (dv - oc_resid) / shop_wmo_mean * 100 if pd.notna(dv) and pd.notna(oc_resid) else float("nan")
                            raw_wmo_rows.append({"sky": sky, "pct_change": raw_pct, "sem_pct": float("nan"), "n_shop_days": n})
                            res_wmo_rows.append({"sky": sky, "pct_change": res_pct, "sem_pct": float("nan"), "n_shop_days": n})

                        c1, c2 = st.columns(2)
                        with c1:
                            st.plotly_chart(
                                plot_sky_pct_bars(
                                    pd.DataFrame(raw_wmo_rows),
                                    f"{sel_wmo_shop} · {sel_wmo_sku} · codes [{codes_label}] — Raw",
                                    1,
                                    subtitle=f"Raw mean daily sales % vs Overcast  ·  Sunny = codes [{codes_label}]",
                                ),
                                use_container_width=True,
                                key="wa_wmo_shop_raw",
                            )
                        with c2:
                            st.plotly_chart(
                                plot_sky_pct_bars(
                                    pd.DataFrame(res_wmo_rows),
                                    f"{sel_wmo_shop} · {sel_wmo_sku} · codes [{codes_label}] — STL",
                                    1,
                                    subtitle=f"STL residual % vs Overcast  ·  Sunny = codes [{codes_label}]",
                                ),
                                use_container_width=True,
                                key="wa_wmo_shop_resid",
                            )

    # ── Feels-Like vs Actual Temperature Gap Analysis ─────────────────────────
    st.markdown("---")
    show_gap = st.checkbox(
        "Show Feels-Like vs Actual Temperature Gap Analysis",
        value=False,
        key="wa_show_gap",
    )
    if show_gap:
        st.markdown("### Feels-Like vs Actual Temperature Gap Effect on Sales")
        st.caption(
            "**Gap = apparent_temperature_mean − temperature** (feels-like minus actual). "
            "Negative = feels colder than thermometer (wind chill); positive = feels warmer (heat index). "
            "**Cold Feel** ≤ −3°C · **Slight Chill** −3 to −1°C · **Similar** −1 to +1°C *(baseline = 0%)* · **Warm Feel** > +1°C. "
            "Reuses STL residuals from the Sky Condition cache — no extra computation. "
            "Left = raw sales · Right = STL residual."
        )

        GAP_SKY_CACHE = "./data/cache/sky_analysis.parquet"
        gap_base_df = None
        if os.path.exists(GAP_SKY_CACHE):
            gap_base_df = _load_ols_cache(GAP_SKY_CACHE)
            if gap_base_df is not None and "mean_sales" not in gap_base_df.columns:
                gap_base_df = None

        if gap_base_df is None:
            st.warning(
                "Sky analysis cache not found. Enable **Show Sky Condition Analysis** above and "
                "let it run first — the gap analysis reuses those STL residuals."
            )
        else:
            with st.spinner("Merging temperature gap data…"):
                gap_df = _cached_gap_analysis(gap_base_df, df_weather)

            if gap_df.empty:
                st.warning("Could not compute gap — check weather data coverage.")
            else:
                _GAP_CATS = ["Cold Feel", "Slight Chill", "Similar", "Warm Feel"]
                _GAP_BL   = "Similar"

                # Per-shop % change vs Similar baseline
                def _shop_gap_pct(df):
                    records = []
                    for cust, grp in df.groupby("customer_code"):
                        route = grp["route"].iloc[0]
                        shop_mean = float(grp["mean_sales"].iloc[0])
                        if shop_mean == 0:
                            continue
                        cat_raw   = grp.groupby("gap_cat")["sales_quantity"].mean()
                        cat_resid = grp.groupby("gap_cat")["residual"].mean()
                        bl_raw    = cat_raw.get(_GAP_BL, float("nan"))
                        bl_resid  = cat_resid.get(_GAP_BL, float("nan"))
                        if pd.isna(bl_raw) or pd.isna(bl_resid):
                            continue
                        for cat in _GAP_CATS:
                            rv = cat_raw.get(cat, float("nan"))
                            dv = cat_resid.get(cat, float("nan"))
                            raw_pct = (rv - bl_raw) / shop_mean * 100 if pd.notna(rv) else float("nan")
                            res_pct = (dv - bl_resid) / shop_mean * 100 if pd.notna(dv) else float("nan")
                            n = int((grp["gap_cat"] == cat).sum())
                            records.append({
                                "customer_code": cust, "route": route, "gap_cat": cat,
                                "raw_pct": raw_pct, "resid_pct": res_pct, "n_days": n,
                                "month": None,
                            })
                    return pd.DataFrame(records)

                shop_gap_df = _shop_gap_pct(gap_df)

                if shop_gap_df.empty:
                    st.warning("Not enough shops have Similar days as baseline.")
                else:
                    n_gap_shops = shop_gap_df["customer_code"].nunique()

                    # Unique dates per gap_cat across all shops (for hover)
                    _gap_date_counts = (
                        gap_df.groupby("gap_cat")["date"].nunique().rename("n_shop_days")
                    )

                    def _agg_gap(df, date_counts=None):
                        agg = (
                            df.groupby("gap_cat")
                            .agg(
                                raw_pct=("raw_pct", "mean"),
                                sem_raw=("raw_pct", "sem"),
                                resid_pct=("resid_pct", "mean"),
                                sem_resid=("resid_pct", "sem"),
                            )
                            .reset_index()
                        )
                        if date_counts is not None:
                            agg = agg.merge(date_counts.reset_index(), on="gap_cat", how="left")
                        else:
                            agg["n_shop_days"] = 0
                        return agg

                    def _make_gap_bar(agg, pct_col, sem_col):
                        return agg.rename(columns={pct_col: "pct_change", sem_col: "sem_pct"})

                    # ── 1. All Shops ──────────────────────────────────────────
                    st.markdown(f"#### All Shops — {n_gap_shops} shops")
                    agg_gap_all = _agg_gap(shop_gap_df, _gap_date_counts)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.plotly_chart(
                            plot_gap_pct_bars(
                                _make_gap_bar(agg_gap_all, "raw_pct", "sem_raw"),
                                "Temperature Gap Effect — Raw Sales",
                                n_gap_shops,
                                subtitle=f"Raw mean sales % vs Similar — {n_gap_shops} shops  ·  error bars = SE across shops",
                            ),
                            use_container_width=True, key="wa_gap_all_raw",
                        )
                    with c2:
                        st.plotly_chart(
                            plot_gap_pct_bars(
                                _make_gap_bar(agg_gap_all, "resid_pct", "sem_resid"),
                                "Temperature Gap Effect — STL Residual",
                                n_gap_shops,
                                subtitle=f"STL residual % vs Similar — {n_gap_shops} shops  ·  error bars = SE across shops",
                            ),
                            use_container_width=True, key="wa_gap_all_resid",
                        )

                    # ── 2. By Territory ───────────────────────────────────────
                    st.markdown("#### By Territory")
                    gap_routes = sorted(shop_gap_df["route"].dropna().unique().tolist())
                    sel_gap_route = st.selectbox(
                        "Select Territory", gap_routes, key="wa_gap_terr_sel"
                    )
                    terr_gap_df = shop_gap_df[shop_gap_df["route"] == sel_gap_route]
                    n_gap_terr  = terr_gap_df["customer_code"].nunique()
                    _terr_date_counts = (
                        gap_df[gap_df["route"] == sel_gap_route]
                        .groupby("gap_cat")["date"].nunique().rename("n_shop_days")
                    )
                    agg_gap_terr = _agg_gap(terr_gap_df, _terr_date_counts)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.plotly_chart(
                            plot_gap_pct_bars(
                                _make_gap_bar(agg_gap_terr, "raw_pct", "sem_raw"),
                                f"Territory {sel_gap_route} — Raw Sales",
                                n_gap_terr,
                                subtitle=f"Raw mean sales % vs Similar — {n_gap_terr} shops  ·  error bars = SE",
                            ),
                            use_container_width=True, key="wa_gap_terr_raw",
                        )
                    with c2:
                        st.plotly_chart(
                            plot_gap_pct_bars(
                                _make_gap_bar(agg_gap_terr, "resid_pct", "sem_resid"),
                                f"Territory {sel_gap_route} — STL Residual",
                                n_gap_terr,
                                subtitle=f"STL residual % vs Similar — {n_gap_terr} shops  ·  error bars = SE",
                            ),
                            use_container_width=True, key="wa_gap_terr_resid",
                        )

                    # ── 3. Monthly breakdown ──────────────────────────────────
                    st.markdown("#### Monthly Breakdown")
                    st.caption(
                        "How the gap effect varies month by month. "
                        "For each month, mean sales per gap category is expressed as % vs the "
                        "Similar baseline for that same month."
                    )

                    # Aggregate all shops → (month, gap_cat) means, then % vs Similar within month
                    _m_agg = (
                        gap_df.groupby(["month", "gap_cat"])
                        .agg(
                            mean_raw=("sales_quantity", "mean"),
                            mean_resid=("residual", "mean"),
                            mean_sales=("mean_sales", "mean"),
                            n_shop_days=("date", "nunique"),
                        )
                        .reset_index()
                    )
                    _m_sim = (
                        _m_agg[_m_agg["gap_cat"] == _GAP_BL]
                        [["month", "mean_raw", "mean_resid", "mean_sales"]]
                        .rename(columns={"mean_raw": "bl_raw", "mean_resid": "bl_resid",
                                         "mean_sales": "bl_mean_sales"})
                    )
                    _m_merged = _m_agg.merge(_m_sim, on="month", how="left")
                    _m_merged["raw_pct"] = (
                        (_m_merged["mean_raw"] - _m_merged["bl_raw"])
                        / _m_merged["bl_mean_sales"] * 100
                    )
                    _m_merged["resid_pct"] = (
                        (_m_merged["mean_resid"] - _m_merged["bl_resid"])
                        / _m_merged["bl_mean_sales"] * 100
                    )

                    agg_monthly_raw = _m_merged.rename(columns={"raw_pct": "pct_change"})[
                        ["month", "gap_cat", "pct_change", "n_shop_days"]
                    ]
                    agg_monthly_resid = _m_merged.rename(columns={"resid_pct": "pct_change"})[
                        ["month", "gap_cat", "pct_change", "n_shop_days"]
                    ]

                    c1, c2 = st.columns(2)
                    with c1:
                        st.plotly_chart(
                            plot_gap_monthly(
                                agg_monthly_raw,
                                f"Monthly Gap Effect — Raw Sales ({n_gap_shops} shops)",
                            ),
                            use_container_width=True, key="wa_gap_monthly_raw",
                        )
                    with c2:
                        st.plotly_chart(
                            plot_gap_monthly(
                                agg_monthly_resid,
                                f"Monthly Gap Effect — STL Residual ({n_gap_shops} shops)",
                            ),
                            use_container_width=True, key="wa_gap_monthly_resid",
                        )

                    # ── 4. By Shop & SKU ──────────────────────────────────────
                    st.markdown("#### By Shop & SKU")
                    cold_rank = (
                        shop_gap_df[shop_gap_df["gap_cat"] == "Cold Feel"]
                        .sort_values("resid_pct")["customer_code"]
                        .tolist()
                    )
                    remaining_gap = [
                        c for c in shop_gap_df["customer_code"].unique()
                        if c not in cold_rank
                    ]
                    sorted_gap_shops = cold_rank + remaining_gap

                    gs1, gs2 = st.columns(2)
                    with gs1:
                        sel_gap_shop = st.selectbox(
                            "Select Shop (sorted by Cold Feel impact, most negative first)",
                            sorted_gap_shops,
                            key="wa_gap_shop_sel",
                        )
                    with gs2:
                        sel_gap_sku = st.selectbox(
                            "Select SKU", ["All"] + _TOP10_SKUS, key="wa_gap_shop_sku"
                        )

                    sku_gap_filter = sel_gap_sku if sel_gap_sku != "All" else None
                    with st.spinner("Loading data…"):
                        single_gap_raw = _cached_single_sky(
                            sellout, df_weather, sel_gap_shop, sku_gap_filter
                        )

                    if single_gap_raw.empty:
                        st.info("No data for this shop / SKU combination (needs ≥14 days).")
                    else:
                        # Merge gap onto single-shop result
                        wx_gap = df_weather[["date", "latitude", "longitude",
                                             "temperature", "apparent_temperature_mean"]].copy()
                        wx_gap["date"] = pd.to_datetime(wx_gap["date"])
                        sg = single_gap_raw.copy()
                        sg["date"] = pd.to_datetime(sg["date"])
                        sg = sg.merge(wx_gap, on=["date", "latitude", "longitude"], how="left")
                        sg["gap"] = sg["apparent_temperature_mean"] - sg["temperature"]
                        sg["gap_cat"] = sg["gap"].apply(
                            lambda g: (
                                "Cold Feel" if pd.notna(g) and float(g) <= -3 else
                                "Slight Chill" if pd.notna(g) and float(g) <= -1 else
                                "Similar" if pd.notna(g) and float(g) <= 1 else
                                "Warm Feel" if pd.notna(g) else None
                            )
                        )
                        sg = sg.dropna(subset=["gap_cat"])

                        gap_raw_m  = sg.groupby("gap_cat")["sales_quantity"].mean()
                        gap_res_m  = sg.groupby("gap_cat")["residual"].mean()
                        shop_gm    = float(sg["mean_sales"].iloc[0])
                        bl_r  = gap_raw_m.get(_GAP_BL, float("nan"))
                        bl_d  = gap_res_m.get(_GAP_BL, float("nan"))

                        _sg_date_counts = sg.groupby("gap_cat")["date"].nunique()
                        rr, dr = [], []
                        for cat in _GAP_CATS:
                            n  = int(_sg_date_counts.get(cat, 0))
                            rv = gap_raw_m.get(cat, float("nan"))
                            dv = gap_res_m.get(cat, float("nan"))
                            rp = (rv - bl_r) / shop_gm * 100 if pd.notna(rv) and pd.notna(bl_r) else float("nan")
                            dp = (dv - bl_d) / shop_gm * 100 if pd.notna(dv) and pd.notna(bl_d) else float("nan")
                            rr.append({"gap_cat": cat, "pct_change": rp, "sem_pct": float("nan"), "n_shop_days": n})
                            dr.append({"gap_cat": cat, "pct_change": dp, "sem_pct": float("nan"), "n_shop_days": n})

                        c1, c2 = st.columns(2)
                        with c1:
                            st.plotly_chart(
                                plot_gap_pct_bars(
                                    pd.DataFrame(rr),
                                    f"{sel_gap_shop} · {sel_gap_sku} — Raw Sales",
                                    1,
                                    subtitle="Raw mean daily sales % vs Similar (Similar = 0)",
                                ),
                                use_container_width=True, key="wa_gap_shop_raw",
                            )
                        with c2:
                            st.plotly_chart(
                                plot_gap_pct_bars(
                                    pd.DataFrame(dr),
                                    f"{sel_gap_shop} · {sel_gap_sku} — STL Residual",
                                    1,
                                    subtitle="STL residual % vs Similar — trend & weekly seasonality removed",
                                ),
                                use_container_width=True, key="wa_gap_shop_resid",
                            )

    # ── Storm / Wind Gust Analysis ────────────────────────────────────────────
    st.markdown("---")
    show_storm = st.checkbox(
        "Show Storm (Wind Gust) Effect on Sales",
        value=False,
        key="wa_show_storm",
    )
    if show_storm:
        st.markdown("### Storm Effect on Sales")
        st.caption(
            "Uses **windgusts_max** (daily peak wind gusts, km/h) — better than mean windspeed for detecting storm conditions. "
            "**Calm** < 25 km/h · **Moderate** 25–40 km/h *(baseline = 0%)* · "
            "**Windy** 40–60 km/h · **Storm** > 60 km/h. "
            "Reuses STL residuals from the Sky Condition cache — no extra computation. "
            "Left = raw sales · Right = STL residual."
        )

        STORM_SKY_CACHE = "./data/cache/sky_analysis.parquet"
        storm_base_df = None
        if os.path.exists(STORM_SKY_CACHE):
            storm_base_df = _load_ols_cache(STORM_SKY_CACHE)
            if storm_base_df is not None and "mean_sales" not in storm_base_df.columns:
                storm_base_df = None

        if storm_base_df is None:
            st.warning(
                "Sky analysis cache not found. Enable **Show Sky Condition Analysis** above "
                "and let it run first — the storm analysis reuses those STL residuals."
            )
        else:
            with st.spinner("Merging wind gust data…"):
                storm_df = _cached_storm_analysis(storm_base_df, df_weather)

            if storm_df.empty:
                st.warning("Could not compute storm analysis — check weather data coverage.")
            else:
                _STORM_CATS = ["Calm", "Moderate", "Windy", "Storm"]
                _STORM_BL   = "Moderate"

                def _shop_storm_pct(df):
                    records = []
                    for cust, grp in df.groupby("customer_code"):
                        route     = grp["route"].iloc[0]
                        shop_mean = float(grp["mean_sales"].iloc[0])
                        if shop_mean == 0:
                            continue
                        cat_raw   = grp.groupby("storm_cat")["sales_quantity"].mean()
                        cat_resid = grp.groupby("storm_cat")["residual"].mean()
                        bl_raw    = cat_raw.get(_STORM_BL, float("nan"))
                        bl_resid  = cat_resid.get(_STORM_BL, float("nan"))
                        if pd.isna(bl_raw) or pd.isna(bl_resid):
                            continue
                        for cat in _STORM_CATS:
                            rv = cat_raw.get(cat, float("nan"))
                            dv = cat_resid.get(cat, float("nan"))
                            raw_pct = (rv - bl_raw) / shop_mean * 100 if pd.notna(rv) else float("nan")
                            res_pct = (dv - bl_resid) / shop_mean * 100 if pd.notna(dv) else float("nan")
                            records.append({
                                "customer_code": cust, "route": route, "storm_cat": cat,
                                "raw_pct": raw_pct, "resid_pct": res_pct,
                            })
                    return pd.DataFrame(records)

                shop_storm_df = _shop_storm_pct(storm_df)

                if shop_storm_df.empty:
                    st.warning("Not enough shops have Moderate wind days as baseline.")
                else:
                    n_storm_shops = shop_storm_df["customer_code"].nunique()
                    _storm_date_counts = (
                        storm_df.groupby("storm_cat")["date"].nunique().rename("n_dates")
                    )

                    def _agg_storm(df, date_counts):
                        agg = (
                            df.groupby("storm_cat")
                            .agg(
                                raw_pct=("raw_pct", "mean"),
                                sem_raw=("raw_pct", "sem"),
                                resid_pct=("resid_pct", "mean"),
                                sem_resid=("resid_pct", "sem"),
                            )
                            .reset_index()
                        )
                        return agg.merge(date_counts.reset_index(), on="storm_cat", how="left")

                    def _make_storm_bar(agg, pct_col, sem_col):
                        return agg.rename(columns={pct_col: "pct_change", sem_col: "sem_pct"})

                    # ── 1. All Shops ──────────────────────────────────────────
                    st.markdown(f"#### All Shops — {n_storm_shops} shops")
                    agg_storm_all = _agg_storm(shop_storm_df, _storm_date_counts)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.plotly_chart(
                            plot_storm_pct_bars(
                                _make_storm_bar(agg_storm_all, "raw_pct", "sem_raw"),
                                "Storm Effect — Raw Sales",
                                n_storm_shops,
                                subtitle=f"Raw mean sales % vs Moderate — {n_storm_shops} shops  ·  error bars = SE across shops",
                            ),
                            use_container_width=True, key="wa_storm_all_raw",
                        )
                    with c2:
                        st.plotly_chart(
                            plot_storm_pct_bars(
                                _make_storm_bar(agg_storm_all, "resid_pct", "sem_resid"),
                                "Storm Effect — STL Residual",
                                n_storm_shops,
                                subtitle=f"STL residual % vs Moderate — {n_storm_shops} shops  ·  error bars = SE across shops",
                            ),
                            use_container_width=True, key="wa_storm_all_resid",
                        )

                    # ── 2. By Territory ───────────────────────────────────────
                    st.markdown("#### By Territory")
                    storm_routes = sorted(shop_storm_df["route"].dropna().unique().tolist())
                    sel_storm_route = st.selectbox(
                        "Select Territory", storm_routes, key="wa_storm_terr_sel"
                    )
                    terr_storm_df = shop_storm_df[shop_storm_df["route"] == sel_storm_route]
                    n_storm_terr  = terr_storm_df["customer_code"].nunique()
                    _terr_storm_dates = (
                        storm_df[storm_df["route"] == sel_storm_route]
                        .groupby("storm_cat")["date"].nunique().rename("n_dates")
                    )
                    agg_storm_terr = _agg_storm(terr_storm_df, _terr_storm_dates)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.plotly_chart(
                            plot_storm_pct_bars(
                                _make_storm_bar(agg_storm_terr, "raw_pct", "sem_raw"),
                                f"Territory {sel_storm_route} — Raw Sales",
                                n_storm_terr,
                                subtitle=f"Raw mean sales % vs Moderate — {n_storm_terr} shops  ·  error bars = SE",
                            ),
                            use_container_width=True, key="wa_storm_terr_raw",
                        )
                    with c2:
                        st.plotly_chart(
                            plot_storm_pct_bars(
                                _make_storm_bar(agg_storm_terr, "resid_pct", "sem_resid"),
                                f"Territory {sel_storm_route} — STL Residual",
                                n_storm_terr,
                                subtitle=f"STL residual % vs Moderate — {n_storm_terr} shops  ·  error bars = SE",
                            ),
                            use_container_width=True, key="wa_storm_terr_resid",
                        )

                    # ── 3. Monthly breakdown ──────────────────────────────────
                    st.markdown("#### Monthly Breakdown")
                    _sm_agg = (
                        storm_df.groupby(["month", "storm_cat"])
                        .agg(
                            mean_raw=("sales_quantity", "mean"),
                            mean_resid=("residual", "mean"),
                            mean_sales=("mean_sales", "mean"),
                            n_dates=("date", "nunique"),
                        )
                        .reset_index()
                    )
                    _sm_sim = (
                        _sm_agg[_sm_agg["storm_cat"] == _STORM_BL]
                        [["month", "mean_raw", "mean_resid", "mean_sales"]]
                        .rename(columns={"mean_raw": "bl_raw", "mean_resid": "bl_resid",
                                         "mean_sales": "bl_mean_sales"})
                    )
                    _sm = _sm_agg.merge(_sm_sim, on="month", how="left")
                    _sm["raw_pct"]   = (_sm["mean_raw"]   - _sm["bl_raw"])   / _sm["bl_mean_sales"] * 100
                    _sm["resid_pct"] = (_sm["mean_resid"] - _sm["bl_resid"]) / _sm["bl_mean_sales"] * 100

                    agg_sm_raw   = _sm.rename(columns={"raw_pct":   "pct_change"})[["month", "storm_cat", "pct_change", "n_dates"]]
                    agg_sm_resid = _sm.rename(columns={"resid_pct": "pct_change"})[["month", "storm_cat", "pct_change", "n_dates"]]
                    c1, c2 = st.columns(2)
                    with c1:
                        st.plotly_chart(
                            plot_storm_monthly(agg_sm_raw,   f"Monthly Storm Effect — Raw Sales ({n_storm_shops} shops)"),
                            use_container_width=True, key="wa_storm_monthly_raw",
                        )
                    with c2:
                        st.plotly_chart(
                            plot_storm_monthly(agg_sm_resid, f"Monthly Storm Effect — STL Residual ({n_storm_shops} shops)"),
                            use_container_width=True, key="wa_storm_monthly_resid",
                        )

                    # ── 4. By Shop & SKU ──────────────────────────────────────
                    st.markdown("#### By Shop & SKU")
                    storm_rank = (
                        shop_storm_df[shop_storm_df["storm_cat"] == "Storm"]
                        .sort_values("resid_pct")["customer_code"]
                        .tolist()
                    )
                    remaining_storm = [
                        c for c in shop_storm_df["customer_code"].unique()
                        if c not in storm_rank
                    ]
                    sorted_storm_shops = storm_rank + remaining_storm

                    st1, st2 = st.columns(2)
                    with st1:
                        sel_storm_shop = st.selectbox(
                            "Select Shop (sorted by Storm impact, most negative first)",
                            sorted_storm_shops,
                            key="wa_storm_shop_sel",
                        )
                    with st2:
                        sel_storm_sku = st.selectbox(
                            "Select SKU", ["All"] + _TOP10_SKUS, key="wa_storm_shop_sku"
                        )

                    sku_storm_filter = sel_storm_sku if sel_storm_sku != "All" else None
                    with st.spinner("Loading data…"):
                        single_storm_raw = _cached_single_sky(
                            sellout, df_weather, sel_storm_shop, sku_storm_filter
                        )

                    if single_storm_raw.empty:
                        st.info("No data for this shop / SKU combination (needs ≥14 days).")
                    else:
                        wx_storm = df_weather[["date", "latitude", "longitude", "windgusts_max"]].copy()
                        wx_storm["date"] = pd.to_datetime(wx_storm["date"])
                        ss = single_storm_raw.copy()
                        ss["date"] = pd.to_datetime(ss["date"])
                        ss = ss.merge(wx_storm, on=["date", "latitude", "longitude"], how="left")
                        ss["storm_cat"] = ss["windgusts_max"].apply(
                            lambda g: (
                                "Calm"     if pd.notna(g) and float(g) < 25 else
                                "Moderate" if pd.notna(g) and float(g) < 40 else
                                "Windy"    if pd.notna(g) and float(g) < 60 else
                                "Storm"    if pd.notna(g) else None
                            )
                        )
                        ss = ss.dropna(subset=["storm_cat"])

                        st_raw_m  = ss.groupby("storm_cat")["sales_quantity"].mean()
                        st_res_m  = ss.groupby("storm_cat")["residual"].mean()
                        shop_sm   = float(ss["mean_sales"].iloc[0])
                        bl_r = st_raw_m.get(_STORM_BL, float("nan"))
                        bl_d = st_res_m.get(_STORM_BL, float("nan"))
                        _ss_dates = ss.groupby("storm_cat")["date"].nunique()

                        rr, dr = [], []
                        for cat in _STORM_CATS:
                            n  = int(_ss_dates.get(cat, 0))
                            rv = st_raw_m.get(cat, float("nan"))
                            dv = st_res_m.get(cat, float("nan"))
                            rp = (rv - bl_r) / shop_sm * 100 if pd.notna(rv) and pd.notna(bl_r) else float("nan")
                            dp = (dv - bl_d) / shop_sm * 100 if pd.notna(dv) and pd.notna(bl_d) else float("nan")
                            rr.append({"storm_cat": cat, "pct_change": rp, "sem_pct": float("nan"), "n_dates": n})
                            dr.append({"storm_cat": cat, "pct_change": dp, "sem_pct": float("nan"), "n_dates": n})

                        c1, c2 = st.columns(2)
                        with c1:
                            st.plotly_chart(
                                plot_storm_pct_bars(
                                    pd.DataFrame(rr),
                                    f"{sel_storm_shop} · {sel_storm_sku} — Raw Sales",
                                    1,
                                    subtitle="Raw mean daily sales % vs Moderate (Moderate = 0)",
                                ),
                                use_container_width=True, key="wa_storm_shop_raw",
                            )
                        with c2:
                            st.plotly_chart(
                                plot_storm_pct_bars(
                                    pd.DataFrame(dr),
                                    f"{sel_storm_shop} · {sel_storm_sku} — STL Residual",
                                    1,
                                    subtitle="STL residual % vs Moderate — trend & weekly seasonality removed",
                                ),
                                use_container_width=True, key="wa_storm_shop_resid",
                            )

    # ── Sunny Day Transition Analysis ─────────────────────────────────────────
    st.markdown("---")
    show_sunny_trans = st.checkbox(
        "Show Sunny Day Transition Analysis",
        value=False,
        key="wa_show_sunny_transition",
    )
    if show_sunny_trans:
        st.markdown("### Sunny Day Transition Analysis")
        st.caption(
            "Two focused comparisons. **Section 1:** average sales on the non-sunny day "
            "immediately before a sunny day vs the sunny day itself. "
            "**Section 2:** the sunny day vs the non-sunny day immediately after. "
            "% change is shown at the top of each chart. Left = Raw sales, Right = STL "
            "(trend & weekly seasonality removed)."
        )

        sky_cache_path = "./data/cache/sky_analysis.parquet"
        trans_sky_df = None
        if os.path.exists(sky_cache_path):
            trans_sky_df = _load_parquet(sky_cache_path)

        if trans_sky_df is None or trans_sky_df.empty:
            st.info(
                "Sky analysis cache not found. Enable **Sky Condition Effect on Sales** "
                "first — it builds the cache that this analysis reuses."
            )
        else:
            # ── WMO code checkboxes — show unique calendar dates ───────────
            _trans_sky_dates = trans_sky_df.copy()
            _trans_sky_dates["date"] = pd.to_datetime(_trans_sky_dates["date"])
            _trans_wmo_counts = {
                c: int(_trans_sky_dates[_trans_sky_dates["weathercode"] == c]["date"].nunique())
                for c in [0, 1, 2]
            }
            st.markdown("**Select Sunny codes:**")
            _twc0, _twc1, _twc2 = st.columns(3)
            with _twc0:
                _tsel_0 = st.checkbox(
                    f"Code 0 — Clear sky  ({_trans_wmo_counts[0]:,} unique days)",
                    value=True, key="wa_trans_wmo_0",
                )
            with _twc1:
                _tsel_1 = st.checkbox(
                    f"Code 1 — Mainly clear  ({_trans_wmo_counts[1]:,} unique days)",
                    value=True, key="wa_trans_wmo_1",
                )
            with _twc2:
                _tsel_2 = st.checkbox(
                    f"Code 2 — Partly cloudy  ({_trans_wmo_counts[2]:,} unique days)",
                    value=True, key="wa_trans_wmo_2",
                )

            _trans_sunny_codes = frozenset(
                c for c, sel in [(0, _tsel_0), (1, _tsel_1), (2, _tsel_2)] if sel
            )

            if not _trans_sunny_codes:
                st.warning("Select at least one code to define Sunny days.")
            else:
                _selected_n = sum(_trans_wmo_counts[c] for c in _trans_sunny_codes)
                st.caption(
                    f"Sunny definition: codes {sorted(_trans_sunny_codes)} → "
                    f"**{_selected_n:,} unique calendar days** tagged as Sunny"
                )

                def _trans_relabel(code, selected):
                    if pd.isna(code):
                        return None
                    c = int(code)
                    if c in selected:
                        return "Sunny"
                    if c == 3:
                        return "Overcast"
                    return "Others"

                _trans_sky_relabeled = trans_sky_df.copy()
                _trans_sky_relabeled["sky"] = _trans_sky_relabeled["weathercode"].apply(
                    lambda c: _trans_relabel(c, _trans_sunny_codes)
                )
                _trans_sky_relabeled = _trans_sky_relabeled.dropna(subset=["sky"])

                trans_df = _cached_sunny_transition(_trans_sky_relabeled)

                if trans_df.empty:
                    st.warning("No transition days found with the selected Sunny codes.")
                else:
                    _TRANS_CATS = ["Day Before Sunny", "Sunny Day", "Day After Sunny"]

                    def _agg_trans_abs(df, cat_a, cat_b, route=None):
                        d = df if route is None else df[df["route"] == route]
                        d = d[d["transition_cat"].isin([cat_a, cat_b])]
                        per = (
                            d.groupby(["customer_code", "transition_cat"])
                            .agg(
                                mean_raw=("sales_quantity", "mean"),
                                mean_resid=("residual", "mean"),
                                mean_sales=("mean_sales", "first"),
                            )
                            .reset_index()
                        )
                        global_dates = (
                            d.groupby("transition_cat")["date"]
                            .nunique()
                            .rename("n_dates")
                            .reset_index()
                        )
                        agg = (
                            per.groupby("transition_cat")
                            .agg(
                                mean_raw=("mean_raw", "mean"),
                                mean_resid=("mean_resid", "mean"),
                                mean_sales=("mean_sales", "mean"),
                                n_shops=("customer_code", "nunique"),
                            )
                            .reset_index()
                        )
                        agg = agg.merge(global_dates, on="transition_cat", how="left")
                        return agg.set_index("transition_cat").reindex([cat_a, cat_b]).reset_index()

                    def _agg_trans_monthly(df, cat_a, cat_b, route=None):
                        """Monthly % change between cat_a and cat_b, averaged across shops."""
                        d = df if route is None else df[df["route"] == route]
                        d = d[d["transition_cat"].isin([cat_a, cat_b])].copy()
                        d["_m"] = pd.to_datetime(d["date"]).dt.month
                        if d.empty:
                            return pd.DataFrame()

                        shop_mean = float(d["mean_sales"].mean()) if not d["mean_sales"].isna().all() else 1.0
                        if shop_mean == 0:
                            shop_mean = 1.0

                        per = (
                            d.groupby(["customer_code", "_m", "transition_cat"])
                            .agg(mean_raw=("sales_quantity", "mean"),
                                 mean_resid=("residual", "mean"))
                            .reset_index()
                        )
                        raw_piv = per.pivot_table(
                            index=["customer_code", "_m"],
                            columns="transition_cat", values="mean_raw",
                        )
                        res_piv = per.pivot_table(
                            index=["customer_code", "_m"],
                            columns="transition_cat", values="mean_resid",
                        )
                        rows = []
                        for idx in raw_piv.index:
                            rrow = raw_piv.loc[idx]
                            drow = res_piv.loc[idx] if idx in res_piv.index else None
                            if drow is None:
                                continue
                            ra = rrow.get(cat_a, float("nan"))
                            rb = rrow.get(cat_b, float("nan"))
                            da = drow.get(cat_a, float("nan"))
                            db = drow.get(cat_b, float("nan"))
                            if pd.isna(ra) or pd.isna(rb):
                                continue
                            rows.append({
                                "month": idx[1],
                                "raw_pct":  (rb - ra) / shop_mean * 100,
                                "stl_pct":  (db - da) / shop_mean * 100 if (pd.notna(da) and pd.notna(db)) else float("nan"),
                            })
                        if not rows:
                            return pd.DataFrame()

                        mdf = pd.DataFrame(rows).groupby("month").mean().reset_index()
                        # unique calendar dates for cat_a per month
                        n_dates = (
                            d[d["transition_cat"] == cat_a]
                            .groupby("_m")["date"].nunique()
                            .rename("n_dates")
                            .reset_index()
                            .rename(columns={"_m": "month"})
                        )
                        mdf = mdf.merge(n_dates, on="month", how="left")
                        all_months = pd.DataFrame({"month": range(1, 13)})
                        return all_months.merge(mdf, on="month", how="left")

                    def _pair_charts(df, cat_a, cat_b, key_prefix, n_shops, route=None):
                        agg  = _agg_trans_abs(df, cat_a, cat_b, route=route)
                        sub  = agg[agg["transition_cat"].isin([cat_a, cat_b])].copy()
                        sub  = sub.set_index("transition_cat").reindex([cat_a, cat_b]).reset_index()
                        shop_mean = float(agg["mean_sales"].mean()) if not agg["mean_sales"].isna().all() else 1.0
                        n_str = f"{n_shops} shops" if n_shops > 1 else "1 shop"

                        raw_df = pd.DataFrame({
                            "transition_cat": sub["transition_cat"],
                            "mean_val":       sub["mean_raw"],
                            "n_dates":        sub["n_dates"],
                        })
                        stl_df = pd.DataFrame({
                            "transition_cat": sub["transition_cat"],
                            "mean_val":       sub["mean_resid"],
                            "n_dates":        sub["n_dates"],
                        })

                        c1, c2 = st.columns(2)
                        with c1:
                            st.plotly_chart(
                                plot_transition_pair(
                                    raw_df,
                                    f"<b>{cat_a} → {cat_b}</b><br><sub>Raw avg daily sales · {n_str}</sub>",
                                ),
                                use_container_width=True, key=f"{key_prefix}_raw",
                            )
                        with c2:
                            st.plotly_chart(
                                plot_transition_pair(
                                    stl_df,
                                    f"<b>{cat_a} → {cat_b}</b><br><sub>STL residual · {n_str}</sub>",
                                    shop_mean=shop_mean,
                                ),
                                use_container_width=True, key=f"{key_prefix}_stl",
                            )

                    # ── All Shops ──────────────────────────────────────────
                    _n_all_shops = trans_df["customer_code"].nunique()
                    st.markdown(f"#### All Shops ({_n_all_shops} shops)")

                    st.markdown("**Section 1 — Before Sunny → Sunny Day**")
                    _pair_charts(trans_df, "Day Before Sunny", "Sunny Day", "wa_trans_all_s1", _n_all_shops)
                    _m1 = _agg_trans_monthly(trans_df, "Day Before Sunny", "Sunny Day")
                    st.plotly_chart(
                        plot_transition_monthly(_m1, "Day Before Sunny", "Sunny Day",
                            "Monthly % Change — Day Before Sunny → Sunny Day"),
                        use_container_width=True, key="wa_trans_all_s1_monthly",
                    )

                    st.markdown("**Section 2 — Sunny Day → Day After Sunny**")
                    _pair_charts(trans_df, "Sunny Day", "Day After Sunny", "wa_trans_all_s2", _n_all_shops)
                    _m2 = _agg_trans_monthly(trans_df, "Sunny Day", "Day After Sunny")
                    st.plotly_chart(
                        plot_transition_monthly(_m2, "Sunny Day", "Day After Sunny",
                            "Monthly % Change — Sunny Day → Day After Sunny"),
                        use_container_width=True, key="wa_trans_all_s2_monthly",
                    )

                    # ── By Territory ───────────────────────────────────────
                    st.markdown("#### By Territory")
                    trans_routes = sorted(trans_df["route"].dropna().unique().tolist())
                    if trans_routes:
                        sel_trans_route = st.selectbox(
                            "Select Territory (Route)", trans_routes, key="wa_trans_route_sel"
                        )
                        _n_r_shops = trans_df[trans_df["route"] == sel_trans_route]["customer_code"].nunique()

                        st.markdown("**Section 1 — Before Sunny → Sunny Day**")
                        _pair_charts(trans_df, "Day Before Sunny", "Sunny Day", "wa_trans_r_s1", _n_r_shops, route=sel_trans_route)
                        _mr1 = _agg_trans_monthly(trans_df, "Day Before Sunny", "Sunny Day", route=sel_trans_route)
                        st.plotly_chart(
                            plot_transition_monthly(_mr1, "Day Before Sunny", "Sunny Day",
                                f"Monthly % Change — Day Before Sunny → Sunny Day ({sel_trans_route})"),
                            use_container_width=True, key="wa_trans_r_s1_monthly",
                        )

                        st.markdown("**Section 2 — Sunny Day → Day After Sunny**")
                        _pair_charts(trans_df, "Sunny Day", "Day After Sunny", "wa_trans_r_s2", _n_r_shops, route=sel_trans_route)
                        _mr2 = _agg_trans_monthly(trans_df, "Sunny Day", "Day After Sunny", route=sel_trans_route)
                        st.plotly_chart(
                            plot_transition_monthly(_mr2, "Sunny Day", "Day After Sunny",
                                f"Monthly % Change — Sunny Day → Day After Sunny ({sel_trans_route})"),
                            use_container_width=True, key="wa_trans_r_s2_monthly",
                        )

                    # ── By Shop ────────────────────────────────────────────
                    st.markdown("#### By Shop")
                    all_trans_shops = sorted(trans_df["customer_code"].unique().tolist())
                    sel_trans_shop = st.selectbox(
                        "Select Shop", all_trans_shops, key="wa_trans_shop_sel"
                    )
                    shop_trans_df = trans_df[trans_df["customer_code"] == sel_trans_shop]
                    if shop_trans_df.empty:
                        st.info("No transition days found for this shop.")
                    else:
                        st.markdown("**Section 1 — Before Sunny → Sunny Day**")
                        _pair_charts(shop_trans_df, "Day Before Sunny", "Sunny Day", "wa_trans_shop_s1", 1)
                        _ms1 = _agg_trans_monthly(shop_trans_df, "Day Before Sunny", "Sunny Day")
                        st.plotly_chart(
                            plot_transition_monthly(_ms1, "Day Before Sunny", "Sunny Day",
                                f"Monthly % Change — Day Before Sunny → Sunny Day ({sel_trans_shop})"),
                            use_container_width=True, key="wa_trans_shop_s1_monthly",
                        )

                        st.markdown("**Section 2 — Sunny Day → Day After Sunny**")
                        _pair_charts(shop_trans_df, "Sunny Day", "Day After Sunny", "wa_trans_shop_s2", 1)
                        _ms2 = _agg_trans_monthly(shop_trans_df, "Sunny Day", "Day After Sunny")
                        st.plotly_chart(
                            plot_transition_monthly(_ms2, "Sunny Day", "Day After Sunny",
                                f"Monthly % Change — Sunny Day → Day After Sunny ({sel_trans_shop})"),
                            use_container_width=True, key="wa_trans_shop_s2_monthly",
                        )

                    # ── Sunny Day × Temperature Combined Analysis ──────────
                    st.markdown("---")
                    st.markdown("#### Sunny Day Sales vs Temperature")
                    st.caption(
                        "For each temperature bin (based on the temperature on the Sunny Day), "
                        "the left chart compares average STL residuals of the Day Before Sunny "
                        "vs the Sunny Day. The right chart shows the % change between them. "
                        "Shows whether warmer or cooler sunny days drive a bigger sales lift."
                    )

                    # Code checkboxes — filter which sunny codes go into temp analysis
                    _sunny_in_trans = trans_df[trans_df["transition_cat"] == "Sunny Day"]
                    _stc_wmo_counts = {
                        c: int(_sunny_in_trans[_sunny_in_trans["weathercode"] == c]["date"].nunique())
                        for c in [0, 1, 2]
                    }
                    st.markdown("**Select Sunny codes:**")
                    _sc0, _sc1, _sc2 = st.columns(3)
                    with _sc0:
                        _stc_sel_0 = st.checkbox(
                            f"Code 0 — Clear sky  ({_stc_wmo_counts[0]:,} unique days)",
                            value=True, key="wa_stc_wmo_0",
                        )
                    with _sc1:
                        _stc_sel_1 = st.checkbox(
                            f"Code 1 — Mainly clear  ({_stc_wmo_counts[1]:,} unique days)",
                            value=True, key="wa_stc_wmo_1",
                        )
                    with _sc2:
                        _stc_sel_2 = st.checkbox(
                            f"Code 2 — Partly cloudy  ({_stc_wmo_counts[2]:,} unique days)",
                            value=True, key="wa_stc_wmo_2",
                        )

                    _stc_codes = frozenset(
                        c for c, sel in [(0, _stc_sel_0), (1, _stc_sel_1), (2, _stc_sel_2)] if sel
                    )

                    if not _stc_codes:
                        st.warning("Select at least one code.")
                    else:
                        # Keep all Day Before rows + only selected-code Sunny Day rows
                        _trans_for_temp = pd.concat([
                            trans_df[trans_df["transition_cat"] != "Sunny Day"],
                            trans_df[
                                (trans_df["transition_cat"] == "Sunny Day") &
                                (trans_df["weathercode"].isin(_stc_codes))
                            ],
                        ], ignore_index=True)

                        _stc_df = _cached_sunny_temp_combined(_trans_for_temp, df_weather)
                        if _stc_df.empty:
                            st.info("Not enough paired data to build the temperature breakdown.")
                        else:
                            st.plotly_chart(
                                plot_sunny_temp_combined(
                                    _stc_df,
                                    f"Sunny Day Sales by Temperature — All Shops  "
                                    f"(codes {sorted(_stc_codes)})",
                                ),
                                use_container_width=True, key="wa_sunny_temp_all",
                            )

                            # By territory
                            if trans_routes:
                                st.markdown("**By Territory**")
                                _sel_stc_route = st.selectbox(
                                    "Select Territory", trans_routes,
                                    key="wa_sunny_temp_route_sel",
                                )
                                _stc_r = _cached_sunny_temp_combined(
                                    _trans_for_temp[_trans_for_temp["route"] == _sel_stc_route],
                                    df_weather,
                                )
                                if not _stc_r.empty:
                                    st.plotly_chart(
                                        plot_sunny_temp_combined(
                                            _stc_r,
                                            f"Sunny Day Sales by Temperature — {_sel_stc_route}  "
                                            f"(codes {sorted(_stc_codes)})",
                                        ),
                                        use_container_width=True, key="wa_sunny_temp_route",
                                    )

    # ── Temperature Swing Analysis ────────────────────────────────────────────
    st.markdown("---")
    show_temp_swing = st.checkbox(
        "Show Temperature Swing Analysis",
        value=False,
        key="wa_show_temp_swing",
    )
    if show_temp_swing:
        st.markdown("### Temperature Swing Analysis")
        st.caption(
            "Does a sudden warm-up or cold-snap change cigarette sales? "
            "This compares STL residuals on days where temperature rose >5°C vs the previous day "
            "(Big Rise), days with little change (Neutral), and days where it fell >5°C (Big Drop). "
            "Baseline = Neutral days."
        )
        with st.spinner("Computing temperature swing analysis…"):
            _ts_df = _cached_temp_swing(sellout, df_weather)

        if _ts_df.empty:
            st.info("Not enough data for temperature swing analysis.")
        else:
            _ts_routes = sorted(_ts_df["route"].dropna().unique().tolist())
            _ts_n_shops = _ts_df["customer_code"].nunique()
            st.plotly_chart(
                plot_temp_swing(
                    _ts_df,
                    f"Temperature Swing vs Sales — All Shops ({_ts_n_shops} shops)",
                ),
                use_container_width=True, key="wa_ts_all",
            )

            st.markdown("##### OLS Regression (with DOW + month controls)")
            with st.spinner("Running OLS…"):
                _ts_ols = _cached_ts_ols(_ts_df)
            _ts_order = ["Big Drop (≤−5°C)", "Big Rise (≥+5°C)"]
            _ts_ols_col, _ = st.columns(2)
            with _ts_ols_col:
                st.plotly_chart(
                    plot_ols_category_effect(
                        _ts_ols, _ts_order, "Neutral (−5 to +5°C)",
                        f"Temperature Swing — OLS Effect — All Shops ({_ts_n_shops} shops)",
                        height=500, bargap=0.45,
                    ),
                    use_container_width=True, key="wa_ts_ols_all",
                )

            # ── Big Drop / Big Rise day table ──────────────────────────────
            st.markdown("##### Big Drop & Big Rise Days")
            _name_map = (
                sellout[["customer_code", "customer_name"]]
                .drop_duplicates("customer_code")
                .set_index("customer_code")["customer_name"]
                .to_dict()
            )
            _ts_events = _ts_df[_ts_df["swing_cat"] != "Neutral (−5 to +5°C)"].copy()
            _ts_events["customer_name"] = _ts_events["customer_code"].map(_name_map)
            _ts_events["date"] = pd.to_datetime(_ts_events["date"]).dt.date
            # One row per (date, customer) — deduplicates multiple SKUs for same shop/date
            _ts_agg_cols = {}
            if "prev_temp" in _ts_events.columns:
                _ts_agg_cols["prev_temp"] = ("prev_temp", "mean")
            if "temperature" in _ts_events.columns:
                _ts_agg_cols["temperature"] = ("temperature", "mean")
            if "temp_delta" in _ts_events.columns:
                _ts_agg_cols["temp_delta"] = ("temp_delta", "mean")
            _ts_events = (
                _ts_events.groupby(["date", "swing_cat", "customer_code", "customer_name"])
                .agg(**_ts_agg_cols)
                .reset_index()
            )
            for _col in ["prev_temp", "temperature", "temp_delta"]:
                if _col in _ts_events.columns:
                    _ts_events[_col] = _ts_events[_col].round(1)
            _ts_events = _ts_events.rename(columns={
                "swing_cat": "Event",
                "prev_temp": "Prev Day Temp (°C)",
                "temperature": "Today Temp (°C)",
                "temp_delta": "Change (°C)",
            })
            _ts_disp_cols = [c for c in
                             ["date", "customer_name", "Event",
                              "Prev Day Temp (°C)", "Today Temp (°C)", "Change (°C)"]
                             if c in _ts_events.columns]
            st.dataframe(
                _ts_events[_ts_disp_cols].sort_values(["date", "customer_name"]),
                use_container_width=True, hide_index=True,
            )

            if _ts_routes:
                st.markdown("**By Territory**")
                _sel_ts_route = st.selectbox(
                    "Select Territory", _ts_routes, key="wa_ts_route_sel"
                )
                _ts_r = _ts_df[_ts_df["route"] == _sel_ts_route]
                if not _ts_r.empty:
                    st.plotly_chart(
                        plot_temp_swing(
                            _ts_r,
                            f"Temperature Swing vs Sales — {_sel_ts_route}",
                        ),
                        use_container_width=True, key="wa_ts_route",
                    )
                    _ts_ols_r = _cached_ts_ols(_ts_r)
                    st.plotly_chart(
                        plot_ols_category_effect(
                            _ts_ols_r, _ts_order, "Neutral (−5 to +5°C)",
                            f"Temperature Swing — OLS Effect — {_sel_ts_route}",
                        ),
                        use_container_width=True, key="wa_ts_ols_route",
                    )

    # ── Rain Streak & First Dry Day Analysis ──────────────────────────────────
    st.markdown("---")
    show_rain_streak = st.checkbox(
        "Show Rain Streak & First Dry Day Analysis",
        value=False,
        key="wa_show_rain_streak",
    )
    if show_rain_streak:
        st.markdown("### Rain Streak & First Dry Day Analysis")
        st.caption(
            "Does rain compound over consecutive days? And does the first dry day after a "
            "long rain streak see a sales bounce? Adjust the rain threshold below. "
            "Baseline = Normal Dry Day."
        )
        _rs_threshold = st.slider(
            "Rain threshold (mm/day) — days above this count as a Rain Day",
            min_value=0.5, max_value=10.0, value=1.0, step=0.5,
            key="wa_rs_threshold",
        )
        with st.spinner("Computing rain streak analysis…"):
            _rs_df = _cached_rain_streak(sellout, df_weather, _rs_threshold)

        if _rs_df.empty:
            st.info("Not enough data for rain streak analysis.")
        else:
            _rs_routes = sorted(_rs_df["route"].dropna().unique().tolist())
            _rs_n_shops = _rs_df["customer_code"].nunique()
            st.plotly_chart(
                plot_rain_streak(
                    _rs_df,
                    f"Rain Streak & First Dry Day — All Shops ({_rs_n_shops} shops)",
                ),
                use_container_width=True, key="wa_rs_all",
            )

            st.markdown("##### OLS Regression (with DOW + month controls)")
            with st.spinner("Running OLS…"):
                _rs_ols = _cached_rs_ols(_rs_df)
            _rs_order = [
                "Normal Dry Day", "Rain Day 1", "Rain Day 2",
                "Rain Day 3", "Rain Day 4+", "First Dry Day (after 3+ rain)",
            ]
            st.plotly_chart(
                plot_ols_category_effect(
                    _rs_ols, _rs_order, "Normal Dry Day",
                    f"Rain Streak — OLS Effect — All Shops ({_rs_n_shops} shops)",
                ),
                use_container_width=True, key="wa_rs_ols_all",
            )

            if _rs_routes:
                st.markdown("**By Territory**")
                _sel_rs_route = st.selectbox(
                    "Select Territory", _rs_routes, key="wa_rs_route_sel"
                )
                _rs_r = _rs_df[_rs_df["route"] == _sel_rs_route]
                if not _rs_r.empty:
                    st.plotly_chart(
                        plot_rain_streak(
                            _rs_r,
                            f"Rain Streak & First Dry Day — {_sel_rs_route}",
                        ),
                        use_container_width=True, key="wa_rs_route",
                    )
                    _rs_ols_r = _cached_rs_ols(_rs_r)
                    st.plotly_chart(
                        plot_ols_category_effect(
                            _rs_ols_r, _rs_order, "Normal Dry Day",
                            f"Rain Streak — OLS Effect — {_sel_rs_route}",
                        ),
                        use_container_width=True, key="wa_rs_ols_route",
                    )

    # ── Rain Intensity Analysis ────────────────────────────────────────────────
    st.markdown("---")
    show_rain_intensity = st.checkbox(
        "Show Rain Intensity Analysis",
        value=False,
        key="wa_show_rain_intensity",
    )
    if show_rain_intensity:
        st.markdown("### Rain Intensity Analysis")
        st.caption(
            "Does it matter HOW HARD it rains, not just WHETHER it rains? "
            "Intensity is calculated as mm of precipitation per rainy hour, separating a "
            "brief downpour from a slow all-day drizzle. Baseline = No Rain days."
        )
        with st.spinner("Computing rain intensity analysis…"):
            _ri_df = _cached_rain_intensity(sellout, df_weather)

        if _ri_df.empty:
            st.info("Not enough data for rain intensity analysis.")
        else:
            _ri_routes = sorted(_ri_df["route"].dropna().unique().tolist())
            _ri_n_shops = _ri_df["customer_code"].nunique()
            st.plotly_chart(
                plot_rain_intensity(
                    _ri_df,
                    f"Rain Intensity vs Sales — All Shops ({_ri_n_shops} shops)",
                ),
                use_container_width=True, key="wa_ri_all",
            )

            st.markdown("##### OLS Regression (with DOW + month controls)")
            with st.spinner("Running OLS…"):
                _ri_ols = _cached_ri_ols(_ri_df)
            _ri_order = ["No Rain", "Drizzle (<1 mm/h)", "Moderate (1–4 mm/h)", "Heavy (>4 mm/h)"]
            st.plotly_chart(
                plot_ols_category_effect(
                    _ri_ols, _ri_order, "No Rain",
                    f"Rain Intensity — OLS Effect — All Shops ({_ri_n_shops} shops)",
                ),
                use_container_width=True, key="wa_ri_ols_all",
            )

            if _ri_routes:
                st.markdown("**By Territory**")
                _sel_ri_route = st.selectbox(
                    "Select Territory", _ri_routes, key="wa_ri_route_sel"
                )
                _ri_r = _ri_df[_ri_df["route"] == _sel_ri_route]
                if not _ri_r.empty:
                    st.plotly_chart(
                        plot_rain_intensity(
                            _ri_r,
                            f"Rain Intensity — {_sel_ri_route}",
                        ),
                        use_container_width=True, key="wa_ri_route",
                    )
                    _ri_ols_r = _cached_ri_ols(_ri_r)
                    st.plotly_chart(
                        plot_ols_category_effect(
                            _ri_ols_r, _ri_order, "No Rain",
                            f"Rain Intensity — OLS Effect — {_sel_ri_route}",
                        ),
                        use_container_width=True, key="wa_ri_ols_route",
                    )

    # ── Sunshine Fraction Analysis ─────────────────────────────────────────────
    st.markdown("---")
    show_sunshine_fraction = st.checkbox(
        "Show Sunshine Fraction Analysis",
        value=False,
        key="wa_show_sunshine_fraction",
    )
    if show_sunshine_fraction:
        st.markdown("### Sunshine Fraction Analysis")
        st.caption(
            "What fraction of daylight hours was actually sunny? "
            "Uses exact sunshine and daylight duration (in seconds) rather than discrete WMO codes, "
            "giving a continuous 0–100% measure of how sunny the day was. "
            "Only dry days are included so cloud cover and rain effects don't contaminate each other. "
            "Baseline = Overcast days (<25% sunshine)."
        )
        _sf_threshold = st.slider(
            "Max precipitation to count as a dry day (mm) — days above this are excluded",
            min_value=0.0, max_value=5.0, value=1.0, step=0.5,
            key="wa_sf_threshold",
        )
        with st.spinner("Computing sunshine fraction analysis…"):
            _sf_df = _cached_sunshine_fraction(sellout, df_weather, _sf_threshold)

        if _sf_df.empty:
            st.info("Not enough data for sunshine fraction analysis.")
        else:
            _sf_routes = sorted(_sf_df["route"].dropna().unique().tolist())
            _sf_n_shops = _sf_df["customer_code"].nunique()
            st.plotly_chart(
                plot_sunshine_fraction(
                    _sf_df,
                    f"Sunshine Fraction vs Sales — All Shops ({_sf_n_shops} shops)",
                ),
                use_container_width=True, key="wa_sf_all",
            )

            st.markdown("##### OLS Regression (with DOW + month controls)")
            with st.spinner("Running OLS…"):
                _sf_ols = _cached_sf_ols(_sf_df)
            _sf_order = [
                "Overcast (<25%)", "Partly Cloudy (25–50%)",
                "Mostly Sunny (50–75%)", "Clear (>75%)",
            ]
            st.plotly_chart(
                plot_ols_category_effect(
                    _sf_ols, _sf_order, "Overcast (<25%)",
                    f"Sunshine Fraction — OLS Effect — All Shops ({_sf_n_shops} shops)",
                ),
                use_container_width=True, key="wa_sf_ols_all",
            )

            if _sf_routes:
                st.markdown("**By Territory**")
                _sel_sf_route = st.selectbox(
                    "Select Territory", _sf_routes, key="wa_sf_route_sel"
                )
                _sf_r = _sf_df[_sf_df["route"] == _sel_sf_route]
                if not _sf_r.empty:
                    st.plotly_chart(
                        plot_sunshine_fraction(
                            _sf_r,
                            f"Sunshine Fraction — {_sel_sf_route}",
                        ),
                        use_container_width=True, key="wa_sf_route",
                    )
                    _sf_ols_r = _cached_sf_ols(_sf_r)
                    st.plotly_chart(
                        plot_ols_category_effect(
                            _sf_ols_r, _sf_order, "Overcast (<25%)",
                            f"Sunshine Fraction — OLS Effect — {_sel_sf_route}",
                        ),
                        use_container_width=True, key="wa_sf_ols_route",
                    )

    # ── Sunshine Threshold Transition Analysis ─────────────────────────────────
    st.markdown("---")
    show_sun_trans = st.checkbox(
        "Show Sunshine Threshold Transition Analysis",
        value=False,
        key="wa_show_sun_thresh_trans",
    )
    if show_sun_trans:
        st.markdown("### Sunshine Threshold Transition Analysis")
        st.caption(
            "Set a sunshine fraction threshold (0 = 0% of daylight sunny, 1 = 100%). "
            "A **Bright Day** is any day where sunshine_duration / daylight_duration ≥ threshold. "
            "For each Bright Day, the analysis looks back (up to 7 days) to find the nearest "
            "preceding day below the threshold — the **Day Before Bright**. "
            "Compares STL residuals and raw sales between those two day types."
        )

        _stt_threshold = st.slider(
            "Sunshine fraction threshold",
            min_value=0.0, max_value=1.0, value=0.8, step=0.05,
            format="%.2f",
            key="wa_stt_threshold",
        )
        st.caption(
            f"Bright Day = days where at least **{_stt_threshold * 100:.0f}%** "
            "of daylight hours were sunny."
        )

        with st.spinner("Computing sunshine transition analysis…"):
            _stt_df = _cached_sunshine_transition(sellout, df_weather, _stt_threshold)

        if _stt_df.empty:
            st.info(
                "No transition pairs found at this threshold. "
                "Try lowering the threshold to include more days."
            )
        else:
            _CAT_A = "Cloudy"
            _CAT_B = "Sunny"

            # ── Options ────────────────────────────────────────────────────
            _opt_c1, _opt_c2 = st.columns(2)
            with _opt_c1:
                _show_stl     = st.checkbox("Show STL residual chart", value=False, key="wa_stt_show_stl")
                _exclude_rain = st.checkbox("Exclude rain days from Bright Days", value=False, key="wa_stt_excl_rain")
            with _opt_c2:
                if _exclude_rain:
                    _rain_excl_thresh = st.slider(
                        "Rain threshold (mm) — days above this are excluded",
                        min_value=0.0, max_value=5.0, value=0.0, step=0.5,
                        key="wa_stt_rain_thresh",
                    )

            # Apply rain exclusion to Bright Days only
            _stt_df_view = _stt_df.copy()
            if _exclude_rain:
                _rainy_mask = (
                    (_stt_df_view["transition_cat"] == _CAT_B) &
                    (_stt_df_view["precipitation"] > _rain_excl_thresh)
                )
                _n_removed = int(_rainy_mask.sum())
                _stt_df_view = _stt_df_view[~_rainy_mask].copy()
                st.caption(
                    f"Rain exclusion active: removed **{_n_removed}** rainy Bright Day rows "
                    f"(precipitation > {_rain_excl_thresh} mm). "
                    f"**{int((_stt_df_view['transition_cat'] == _CAT_B).sum())}** Bright Days remain."
                )

            _stt_n_shops = _stt_df_view["customer_code"].nunique()
            _stt_routes  = sorted(_stt_df_view["route"].dropna().unique().tolist())
            _CAT_C = "Day After Bright"

            def _stt_agg_abs(df, cat_a, cat_b, route=None):
                """Aggregate mean raw/STL per transition category pair."""
                full = df if route is None else df[df["route"] == route]
                # Always count Bright Day dates as reference n (before filtering)
                n_transitions = int(full[full["transition_cat"] == _CAT_B]["date"].nunique())
                d = full[full["transition_cat"].isin([cat_a, cat_b])]
                per = (
                    d.groupby(["customer_code", "transition_cat"])
                    .agg(
                        mean_raw=("sales_quantity", "mean"),
                        mean_resid=("residual", "mean"),
                        mean_sales=("mean_sales", "first"),
                    )
                    .reset_index()
                )
                agg = (
                    per.groupby("transition_cat")
                    .agg(
                        mean_raw=("mean_raw", "mean"),
                        mean_resid=("mean_resid", "mean"),
                        mean_sales=("mean_sales", "mean"),
                        n_shops=("customer_code", "nunique"),
                    )
                    .reset_index()
                )
                agg["n_dates"] = n_transitions
                return agg.set_index("transition_cat").reindex([cat_a, cat_b]).reset_index()

            def _stt_agg_monthly(df, cat_a, cat_b, route=None):
                """Monthly % change from cat_a to cat_b."""
                full = df if route is None else df[df["route"] == route]
                d = full[full["transition_cat"].isin([cat_a, cat_b])].copy()
                d["_m"] = pd.to_datetime(d["date"]).dt.month
                if d.empty:
                    return pd.DataFrame()

                shop_mean = float(d["mean_sales"].mean()) if not d["mean_sales"].isna().all() else 1.0
                if shop_mean == 0:
                    shop_mean = 1.0

                per = (
                    d.groupby(["customer_code", "_m", "transition_cat"])
                    .agg(mean_raw=("sales_quantity", "mean"),
                         mean_resid=("residual", "mean"))
                    .reset_index()
                )
                raw_piv = per.pivot_table(
                    index=["customer_code", "_m"],
                    columns="transition_cat", values="mean_raw",
                )
                res_piv = per.pivot_table(
                    index=["customer_code", "_m"],
                    columns="transition_cat", values="mean_resid",
                )
                rows = []
                for idx in raw_piv.index:
                    rrow = raw_piv.loc[idx]
                    drow = res_piv.loc[idx] if idx in res_piv.index else None
                    if drow is None:
                        continue
                    ra = rrow.get(cat_a, float("nan"))
                    rb = rrow.get(cat_b, float("nan"))
                    da = drow.get(cat_a, float("nan"))
                    db = drow.get(cat_b, float("nan"))
                    if pd.isna(ra) or pd.isna(rb):
                        continue
                    rows.append({
                        "month":   idx[1],
                        "raw_pct": (rb - ra) / shop_mean * 100,
                        "stl_pct": (db - da) / shop_mean * 100
                                   if (pd.notna(da) and pd.notna(db)) else float("nan"),
                    })
                if not rows:
                    return pd.DataFrame()

                mdf = pd.DataFrame(rows).groupby("month").mean().reset_index()
                _bright_tmp = full[full["transition_cat"] == _CAT_B].copy()
                _bright_tmp["_m"] = pd.to_datetime(_bright_tmp["date"]).dt.month
                n_dates = (
                    _bright_tmp.groupby("_m")["date"].nunique()
                    .rename("n_dates").reset_index()
                    .rename(columns={"_m": "month"})
                )
                mdf = mdf.merge(n_dates, on="month", how="left")
                all_months = pd.DataFrame({"month": range(1, 13)})
                return all_months.merge(mdf, on="month", how="left")

            def _stt_pair_charts(df, cat_a, cat_b, key_prefix, n_shops, route=None):
                """Render Raw (always) + optional STL pair chart for one transition."""
                agg  = _stt_agg_abs(df, cat_a, cat_b, route=route)
                sub  = agg[agg["transition_cat"].isin([cat_a, cat_b])].copy()
                sub  = sub.set_index("transition_cat").reindex([cat_a, cat_b]).reset_index()
                shop_mean = float(agg["mean_sales"].mean()) if not agg["mean_sales"].isna().all() else 1.0
                n_str = f"{n_shops} shops" if n_shops > 1 else "1 shop"
                raw_df = pd.DataFrame({
                    "transition_cat": sub["transition_cat"],
                    "mean_val":       sub["mean_raw"],
                    "n_dates":        sub["n_dates"],
                })
                stl_df = pd.DataFrame({
                    "transition_cat": sub["transition_cat"],
                    "mean_val":       sub["mean_resid"],
                    "n_dates":        sub["n_dates"],
                })
                _col_raw, _col_stl = st.columns(2)
                with _col_raw:
                    st.plotly_chart(
                        plot_transition_pair(
                            raw_df,
                            f"<b>{cat_a} → {cat_b}</b>"
                            f"<br><sub>Raw avg daily sales · {n_str}</sub>",
                        ),
                        use_container_width=True, key=f"{key_prefix}_raw",
                    )
                if _show_stl:
                    with _col_stl:
                        st.plotly_chart(
                            plot_transition_pair(
                                stl_df,
                                f"<b>{cat_a} → {cat_b}</b>"
                                f"<br><sub>STL residual · {n_str}</sub>",
                                shop_mean=shop_mean,
                            ),
                            use_container_width=True, key=f"{key_prefix}_stl",
                        )

            # ── All Shops ──────────────────────────────────────────────────
            st.markdown(f"#### All Shops ({_stt_n_shops} shops)")

            st.markdown("**Section 1 — Day Before Bright → Bright Day**")
            _stt_pair_charts(_stt_df_view, _CAT_A, _CAT_B, "wa_stt_all_s1", _stt_n_shops)
            _stt_m1 = _stt_agg_monthly(_stt_df_view, _CAT_A, _CAT_B)
            st.plotly_chart(
                plot_transition_monthly(
                    _stt_m1, _CAT_A, _CAT_B,
                    "",
                    show_n_axis=False,
                ),
                use_container_width=True, key="wa_stt_all_s1_monthly",
            )

            _s2_vs_before = st.checkbox(
                "Compare Day After Bright against Day Before Bright (instead of Bright Day)",
                key="wa_stt_s2_vs_before",
            )
            _s2_cat_a = _CAT_A if _s2_vs_before else _CAT_B

            st.markdown(f"**Section 2 — {_s2_cat_a} → {_CAT_C}**")
            _stt_pair_charts(_stt_df_view, _s2_cat_a, _CAT_C, "wa_stt_all_s2", _stt_n_shops)
            _stt_m2 = _stt_agg_monthly(_stt_df_view, _s2_cat_a, _CAT_C)
            st.plotly_chart(
                plot_transition_monthly(
                    _stt_m2, _s2_cat_a, _CAT_C,
                    f"Monthly % Change — {_s2_cat_a} → {_CAT_C}  (threshold = {_stt_threshold:.0%})",
                    show_n_axis=False,
                ),
                use_container_width=True, key="wa_stt_all_s2_monthly",
            )

            # ── By Territory ───────────────────────────────────────────────
            if _stt_routes:
                st.markdown("**By Territory**")
                _sel_stt_route = st.selectbox(
                    "Select Territory", _stt_routes, key="wa_stt_route_sel"
                )
                _stt_r = _stt_df_view[_stt_df_view["route"] == _sel_stt_route]
                if not _stt_r.empty:
                    _stt_r_shops = _stt_r["customer_code"].nunique()

                    st.markdown("**Section 1 — Day Before Bright → Bright Day**")
                    _stt_pair_charts(_stt_r, _CAT_A, _CAT_B, "wa_stt_route_s1", _stt_r_shops, route=_sel_stt_route)
                    _stt_r_m1 = _stt_agg_monthly(_stt_r, _CAT_A, _CAT_B, route=_sel_stt_route)
                    st.plotly_chart(
                        plot_transition_monthly(
                            _stt_r_m1, _CAT_A, _CAT_B,
                            f"Monthly % Change — {_sel_stt_route} · {_CAT_A} → {_CAT_B}",
                            show_n_axis=False,
                        ),
                        use_container_width=True, key="wa_stt_route_s1_monthly",
                    )

                    st.markdown(f"**Section 2 — {_s2_cat_a} → {_CAT_C}**")
                    _stt_pair_charts(_stt_r, _s2_cat_a, _CAT_C, "wa_stt_route_s2", _stt_r_shops, route=_sel_stt_route)
                    _stt_r_m2 = _stt_agg_monthly(_stt_r, _s2_cat_a, _CAT_C, route=_sel_stt_route)
                    st.plotly_chart(
                        plot_transition_monthly(
                            _stt_r_m2, _s2_cat_a, _CAT_C,
                            f"Monthly % Change — {_sel_stt_route} · {_s2_cat_a} → {_CAT_C}",
                            show_n_axis=False,
                        ),
                        use_container_width=True, key="wa_stt_route_s2_monthly",
                    )

