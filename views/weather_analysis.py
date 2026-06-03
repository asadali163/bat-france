import os
import streamlit as st
import pandas as pd
from services.data_loader import load_weather_data
from services.filters import get_customer_list_weather, get_fmc_only
from charts.weather_charts import (
    plot_customer_weather,
    plot_rain_band_chart,
    plot_ols_rain_chart,
    plot_ols_rain_effect,
    plot_temp_contribution,
    plot_prophet_seasonality,
)
from services.porcessors import (
    rain_band_processor,
    ols_rain_processor,
    run_ols_rain_all_shops,
    aggregate_ols_rain,
    run_prophet_ols_all_shops,
    compute_temp_contribution,
    compute_prophet_seasonality,
)

OLS_CACHE_PATH        = "./data/cache/ols_rain_all.parquet"
PROPHET_CACHE_PATH    = "./data/cache/prophet_ols_all.parquet"
SEASONALITY_CACHE_PATH = "./data/cache/prophet_seasonality.parquet"


# ── Cached helpers ────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _merge_weather(sellout_fmc: pd.DataFrame, df_weather: pd.DataFrame) -> pd.DataFrame:
    merged = sellout_fmc.merge(df_weather, on=["date", "latitude", "longitude"], how="left")
    merged["is_rain"] = merged["precipitation"] > 4
    return merged


@st.cache_data(show_spinner=False)
def _cached_weather_chart(customer_df: pd.DataFrame, rain_range: tuple, robust: bool):
    return plot_customer_weather(customer_df, rain_range, robust)


@st.cache_data(show_spinner=False)
def _cached_rain_band(customer_df: pd.DataFrame):
    return rain_band_processor(customer_df)


@st.cache_data(show_spinner=False)
def _cached_ols(customer_df: pd.DataFrame):
    return ols_rain_processor(customer_df)


@st.cache_data(show_spinner=False)
def _load_ols_cache(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def _cached_agg_ols(results_df: pd.DataFrame, route=None) -> pd.DataFrame:
    return aggregate_ols_rain(results_df, route)


@st.cache_data(show_spinner=False)
def _load_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def _cached_temp_contribution(prophet_df, sellout, df_weather, route=None):
    return compute_temp_contribution(prophet_df, sellout, df_weather, route)


@st.cache_data(show_spinner=False)
def _cached_seasonality(seasonality_df, route=None):
    return compute_prophet_seasonality(seasonality_df, route)


# ── Main render ───────────────────────────────────────────────────────────────

def render(sellout, sellin):
    df_weather = load_weather_data()

    sellin  = get_fmc_only(sellin)
    sellout = get_fmc_only(sellout)

    # ── Customer-level analysis ───────────────────────────────────────────────
    sellout_merged = _merge_weather(sellout, df_weather)

    col1, col2 = st.columns(2)
    with col1:
        customer_list    = get_customer_list_weather(sellout_merged)
        selected_customer = st.selectbox(
            "Select Customer", customer_list, key="customer_weather"
        )

    selected_customer_df = sellout_merged[
        sellout_merged["customer_code"] == selected_customer
    ]

    with col2:
        customer_max_rain = float(selected_customer_df["precipitation"].max())
        rain_range = st.slider(
            "Rain Range (mm)",
            min_value=0.0,
            max_value=customer_max_rain,
            value=(0.0, customer_max_rain),
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

    if os.path.exists(OLS_CACHE_PATH):
        results_df = _load_ols_cache(OLS_CACHE_PATH)
    else:
        st.info(
            "No cached results found. Running OLS for all shops — "
            "this takes ~15 minutes and will be saved for future visits."
        )
        with st.spinner("Running OLS analysis for all shops…"):
            results_df = run_ols_rain_all_shops(sellout, df_weather)
            os.makedirs(os.path.dirname(OLS_CACHE_PATH), exist_ok=True)
            results_df.to_parquet(OLS_CACHE_PATH, index=False)
        st.success("Analysis complete. Results saved to cache.")

    if results_df.empty:
        st.warning("No results — check that weather and sales data overlap.")
        return

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
        n_route   = int(results_df[results_df["route"] == selected_route]["customer_code"].nunique())
        agg_route = _cached_agg_ols(results_df, route=selected_route)
        fig_route = plot_ols_rain_effect(
            agg_route,
            f"<b>Territory {selected_route} — {n_route} shops</b><br>"
            "<sub>Mean % effect on sales by rainfall band vs dry days (error bars = 95% CI)</sub>",
        )
        st.plotly_chart(fig_route, use_container_width=True, key="wa_ols_route_chart")

    st.markdown("---")
    if st.button("🔄 Regenerate Rain OLS Cache", key="wa_ols_regen"):
        if os.path.exists(OLS_CACHE_PATH):
            os.remove(OLS_CACHE_PATH)
        _load_ols_cache.clear()
        _cached_agg_ols.clear()
        st.rerun()

    # ── Temperature Effect ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Temperature Effect on Sales")
    st.caption(
        "Prophet regressor β × actual temperature = estimated sales units "
        "added/removed by temperature each month. "
        "Averaged across all FMC shops."
    )

    both_cached = os.path.exists(PROPHET_CACHE_PATH) and os.path.exists(SEASONALITY_CACHE_PATH)
    if both_cached:
        prophet_df     = _load_parquet(PROPHET_CACHE_PATH)
        seasonality_df = _load_parquet(SEASONALITY_CACHE_PATH)
    else:
        st.info(
            "No Prophet cache found. Running Prophet + OLS for all shops — "
            "this takes ~15–20 minutes and will be saved for future visits."
        )
        prog_bar  = st.progress(0)
        prog_text = st.empty()

        def _on_progress(current, total, fitted, skipped):
            prog_bar.progress(current / total)
            prog_text.text(
                f"Shop {current} / {total}  —  "
                f"fitted: {fitted}  |  skipped: {skipped}"
            )

        prophet_df, seasonality_df = run_prophet_ols_all_shops(
            sellout, df_weather, progress_callback=_on_progress
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
        n_shops_p  = prophet_df["customer_code"].nunique()
        temp_routes = sorted(prophet_df["route"].dropna().unique().tolist())

        # ── Temperature contribution chart ────────────────────────────────────
        t_avg, s_avg, n_c = _cached_temp_contribution(prophet_df, sellout, df_weather)
        fig_temp_all = plot_temp_contribution(
            t_avg, s_avg, n_c,
            f"Prophet Temperature Regressor Contribution — Avg Across {n_c} Customers",
        )
        st.plotly_chart(fig_temp_all, use_container_width=True, key="wa_temp_all")

        st.markdown("#### Temperature Contribution by Territory")
        if temp_routes:
            sel_temp_route = st.selectbox(
                "Select Territory (Route)", temp_routes, key="wa_temp_route"
            )
            t_r, s_r, n_r = _cached_temp_contribution(
                prophet_df, sellout, df_weather, route=sel_temp_route
            )
            fig_temp_route = plot_temp_contribution(
                t_r, s_r, n_r,
                f"Territory {sel_temp_route} — {n_r} Customers",
            )
            st.plotly_chart(fig_temp_route, use_container_width=True, key="wa_temp_route_chart")

        # ── Prophet yearly seasonality chart ──────────────────────────────────
        st.markdown("---")
        st.markdown("### Prophet Yearly Seasonality")
        st.caption(
            "Average yearly seasonality component from Prophet across all shops. "
            "Positive = above-average sales period; negative = below-average. "
            "This is after controlling for temperature, rain, and weekday effects."
        )

        seas_avg = _cached_seasonality(seasonality_df)
        if not seas_avg.empty:
            fig_seas = plot_prophet_seasonality(
                seas_avg, n_shops_p,
                f"Prophet Yearly Seasonality — Avg Across {n_shops_p} Shops",
            )
            st.plotly_chart(fig_seas, use_container_width=True, key="wa_seas_all")

        st.markdown("#### Seasonality by Territory")
        if temp_routes:
            sel_seas_route = st.selectbox(
                "Select Territory (Route)", temp_routes, key="wa_seas_route"
            )
            seas_r = _cached_seasonality(seasonality_df, route=sel_seas_route)
            n_r_s  = int(seasonality_df[seasonality_df["route"] == sel_seas_route]
                         ["customer_code"].nunique())
            fig_seas_r = plot_prophet_seasonality(
                seas_r, n_r_s,
                f"Territory {sel_seas_route} — {n_r_s} Shops",
            )
            st.plotly_chart(fig_seas_r, use_container_width=True, key="wa_seas_route_chart")

    st.markdown("---")
    if st.button("🔄 Regenerate Prophet Cache", key="wa_prophet_regen"):
        for p in [PROPHET_CACHE_PATH, SEASONALITY_CACHE_PATH]:
            if os.path.exists(p):
                os.remove(p)
        _load_parquet.clear()
        _cached_temp_contribution.clear()
        _cached_seasonality.clear()
        st.rerun()
