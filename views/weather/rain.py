import os
import streamlit as st
import pandas as pd
import calendar as _cal
from charts.weather_charts import (
    plot_customer_weather,
    plot_rain_band_chart,
    plot_ols_rain_chart,
    plot_ols_rain_effect,
    plot_ols_rain_band_effect,
    plot_daily_rainfall,
    plot_rain_streak,
    plot_ols_category_effect,
    plot_rain_intensity,
)
from services.filters import get_customer_list_weather
from services.porcessors import (
    rain_band_processor,
    ols_rain_processor,
    run_ols_rain_all_shops,
    aggregate_ols_rain,
    run_ols_rain_band_all_shops,
    aggregate_ols_rain_band,
    compute_prophet_rain_curve,
    compute_rain_streak_analysis,
    compute_rain_streak_ols,
    compute_rain_intensity_analysis,
    compute_rain_intensity_ols,
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
def _load_parquet(path: str) -> pd.DataFrame:
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
def _cached_prophet_rain_curve(prophet_df, sellout, route=None):
    return compute_prophet_rain_curve(prophet_df, sellout, route)


@st.cache_data(show_spinner=False)
def _cached_rain_streak(sellout: pd.DataFrame, df_weather: pd.DataFrame, threshold: float = 1.0) -> pd.DataFrame:
    return compute_rain_streak_analysis(sellout, df_weather, rain_threshold_mm=threshold)


@st.cache_data(show_spinner=False)
def _cached_rs_ols(base_df: pd.DataFrame) -> pd.DataFrame:
    return compute_rain_streak_ols(base_df)


@st.cache_data(show_spinner=False)
def _cached_rain_intensity(sellout: pd.DataFrame, df_weather: pd.DataFrame) -> pd.DataFrame:
    return compute_rain_intensity_analysis(sellout, df_weather)


@st.cache_data(show_spinner=False)
def _cached_ri_ols(base_df: pd.DataFrame) -> pd.DataFrame:
    return compute_rain_intensity_ols(base_df)


def render_rain(sellout, df_weather, sellin):
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
