import os
import streamlit as st
import pandas as pd
from charts.weather_charts import (
    plot_ols_temp_effect,
    plot_temp_contribution,
    plot_prophet_seasonality,
    plot_temp_swing,
    plot_ols_category_effect,
)
from services.porcessors import (
    run_prophet_ols_all_shops,
    run_prophet_temp_all_shops,
    compute_prophet_temp_curve,
    compute_temp_contribution,
    compute_prophet_seasonality,
    compute_temp_swing_analysis,
    compute_temp_swing_ols,
    run_ols_temp_all_shops,
    aggregate_ols_temp,
    run_ols_fl_all_shops,
    aggregate_ols_fl,
    run_prophet_fl_all_shops,
    compute_prophet_fl_curve,
)

OLS_TEMP_CACHE_PATH = "./data/cache/ols_temp_all.parquet"
OLS_FL_CACHE_PATH = "./data/cache/ols_fl_all.parquet"
PROPHET_FL_CACHE_PATH = "./data/cache/prophet_fl_all.parquet"
PROPHET_CACHE_PATH = "./data/cache/prophet_ols_all.parquet"
SEASONALITY_CACHE_PATH = "./data/cache/prophet_seasonality.parquet"
PROPHET_TEMP_CACHE_PATH = "./data/cache/prophet_temp_all.parquet"


# ── Cached helpers ────────────────────────────────────────────────────────────


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
def _cached_temp_contribution(prophet_df, sellout, df_weather, route=None):
    return compute_temp_contribution(prophet_df, sellout, df_weather, route)


@st.cache_data(show_spinner=False)
def _cached_temp_swing(sellout: pd.DataFrame, df_weather: pd.DataFrame, _v=2) -> pd.DataFrame:
    return compute_temp_swing_analysis(sellout, df_weather)


@st.cache_data(show_spinner=False)
def _cached_ts_ols(base_df: pd.DataFrame) -> pd.DataFrame:
    return compute_temp_swing_ols(base_df)


@st.cache_data(show_spinner=False)
def _cached_seasonality(seasonality_df, route=None):
    return compute_prophet_seasonality(seasonality_df, route)


@st.cache_data(show_spinner=False)
def _load_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


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
def _cached_fl_contribution(prophet_df, sellout, df_weather, route=None):
    return compute_temp_contribution(
        prophet_df, sellout, df_weather, route,
        beta_col="prophet_apparent_temperature_mean",
        temp_col="apparent_temperature_mean",
    )


def render_temp(sellout, df_weather, sellin):
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
