import os
import streamlit as st
import pandas as pd
from charts.weather_charts import (
    plot_sunshine_fraction,
    plot_ols_category_effect,
    plot_transition_pair,
    plot_transition_monthly,
)
from services.porcessors import (
    compute_sunshine_fraction_analysis,
    compute_sunshine_fraction_ols,
    compute_sunshine_transition_analysis,
)


# ── Cached helpers ────────────────────────────────────────────────────────────


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


def render_sunshine(sellout, df_weather):
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
