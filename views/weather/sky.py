import os
import streamlit as st
import pandas as pd
from charts.weather_charts import (
    plot_sky_pct_bars,
    plot_sky_territory_bars,
    plot_sky_shop_bars,
    plot_gap_pct_bars,
    plot_gap_monthly,
    plot_storm_pct_bars,
    plot_storm_monthly,
    plot_transition_pair,
    plot_transition_monthly,
    plot_sunny_temp_combined,
    plot_snow_analysis,
)
from services.porcessors import (
    compute_sky_analysis,
    compute_customer_sku_sky,
    compute_gap_analysis,
    compute_storm_analysis,
    compute_sunny_transition_analysis,
    compute_sunny_temp_combined,
    compute_snow_analysis,
    compute_snow_ols,
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


# ── Cached helpers ────────────────────────────────────────────────────────────


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
def _cached_sunny_temp_combined(trans_df: pd.DataFrame, df_weather: pd.DataFrame, _v=3) -> pd.DataFrame:
    return compute_sunny_temp_combined(trans_df, df_weather)


@st.cache_data(show_spinner=False)
def _cached_snow(sellout: pd.DataFrame, df_weather: pd.DataFrame) -> pd.DataFrame:
    return compute_snow_analysis(sellout, df_weather)


@st.cache_data(show_spinner=False)
def _cached_sn_ols(base_df: pd.DataFrame) -> pd.DataFrame:
    return compute_snow_ols(base_df)


@st.cache_data(show_spinner=False)
def _load_ols_cache(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def render_sky(sellout, df_weather):
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
