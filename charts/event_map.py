import pandas as pd
import folium


def plot_event_map(shop_lat, shop_lon, shop_name, df_events, max_distance_m):
    m = folium.Map(location=[shop_lat, shop_lon], zoom_start=14)

    # Distance radius circle
    folium.Circle(
        location=[shop_lat, shop_lon],
        radius=max_distance_m,
        color="steelblue",
        fill=True,
        fill_opacity=0.1,
    ).add_to(m)

    # Shop marker
    folium.Marker(
        location=[shop_lat, shop_lon],
        popup=folium.Popup(f"<b>{shop_name}</b>", max_width=200),
        tooltip=shop_name,
        icon=folium.Icon(color="blue", icon="store", prefix="fa"),
    ).add_to(m)

    # Event markers
    for _, row in df_events.iterrows():
        popup_html = f"""
            <b>{row['name']}</b><br>
            📅 {row['date'].date()} {row.get('time', '')}<br>
            📍 {row['venue']}<br>
            👥 Capacity: {row.get('estimated_capacity', 'N/A')}<br>
            📏 {int(row['distance_m'])} m away
        """
        folium.Marker(
            location=[row["venue_lat"], row["venue_lon"]],
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=row["name"],
            icon=folium.Icon(color="red", icon="flag", prefix="fa"),
        ).add_to(m)

    return m


def plot_shops_for_event(event_name, event_lat, event_lon, df_shops, max_distance_m):
    m = folium.Map(location=[event_lat, event_lon], zoom_start=14)

    # Distance radius circle
    folium.Circle(
        location=[event_lat, event_lon],
        radius=max_distance_m,
        color="crimson",
        fill=True,
        fill_opacity=0.1,
    ).add_to(m)

    # Event marker
    folium.Marker(
        location=[event_lat, event_lon],
        popup=folium.Popup(f"<b>{event_name}</b>", max_width=200),
        tooltip=event_name,
        icon=folium.Icon(color="red", icon="flag", prefix="fa"),
    ).add_to(m)

    # Shop markers
    seen = set()
    for _, row in df_shops.iterrows():
        key = (row["shop_lat"], row["shop_lon"])
        if key in seen:
            continue
        seen.add(key)
        popup_html = f"""
            <b>{row.get('customer_name', 'Shop')}</b><br>
            📏 {int(row['distance_m'])} m from event
        """
        folium.Marker(
            location=[row["shop_lat"], row["shop_lon"]],
            popup=folium.Popup(popup_html, max_width=200),
            tooltip=row.get("customer_name", "Shop"),
            icon=folium.Icon(color="blue", icon="store", prefix="fa"),
        ).add_to(m)

    return m


_EVT_COLORS = {"district": "green", "in_range": "blue", "too_far": "orange"}

_LEGEND_HTML = """
<div style="position:fixed;bottom:30px;left:30px;z-index:1000;background:white;
            padding:10px 14px;border-radius:8px;font-size:12px;line-height:1.8;
            box-shadow:2px 2px 6px rgba(0,0,0,0.3)">
  <div><span style="color:crimson;font-size:16px">&#9679;</span> Local radius (500m)</div>
  <div><span style="color:crimson;font-size:16px">&#9679;</span> Outlet</div>
  <div><span style="color:blue;font-size:16px">&#9679;</span> Local (in range)</div>
  <div><span style="color:orange;font-size:16px">&#9679;</span> Local (too far)</div>
  <div><span style="color:green;font-size:16px">&#9679;</span> District (always on)</div>
</div>
"""

_MAP_FOOTER = (
    "<div style='text-align:center;font-size:11px;color:#555;padding:6px 10px;"
    "background:#f8f8f8;border-top:1px solid #ddd'>"
    "<span style='color:crimson'>&#9679;</span> Red&nbsp;= outlet + catchment (500m)&nbsp;|&nbsp;"
    "<span style='color:green'>&#9679;</span> Green&nbsp;= district events (always on)&nbsp;|&nbsp;"
    "<span style='color:blue'>&#9679;</span> Blue&nbsp;= local events (in range) + blue circles&nbsp;= impact radius&nbsp;|&nbsp;"
    "<span style='color:orange'>&#9679;</span> Orange&nbsp;= local events (too far)&nbsp;|&nbsp;"
    "Included if red + blue circles overlap</div>"
)


def plot_event_map_v2(shop_lat, shop_lon, shop_name, df_events):
    """Map with event impact-radius circles; events coloured by district/in_range/too_far."""
    m = folium.Map(location=[shop_lat, shop_lon], zoom_start=14)

    # Shop 500 m catchment circle
    folium.Circle(
        location=[shop_lat, shop_lon],
        radius=500,
        color="crimson",
        fill=True,
        fill_opacity=0.08,
        weight=2,
        tooltip="Local radius (500m)",
    ).add_to(m)

    # Shop marker
    folium.CircleMarker(
        location=[shop_lat, shop_lon],
        radius=9,
        color="crimson",
        fill=True,
        fill_color="crimson",
        fill_opacity=1.0,
        tooltip=shop_name,
        popup=folium.Popup(f"<b>{shop_name}</b><br>Outlet", max_width=200),
    ).add_to(m)

    for _, row in df_events.iterrows():
        evt_color = _EVT_COLORS.get(row.get("event_type", "too_far"), "gray")
        cap_str = f"{int(row['estimated_capacity']):,}" if pd.notna(row.get("estimated_capacity")) else "N/A"
        popup_html = (
            f"<b>{row['name']}</b><br>"
            f"&#128197; {row['date'].date()}&nbsp;{row.get('time','')}<br>"
            f"&#128205; {row['venue']}<br>"
            f"&#128101; Capacity: {cap_str}<br>"
            f"&#128207; {int(row['distance_m'])} m away<br>"
            f"&#127919; Impact radius: {800 if row.get('source') == 'df_e_r' else int(row['impact_radius_m'])} m<br>"
            f"<i>{row.get('event_type','').replace('_',' ').title()}</i>"
        )
        if row.get("event_type") in ("in_range", "district"):
            circle_radius = 800 if row.get("source") == "df_e_r" else int(row["impact_radius_m"])
            folium.Circle(
                location=[row["venue_lat"], row["venue_lon"]],
                radius=circle_radius,
                color=evt_color,
                fill=True,
                fill_opacity=0.10,
                weight=1.5,
            ).add_to(m)

        folium.CircleMarker(
            location=[row["venue_lat"], row["venue_lon"]],
            radius=7,
            color=evt_color,
            fill=True,
            fill_color=evt_color,
            fill_opacity=0.9,
            tooltip=row["name"],
            popup=folium.Popup(popup_html, max_width=270),
        ).add_to(m)

    m.get_root().html.add_child(folium.Element(_LEGEND_HTML))
    m.get_root().html.add_child(folium.Element(_MAP_FOOTER))
    return m
