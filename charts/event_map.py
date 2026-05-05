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
