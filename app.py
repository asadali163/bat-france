import streamlit as st
from views import weather_analysis, event_analysis, forecasting
from services.data_loader import load_sell_data


st.set_page_config(
    page_title="Streamlit App",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    sellin, sellout = load_sell_data()
    st.title("Streamlit App")
    st.caption(
        f"Sell-in: **{len(sellin):,}** rows | "
        f"Sell-out: **{len(sellout):,}** rows | "
        f"Customers: **{sellin['customer_code'].nunique()}** | "
        f"SKUs: **{sellin['sku_code'].nunique()}**"
    )

    tab_analysis, events_analysis, forecasting_analysis = st.tabs(
        ["Weather Analysis", "Event Analysis", "Forecasting"]
    )

    with tab_analysis:
        weather_analysis.render(sellout=sellout, sellin=sellin)

    with events_analysis:
        event_analysis.render(sellout=sellout, sellin=sellin)

    with forecasting_analysis:
        forecasting.render(sellout, sellin)


if __name__ == "__main__":
    main()
