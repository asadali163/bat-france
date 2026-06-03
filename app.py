import streamlit as st
from views import weather_analysis, event_analysis, event_analysis2, forecasting
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

    # tab_analysis, tab_events, tab_events2, tab_forecasting = st.tabs(
    #     ["Weather Analysis", "Event Analysis 2", "Event Analysis", "Forecasting"]
    # )

    tab_analysis, tab_events2 = st.tabs(["Weather Analysis", "Event Analysis"])

    with tab_analysis:
        weather_analysis.render(sellout=sellout, sellin=sellin)

    # with tab_events:
    #     event_analysis.render(sellout=sellout, sellin=sellin)

    with tab_events2:
        event_analysis2.render(sellout=sellout, sellin=sellin)

    # with tab_forecasting:
    #     pass


if __name__ == "__main__":
    main()
