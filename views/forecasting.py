import streamlit as st
import pandas as pd
from services.filters import get_fmc_only, get_customer_list_events


def render(sellin: pd.DataFrame, sellout: pd.DataFrame):
    sellin = get_fmc_only(sellin)
    sellout = get_fmc_only(sellout)

    col1, col2 = st.columns(2)
    with col1:
        customer_list = get_customer_list_events(sellout)
        selected_customer = st.selectbox(
            "Select Customer", customer_list, key="customer_forecasting"
        )
        print("Selected Customer is : ", selected_customer)
    with col2:
        st.write("Second Column")
