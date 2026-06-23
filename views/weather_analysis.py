import streamlit as st
import pandas as pd
from services.data_loader import load_weather_data
from services.filters import get_fmc_only

from views.weather.rain import render_rain
from views.weather.temp import render_temp
from views.weather.wind import render_wind
from views.weather.sky import render_sky
from views.weather.sunshine import render_sunshine


def render(sellout, sellin):
    df_weather = load_weather_data()
    sellin = get_fmc_only(sellin)
    sellout = get_fmc_only(sellout)

    render_rain(sellout, df_weather, sellin)
    render_temp(sellout, df_weather, sellin)
    render_wind(sellout, df_weather, sellin)
    render_sky(sellout, df_weather)
    render_sunshine(sellout, df_weather)
