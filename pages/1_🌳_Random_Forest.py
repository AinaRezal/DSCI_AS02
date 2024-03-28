import streamlit as st
import pandas as pd
import plotly_express as px
import pydeck as pdk

st.set_page_config(layout="wide",
                   initial_sidebar_state="collapsed",
                   page_title='Random Forest'
                   )

st.markdown("<h1 style='text-align: center; color: black;'>Random Forest</h1>", unsafe_allow_html=True)

df_eq = pd.read_csv('cleaned_earthquake.csv')
df_dis = pd.read_csv('cleaned_disaster.csv')