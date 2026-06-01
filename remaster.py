import pandas as pd
import streamlit as st

st.title("Mon Dashboard")
df = pd.read_csv("final.csv")
st.dataframe(df)
st.line_chart(df)
