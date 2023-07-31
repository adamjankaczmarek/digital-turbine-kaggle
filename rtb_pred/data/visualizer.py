import pandas as pd
import pandas_profiling
import streamlit as st
from streamlit_pandas_profiling import st_profile_report
from ydata_profiling import ProfileReport

df = pd.read_csv("../../data/train_data.csv")[:10000]

profile = ProfileReport(
    df,
    title="RTB",
)


st.title("Pandas Profiling in Streamlit!")
st.write(df)
st_profile_report(profile)
