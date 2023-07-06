import streamlit as st
from ydata_profiling import profile_report
import pandas as pd
import base64
from streamlit_pandas_profiling import st_profile_report

if st.session_state.uploaded_dataset is not None:
    df = st.session_state.uploaded_dataset

st.header("EDA Profiling for dataset")
with st.sidebar:
    st.image("Profile_2.png")
    st.title("TANGY MODELLER")
    choice = st.radio("Clear Data", ["Page","Clear"])
    st.info("Free to use app, Cheers! - By Tanmay Ghosh")
if choice == "Clear":
    if st.button("Clear All Data"):
        st.session_state.uploaded_dataset = None
        st.success("Data removed sucessfully")  
st.title("Show Graphs - need to code")
if st.button("Show Profiling"):
        df = st.session_state.uploaded_dataset
        st.dataframe(df)
        profile_df = df.profile_report()
        st_profile_report(profile_df)

if st.button("Download report"):
    df = st.session_state.uploaded_dataset
    profile_df = df.profile_report()
    profile_html = st_profile_report(profile_df)
    file_path = "Profile_report.html"
    profile_html.to_file(file_path)
    st.session_state["download_file_path"] = file_path
    st.success("Report download link generated successfully")
    if "download_file_path" in st.session_state:
        file_path = st.session_state["download_file_path"]
        st.markdown(f'<a href="{file_path}" download>Download Profile Report</a>', unsafe_allow_html=True)

