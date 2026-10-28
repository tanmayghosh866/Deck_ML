import streamlit as st
import data_profiling
from data_profiling import ProfileReport
import pandas as pd
import base64
from io import BytesIO
import streamlit.components.v1 as components

if st.session_state.uploaded_dataset is not None:
    df = st.session_state.uploaded_dataset
st.text("test text")
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
# if st.button("Show Profiling"):
#         df = st.session_state.uploaded_dataset
#         st.dataframe(df)
#         profile_df = df.profile_report()
#         st_profile_report(profile_df)
        
if st.button("Show Profiling"):
    df = st.session_state.uploaded_dataset
    st.dataframe(df)  
    with st.spinner("Generating profiling report..."):
        # 1. Generate the ydata-profiling report object
        profile_df = ProfileReport(df, title="Dataset Profile Report", explorative=True)
        profile_html = profile_df.to_html()
        components.html(profile_html, height=800, scrolling=True)

if st.button("Download report"):
    df = st.session_state.uploaded_dataset
    profile_df = ProfileReport(df, title="Dataset Profile Report", explorative=True)
    html_report = profile_df.to_html()
    href = f'data:text/html;base64,{base64.b64encode(html_report.encode()).decode()}'
    st.markdown(f'<a href="{href}" download>Download Profile Report</a>', unsafe_allow_html=True)
    

st.text("Note: The profiling report may take some time to generate for large datasets. Please be patient while the report is being created.")

