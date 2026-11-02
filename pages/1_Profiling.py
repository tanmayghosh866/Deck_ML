import streamlit as st
import data_profiling
from data_profiling import ProfileReport
import pandas as pd
import base64
from io import BytesIO
import streamlit.components.v1 as components

st.set_page_config(
    page_icon="📊",
    layout="wide",  # Options: "centered" or "wide"
    initial_sidebar_state="expanded"  # Optional: "auto", "expanded", "collapsed"
)

try:
    if st.session_state["df_manipulated"] is not None:
        df = st.session_state["df_manipulated"]
    else:
        df = None
except Exception as e:
        st.warning(f"Error loading dataset: Please load the dataset through the Homepage")


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

if st.button("Show Profiling"):
    df = st.session_state["df_manipulated"]
    st.dataframe(df)  
    with st.spinner("Generating profiling report..."):
        # 1. Generate the ydata-profiling report object
        profile_df = ProfileReport(df, title="Dataset Profile Report", explorative=True)
        profile_html = profile_df.to_html()
        components.html(profile_html, height=800, scrolling=True)

if st.button("Download report"):
    df = st.session_state["df_manipulated"]
    profile_df = ProfileReport(df, title="Dataset Profile Report", explorative=True)
    html_report = profile_df.to_html()
    href = f'data:text/html;base64,{base64.b64encode(html_report.encode()).decode()}'
    st.markdown(f'<a href="{href}" download>Download Profile Report</a>', unsafe_allow_html=True)
    

st.text("Note: The profiling report may take some time to generate for large datasets. Please be patient while the report is being created.")

