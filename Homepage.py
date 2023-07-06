import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split

app_name = "TANGY MODELLER"
st.write(f"<h1 style='text-align: center;'>{app_name}</h1>", unsafe_allow_html=True)

if 'uploaded_dataset' not in st.session_state:
    st.session_state.uploaded_dataset = None

with st.sidebar: 
    st.image("icon.png")
    st.title("TANGY MODELLER")
    choice = st.radio("Select file to upload here", ["Upload","Download","Clear"])
    st.info("Free to use app, Cheers! - By Tanmay Ghosh")

if choice == "Upload":
    st.title("Upload Your Dataset")
    st.success("""This app is in real-time and does not store any data in any manner to respect data privacy. Please ensure to save the model before ending session""")
    uploaded_dataset = st.file_uploader("Upload a CSV file", type="csv")
    

    if uploaded_dataset is not None:
        df_total = pd.read_csv(uploaded_dataset, index_col=None)
        train_data, test_data = train_test_split(df_total, test_size=0.7, random_state=42)
        df = train_data
        st.session_state.uploaded_dataset = df
        st.dataframe(df)
if choice == "Download":
    st.title("Download your predicted dataset")
    st.text("Visit this section only once the model is run and best model is downloaded")
    st.warning("Predicted output dataset feature will be added in future using the trained model")
    uploaded_dataset = st.session_state.get("uploaded_dataset")
    if uploaded_dataset is not None:
        st.dataframe(uploaded_dataset)
    uploaded_file_test = st.file_uploader("Upload a testing file", type="csv")
    if uploaded_file_test is not None:
        st.dataframe(uploaded_file_test)
        # add test to global and variable for testing
    uploaded_model = st.file_uploader("Upload the model", type='pkl')
    if uploaded_model is not None:
        st.text("Predicted output dataset feature will be added in future using the trained model")
        # add model passing using Pickle Library
if choice== "Clear":
    if st.button("Clear All Data"):
        st.session_state.uploaded_dataset = None
        st.success("All Data Removed Sucessfully")
        # Run model for prediction code here
