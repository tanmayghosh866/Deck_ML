import streamlit as st
from pycaret.classification import setup, compare_models, pull, save_model, load_model
import pandas as pd
import joblib
if st.session_state.uploaded_dataset is not None:
    df = st.session_state.uploaded_dataset
else:
    pass
df = st.session_state.uploaded_dataset
st.title("Regression Modelling for Numerical Target Variables")
with st.sidebar:
    st.image("reg.png")
    st.title("TANGY MODELLER")
    st.info("Free to use app, Cheers! - By Tanmay Ghosh")
    choice = st.radio("Clear Data", ["Page","Clear"])
    if choice == "Clear":
          if st.button("Clear All Data"):
            st.session_state.uploaded_dataset = None
            st.success("Data removed sucessfully")
if df is None:
    st.warning("Please Upload Data")
else:
    numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    st.markdown("#### Machine Learning Algorithm Comparison")
    chosen_target = st.selectbox('Choose the Target Column', categorical_columns)
    st.dataframe(df)
@st.cache_data

def run_model_comparison(df, chosen_target):
    setup(df, target=chosen_target)
    setup_df = pull()
    # Get the list of available models - hardcoded
    model_names = ['dt','ada','llar','br','dummy','lar','lasso','lr','ridge','en','et','omp','xgboost','catboost','gbr','rf','huber']
    # Run model comparison with progress tracking - hardcoded
    progress_bar = st.progress(0)
    progress_text = st.empty()
    num_models = len(model_names)
    for i, model_names in enumerate(model_names):
        progress_bar.progress((i + 1) / num_models)
        progress_text.text(f"Running Model {i + 1}/{num_models}")
    st.success("Comparison of various models of Regression")
    st.info("Please note, Modelling takes time depending on number of columns, please do not refresh while the app shows running in the top right, average time : 20-25 minutes")
    st.dataframe(setup_df)
    st.info("These are the ML models")
    setup(df, target=chosen_target)
    setup_df = pull()
    best_model = compare_models()
    compare_df = pull()
    st.dataframe(compare_df)
    best_model_info = compare_df.iloc[0]
    st.text(f"The best model is: {best_model_info}")

# Code to use to pull best model
code = """
def Pull_model():
    return model.pkl
    Code will be added here in future for reference
"""
st.text("Code to Pull downloaded model in Jupyter Notebook")
st.code(code, language='python')

# Run model comparison with progress bar
if st.button('Run Model Comparison'):
    run_model_comparison(df, chosen_target)
