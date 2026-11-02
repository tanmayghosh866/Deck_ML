import streamlit as st
import pandas as pd

st.set_page_config(
    page_icon="📊",
    layout="wide",  # Options: "centered" or "wide"
    initial_sidebar_state="expanded"  # Optional: "auto", "expanded", "collapsed"
)


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
        df_total = pd.read_csv(uploaded_dataset, index_col=None, sep=r'[,;|]', engine='python')
        def clean_dataframe(df, chars_to_remove=r'["/]'):
            df.columns = df.columns.str.replace(chars_to_remove, '', regex=True)
            for col in df.select_dtypes(include=['object', 'string']).columns:
                df[col] = df[col].str.replace(chars_to_remove, '', regex=True)

            return df
        df = clean_dataframe(df_total)

        st.session_state.uploaded_dataset = df_total
        st.dataframe(df_total)
        

        # 1. Initialize session state for the dataframe if it doesn't exist
        if "df_manipulated" not in st.session_state:
            st.session_state["df_manipulated"] = None

        st.header("CSV Data Manipulator")
        st.divider()
        # 2. File Uploader
        uploaded_file = df_total

        if uploaded_file is not None:
            # Read the original CSV only once (cached) to prevent re-reading on every rerun

            df_raw = uploaded_file

            st.subheader("Data Manipulation Tools")

            # 3. Step 1: Remove Columns
            all_columns = df_raw.columns.tolist()
            columns_to_keep = st.multiselect(
                "Select columns to KEEP:", 
                options=all_columns, 
                default=all_columns
            )

            if columns_to_keep:
                # Filter dataframe to kept columns
                df_filtered = df_raw[columns_to_keep].copy()

                # 4. Step 2: Change Column Headers & Edit Cells inline
                st.write("Edit cell values directly, or double-click headers in the editor below to rename them:")

                # st.data_editor allows both cell editing and column renaming via the UI
                edited_df = st.data_editor(
                    df_filtered, 
                    key="csv_editor", 
                    use_container_width=True,
                    num_rows="dynamic" # Allows users to add/delete rows if needed
                )

                # 5. Save Button to commit changes to Session State
                if st.button("Save Changes to Session State", type="primary"):
                    st.session_state["df_manipulated"] = edited_df
                    st.success("Dataframe saved successfully! Available across other pages via `st.session_state['df_manipulated']`.")

        # 6. Preview current global state
        if st.session_state["df_manipulated"] is not None:
            st.subheader("Current Global Dataframe State (Available to other pages)")
            st.dataframe(st.session_state["df_manipulated"])

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
