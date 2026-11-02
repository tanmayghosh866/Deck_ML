import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from imblearn.over_sampling import SMOTE, RandomOverSampler


st.set_page_config(
    page_icon="📊",
    layout="wide",  # Options: "centered" or "wide"
    initial_sidebar_state="expanded"  # Optional: "auto", "expanded", "collapsed"
)


        
    
st.title("Classification Modelling for Categorical Target Variables")
with st.sidebar:
    st.image("class.png")
    st.title("TANGY MODELLER")
    st.info("Free to use app, Cheers! - By Tanmay Ghosh")
    choice = st.radio("Clear Data", ["Page","Clear"])
    if choice == "Clear":
          if st.button("Clear All Data"):
            st.session_state.uploaded_dataset = None
            st.success("Data removed sucessfully")
            
try:
    if st.session_state["df_manipulated"] is not None:
        df = st.session_state["df_manipulated"]
    else:
        df = None
except Exception as e:
        st.warning(f"Error loading dataset: Please load the dataset through the Homepage")
else:
    numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    st.markdown("#### Machine Learning Algorithm Comparison")
    chosen_target = st.selectbox('Choose the Target Column', categorical_columns)
    st.dataframe(df)
    
# Tasks:
    
    
# If there is class imbalance First Handle class imbalance using SMOTE or Random Oversampling, then apply the following classification models:
# Then measure the accuracy, precision, recall, F1-score, and AUC-ROC for each model and compare them to select the best model for your dataset.
# create plots for each model's performance metrics and display them in the Streamlit app.
# at the end create a summary table with the performance metrics of each model and display it in the Streamlit app.
    
# Logistic Regression
# k-Nearest Neighbors (KNN)
# Decision Tree Classifier
# Naive Bayes Classifier
# Gradient Boosting Classifier
# Random Forest Classifier

if st.button("Run Model Comparison"):
    # Separate X and y
    X = df.drop(columns=[chosen_target])
    y = df[chosen_target]
    # Remove rows with missing values
    data = pd.concat([X, y], axis=1).dropna()
    X = data.drop(columns=[chosen_target])
    y = data[chosen_target]
    
    # Encode categorical independent variables
    X = pd.get_dummies(X, drop_first=True)
    st.dataframe(X)
    # Encode target variable
    le = LabelEncoder()
    y = le.fit_transform(y)
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    st.success("Dataset prepared successfully!")
    
    # Check class distribution

    class_counts = pd.Series(y_train).value_counts()
    imbalance_ratio = class_counts.max() / class_counts.min()
    st.markdown("### Class Imbalance Detection")
    fig, ax = plt.subplots()
    sns.barplot(
        x=class_counts.index.astype(str),
        y=class_counts.values,
        ax=ax
    )
    ax.set_title("Class Distribution Before Resampling")
    ax.set_xlabel("Classes")
    ax.set_ylabel("Count")
    st.pyplot(fig)
    if imbalance_ratio > 2:
        st.warning("Class imbalance detected!")
        resampling_method = st.selectbox(
            "Choose Imbalance Handling Technique",
            ["SMOTE", "Random Oversampling"]
        )
    else:
        st.success("No significant class imbalance detected.")
        resampling_method = st.selectbox(
            "Choose Imbalance Handling Technique",
            ["No Resampling"]
        )
        
    if resampling_method == "SMOTE":
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(
            X_train,
            y_train
        )

        st.success("SMOTE applied successfully!")

    elif resampling_method == "Random Oversampling":

        ros = RandomOverSampler(random_state=42)
        X_train, y_train = ros.fit_resample(
                    X_train,
                    y_train
                )
        st.success("Random Oversampling applied successfully!")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    models = {

                "Logistic Regression": LogisticRegression(
                    max_iter=1000,
                    random_state=42),

                "KNN": KNeighborsClassifier(
                    n_neighbors=5),

                "Decision Tree": DecisionTreeClassifier(
                    random_state=42),

                "Naive Bayes": GaussianNB(),

                "Gradient Boosting": GradientBoostingClassifier(
                    random_state=42),

                "Random Forest": RandomForestClassifier(
                    n_estimators=100,
                    random_state=42)

    }
    results = []

    trained_models = {}

    for model_name, model in models.items():

        st.markdown(f"### {model_name}")
        st.divider()
        # Train model
        model.fit(X_train_scaled, y_train)
        # Predictions
        y_pred = model.predict(X_test_scaled)
        # Probability predictions
        y_prob = model.predict_proba(X_test_scaled)
        # Evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(
                    y_test,
                    y_pred,
                    average='weighted',
                    zero_division=0
        )

        recall = recall_score(
                    y_test,
                    y_pred,
                    average='weighted',
                    zero_division=0
                )

        f1 = f1_score(
                    y_test,
                    y_pred,
                    average='weighted',
                    zero_division=0
                )

                # AUC ROC

        try:

            if len(np.unique(y_test)) == 2:

                        auc = roc_auc_score(
                            y_test,
                            y_prob[:, 1]
                        )

            else:

                        auc = roc_auc_score(
                            y_test,
                            y_prob,
                            multi_class='ovr',
                            average='weighted',
                            labels=model.classes_
                        )

        except ValueError:
                    auc = np.nan

                # Store results

        results.append({
                    "Model": model_name,
                    "Accuracy": accuracy,
                    "Precision": precision,
                    "Recall": recall,
                    "F1 Score": f1,
                    "AUC ROC": auc

                })

        trained_models[model_name] = model

        # Display metrics

        col1, col2, col3, col4, col5 = st.columns(5)

        col1.metric("Accuracy", f"{accuracy:.3f}")
        col2.metric("Precision", f"{precision:.3f}")
        col3.metric("Recall", f"{recall:.3f}")
        col4.metric("F1 Score", f"{f1:.3f}")
        col5.metric("AUC ROC", f"{auc:.3f}" if not np.isnan(auc) else "N/A")
        # Plot performance metric
        metric_names = [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "AUC ROC"
        ]
        metric_values = [
            accuracy,
            precision,
            recall,
            f1,
            auc
        ]
        fig, ax = plt.subplots(figsize=(4, 4))
        sns.barplot(
            x=metric_names,
            y=metric_values,
            ax=ax)
        
        ax.set_title(f"{model_name} - Performance Metrics")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        ax.set_xlabel("Evaluation Metrics")
        for container in ax.containers:
            ax.bar_label(
                container,
                fmt="%.2f"
            )
        plt.xticks(rotation=20)
        st.pyplot(fig)
        plt.close(fig)
        
        st.divider()

    st.markdown("## Final Model Comparison")

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
                by="F1 Score",
                ascending=False
            )

    st.dataframe(
                results_df.style.format({
                    "Accuracy": "{:.3f}",
                    "Precision": "{:.3f}",
                    "Recall": "{:.3f}",
                    "F1 Score": "{:.3f}",
                    "AUC ROC": "{:.3f}"
                }),
                width='stretch'
)

        # Best model based on F1 Score
    best_model = results_df.iloc[0]["Model"]
    st.success(f"Best Performing Model based on F1 Score: {best_model}")