import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor, plot_tree

st.set_page_config(
    page_icon="📊",
    layout="wide",  # Options: "centered" or "wide"
    initial_sidebar_state="expanded"  # Optional: "auto", "expanded", "collapsed"
)

if st.session_state.uploaded_dataset is not None:
    try:
        df = st.session_state.uploaded_dataset
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
else:
    st.error(f"Error loading dataset:")


st.markdown(
    """
    <div style="
        background-color: #4CB5F5; 
        padding: 15px; 
        border-radius: 8px; 
        text-align: center;
    ">
        <h1 style="color: #FFFFFF; margin: 0; font-family: sans-serif;">
            Regression Modelling for Numerical Target Variables
        </h1>
    </div>
    """,
    unsafe_allow_html=True
)


st.dataframe(df)
num_cols = df.select_dtypes(include=['number']).columns.tolist()
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

st.subheader("Comparison of various Regression Models")
st.write("#### Select the Target Column for Regression Modelling")


rem_list = []

for num_col in num_cols:
    if df[num_col].nunique() == len(df):
        rem_list.append(num_col)

# linear regression
# polynomial regression
# Decision Tree regression

st.success("#### Linear Regression")
X_inter = df.drop(columns=rem_list)
target_col = st.selectbox("Select Target Column", X_inter.select_dtypes(include=['number']).columns.tolist())
num_cols_new = X_inter.select_dtypes(include=['number']).columns.tolist()
df_reg = X_inter[num_cols_new]
reg_model = LinearRegression()

X = df_reg.drop(columns=[target_col])
y = df_reg[target_col]

test_size = st.slider("Select Test Size (as a fraction of the dataset)", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
st.info(f"Selected Test Size: {test_size}, Selected Target Column: {target_col}, Number of Features: {X.shape[1]}, Number of Samples: {X.shape[0]}")
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=test_size, random_state=42)
reg_model.fit(train_X, train_y)
y_pred = reg_model.predict(test_X)
mse = mean_squared_error(test_y, y_pred)
mae = mean_absolute_error(test_y, y_pred)
r2 = r2_score(test_y, y_pred)

st.write(f"Mean Squared Error (MSE): {mse:.4f}")
st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
st.write(f"R-squared (R2): {r2:.4f}")

st.write("#### Coefficients of the Linear Regression Model")
coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': reg_model.coef_})
st.dataframe(coefficients)

st.write("#### Visualizing the Coefficients")
plt.figure(figsize=(10, 6))
sns.barplot(data=coefficients, x='Coefficient', y='Feature')
plt.title("Coefficients of the Linear Regression Model")
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
st.pyplot(plt)


st.success("#### Polynomial Regression")

# create polynomial features up to degree 3
    

scores=[]

for i in range(1,5):
    poly=PolynomialFeatures(degree=i)
    scaler=StandardScaler()
    
    X_train_poly1 = poly.fit_transform(train_X)
    X_test_poly1 = poly.transform(test_X)
    scaler.fit(X_train_poly1)


    # Step 2: Fit linear regression
    model1 = LinearRegression()
    model1.fit(X_train_poly1, train_y)


    # Step 3: Predict
    y_train_pred1 = model1.predict(X_train_poly1)
    y_test_pred1 = model1.predict(X_test_poly1)


    # Step 4: Calculate errors (MSE = Mean Squared Error)
    train_mse_1 = np.mean((train_y - y_train_pred1) ** 2)
    test_mse_1 = np.mean((test_y - y_test_pred1) ** 2)

    # Step 5: Calculate weight magnitude
    weight_norm_1 = np.sqrt(np.sum(model1.coef_ ** 2))
    
    col1, col2, col3 = st.columns(3)
    col1.write(f"Degree_{i} >>>> Train MSE: {train_mse_1:.4f}")
    col2.write(f"Test MSE: \n {test_mse_1:.4f}")
    col3.write(f"Weight Norm: \n{weight_norm_1:.2f}")

st.success("#### Decision Tree Regression with pre set parameters")

# Cache the fitted grid search in session state; keyed by dataset + target + test size
_gs_key = f"dt_best_model_{target_col}_{test_size}_{len(train_X)}_{X.shape[1]}"

if "dt_grid_result" not in st.session_state or st.session_state.dt_grid_result is None \
        or st.session_state.get("dt_grid_key") != _gs_key:
    st.session_state.dt_grid_result = None

run_grid = st.button("Run Grid Search", type="primary", key="btn_run_grid_search")
if run_grid:
    st.session_state.dt_grid_result = "running"

param_grid = {
    "max_depth": [2, 4, 6, 8, 10],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "criterion": ["squared_error", "absolute_error"],
}
st.info("total number of combinations: 5*3*3*2 = 90, so it will take time to run, please be patient")

best_model = st.session_state.get("dt_best_model")

if st.session_state.dt_grid_result == "running":
    grid_search = GridSearchCV(
        estimator=DecisionTreeRegressor(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    with st.spinner("Running Grid Search... this can take 20-25 minutes, do not refresh"):
        grid_search.fit(train_X, train_y)
    st.session_state.dt_best_model = grid_search.best_estimator_
    st.session_state.dt_best_params = grid_search.best_params_
    st.session_state.dt_grid_result = "done"
    st.session_state.dt_grid_key = _gs_key
    st.session_state.dt_grid_result = "done"
    best_model = st.session_state.dt_best_model

if best_model is None:
    st.warning("Grid Search has not been run yet. Press the 'Run Grid Search' button above to fit the Decision Tree.")
    st.stop()

st.info(f"Best Parameters Found: {st.session_state.get('dt_best_params')}\n")

# 4. Predict on Test Data
y_pred = best_model.predict(test_X)

# 5. Evaluate Model Performance
mse = mean_squared_error(test_y, y_pred)
mae = mean_absolute_error(test_y, y_pred)
r2 = r2_score(test_y, y_pred)
st.info(f"Mean Squared Error (MSE):  {mse:.4f}")
st.info(f"Mean Absolute Error (MAE): {mae:.4f}")
st.info(f"R-squared (R2 Score):      {r2:.4f}")

# 6. Visualize the Regressor Fit (single selected feature, since the model is multi-feature)
plot_feature = st.selectbox("Select Feature to Visualize Against Target", train_X.columns.tolist())
feat_idx = train_X.columns.get_loc(plot_feature)
X_grid = np.linspace(train_X[plot_feature].min(), train_X[plot_feature].max(), 300).reshape(-1, 1)
# Build a full-feature grid: keep the selected feature varying, fix all others at their mean
X_grid_full = np.tile(train_X.mean().values, (len(X_grid), 1))
X_grid_full[:, feat_idx] = X_grid.ravel()
y_grid_pred = best_model.predict(X_grid_full)
plt.figure(figsize=(10, 6))
plt.scatter(train_X[plot_feature], train_y, s=20, edgecolor="black", c="darkorange", label="Data")
plt.plot(X_grid.ravel(), y_grid_pred, color="cornflowerblue", label="Prediction", linewidth=2)
plt.xlabel(plot_feature)
plt.ylabel("Target (y)")
plt.title("Decision Tree Regression Fit")
plt.legend()
st.pyplot(plt)

# 7. Plot the Tree Structure (Limited to depth=3 for readability)
plt.figure(figsize=(15, 8))
plot_tree(
    best_model,
    max_depth=3,
    feature_names=train_X.columns.tolist(),
    filled=True,
    rounded=True,
    fontsize=10,
)
plt.title("Trained Decision Tree Structure (Truncated View)")
st.pyplot(plt)

##################################################################################################



