import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


# -------------------------------------------------------
# Page settings
# -------------------------------------------------------

st.set_page_config(
    page_title="Time Series Data Explorer",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Time Series Data Explorer and Prediction App")
st.write(
    """
    This app allows you to upload a CSV file, explore the data, view time trends,
    calculate rolling averages, decompose a time series, and build a simple prediction model.
    """
)

st.info(
    """
    Please upload a CSV file. The app will automatically read your columns and allow you
    to choose the date column, target variable, and grouping variable.
    """
)


# -------------------------------------------------------
# Helper functions
# -------------------------------------------------------

def detect_date_columns(data):
    possible_date_columns = []

    for col in data.columns:
        try:
            converted = pd.to_datetime(data[col], errors="coerce")
            valid_dates = converted.notna().mean()

            if valid_dates >= 0.70:
                possible_date_columns.append(col)
        except Exception:
            pass

    return possible_date_columns


def safe_numeric_columns(data):
    return data.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()


def safe_categorical_columns(data):
    return data.select_dtypes(include=["object", "category", "bool"]).columns.tolist()


def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# -------------------------------------------------------
# File upload
# -------------------------------------------------------

uploaded_file = st.file_uploader("📤 Upload your CSV file", type=["csv"])

if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read the file. Error: {e}")
        st.stop()

    st.success("✅ File uploaded successfully!")

    # -------------------------------------------------------
    # Data preview
    # -------------------------------------------------------

    st.header("1. Dataset Preview")

    st.write("### First 5 rows")
    st.dataframe(df.head())

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    with col3:
        st.metric("Duplicate rows", df.duplicated().sum())

    st.write("### Column names")
    st.write(list(df.columns))

    # -------------------------------------------------------
    # Missing values
    # -------------------------------------------------------

    st.header("2. Missing Values and Data Types")

    missing_values = df.isnull().sum().reset_index()
    missing_values.columns = ["Column", "Missing values"]

    st.write("### Missing values per column")
    st.dataframe(missing_values)

    st.write("### Data types")
    data_types = pd.DataFrame(df.dtypes, columns=["Data type"])
    st.dataframe(data_types)

    # -------------------------------------------------------
    # Column selection
    # -------------------------------------------------------

    st.header("3. Select Important Columns")

    date_candidates = detect_date_columns(df)
    numeric_columns = safe_numeric_columns(df)
    categorical_columns = safe_categorical_columns(df)

    if len(date_candidates) > 0:
        default_date_index = 0
        date_column = st.selectbox(
            "📅 Select the date column",
            options=date_candidates,
            index=default_date_index
        )
    else:
        st.warning("No clear date column was detected. Please select one manually.")
        date_column = st.selectbox(
            "📅 Select the date column",
            options=df.columns
        )

    if len(numeric_columns) == 0:
        st.error("No numeric columns were found. A numeric target is needed for prediction.")
        st.stop()

    target_column = st.selectbox(
        "🎯 Select the target variable to analyse and predict",
        options=numeric_columns
    )

    group_column = None

    if len(categorical_columns) > 0:
        group_column = st.selectbox(
            "🏥 Select a grouping column, for example district, facility, region, or category",
            options=["None"] + categorical_columns
        )

        if group_column == "None":
            group_column = None

    # Convert date column
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
    df = df.dropna(subset=[date_column])
    df = df.sort_values(date_column)

    # -------------------------------------------------------
    # Basic statistics
    # -------------------------------------------------------

    st.header("4. Summary Statistics")

    st.write("### Numeric summary")
    st.dataframe(df.describe())

    if group_column is not None:
        st.write(f"### Unique values in {group_column}")
        st.write(df[group_column].unique())

    st.write("### Time period")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Start date", str(df[date_column].min().date()))

    with col2:
        st.metric("End date", str(df[date_column].max().date()))

    # -------------------------------------------------------
    # Monthly aggregation
    # -------------------------------------------------------

    st.header("5. Time Trend Analysis")

    df["year_month"] = df[date_column].dt.to_period("M").dt.to_timestamp()

    monthly = (
        df.groupby("year_month")[target_column]
        .sum()
        .reset_index()
        .rename(columns={"year_month": "date", target_column: "target_total"})
    )

    st.write("### Monthly aggregated data")
    st.dataframe(monthly.head())

    # Plot target over time
    st.write(f"### Monthly trend of {target_column}")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(monthly["date"], monthly["target_total"], marker="o")
    ax.set_title(f"Monthly Trend of {target_column}")
    ax.set_xlabel("Date")
    ax.set_ylabel(target_column)
    ax.grid(True)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # -------------------------------------------------------
    # Rolling average
    # -------------------------------------------------------

    st.header("6. Rolling Average")

    rolling_window = st.slider(
        "Select rolling average window in months",
        min_value=2,
        max_value=12,
        value=3
    )

    monthly[f"{rolling_window}_month_rolling_average"] = (
        monthly["target_total"].rolling(window=rolling_window).mean()
    )

    st.write("### Data with rolling average")
    st.dataframe(monthly.head(12))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        monthly["date"],
        monthly["target_total"],
        marker="o",
        label="Original"
    )
    ax.plot(
        monthly["date"],
        monthly[f"{rolling_window}_month_rolling_average"],
        marker="o",
        label=f"{rolling_window}-month rolling average"
    )
    ax.set_title(f"{target_column} with {rolling_window}-Month Rolling Average")
    ax.set_xlabel("Date")
    ax.set_ylabel(target_column)
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # -------------------------------------------------------
    # Grouped trend
    # -------------------------------------------------------

    if group_column is not None:
        st.header("7. Trend by Group")

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(
            data=df,
            x=date_column,
            y=target_column,
            hue=group_column,
            marker="o",
            ax=ax
        )
        ax.set_title(f"{target_column} Over Time by {group_column}")
        ax.set_xlabel("Date")
        ax.set_ylabel(target_column)
        ax.grid(True)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # -------------------------------------------------------
    # Time-series decomposition
    # -------------------------------------------------------

    st.header("8. Time-Series Decomposition")

    st.write(
        """
        Decomposition separates the time series into:
        - Trend
        - Seasonality
        - Residuals
        """
    )

    decomposition_period = st.selectbox(
        "Select decomposition period",
        options=[3, 6, 12],
        index=2
    )

    if len(monthly) >= decomposition_period * 2:
        try:
            monthly_indexed = monthly.set_index("date")

            decomposition = seasonal_decompose(
                monthly_indexed["target_total"],
                model="additive",
                period=decomposition_period
            )

            fig = decomposition.plot()
            fig.set_size_inches(12, 8)
            plt.suptitle(f"Time Series Decomposition: {target_column}", y=1.02)
            st.pyplot(fig)

        except Exception as e:
            st.warning(f"Decomposition could not be completed. Error: {e}")
    else:
        st.warning(
            f"Not enough monthly data for decomposition. You need at least "
            f"{decomposition_period * 2} monthly observations."
        )

    # -------------------------------------------------------
    # Stationarity test
    # -------------------------------------------------------

    st.header("9. Stationarity Test")

    st.write(
        """
        The Augmented Dickey-Fuller test checks whether the time series is stationary.
        A p-value of 0.05 or below usually suggests that the series is stationary.
        """
    )

    try:
        clean_series = monthly["target_total"].dropna()

        if len(clean_series) > 10:
            result = adfuller(clean_series)

            st.write(f"ADF Statistic: **{result[0]:.4f}**")
            st.write(f"p-value: **{result[1]:.4f}**")

            if result[1] <= 0.05:
                st.success("The time series is likely stationary.")
            else:
                st.warning("The time series is likely not stationary.")
        else:
            st.warning("Not enough observations to run the ADF test.")

    except Exception as e:
        st.warning(f"Stationarity test could not be completed. Error: {e}")

    # -------------------------------------------------------
    # Prediction model
    # -------------------------------------------------------

    st.header("10. Prediction Model")

    st.write(
        """
        This section builds a simple machine learning model using the selected target variable.
        The app creates time-based features such as year, month, lag values, and rolling averages.
        """
    )

    model_df = df.copy()
    model_df = model_df.sort_values(date_column)

    model_df["year"] = model_df[date_column].dt.year
    model_df["month"] = model_df[date_column].dt.month
    model_df["quarter"] = model_df[date_column].dt.quarter

    if group_column is not None:
        model_df = model_df.sort_values([group_column, date_column])

        model_df[f"{target_column}_lag_1"] = (
            model_df.groupby(group_column)[target_column].shift(1)
        )

        model_df[f"{target_column}_lag_2"] = (
            model_df.groupby(group_column)[target_column].shift(2)
        )

        model_df[f"{target_column}_rolling_3"] = (
            model_df.groupby(group_column)[target_column]
            .shift(1)
            .rolling(window=3)
            .mean()
        )
    else:
        model_df[f"{target_column}_lag_1"] = model_df[target_column].shift(1)
        model_df[f"{target_column}_lag_2"] = model_df[target_column].shift(2)
        model_df[f"{target_column}_rolling_3"] = (
            model_df[target_column].shift(1).rolling(window=3).mean()
        )

    model_df = model_df.dropna()

    if model_df.shape[0] < 20:
        st.warning(
            "The dataset is too small after creating lag variables. "
            "Please upload a dataset with more rows for modelling."
        )
        st.stop()

    exclude_columns = [target_column, date_column]

    if "year_month" in model_df.columns:
        exclude_columns.append("year_month")

    feature_columns = [col for col in model_df.columns if col not in exclude_columns]

    X = model_df[feature_columns]
    y = model_df[target_column]

    numeric_features = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features)
        ]
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    test_size = st.slider(
        "Select test size percentage",
        min_value=10,
        max_value=40,
        value=20
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size / 100,
        random_state=42
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = calculate_rmse(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write("### Model performance")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("MAE", round(mae, 3))

    with col2:
        st.metric("RMSE", round(rmse, 3))

    with col3:
        st.metric("R² Score", round(r2, 3))

    results = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": y_pred
    })

    st.write("### Actual vs Predicted values")
    st.dataframe(results.head(20))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(results["Actual"], results["Predicted"])
    ax.set_title("Actual vs Predicted Values")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.grid(True)
    st.pyplot(fig)

    # -------------------------------------------------------
    # Download outputs
    # -------------------------------------------------------

    st.header("11. Download Processed Files")

    csv_monthly = monthly.to_csv(index=False).encode("utf-8")
    csv_model = model_df.to_csv(index=False).encode("utf-8")
    csv_results = results.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download monthly summary data",
        data=csv_monthly,
        file_name="monthly_summary.csv",
        mime="text/csv"
    )

    st.download_button(
        label="⬇️ Download model-ready data",
        data=csv_model,
        file_name="model_ready_data.csv",
        mime="text/csv"
    )

    st.download_button(
        label="⬇️ Download prediction results",
        data=csv_results,
        file_name="prediction_results.csv",
        mime="text/csv"
    )

else:
    st.warning("Please upload a CSV file to begin.")
