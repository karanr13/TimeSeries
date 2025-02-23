import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Streamlit Page Configuration

st.set_page_config(page_title="Time Series Forecasting", layout="wide")
st.title("Time Series Forecasting & Analysis")
st.image("https://www.investopedia.com/thmb/IEG4YEY-j4PXx1U3TyyHU0YEq88=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/TermDefinitions_TimeSeries__V1_CT-e4cb9fe23caf415491b1e850a1be300b.png", width=1000)
st.markdown(
    """
    <style>
    /* Sidebar: Keep Blue Background */
    [data-testid="stSidebar"] {
        background-color: #1E3A8A; /* Dark Blue */
        color: white;
    }

    /* Main Content: Black Background & Light Blue Borders */
    [data-testid="stAppViewContainer"] {
        background-color: black; /* Main Black Background */
        padding: 20px;
    }

    /* Add Light Blue Borders Around Content */
    [data-testid="stAppViewContainer"] > div {
        border: 4px solid #4FC3F7; /* Light Blue Border */
        border-radius: 15px;
        padding: 25px;
        box-shadow: 5px 5px 15px rgba(79, 195, 247, 0.5);
    }

    /* Clock Icon Background */
    body {
        background-image: url('https://img.icons8.com/clouds/512/apple-clock.png');
        background-size: 100px;
        background-repeat: repeat;
        opacity: 0.95;
    }

    /* Adjust Sidebar Text Color */
    [data-testid="stSidebar"] * {
        color: white !important;
    }

    /* Adjust Main Content Text Color */
    [data-testid="stAppViewContainer"] * {
        color: white !important;
    }

    /* Customize Headers */
    h1, h2, h3 {
        color: #4FC3F7 !important; /* Light Blue Headers */
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
    /* Main Content: Black Background */
    [data-testid="stAppViewContainer"] {
        background-color: black;
        padding: 20px;
    }

    /* Add Blurred Sky Blue Borders */
    [data-testid="stAppViewContainer"] > div {
        border: 4px solid rgba(135, 206, 235, 0.6); /* Sky Blue Border with Transparency */
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0px 0px 20px 10px rgba(135, 206, 235, 0.5); /* Blurred Effect */
    }
    </style>
    """,
    unsafe_allow_html=True
    )

st.markdown("This app performs advanced time series forecasting using ARIMA/SARIMA models.")
st.header("📌 Real-Life Applications of Time Series Forecasting")
st.write("""
1. **Stock Market Analysis**: Investors use time series forecasting to predict stock price movements, helping them make informed decisions on buying or selling stocks.
2. **Weather Forecasting**: Meteorologists analyze past temperature, humidity, and pressure trends to make short- and long-term weather predictions.
3. **Sales & Demand Forecasting**: Businesses use historical sales data to forecast demand, helping with inventory management and supply chain optimization.
4. **Healthcare & Disease Prediction**: Hospitals and governments analyze disease trends to predict outbreaks, enabling better resource allocation and policy-making.
""")
# Introduction to Time Series Data
st.write("""
### Understanding Time Series Data
Time series data consists of observations recorded sequentially over time. It is characterized by:
- **Trend**: The long-term movement of the data.
- **Seasonality**: Repeating patterns at fixed intervals.
- **Cyclic Behavior**: Fluctuations that are not of fixed frequency.
- **Irregular Components**: Random noise or anomalies.

A good time series dataset should:
- Have a **datetime index** or a timestamp column.
- Be **regularly spaced** (daily, monthly, yearly, etc.).
- Contain **enough data points** to capture meaningful patterns.
- Be checked for **missing values** and handled appropriately.

Common Applications:
- **Stock Market Analysis**
- **Weather Forecasting**
- **Economic Trends & GDP Analysis**
- **Sales Predictions**
""")

# Function to check if data is a time series
def check_timeseries(df):
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.iloc[:, 0], format='mixed', errors='coerce')
            df.set_index(df.index, inplace=True)
            df = df.iloc[:, 1]
            return df.dropna()
        except Exception as e:
            return str(e)
    return df.dropna()

# Function to test stationarity using ADF and KPSS tests
def test_stationarity(ts):
    adf_result = adfuller(ts.dropna())
    kpss_result = kpss(ts.dropna(), regression='c')
    
    st.write("### Augmented Dickey-Fuller Test (ADF)")
    st.write("**Hypothesis:** H₀: The time series has a unit root (non-stationary), H₁: The time series is stationary.")
    st.write(f"Test Statistic: {adf_result[0]}")
    st.write(f"p-value: {adf_result[1]}")
    st.write(f"Critical Values: {adf_result[4]}")
    if adf_result[1] < 0.05:
        st.write("**Decision:** Reject H₀. The series is stationary.")
    else:
        st.write("**Decision:** Fail to reject H₀. The series is non-stationary.")
    
    st.write("### KPSS Test")
    st.write("**Hypothesis:** H₀: The time series is stationary, H₁: The time series has a unit root (non-stationary).")
    st.write(f"Test Statistic: {kpss_result[0]}")
    st.write(f"p-value: {kpss_result[1]}")
    st.write(f"Critical Values: {kpss_result[3]}")
    if kpss_result[1] < 0.05:
        st.write("**Decision:** Reject H₀. The series is non-stationary.")
    else:
        st.write("**Decision:** Fail to reject H₀. The series is stationary.")
    
    return adf_result[1], kpss_result[1]

# Function to apply Ljung-Box test for residual autocorrelation
def ljung_box_test(residuals, lags=10):
    result = acorr_ljungbox(residuals.dropna(), lags=[lags], return_df=True)
    st.write("### Ljung-Box Test: Checking for Residual Autocorrelation")
    st.write("**Hypothesis:** H₀: Residuals are white noise (no autocorrelation). H₁: Residuals show autocorrelation.")
    st.write(result)
    if result['lb_pvalue'].values[0] < 0.05:
        st.write("**Decision:** Reject H₀. Residuals show autocorrelation, indicating a poor model fit.")
    else:
        st.write("**Decision:** Fail to reject H₀. Residuals are white noise, indicating a good model fit.")


st.header("📌 Understanding ACF & PACF for ARIMA Model Selection")

st.subheader("1️⃣ What are Lags?")
st.write("""
- A **lag** represents a previous time step in a time series.
- **Lag 1** means the value from one time step before, **lag 2** means two steps before, and so on.
- In **ACF and PACF plots**, the x-axis represents the lag number, and the y-axis represents correlation strength.
""")

st.subheader("2️⃣ How to Use ACF & PACF to Select ARIMA Parameters?")
st.write("""
ARIMA consists of three components:
- **p (Autoregressive Order - AR)**
- **d (Differencing Order)**
- **q (Moving Average Order - MA)**
""")

st.subheader("🔹 ACF (Autocorrelation Function) – Determines `q` (MA Order)")
st.write("""
- ACF measures how current values relate to past values.
- If **ACF drops off sharply after a few lags**, it suggests an **MA(q) model**.
- If **ACF decreases slowly (gradual decay)**, then the series might need **differencing (`d`)** before selecting `q`.

**MA Process (Moving Average):**
- If ACF **cuts off** after a certain lag (`q`) and becomes insignificant → Choose that lag as `q`.
""")

st.subheader("🔹 PACF (Partial Autocorrelation Function) – Determines `p` (AR Order)")
st.write("""
- PACF removes the influence of intermediate lags and shows only the direct correlation.
- If PACF **cuts off sharply** after a few lags → suggests an **AR(p) model**.
- If PACF shows a slow decay → differencing may be needed (`d`).

**AR Process (Autoregressive):**
- If PACF **cuts off** at a certain lag (`p`) and becomes insignificant → Choose that lag as `p`.
""")

st.subheader("📊 How to Select (p, q) from the Graphs?")
st.write("""
| Scenario | ACF Behavior | PACF Behavior | Suggested Model |
|----------|-------------|--------------|----------------|
| AR Model (p) | ACF decays gradually | PACF cuts off at lag p | AR(p) |
| MA Model (q) | ACF cuts off at lag q | PACF decays gradually | MA(q) |
| ARMA Model (p, q) | ACF decays gradually | PACF decays gradually | ARMA(p, q) |
""")

st.subheader("🔍 Example Interpretation")
st.write("""
1. **If PACF cuts off at lag 2, but ACF decays gradually → AR(2) model.**
2. **If ACF cuts off at lag 3, but PACF decays gradually → MA(3) model.**
3. **If both ACF and PACF decay slowly, differencing (`d`) may be required.**
""")

st.subheader("📌 Graphs to Identify AR and MA")
st.write("""
- **AR Model** → Look at **PACF** for `p`.
- **MA Model** → Look at **ACF** for `q`.
- **If neither ACF nor PACF cuts off immediately**, apply **differencing (d)**.

🔹 After choosing (p, d, q), apply **ARIMA(p, d, q)** for forecasting.
""")



# Upload and analyze data
uploaded_file = st.file_uploader("📂 Upload CSV File (Time Series Data)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Convert to time series format if necessary
    df = check_timeseries(df)  

    if isinstance(df, str):
        st.error("Error in data: " + df)
    else:
        st.subheader("📈 Data Overview & Trends")
        st.line_chart(df)

        diff_order = st.slider("Select Number of Differencing", min_value=0, max_value=3, value=1)
        df = df.diff(periods=diff_order).dropna()

        st.subheader("📊 Stationarity Tests")
        test_stationarity(df)

        st.subheader("📉 ACF and PACF Plots")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        plot_acf(df, ax=axes[0])
        plot_pacf(df, ax=axes[1])
        axes[0].set_title("Autocorrelation Function (ACF)")
        axes[1].set_title("Partial Autocorrelation Function (PACF)")
        st.pyplot(fig)

        # ARIMA / SARIMA Model Selection
        seasonal = st.radio("Does your data have seasonality?", [True, False])

        st.subheader("🤖 Best ARIMA/SARIMA Model")

        if seasonal:
            seasonal_period = st.number_input("Set Seasonal Period (e.g., 12 for monthly, 4 for quarterly)", min_value=1, value=12)
            model = auto_arima(df, seasonal=True, m=seasonal_period, stepwise=True, suppress_warnings=True)  # SARIMA
            model_name = f"SARIMA{model.order} x {model.seasonal_order}"
        else:
            model = auto_arima(df, seasonal=False, stepwise=True, suppress_warnings=True)  # ARIMA
            model_name = f"ARIMA{model.order}"  # Force the correct display name

        st.write(f"**Selected Model:** {model_name}")  # Corrected model name display
        st.write(model.summary())  # Model summary

        # 🔍 Learning Note: Why does SARIMAX appear even when seasonality is False?
        with st.expander("🔎 **Why does it say SARIMAX even if seasonality is False?**"):
            st.write("""
            - `auto_arima` internally uses `SARIMAX` from `statsmodels`, even when no seasonality is specified.
            - `SARIMAX(1,0,0)` is functionally the same as `ARIMA(1,0,0)` when there are no seasonal components.
            - The `SARIMAX` label in the summary does **not** change, even if `seasonal=False`.
            """)

        # Forecasting
        forecast_period = st.slider("Select Forecasting Period", min_value=1, max_value=len(df) // 2, value=12)
        forecast_values, conf_int = model.predict(n_periods=forecast_period, return_conf_int=True)

        # Plot Forecasted Values
        st.subheader("📈 Forecasted Values")
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df, label='Actual Data', color='blue')
        forecast_index = pd.date_range(df.index[-1], periods=forecast_period, freq='D')
        plt.plot(forecast_index, forecast_values, label='Forecast', linestyle='dashed', color='red')
        plt.fill_between(forecast_index, conf_int[:, 0], conf_int[:, 1], color='gray', alpha=0.3, label='Confidence Interval')
        plt.legend()
        st.pyplot(plt)

        # Ljung-Box Test for Residuals
        st.subheader("🔍 Ljung-Box Test Results")
        ljung_test = acorr_ljungbox(model.resid(), lags=[10], return_df=True)
        st.write(ljung_test)

        # Decision Based on Ljung-Box Test
        p_value = ljung_test['lb_pvalue'].values[0]
        if p_value < 0.05:
            st.write("**Decision:** Reject H₀ → Residuals show autocorrelation, indicating a poor model fit.")
        else:
            st.write("**Decision:** Fail to reject H₀ → Residuals are white noise, indicating a good model fit.")

        # Error Metrics
        st.subheader("📊 Error Metrics")
        mse = mean_squared_error(df[-forecast_period:], forecast_values)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(df[-forecast_period:], forecast_values)

        st.write(f"MSE: {mse} (Lower is better)")
        st.write(f"RMSE: {rmse} (Lower is better)")
        st.write(f"MAE: {mae} (Lower is better)")

# Next Steps Section
st.header("📍 Next Steps")
st.write("""
- **Further Model Tuning**: Experiment with different parameters and seasonal adjustments.
- **Feature Engineering**: Include external factors such as economic indicators, holidays, or weather data.
- **Deploying the Model**: Use the trained model in production for real-time forecasting.
- **Automating Forecasting**: Schedule periodic data updates and automatic retraining.
""")
