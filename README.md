# Stock-Prediction-Dataset
# Facebook Prophet: Time Series Forecasting

## Project Overview
Prophet is an open-source library developed by Facebook’s Core Data Science team for forecasting time series data. It is based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, along with holiday effects. Prophet is designed to handle missing data, outliers, and trend shifts effectively, making it ideal for time series with strong seasonal components and multiple seasons of historical data.

### Key Features
- **Accurate and Fast**: Reliable forecasts for planning and goal setting.
- **Fully Automatic**: Generates forecasts with minimal manual intervention.
- **Tunable Forecasts**: Allows adjustments through interpretable parameters.
- **Multi-language Support**: Available in both Python and R.
- **Seasonality Handling**: Supports multiple seasonal variations.
- **Robustness**: Handles outliers and missing data seamlessly.

## Installation

Prophet can be installed using pip:
```bash
pip install prophet
```
This will install the required dependencies such as `cmdstanpy`, `numpy`, `matplotlib`, and `pandas`. Ensure your Python environment is properly configured.

Additionally, install the `yfinance` package for retrieving financial market data:
```bash
pip install yfinance --upgrade --no-cache-dir
```

## Example: Stock Price Forecasting

This example demonstrates how to use Prophet to forecast historical stock prices from Yahoo Finance.

### Prerequisites
Ensure the following Python libraries are installed:
- `pandas`
- `prophet`
- `yfinance`
- `matplotlib`

### Code Implementation

#### Import Libraries
```python
import pandas as pd
from prophet import Prophet
import yfinance as yf
from datetime import timedelta
import matplotlib.pyplot as plt
```

#### Download Stock Data
```python
# Define stock ticker and date range
stock = '^GSPC'  # Example: S&P 500
start = '1900-01-01'
yesterday = pd.to_datetime("today") - timedelta(days=1)

# Fetch data
df = yf.download(stock, start=start, end=yesterday, auto_adjust=True, progress=True)

# Select the 'Close' price column
df = df[['Close']]
```

#### Prepare Data for Prophet
Prophet requires the dataset to have columns named `ds` (date) and `y` (value to forecast).
```python
# Reset index and rename columns
df.reset_index(inplace=True)
df.rename(columns={"Date": "ds", "Close": "y"}, inplace=True)

# Ensure date format is correct
df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)
```

#### Fit the Prophet Model
```python
# Initialize and fit the model
model = Prophet()
model.fit(df)
```

#### Make Future Predictions
```python
# Define future periods for prediction
future = model.make_future_dataframe(periods=151)
forecast = model.predict(future)

# Display the forecast
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
```

#### Visualize the Results
```python
# Plot forecasted data
fig = model.plot(forecast)
plt.show()
```

## Sample Output
The output includes the predicted values (`yhat`) along with the lower and upper confidence intervals (`yhat_lower` and `yhat_upper`). For example:

| ds         | yhat     | yhat_lower | yhat_upper |
|------------|----------|------------|------------|
| 2025-03-24 | 17.309053| 16.216344  | 18.440743  |
| 2025-03-25 | 17.313038| 16.297757  | 18.464247  |

## Conclusion
Facebook Prophet is a powerful and versatile tool for time series forecasting. Its ease of use, combined with robust handling of seasonal effects, missing data, and outliers, makes it a valuable resource for data scientists and analysts. Whether you are forecasting stock prices or other types of time series data, Prophet provides a reliable and customizable solution.

