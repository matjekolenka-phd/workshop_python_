# workshop_python_
Building my first time ever App
https://workshoppython-au2wwzfwyl9pp8evapppfwl.streamlit.app/
# 🧠 Time Series Data Explorer and Prediction App

This is a Streamlit application for exploring time-series or panel data.

The app allows users to upload a CSV file, explore trends over time, calculate rolling averages, perform time-series decomposition, run a stationarity test, and build a simple prediction model.

## Main Features

- Upload CSV data
- Preview the dataset
- Check missing values and duplicates
- Select a date column
- Select a target variable
- Select a grouping variable such as district, facility, or region
- Plot monthly trends
- Calculate rolling averages
- Plot trends by group
- Decompose the time series into trend, seasonality, and residuals
- Run the Augmented Dickey-Fuller stationarity test
- Build a prediction model using Random Forest
- Download processed datasets and prediction results

## Files in This Repository

This repository contains:

- `app.py` - the main Streamlit application
- `requirements.txt` - Python packages required to run the app
- `README.md` - description of the project

## How to Run the App Locally

First, install the required packages:

```bash
pip install -r requirements.txt
