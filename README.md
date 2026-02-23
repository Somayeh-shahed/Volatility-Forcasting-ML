# 📊 Financial Volatility Forecasting: From Baseline to "Lag-Killer"

This repository showcases the evolution of a Machine Learning pipeline designed to predict market risk (Volatility) in financial assets like Gold. The project transitions from a basic predictive model to an advanced, physics-inspired engine that eliminates time-series lagging.

## 📌 Project Overview

The goal of this project is to forecast EWMA Volatility. In financial markets, predicting when the market will become unstable is as crucial as predicting price direction, especially for dynamic risk management and stop-loss optimization.

## 📂 Repository Structure

1. tse-volatility-forcasting-v1

* Description: My initial approach using standard historical lags.

* Status: ⚠️ Legacy.

* Key Issue: Suffered from a 1-day lag effect, where the model merely echoed the previous day's volatility instead of anticipating the next move.

2. tse-volatility-forcasting-v2

 Description: The production-ready version with advanced feature engineering.

 Improvements:

* Introduced Velocity and Acceleration features to capture momentum.

* Shifted target to Volatility Differentials (\Delta) to force the model to learn changes, not just levels.

* Performance: Successfully eliminated the lag, showing real-time responsiveness to market shocks (e.g., the 10% gold price drop in Feb 2026).

## 🚀 Key Technical Features

* Hybrid Ensemble: Combines Linear Regression (for trend momentum) and XGBoost (for non-linear shock detection).

* Advanced Feature Engineering:

* vol_velocity: Rate of change in risk.

* vol_acceleration: The "shaping" of the volatility curve.


## 📈 Real-World Application

 It helps traders:

* Set Dynamic Stop-Losses based on predicted risk.

* Reduce position sizes during high-acceleration volatility periods.

* Identify "Calm before the storm" scenarios in asset prices.

## 👨‍💻 Author

Developed as a comprehensive study on Financial Data Science and Time-Series Forecasting.
