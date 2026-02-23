# Volatility-Forcasting-ML


This project implements an advanced Machine Learning pipeline to forecast Financial Volatility (EWMA). Unlike standard models that suffer from systemic prediction lag, this repository introduces a physics-inspired feature engineering method to detect market shifts in real-time.

## 🎯 The Core Problem: The Lag Effect 

Most time-series models (like basic XGBoost or RNNs) tend to act as "echoes"—they simply repeat yesterday's value as today's prediction. In financial markets, this 1-day lag can be catastrophic for risk management.

## 🚀 The Solution: Differential Engineering

To solve this, I implemented Velocity and Acceleration features:
* Volatility Velocity: The first derivative of risk change (V = Vol_{t-1} - Vol_{t-2}).
* Volatility Acceleration: The second derivative, capturing how fast the risk is gaining momentum.

By shifting the target from absolute values to Differentials (\Delta Volatility), the model learns to predict the movement of risk rather than just its level.

## 🛠 Tech Stack

* Python 3.x
* pytse_client
* XGBoost Regressor: For capturing non-linear market shocks.
* Linear Regression: Used as a momentum-based baseline.
* Pandas & NumPy: For high-performance data manipulation.
* Matplotlib: For dual-model visualization.

## 📊 Key Results

* Lag Elimination: As shown in the visualizations, the model now anticipates spikes simultaneously with the actual market movements.
* Precision: Achieved a significantly low RMSE (~0.0055) during high-volatility periods (e.g., gold price shocks).
* Use Case: Ideal for Dynamic Stop-Loss placement and position sizing in one-way markets (like the Tehran Stock Exchange).

## 📝 How to Use

1. Clone the repo: git clone https://github.com/YOUR_USERNAME/Volatility-ML-Forecasting.git
2. Install dependencies: pip install pandas xgboost scikit-learn jdatetime
3. Run the main script: python main.py
