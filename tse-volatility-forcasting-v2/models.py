
import pandas as pd
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

def prepare_features_and_target(df):
    """Prepare features and target variable for modeling.
    Key features:
    - calculates EWMA volatility
    -Extracts velocity and acceleration of risk
    -Transforms target into 'Volatility Difference' to eliminate lag effect"""
    df['Volatility'] = df['Log_Ret'].ewm(span=5).std()

    
    for i in range(1, 4):
        df[f'lag_vol_{i}'] = df['Volatility'].shift(i)
    
    df['vol_velocity'] = df['lag_vol_1'] - df['lag_vol_2']
    df['vol_acceleration'] = (df['lag_vol_1'] - df['lag_vol_2']) - (df['lag_vol_2'] - df['lag_vol_3'])

    
    df['Vol_Diff_Target'] = df['Volatility'] - df['lag_vol_1']

    df.dropna(inplace=True)
    
    feature_cols = ['lag_vol_2', 'vol_velocity', 'vol_acceleration']
    return df, feature_cols

def train_models(X_train, y_train):
    """Train Linear Regression and XGBoost models."""
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3)
    xgb_model.fit(X_train, y_train)

    return lr_model, xgb_model