import numpy as np
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def prepare_features(df):
    #EWMA Volatility estimation
    df['Volatility'] = df['Log_Ret'].ewm(span=5).std()

    #Generating lag features
    for i in range(1, 4):
        df[f'lag_vol_{i}'] = df['Volatility'].shift(i)

    df.dropna(inplace=True)
    return df



def train_and_compare(df):

    features = [c for c in df.columns if 'lag_vol' in c]
    X = df[features]
    y = df['Volatility']

    # Train-test split
    split = int(len(df) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    #MODEL ۱: XGBoost
    xgb = XGBRegressor(n_estimators=100)
    xgb.fit(X_train, y_train)
    xgb_preds = xgb.predict(X_test)

    #MODEL ۲: Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)

    #Evaluate performance
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))
    
    print(f"XGBoost RMSE: {xgb_rmse}")
    print(f"Linear Regression RMSE: {lr_rmse}")
    
    return xgb_preds, lr_preds, y_test, xgb_rmse, lr_rmse





