import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np
from data_loader import load_and_clean_data
from analysis import run_statistical_tests, detect_anomalies
from models import prepare_features_and_target, train_models


st_name = "طلا"
df = load_and_clean_data(st_name)
run_statistical_tests(df)
df = detect_anomalies(df)


df, features = prepare_features_and_target(df)


split = int(len(df) * 0.8)
train, test = df.iloc[:split], df.iloc[split:]

X_train, y_train = train[features], train['Vol_Diff_Target']
X_test, y_test = test[features], test['Vol_Diff_Target']


lr_model, xgb_model = train_models(X_train, y_train)


lr_diff_pred = lr_model.predict(X_test)
xgb_diff_pred = xgb_model.predict(X_test)

lr_final_pred = lr_diff_pred + test['lag_vol_1'].values
xgb_final_pred = xgb_diff_pred + test['lag_vol_1'].values
actual_vol = test['Volatility'].values


lr_rmse = np.sqrt(mean_squared_error(actual_vol, lr_final_pred))
print(f"Linear Regression RMSE: {lr_rmse:.6f}")

xgb_rmse = np.sqrt(mean_squared_error(actual_vol, xgb_final_pred))
print(f"XGBoost RMSE: {xgb_rmse:.6f}")


plt.figure(figsize=(12, 6))
plt.plot(y_test.index, actual_vol, label='Actual Volatility', color='blue', alpha=0.7)
plt.plot(y_test.index, lr_final_pred, label='Linear Regression (Diff-based)', color='green', linestyle='--')
plt.plot(y_test.index, xgb_final_pred, label='XGBoost Prediction', color='orange')
plt.title('Volatility Prediction: Handling the Lag Effect')
plt.legend()
plt.show()




