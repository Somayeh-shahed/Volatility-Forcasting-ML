from statsmodels.tsa.stattools import adfuller
from sklearn.ensemble import IsolationForest

def run_statistical_tests(df):
    print("--- Running ADF Test ---")
    result = adfuller(df['Log_Ret'])
    p_value = result[1]
    print(f"P-Value: {p_value:.4f}")
    return p_value <= 0.05


def detect_anomalies(df):
    print("--- Detecting Anomalies ---")
    
    # Fit the model and predict anomalies
    model = IsolationForest(contamination=0.02, random_state=42)
    df['anomaly'] = model.fit_predict(df[['volume', 'Log_Ret']])
    return df














