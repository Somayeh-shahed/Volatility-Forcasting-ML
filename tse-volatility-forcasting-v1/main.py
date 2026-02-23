import matplotlib.pyplot as plt
from data_loader import load_and_clean_data
from analysis import run_statistical_tests, detect_anomalies
from models import prepare_features, train_and_compare



st_name = "طلا"
df = load_and_clean_data(st_name)


if run_statistical_tests(df):
    print("Data is stationary. Proceeding...")
    df = detect_anomalies(df)
    df = prepare_features(df)

    xgb_p, lr_p, y_test, xgb_err, lr_err = train_and_compare(df)

    plt.plot(y_test.index, y_test.values, label="Actual")
    plt.plot(y_test.index, xgb_p, label="XGBoost Prediction")
    plt.plot(y_test.index, lr_p, label="Linear Regression Prediction")
    plt.savefig("predictions.pdf", format='pdf', bbox_inches='tight')
    plt.legend()
    plt.show()
else:
    print("Data needs differencing!")