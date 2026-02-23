import pandas as pd
import numpy as np
import pytse_client as tse


def load_and_clean_data(stock):
    ticker = tse.download(symbols=stock, write_to_csv=True) #Dict
    df_ticker = pd.DataFrame(ticker[stock])

    #SET INDEX AND SORT
    df_ticker.set_index('date', inplace=True)
    df_ticker.sort_index(inplace=True)

    #SET LOG RETURN
    df_ticker['Log_Ret'] = np.log(df_ticker['adjClose'] / df_ticker['adjClose'].shift(1))
    df_ticker.dropna(subset=['Log_Ret'], inplace=True)
    return df_ticker





