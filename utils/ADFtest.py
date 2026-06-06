import pandas as pd
import numpy as np
import os
from statsmodels.tsa.stattools import adfuller
from arch.unitroot import ADF

def calculate_ADF(root_path,data_path):
    df_raw = pd.read_csv(os.path.join(root_path,data_path))
    cols = list(df_raw.columns)
    cols.remove('date')
    df_raw = df_raw[cols]
    adf_list = []
    for i in cols:
        df_data = df_raw[i]
        adf = adfuller(df_data, maxlag = 1)
        print(adf)
        adf_list.append(adf)
    return np.array(adf_list)

def calculate_target_ADF(root_path,data_path,target='OT'):
    df_raw = pd.read_csv(os.path.join(root_path,data_path))
    target_cols = target.split(',')
    # df_data = df_raw[target]
    df_raw = df_raw[target_cols]
    adf_list = []
    for i in target_cols:
        df_data = df_raw[i]
        adf = adfuller(df_data, maxlag = 1)
        # print(adf)
        adf_list.append(adf)
    return np.array(adf_list)

def archADF(root_path, data_path):
    df = pd.read_csv(os.path.join(root_path,data_path))
    cols = df.columns[1:]
    stats = 0
    for target_col in cols:
        series = df[target_col].values
        adf = ADF(series)
        stat = adf.stat
        stats += stat
    return stats/len(cols)

if __name__ == '__main__':

    ADFmetric = archADF(root_path="./dataset/exchange_rate/",data_path="exchange_rate.csv")
    print("Exchange ADF metric", ADFmetric)

    ADFmetric = archADF(root_path="./dataset/illness/",data_path="national_illness.csv") 
    print("Illness ADF metric", ADFmetric)

    ADFmetric = archADF(root_path="./dataset/ETT-small/",data_path="ETTm2.csv")
    print("ETTm2 ADF metric", ADFmetric)

    ADFmetric = archADF(root_path="./dataset/electricity/",data_path="electricity.csv")
    print("Electricity ADF metric", ADFmetric)

    ADFmetric = archADF(root_path="./dataset/traffic/",data_path="traffic.csv")
    print("Traffic ADF metric", ADFmetric)

    ADFmetric = archADF(root_path="./dataset/weather/",data_path="weather.csv")
    print("Weather ADF metric", ADFmetric)
