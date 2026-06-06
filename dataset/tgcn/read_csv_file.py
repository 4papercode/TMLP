
import pandas as pd
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar

file_path = 'AZPS.csv'
csv_file = pd.read_csv(file_path)
# csv_file = csv_file.drop(columns=['Forecast_Demand_MWh', 'Adjusted_Generation_MWh', ''])
# csv_file = csv_file[['Year', 'Month', 'Day', 'Hour', 'Adjusted_Demand_MWh', 'Total_Population']]
# 'Hour_sin','Hour_cos','Month_sin','Month_cos','T2','Q2','WSPD','GLW','SWDOWN','Weekday','Holiday'
csv_file = csv_file[['Year', 'Month', 'Day', 'Hour','T2','Q2','WSPD','GLW','SWDOWN', 'Adjusted_Generation_MWh']]
print(f"{csv_file.head}")


csv_file['Hour_sin']  = np.sin(2 * np.pi * csv_file['Hour']  / 24.0)
csv_file['Hour_cos']  = np.cos(2 * np.pi * csv_file['Hour']  / 24.0)
csv_file['Month_sin'] = np.sin(2 * np.pi * (csv_file['Month']-1) / 12.0)  # Month 1~12 -> 0~11
csv_file['Month_cos'] = np.cos(2 * np.pi * (csv_file['Month']-1) / 12.0)
csv_file['Date1'] = pd.to_datetime(csv_file[['Year','Month','Day']])
csv_file['Weekday'] = (csv_file['Date1'].dt.weekday < 5).astype(int)
cal = USFederalHolidayCalendar()
csv_file['Holiday'] = csv_file['Date1'].isin(cal.holidays(start=csv_file['Date1'].min(),
                                                end=csv_file['Date1'].max())).astype(int)

csv_file.to_csv("clean_data.csv", index=False)
