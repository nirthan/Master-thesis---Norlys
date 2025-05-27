# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 08:43:11 2025

@author: Nirthan
"""

import pandas as pd

# --------------- Load Data related to LSTM --------------------

FE_DE_LSTM = pd.read_excel("FE_DE_LSTM_test_results.xlsx")
FE_FR_LSTM = pd.read_excel("FE_FR__LSTM_test_results.xlsx")

Raw_DE_LSTM = pd.read_excel("Raw_DE_LSTM_test_results.xlsx")
Raw_FR_LSTM = pd.read_excel("Raw_FR_LSTM_test_results.xlsx")

Raw_DE_LSTM = Raw_DE_LSTM.iloc[4:].reset_index(drop=True)
Raw_FR_LSTM = Raw_FR_LSTM.iloc[4:].reset_index(drop=True)

FE_DE_LSTM = FE_DE_LSTM.drop(columns=['Timestamp'])  

FE_DE_LSTM.rename(columns={"Actual": "DE_actual", "Predicted": "DE_pred"}, inplace=True)
FE_FR_LSTM.rename(columns={"Actual": "FR_actual", "Predicted": "FR_pred"}, inplace=True)

Raw_DE_LSTM.rename(columns={"Actual": "DE_actual", "Predicted": "DE_pred"}, inplace=True)
Raw_FR_LSTM.rename(columns={"Actual": "FR_actual", "Predicted": "FR_pred"}, inplace=True)


# --------------- Load Data related to XGBoost --------------------

FE_DE_XGBoost = pd.read_excel("FE_DE_XGBoost_test_results.xlsx")
FE_FR_XGBoost = pd.read_excel("FE_FR__XGBoost_test_results.xlsx")

Raw_DE_XGBoost = pd.read_excel("Raw_DE_XGBoost_test_results.xlsx")
Raw_FR_XGBoost = pd.read_excel("Raw_FR_XGBoost_test_results.xlsx")

FE_DE_XGBoost = FE_DE_XGBoost.drop(columns=['Timestamp'])  

FE_DE_XGBoost.rename(columns={"Actual": "DE_actual", "Predicted": "DE_pred"}, inplace=True)
FE_FR_XGBoost.rename(columns={"Actual": "FR_actual", "Predicted": "FR_pred"}, inplace=True)

Raw_DE_XGBoost.rename(columns={"Actual": "DE_actual", "Predicted": "DE_pred"}, inplace=True)
Raw_FR_XGBoost.rename(columns={"Actual": "FR_actual", "Predicted": "FR_pred"}, inplace=True)

# --------------- Choose which model/data to analyze --------------------

# For LSTM Raw data:
df = pd.concat([Raw_DE_LSTM, Raw_FR_LSTM], axis=1) 

# Alternative options:
# df = pd.concat([FE_FR_LSTM, FE_DE_LSTM], axis=1)         # LSTM on feature engineered data
# df = pd.concat([Raw_DE_XGBoost, Raw_FR_XGBoost], axis=1) # XGBoost on raw data
# df = pd.concat([FE_DE_XGBoost, FE_FR_XGBoost], axis=1)   # XGBoost on feature engineered data

# --------------- Profit and Spread Calculations --------------------

# Predicted spread and profit
df["Spread_pred"] = df["FR_pred"] - df["DE_pred"]
df["Profit_pred"] = df.apply(lambda row: row["FR_actual"] - row["DE_actual"] if row["Spread_pred"] > 0
                             else row["DE_actual"] - row["FR_actual"] if row["Spread_pred"] < 0
                             else 0, axis=1)

# Actual spread and theoretical max profit
df["Spread_actual"] = df["FR_actual"] - df["DE_actual"]
df["Profit_actual"] = df.apply(lambda row: row["FR_actual"] - row["DE_actual"] if row["Spread_actual"] > 0
                               else row["DE_actual"] - row["FR_actual"] if row["Spread_actual"] < 0
                               else 0, axis=1)
# Total profits
total_profit = df["Profit_pred"].sum()
total_profit_actual = df["Profit_actual"].sum()

# Cumulative profit
df["CumProfit_pred"] = df["Profit_pred"].cumsum()
df["CumProfit_actual"] = df["Profit_actual"].cumsum()

# Negative profit cases
negative_profit_count = (df["Profit_pred"] < 0).sum()
total_rows = len(df)
negative_profit_percent = (negative_profit_count / total_rows) * 100

print(f"Number of negative profit cases: {negative_profit_count}")
print(f"Percentage of negative profit cases: {round(negative_profit_percent, 2)}%")

