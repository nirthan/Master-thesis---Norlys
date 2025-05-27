# -*- coding: utf-8 -*-
"""
Created on Wed Apr 9 15:01:57 2025

@author: Nirthan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import re
from sklearn.preprocessing  import StandardScaler

Norlys_df = pd.read_excel("Norlys_full.xlsx", sheet_name="Sheet1")


#--------------------------- Calculating Mean Values for Forecast Variables ---------------------------


# Residual load for DE 
Norlys_df['Res_Load_Mean_DE'] = Norlys_df[['Res_Point_Op_DE', 'Res_Point_Ens_DE', 'Res_Volue_Ens_DE']].mean(axis=1)
Norlys_df = Norlys_df.drop(columns=['Res_Point_Op_DE', 'Res_Point_Ens_DE', 'Res_Volue_Ens_DE'])

# Residual load for FR 
Norlys_df['Res_Load_Mean_FR'] = Norlys_df[['Res_Point_Op_FR', 'Res_Point_Ens_FR', 'Res_Volue_Ens_FR']].mean(axis=1)
Norlys_df = Norlys_df.drop(columns=['Res_Point_Op_FR', 'Res_Point_Ens_FR', 'Res_Volue_Ens_FR'])


# Wind forecast for DE 
Norlys_df['Wind_Mean_DE'] = Norlys_df[['Wind_Point_Op_DE', 'Wind_Point_Ens_DE', 'Wind_Volue_Ens_DE','Wind_Meta_Op_DE']].mean(axis=1)
Norlys_df = Norlys_df.drop(columns=['Wind_Point_Op_DE', 'Wind_Point_Ens_DE', 'Wind_Volue_Ens_DE','Wind_Meta_Op_DE'])

# Wind forecast for FR
Norlys_df['Wind_Mean_FR'] = Norlys_df[['Wind_Point_Op_FR', 'Wind_Point_Ens_FR', 'Wind_Volue_Ens_FR','Wind_Meta_Op_FR']].mean(axis=1)
Norlys_df = Norlys_df.drop(columns=['Wind_Point_Op_FR', 'Wind_Point_Ens_FR', 'Wind_Volue_Ens_FR','Wind_Meta_Op_FR'])


# Solar forecast for DE 
Norlys_df['Solar_Mean_DE'] = Norlys_df[['Solar_Point_Ens_DE', 'Solar_Volue_Ens_DE']].mean(axis=1)
Norlys_df = Norlys_df.drop(columns=['Solar_Point_Ens_DE', 'Solar_Volue_Ens_DE'])

# Solar forecast for FR 
Norlys_df['Solar_Mean_FR'] = Norlys_df[['Solar_Point_Ens_FR', 'Solar_Volue_Ens_FR']].mean(axis=1)
Norlys_df = Norlys_df.drop(columns=['Solar_Point_Ens_FR', 'Solar_Volue_Ens_FR'])

# Hydro forecast – only keep total hydro (DE and FR)
Norlys_df = Norlys_df.drop(columns=['HydroPre_Point_Ens_DE', 'HydroPre_Volue_Op_DE','HydroPre_Volue_Ens_DE','HydroROR_Volue_DE'])
Norlys_df = Norlys_df.drop(columns=['HydroPre_Point_Ens_FR', 'HydroPre_Volue_Op_FR','HydroPre_Volue_Ens_FR','HydroROR_Volue_FR'])

#--------------------------- Availability Variables ---------------------------

# Avavability for DE 
Norlys_df['Ava_Coal_Mean_DE'] = Norlys_df[['Ava_Coal_Point_DE', 'Ava_Coal_Volue_DE']].mean(axis=1)
Norlys_df = Norlys_df.drop(columns=['Ava_Coal_Point_DE', 'Ava_Coal_Volue_DE'])

Norlys_df['Ava_Gas_Mean_DE'] = Norlys_df[['Ava_Gas_Point_DE', 'Ava_Gas_Volue_DE']].mean(axis=1)
Norlys_df = Norlys_df.drop(columns=['Ava_Gas_Point_DE', 'Ava_Gas_Volue_DE'])

Norlys_df['Ava_HydroPump_Mean_DE'] = Norlys_df[['Ava_HydroPump_Point_DE', 'Ava_HydroPump_Volue_DE']].mean(axis=1)
Norlys_df = Norlys_df.drop(columns=['Ava_HydroPump_Point_DE', 'Ava_HydroPump_Volue_DE'])

Norlys_df['Ava_HydroRes_Mean_DE'] = Norlys_df[['Ava_HydroRes_Point_DE', 'Ava_HydroRes_Volue_DE']].mean(axis=1)
Norlys_df = Norlys_df.drop(columns=['Ava_HydroRes_Point_DE', 'Ava_HydroRes_Volue_DE'])

Norlys_df['Ava_Lignite_Mean_DE'] = Norlys_df[['Ava_Lignite_Point_DE', 'Ava_Lignite_Volue_DE']].mean(axis=1)
Norlys_df = Norlys_df.drop(columns=['Ava_Lignite_Point_DE', 'Ava_Lignite_Volue_DE'])


#Avavability for FR
Norlys_df['Ava_Nuclear_Mean_FR'] = Norlys_df[['Ava_Nuclear_Point_FR', 'Ava_Nuclear_Volue_FR']].mean(axis=1)
Norlys_df = Norlys_df.drop(columns=['Ava_Nuclear_Point_FR', 'Ava_Nuclear_Volue_FR'])


#--------------------------- Actual Production ---------------------------

# Keep only total wind production for DE and FR
Norlys_df = Norlys_df.drop(columns=['Actp_Wind_On_DE', 'Actp_Wind_Off_DE'])
Norlys_df = Norlys_df.drop(columns=['Actp_Wind_On_FR', 'Actp_Wind_Off_FR'])

#--------------------------- Temperature ---------------------------

# Temperature forecast and normal for DE
Norlys_df['Temp_Fore_Mean_DE'] = Norlys_df[['Temp_Fore_Volue_DE', 'Temp_Fore_Point_DE']].mean(axis=1)
Norlys_df = Norlys_df.drop(columns=['Temp_Fore_Volue_DE', 'Temp_Fore_Point_DE'])

Norlys_df['Temp_Norm_Mean_DE'] = Norlys_df[['Temp_Norm_Volue_DE', 'Temp_Norm_Point_DE']].mean(axis=1)
Norlys_df = Norlys_df.drop(columns=['Temp_Norm_Volue_DE', 'Temp_Norm_Point_DE'])

# Temperature forecast and normal for FR
Norlys_df['Temp_Fore_Mean_FR'] = Norlys_df[['Temp_Fore_Volue_FR', 'Temp_Fore_Point_FR']].mean(axis=1)
Norlys_df = Norlys_df.drop(columns=['Temp_Fore_Volue_FR', 'Temp_Fore_Point_FR'])

Norlys_df['Temp_Norm_Mean_FR'] = Norlys_df[['Temp_Norm_Volue_FR', 'Temp_Norm_Point_FR']].mean(axis=1)
Norlys_df = Norlys_df.drop(columns=['Temp_Norm_Volue_FR', 'Temp_Norm_Point_FR'])




#--------------------------- Feature Creation ---------------------------

# Rename 'Time' column to 'Date'
Norlys_df.rename(columns={"Time": "Date"}, inplace=True)
Norlys_df.set_index("Date", inplace=True)

# Extract hour from datetime index
Norlys_df['Hours'] = Norlys_df.index.hour

# Add weekday column (1 = Monday, ..., 7 = Sunday)
Norlys_df['Weekday'] = Norlys_df.index.weekday + 1

# Create dummy variable for weekday vs weekend (1 = Weekday, 0 = Weekend)
Norlys_df['Weekday'] = (Norlys_df.index.dayofweek < 5).astype(int)



#--------------------------- Months and Seasons ---------------------------

# Add month column (1 = January, ..., 12 = December)
Norlys_df['Months'] = Norlys_df.index.month


# Map months to seasons
def get_season_numeric(month):
    if month in [3, 4, 5]:
        return 'Spring'  # Spring
    elif month in [6, 7, 8]:
        return 'Summer'  # Summer
    elif month in [9, 10, 11]:
        return 'Fall'  # Fall
    else:
        return 'Winter'  # Winter

# Apply season mapping
Norlys_df['season'] = Norlys_df['Months'].apply(get_season_numeric)

# One-hot encode season variable
Norlys_df = pd.get_dummies(Norlys_df, columns=['season'], dtype=int)

# Add calendar week number
Norlys_df['week'] = Norlys_df.index.isocalendar().week


#-----------24 hour rolling variance for spot prices (DE and FR)------------

# Adding 24-hour rolling variance of spot prices for DE and FR (excluding the current hour)
Norlys_df['Spot_DE_rolling_var_24'] = (
    Norlys_df['Spot_DE']
    .shift(1)  # Exclude the current hour
    .rolling(window=24, min_periods=24)
    .var(ddof=1)
)

Norlys_df['Spot_FR_rolling_var_24'] = (
    Norlys_df['Spot_FR']
    .shift(1)  # Exclude the current hour
    .rolling(window=24, min_periods=24)
    .var(ddof=1)
)

#--------------Holidays-----------

# Function to retrieve public holidays using the Nager.Date API
def get_holidays(years, country_code):
    holidays = []
    for year in years:
        url = f"https://date.nager.at/api/v3/PublicHolidays/{year}/{country_code}"
        resp = requests.get(url)
        if resp.ok:
            holidays += [h['date'] for h in resp.json()]
    return pd.to_datetime(holidays)

# Public holidays for the selected years
years = [2023, 2024]
french_holidays = get_holidays(years, "FR")
german_holidays = get_holidays(years, "DE")

# Create holiday indicators 
Norlys_df['holiday_indicator_FR'] = Norlys_df.index.normalize().isin(french_holidays.normalize()).astype(int)
Norlys_df['holiday_indicator_DE'] = Norlys_df.index.normalize().isin(german_holidays.normalize()).astype(int)


#---------------Total exchange --------------------

# Total exchange calculation for DE
exchange_cols_DE = [
    "Exchange_DE-AT", "Exchange_DE-BE", "Exchange_DE-CH",
    "Exchange_DE-CZ", "Exchange_DE-DK1", "Exchange_DE-DK2",
    "Exchange_DE-FR", "Exchange_DE-NL", "Exchange_DE-NO2",
    "Exchange_DE-PL", "Exchange_DE-SE4"
]
Norlys_df['Total_Exchange_DE'] = Norlys_df[exchange_cols_DE].sum(axis=1)

# Total exchange calculation for FR
exchange_cols_FR = [
    "Exchange_FR-BE", "Exchange_FR-CH", "Exchange_FR-DE",
    "Exchange_FR-ES", "Exchange_FR-GB", "Exchange_FR-ITNORD",
    "Exchange_EL1_FR", "Exchange_IF1_FR", "Exchange_IF2_FR"
]
Norlys_df['Total_Exchange_FR'] = Norlys_df[exchange_cols_FR].sum(axis=1)


#--------------------------Renewable and non renewable --------------
column_names = [
    "Actp_Bio_DE", "Actp_Bio_FR", "Actp_Coal_DE", "Actp_Coal_FR", "Actp_Gas_DE", "Actp_Gas_FR", 
    "Actp_Hydro_Pump_DE", "Actp_Hydro_Pump_FR", "Actp_Hydro_Res_DE", "Actp_Hydro_Res_FR", 
    "Actp_Hydro_Ror_DE", "Actp_Hydro_Ror_FR", "Actp_Lignite_DE", "Actp_Nuclear_FR", 
    "Actp_Oil_DE", "Actp_Oil_FR", "Actp_Solar_DE", "Actp_Solar_FR", 
    "Actp_Wind_Tot_DE", "Actp_Wind_Tot_FR"
]

# Separate into DE and FR columns
columns_DE = [col for col in column_names if "_DE" in col]
columns_FR = [col for col in column_names if "_FR" in col]

# Define renewable and non-renewable categories
renewables = ["Bio", "Hydro", "Solar", "Wind"]
non_renewables = ["Coal", "Gas", "Lignite", "Oil", "Nuclear"]

# Identify renewable and non-renewable columns for DE and FR
renewable_DE = [col for col in columns_DE if any(r in col for r in renewables)]
non_renewable_DE = [col for col in columns_DE if any(r in col for r in non_renewables)]

renewable_FR = [col for col in columns_FR if any(r in col for r in renewables)]
non_renewable_FR = [col for col in columns_FR if any(r in col for r in non_renewables)]

# Calculate total renewable and non-renewable production for DE and FR
Norlys_df['Renewable_DE'] = Norlys_df[renewable_DE].sum(axis=1)
Norlys_df['Non_renewable_DE'] = Norlys_df[non_renewable_DE].sum(axis=1)


Norlys_df['Renewable_FR'] = Norlys_df[renewable_FR].sum(axis=1)
Norlys_df['Non_renewable_FR'] = Norlys_df[non_renewable_FR].sum(axis=1)

# Calculate total production
Norlys_df['Total_Production_DE'] = Norlys_df['Renewable_DE'] + Norlys_df['Non_renewable_DE']
Norlys_df['Total_Production_FR'] = Norlys_df['Renewable_FR'] + Norlys_df['Non_renewable_FR']

# Estimate total consumption (load + solar + wind)
Norlys_df['Total_Consumption_FR'] = Norlys_df['Res_Load_Mean_FR'] + Norlys_df["Actp_Wind_Tot_FR"] + Norlys_df["Actp_Solar_FR"]
Norlys_df['Total_Consumption_DE'] = Norlys_df['Res_Load_Mean_DE'] + Norlys_df["Actp_Wind_Tot_DE"] + Norlys_df["Actp_Solar_DE"]

# Difference in renewable production between DE and FR
Norlys_df['Renewable_DE_minus_FR'] = Norlys_df['Renewable_DE'] - Norlys_df['Renewable_FR']


#---------------------------Interaction terms---------------------

scaler = StandardScaler()

# Computing scaled interaktion terms DE and FR 
wind_de_scaled = scaler.fit_transform(Norlys_df[['Actp_Wind_Tot_DE']])
solar_de_scaled = scaler.fit_transform(Norlys_df[['Actp_Solar_DE']])
Norlys_df['Interaction_Wind_Solar_DE'] = (wind_de_scaled * solar_de_scaled).flatten()


wind_fr_scaled = scaler.fit_transform(Norlys_df[['Actp_Wind_Tot_FR']])
solar_fr_scaled = scaler.fit_transform(Norlys_df[['Actp_Solar_FR']])
Norlys_df['Interaction_Wind_Solar_FR'] = (wind_fr_scaled * solar_fr_scaled).flatten()

# Create lagged spot price variables for DE and FR
Norlys_df['Spot_DE_lag1'] = Norlys_df['Spot_DE'].shift(1)
Norlys_df['Spot_DE_lag2'] = Norlys_df['Spot_DE'].shift(2)

Norlys_df['Spot_FR_lag1'] = Norlys_df['Spot_FR'].shift(1)
Norlys_df['Spot_FR_lag2'] = Norlys_df['Spot_FR'].shift(2)