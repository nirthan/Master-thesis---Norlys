# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 19:14:10 2025

@author: Nirthan
"""


import random
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,root_mean_squared_error,mean_absolute_error
from pyswarms.single import GlobalBestPSO 

#------------------------Load and Preprocess Data--------------------------

# Load the dataset 
Norlys_df = pd.read_excel("FE_DE_data.xlsx", sheet_name="Sheet1")
Norlys_df.rename(columns={'Timer': 'Hours'}, inplace=True)
Norlys_df.set_index("Date", inplace=True)
Norlys_df = Norlys_df.apply(pd.to_numeric, errors='coerce')


#-----------------------------Data split-----------------------


# Create a working copy of the original dataset
Norlys_modeldf=Norlys_df.copy()

np.random.seed(42)
random.seed(42)

# Define sizes for training, validation, and testing sets
train_size = int(0.65 * len(Norlys_modeldf))
valid_size = int(0.15 * len(Norlys_modeldf))

train_data = Norlys_modeldf[:train_size]
valid_data = Norlys_modeldf[train_size:train_size + valid_size]
test_data = Norlys_modeldf[train_size + valid_size:]

# Separate features and target variable for each dataset split
X_train = train_data.drop(columns=['Spot_DE'])  
y_train = train_data['Spot_DE']

X_val = valid_data.drop(columns=['Spot_DE'])
y_val = valid_data['Spot_DE']

X_test = test_data.drop(columns=['Spot_DE'])
y_test = test_data['Spot_DE']

#---------------------------Objective Function------------------------


def objective_function(params):
    # Chosen hyperparameters to be tuned
    learning_rate = float(params[0, 0])
    max_depth = int(params[0, 1])
    colsample_bytree = float(params[0, 2])
    subsample = float(params[0, 3])
    gamma = float(params[0, 4])
    reg_lambda = float(params[0, 5])

    print(f"Learning Rate: {learning_rate}, Max Depth: {max_depth}, Colsample By Tree: {colsample_bytree}, Subsample: {subsample}, Gamma: {gamma}, Reg Lambda: {reg_lambda}")
    
    # Define Raw XGBoost model
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        eval_metric='rmse',
        tree_method='exact',
        early_stopping_rounds=10,
        n_estimators=100,
        learning_rate=learning_rate,
        max_depth=max_depth,
        colsample_bytree=colsample_bytree,
        subsample=subsample,
        gamma=gamma,
        reg_lambda=reg_lambda,
        random_state=42
    )

    # Train the model
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    # Predict on validation set
    y_pred = xgb_model.predict(X_val)
    
    # Compute evaluation metrics
    rmse = root_mean_squared_error(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
   
   # Print metrics 
    print(f"RMSE: {rmse}, MSE: {mse}, MAE: {mae}")
    
    return rmse

#----------------------------PSO Hyperparamter tuning-----------------------------

# Define the bounds for hyperparameters
bounds = (
    [0.001, 5, 0.5, 0.5, 0, 1],    
    [0.3, 10, 1, 1, 2, 10]   # Upper bounds
)


optimizer = GlobalBestPSO(n_particles=24, dimensions=6, options={'c1': 2, 'c2': 2, 'w': 0.9}, bounds=bounds)

# Perform PSO optimization
best_rmse, best_pso = optimizer.optimize(objective_function, iters=20)

# Print best parameters found by PSO
print("Best hyperparameters found:")
print(f"Learning Rate: {best_pso[0]}")
print(f"Max Depth: {int(best_pso[1])}")
print(f"Colsample by Tree: {best_pso[2]}")
print(f"Subsample: {best_pso[3]}")
print(f"Gamma: {best_pso[4]}")
print(f"Reg Lambda: {best_pso[5]}")
print(f"Best RMSE: {best_rmse}")


#-------------------------------Rolling forecast data split----------------

# Best hyperparamter from PSO     
best_learning_rate= best_pso[0]
best_max_depth= int(best_pso[1])
best_colsample_bytree= best_pso[2]
best_subsample=best_pso[3]
best_gamma= best_pso[4]
best_reg_lambda= best_pso[5]

# Define Rolling train and test split 
Rolling_train_size = int(0.8 * len(Norlys_modeldf))

# Split the data into train and test
Rolling_train = Norlys_modeldf[:Rolling_train_size]
Rolling_test = Norlys_modeldf[Rolling_train_size:]

X_train_rolling = Rolling_train.drop(columns=['Spot_DE'])  
y_train_rolling = Rolling_train['Spot_DE']

X_test_rolling = Rolling_test.drop(columns=['Spot_DE'])
y_test_rolling = Rolling_test['Spot_DE']


#--------------------------- Feature Importance Analysis ---------------------------#

# Initialize the best model with tuned hyperparameters
best_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    eval_metric='rmse',
    tree_method='exact',
    n_estimators=100,
    learning_rate=best_learning_rate,
    max_depth=int(best_max_depth),
    colsample_bytree=best_colsample_bytree,
    subsample=best_subsample,
    gamma=best_gamma,
    reg_lambda=best_reg_lambda,
    random_state=42
)

# Train the model to the training data
best_model.fit(X_train_rolling, y_train_rolling)


# Get the feature importance from the model
feature_importance = best_model.feature_importances_

sorted_idx = np.argsort(feature_importance)[::-1]

booster = best_model.get_booster()

booster_features = booster.feature_names

importance_gain = booster.get_score(importance_type='gain')
importance_weight = booster.get_score(importance_type='weight')
importance_cover = booster.get_score(importance_type='cover')

importance_df = pd.DataFrame({
    'Feature': booster_features,
    'Gain': [importance_gain.get(feat, 0) for feat in booster_features],
    'Weight': [importance_weight.get(feat, 0) for feat in booster_features],
    'Cover': [importance_cover.get(feat, 0) for feat in booster_features],
    'Default_Importance': best_model.feature_importances_  
})

# Sort features based on frequency (Weight)
importance_df_weights_sorted = importance_df.sort_values(by='Weight',ascending=False)

#---------------------- Display Top 10 Most Frequently Used Features ----------------------#

top_10_weight = importance_df_weights_sorted.head(10)
print("\nTop 10 features by weight:")
print(top_10_weight)


#-----------------------------------Rolling forecast loop-----------------------

step_size = 168
all_predictions = []
all_actuals = []
test_start_idx = 0

# Rolling loop
while test_start_idx < len(X_test_rolling):  

    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        eval_metric='rmse',
        tree_method='exact',
        n_estimators=100,
        learning_rate=best_learning_rate,
        max_depth=int(best_max_depth),
        colsample_bytree=best_colsample_bytree,
        subsample=best_subsample,
        gamma=best_gamma,
        reg_lambda=best_reg_lambda,
        random_state=42
    )
    model.fit(X_train_rolling, y_train_rolling)

    remaining_steps = len(X_test_rolling) - test_start_idx
    steps_to_predict = min(step_size, remaining_steps)

    X_test_block = X_test_rolling.iloc[test_start_idx:test_start_idx + steps_to_predict]
    y_test_block = y_test_rolling.iloc[test_start_idx:test_start_idx + steps_to_predict]

    y_pred_block = model.predict(X_test_block)

    all_predictions.extend(y_pred_block)
    all_actuals.extend(y_test_block)

    test_start_idx += steps_to_predict

    if steps_to_predict == step_size:  
        X_train_rolling = pd.concat([X_train_rolling.iloc[step_size:], X_test_block])
        y_train_rolling = pd.concat([y_train_rolling.iloc[step_size:], y_test_block])


# Save results
results_xgb = pd.DataFrame({
    'Actual': all_actuals,
    'Predicted': all_predictions
})

# Align result index with corresponding dates
results_xgb.index = y_test_rolling.index[:len(results_xgb)]


#--------------------------Evaluation and Plotting---------------------

# Calculate and print metrics
rmse = np.sqrt(mean_squared_error(results_xgb['Actual'], results_xgb['Predicted']))
mae = mean_absolute_error(results_xgb['Actual'], results_xgb['Predicted'])
mse = mean_squared_error(results_xgb['Actual'], results_xgb['Predicted'])

print(f"RMSE: {rmse:.5f}")
print(f"MAE: {mae:5f}")
print(f"MSE: {mse:.5f}")


# Plot actual vs predicted values
plt.figure(figsize=(12, 6))
plt.plot(results_xgb.index, results_xgb['Actual'], label='Actual', linewidth=2)
plt.plot(results_xgb.index, results_xgb['Predicted'], label='Predicted', linewidth=2)
plt.legend(loc='upper left',fontsize=16)  
plt.title('Actual vs Predicted', fontsize=20)
plt.xlabel('Date (Hourly)', fontsize=18)
plt.ylabel('Spot Price for DE', fontsize=20)
plt.xticks(rotation=45, fontsize=18)
plt.yticks(fontsize=18)
plt.grid(True)
plt.tight_layout()  
plt.show()