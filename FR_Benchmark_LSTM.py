# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 23:43:25 2025

@author: Nirthan
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K

from pyswarms.single import GlobalBestPSO


# ------------------------- Load and Preprocess Data -------------------------

# Load dataset
FR_data = pd.read_excel("Benchmark_model_data.xlsx", sheet_name="Sheet1")
date_range = pd.date_range(start='2023-01-01 00:00:00', periods=len(FR_data), freq='H')
FR_data['Timestamp'] = date_range
FR_data.set_index("Timestamp", inplace=True)

# Split into features (X) and target (Y)
X_variable = FR_data.drop(columns=['Spot_FR'])
Y_variable = FR_data[['Spot_FR']]

# Standardize features and target
scaler = StandardScaler()
X_variable_scaled = scaler.fit_transform(X_variable)
X_variable_scaled = pd.DataFrame(X_variable_scaled, columns=X_variable.columns)

Y_scaler = StandardScaler()        
Y_variable_scaled = Y_scaler.fit_transform(Y_variable)
Y_variable_scaled = pd.DataFrame(Y_variable_scaled, columns=Y_variable.columns)

# ------------------------- Sequence Preparation -------------------------
def create_sequences(X, Y, time_step):
    X_seq, Y_seq = [], []
    for i in range(len(X) - time_step):
        X_seq.append(X[i:(i + time_step), :])
        Y_seq.append(Y[i + time_step, :])
    return np.array(X_seq), np.array(Y_seq)

time_step = 24
X_seq, Y_seq = create_sequences(X_variable_scaled.values, Y_variable_scaled.values, time_step)

# ------------------------- Data Split -------------------------
np.random.seed(42)
random.seed(42)

train_size = int(0.65 * len(X_seq))
val_size = int(0.15 * len(X_seq))

X_train = X_seq[:train_size]
Y_train = Y_seq[:train_size]
X_val = X_seq[train_size:train_size + val_size]
Y_val = Y_seq[train_size:train_size + val_size]

# ------------------------- Objective Function -------------------------
# RMSE metric
def RMSE(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def objective_function(params):
    batch_size = int(params[0, 0])
    n_units = int(params[0, 1])
    learning_rate = params[0, 2]
    recurrent_dropout_rate = params[0, 3]

    model = Sequential()
    model.add(LSTM(units=n_units,
                   input_shape=(X_train.shape[1], X_train.shape[2]),
                   rrecurrent_dropout=recurrent_dropout_rate))
    model.add(Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mean_squared_error', metrics=[RMSE])

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, Y_train, epochs=50, batch_size=batch_size, validation_data=(X_val, Y_val),
              callbacks=[early_stopping], verbose=0)

    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(Y_val, y_pred))
    print("Validation RMSE:", rmse)
    return rmse

# ------------------------- PSO Hyperparameter Tuning -------------------------
lower_bound = np.array([16, 64, 0.0001, 0.1]) # lower bounds
upper_bound = np.array([128, 512, 0.005, 0.2]) # upper bounds
bounds = (lower_bound, upper_bound)

options = {'c1': 2, 'c2': 2, 'w': 0.9}
optimizer = GlobalBestPSO(n_particles=16, dimensions=4, options=options, bounds=bounds)

best_cost, best_position = optimizer.optimize(objective_function, iters=50)

print(f"Best RMSE: {best_cost}")
print(f"Best Hyperparameters: {best_position}")

# ------------------------- Rolling Forecasting -------------------------
# Best hyperparameters from PSO
best_batch_size = int(best_position[0])
best_units = int(best_position[1])
best_learning_rate = best_position[2]
best_dropout_rate = best_position[3]

rolling_train_size = int(0.8 * len(X_seq))
rolling_train_data = X_seq[:rolling_train_size]
rolling_train_labels = Y_seq[:rolling_train_size]
rolling_test_data = X_seq[rolling_train_size:]
rolling_test_labels = Y_seq[rolling_train_size:]

def create_lstm_model(input_shape, units, learning_rate, recurrent_dropout_rate):
    model = Sequential()
    model.add(LSTM(units=units,
                   input_shape=input_shape,
                   recurrent_dropout=recurrent_dropout_rate))  
    model.add(Dense(1))  
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mean_squared_error')
    return model

# Rolling loop
all_predictions = []
all_actuals = []
test_start_idx = 0

while test_start_idx < len(rolling_test_data):
    model = create_lstm_model(
        input_shape=(rolling_train_data.shape[1], rolling_train_data.shape[2]),
        units=best_units,
        learning_rate=best_learning_rate,
        recurrent_dropout_rate=best_dropout_rate)

    model.fit(rolling_train_data, rolling_train_labels, epochs=10, batch_size=best_batch_size, verbose=0)

    remaining_steps = len(rolling_test_data) - test_start_idx
    steps_to_predict = min(168, remaining_steps)

    X_test_block = rolling_test_data[test_start_idx:test_start_idx + steps_to_predict]
    y_test_block = rolling_test_labels[test_start_idx:test_start_idx + steps_to_predict]
    y_pred_block = model.predict(X_test_block)

    all_predictions.extend(y_pred_block)
    all_actuals.extend(y_test_block)

    test_start_idx += steps_to_predict

    if steps_to_predict == 168:
        rolling_train_data = np.concatenate([rolling_train_data[168:], X_test_block], axis=0)
        rolling_train_labels = np.concatenate([rolling_train_labels[168:], y_test_block], axis=0)

# Save results
results_df = pd.DataFrame({
    'Actual': np.squeeze(all_actuals),
    'Predicted': np.squeeze(all_predictions)
})

# ------------------------- Evaluation & Plotting -------------------------

# Inverse scaling
actual_unscaled = Y_scaler.inverse_transform(np.array(all_actuals).reshape(-1, 1))
predicted_unscaled = Y_scaler.inverse_transform(np.array(all_predictions).reshape(-1, 1))

# Create final DataFrame
target_length = 3504
last_20_percent_index = len(FR_data) - target_length
last_20_percent_index = FR_data.index[last_20_percent_index:]

unscaled_results_df = pd.DataFrame({
    'Timestamp': last_20_percent_index[:len(actual_unscaled)],
    'Actual': np.squeeze(actual_unscaled),
    'Predicted': np.squeeze(predicted_unscaled)
})
unscaled_results_df.set_index('Timestamp', inplace=True)

# Metrics
mse = mean_squared_error(unscaled_results_df['Actual'], unscaled_results_df['Predicted'])
rmse = np.sqrt(mse)
mae = mean_absolute_error(unscaled_results_df['Actual'], unscaled_results_df['Predicted'])

print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")

# Plot predictions
plt.figure(figsize=(12, 6))
plt.plot(unscaled_results_df.index, unscaled_results_df['Actual'], label='Actual', linewidth=2)
plt.plot(unscaled_results_df.index, unscaled_results_df['Predicted'], label='Predicted', color='red', linewidth=2)
plt.legend(loc='upper left',fontsize=16)  
plt.title('Actual vs Predicted', fontsize=20)
plt.xlabel('Date (Hourly)', fontsize=18)
plt.ylabel('Spot Price for FR', fontsize=20)
plt.xticks(rotation=45, fontsize=18)  
plt.yticks(fontsize=18)
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()  
plt.show()

