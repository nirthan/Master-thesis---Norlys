# ⚡ Feature Engineering in Electricity Spot Price Forecasting  
### Using XGBoost and LSTM Models

## 📘 Overview

This master's thesis investigates the extent to which **feature engineering (FE)** can improve the performance of machine learning models — specifically **XGBoost** and **LSTM** — in forecasting **electricity spot prices** in **Germany (DE)** and **France (FR)**.

To achieve this, we developed two types of models:
- **Benchmark Models**: Using all available features.
- **Feature Engineered Models**: Constructed via systematic feature selection and transformation to reduce redundancy and improve accuracy.

## 🎯 Objectives

- Evaluate the performance difference between raw and feature-engineered datasets.
- Compare XGBoost and LSTM forecasting capabilities using metrics such as MAE, MSE, and RMSE.
- Assess practical application via a **spread trading strategy**.
- Validate performance improvements using **Diebold-Mariano tests**.

## 🧪 Methods

- **Data Preprocessing**:
  - Removed highly correlated features.
  - Constructed new informative features through domain knowledge and iterative testing.
- **Modeling Approaches**:
  - Gradient Boosted Trees (XGBoost)
  - Recurrent Neural Networks (LSTM)
- **Evaluation Metrics**:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - Cumulative profit and directional accuracy in a spread trading setup
- **Statistical Testing**:
  - Diebold-Mariano test for predictive accuracy comparison.

## 📊 Key Results

| Model   | Metric | DE (Raw) | DE (FE) | FR (Raw) | FR (FE) |
|---------|--------|----------|---------|----------|---------|
| **XGBoost** | MAE    | 8.6145   | 6.8916  | 7.1812   | 5.8538  |
|         | MSE    | 227.44   | 156.61  | 120.35   | 81.06   |
|         | RMSE   | 15.0813  | 12.5145 | 10.9705  | 9.0031  |
| **LSTM**    | MAE    | 14.1548  | 13.7150 | 12.9010  | 11.0903 |
|         | MSE    | 719.87   | 699.83  | 287.00   | 215.39  |
|         | RMSE   | 26.8304  | 26.4544 | 16.9411  | 14.6764 |

- **Statistical Significance** (Diebold-Mariano Test):
  - XGBoost DE: *p* = 0.0002
  - XGBoost FR: *p* = 0.0063

- **Spread Strategy**:
  - LSTM (FE): **€82,908 EUR/MWh**, error rate: 15.86%
  - LSTM (Raw): €80,595 EUR/MWh, error rate: 18.00%
  - XGBoost (FE): €84,336 EUR/MWh, error rate: 14.36%
  - XGBoost (Raw): **€84,397 EUR/MWh**, error rate: 14.22%

## 🔍 Insights

- **Feature engineering consistently improved forecasting accuracy** in both XGBoost and LSTM models.
- For XGBoost, improvements were **statistically significant**.
- For LSTM, although statistical testing was not feasible due to test set differences, FE models consistently outperformed benchmark models across metrics.
- However, **better accuracy does not always equate to better decision-making performance**, especially in trading strategies where predicting the correct direction matters.

## 🌍 Implications

- Feature engineering is a **valuable step in time series forecasting pipelines**.
- The results support further investment in intelligent feature design in energy forecasting models.
- Forecast accuracy is essential, but evaluation should also include **practical outcomes**, like profitability and directional correctness.
