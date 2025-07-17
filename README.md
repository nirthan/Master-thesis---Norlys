# ⚡ Feature Engineering in Electricity Spot Price Forecasting: Using XGBoost and LSTM Models

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

## 📊 Key Findings

### Model Performance and Feature Engineering

This project evaluated the impact of feature engineering on two machine learning models — XGBoost and LSTM — for forecasting electricity spot prices in Germany and France.

- Both models were trained with two sets of features:
  - **Raw models**: using all available features without modification.
  - **Feature-engineered (FE) models**: using a carefully selected and created subset of features to reduce redundancy and improve model simplicity.

- The feature-engineered models consistently **outperformed the raw models** across multiple accuracy metrics, showing reduced forecasting errors.

- Statistical testing confirmed that the improvements for the XGBoost model were **statistically significant** in both countries.

- For the LSTM models, while formal statistical testing was limited due to dataset differences, the performance metrics consistently favored the feature-engineered approach.

### Practical Impact on Trading Strategy

- Applying the forecasting models to a trading (spread) strategy revealed that:
  - Improvements in forecast accuracy do not always translate directly to better decision-making outcomes.
  - The feature-engineered LSTM model notably achieved higher cumulative profits and lower error rates compared to its raw counterpart.
  - The XGBoost model showed marginal differences in profitability despite better forecast accuracy, highlighting the complexity of linking forecast improvements to financial gains.

### Conclusion

Overall, the study demonstrates that **feature engineering enhances the predictive power** of both XGBoost and LSTM models in electricity spot price forecasting. These improvements can contribute to more accurate and reliable forecasts, which are essential for effective market decision-making and energy trading.

## 🔍 Insights

- **Feature engineering consistently improved forecasting accuracy** in both XGBoost and LSTM models.
- For XGBoost, improvements were **statistically significant**.
- For LSTM, although statistical testing was not feasible due to test set differences, FE models consistently outperformed benchmark models across metrics.
- However, **better accuracy does not always equate to better decision-making performance**, especially in trading strategies where predicting the correct direction matters.

## 🌍 Implications

- Feature engineering is a **valuable step in time series forecasting pipelines**.
- The results support further investment in intelligent feature design in energy forecasting models.
- Forecast accuracy is essential, but evaluation should also include **practical outcomes**, like profitability and directional correctness.
