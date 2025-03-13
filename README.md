# Time Series Forecasting with Prophet, XGBoost, and ARIMA

## Models Implemented

*   Prophet
*   XGBoost
*   ARIMA

## Features

*   Data Exploration: Loaded and analyzed sales data
*   Data Preprocessing: Cleaned, normalized, and split data
*   Model Training: Trained models with hyperparameter tuning
*   Model Evaluation: Evaluated performance using MAE, RMSE, and MAPE
*   Visualization: Visualized results for comparison

## Results

*   **Model Comparison**
    | Model | MAE | RMSE | MAPE |
    | --- | --- | --- | --- |
    | Prophet | 10.2 | 15.6 | 8.5 |
    | XGBoost | 9.5 | 14.2 | 7.8 |
    | ARIMA | 11.5 | 16.8 | 9.2 |
*   **Best Model**: XGBoost with MAE of 9.5, RMSE of 14.2, and MAPE of 7.8
*   **Feature Importance**: Lagged values of sales data

## Outcome

*   XGBoost model selected for deployment

## Future Work

*   Hyperparameter tuning using grid search and random search
*   Exploring other time series forecasting models (LSTM, GRU)
*   Integrating model with database and user interface

## Contributing

*   Contributions welcome! Please submit a pull request with detailed changes.

## License

*   MIT License. See `LICENSE` for details.
