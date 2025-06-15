# Time Series Sales Forecasting

A comprehensive Python solution for sales forecasting using multiple time series forecasting approaches.

## Overview

This project implements an end-to-end pipeline for sales forecasting, comparing three popular methods: Facebook Prophet, XGBoost, and SARIMAX. The solution includes data preprocessing, feature engineering, model training, evaluation, visualization, and a deployment-ready API.

## Features

- **Complete Data Pipeline**: From raw data to production-ready predictions
- **Multiple Models**: Implementation of statistical (SARIMAX), machine learning (XGBoost), and specialized forecasting (Prophet) models
- **Advanced Feature Engineering**: Time-based features, lagged values, rolling statistics
- **Comprehensive Evaluation**: Multiple metrics (RMSE, MAE, MAPE, R²) and cross-validation
- **Rich Visualizations**: Time series plots, seasonality analysis, model comparisons, residual diagnostics
- **Model Explainability**: Feature importance analysis and model diagnostics
- **Production-Ready API**: Flask-based REST API with proper error handling and logging
- **Best Practices**: Proper train/test splitting, scaling, model persistence, and metadata tracking

## Pipeline Steps

1. **Data Collection and Preparation**
2. **Feature Engineering**
3. **Model Training**
4. **Model Evaluation and Analysis**
5. **Deployment**

## Results

## Model Performance

Model 	RMSE 	MAE 	MAPE (%) 	R² 

XGBoost 	126.4 	94.7 	8.21 	0.857

Prophet 	153.8 	115.2 	10.05 	0.794

SARIMAX 	165.1 	127.3 	11.32 	0.761

XGBoost consistently outperformed the other models, achieving approximately:

  - 18% lower RMSE than Prophet
    
  - 24% lower RMSE than SARIMAX
    
  - 8.21% mean absolute percentage error

## Outcome

*   XGBoost model selected for deployment
## License

This project is licensed under the MIT License - see the LICENSE file for details.




## Future Work

- Ensemble methods:
Combining the forecasts of all three models could provide an additional increase in accuracy.
Weighted averaging based on the forecast horizon

- External data:
Absence of external regressors (economic indicators, weather, marketing campaigns)
Potentially overlooked explanatory variables

- Exploring other time series forecasting models (LSTM, GRU)

## Contributing

*   Contributions welcome! Please submit a pull request with detailed changes.

## License

*   MIT License. See `LICENSE` for details.
