#Step 1: Data Collection and Preparation
import pandas as pd
import numpy as np
from datetime import datetime

# Load data from a CSV file and parse the 'date' column as datetime
data = pd.read_csv('sales_data.csv', parse_dates=['date'])

# Sort the data by date to maintain the temporal order
data.sort_values('date', inplace=True)

# Data cleaning: Remove duplicates and fill missing values using forward fill
data.drop_duplicates(inplace=True)
data.fillna(method='ffill', inplace=True)  # Alternatively, other methods like interpolation can be used

# Feature engineering: Extract day of the week, month, and year from the date
data['day_of_week'] = data['date'].dt.dayofweek
data['month'] = data['date'].dt.month
data['year'] = data['date'].dt.year

# Create lag features (e.g., sales from the previous day) and rolling mean features
data['lag_1'] = data['sales'].shift(1)
data['rolling_mean_7'] = data['sales'].rolling(window=7).mean()

# Drop rows with NaN values that resulted from creating lag features
data.dropna(inplace=True)

#Step 2: Data Splitting
# Split the data into training and testing sets based on a cutoff date
train = data[data['date'] < '2022-01-01']
test = data[data['date'] >= '2022-01-01']

# Separate features and target variable for machine learning models
features = ['day_of_week', 'month', 'year', 'lag_1', 'rolling_mean_7']
X_train, y_train = train[features], train['sales']
X_test, y_test = test[features], test['sales']

#Step 3: Model Training and Comparison
from fbprophet import Prophet

# Prepare data for Prophet (requires columns 'ds' for date and 'y' for target)
prophet_df = train[['date', 'sales']].rename(columns={'date': 'ds', 'sales': 'y'})

# Initialize and train the Prophet model
prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
prophet_model.fit(prophet_df)

# Make predictions for the test period
future = prophet_model.make_future_dataframe(periods=len(test))
forecast = prophet_model.predict(future)

import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Create DMatrix objects for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Define XGBoost parameters
params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'eta': 0.1,
    'max_depth': 5,
    'seed': 42
}

# Train the XGBoost model
xgb_model = xgb.train(params, dtrain, num_boost_round=100)

# Make predictions on the test set
preds_xgb = xgb_model.predict(dtest)

import statsmodels.api as sm

# Prepare time series data for ARIMA
train_ts = train.set_index('date')['sales']

# Fit a SARIMAX model with fixed parameters (p,d,q) and seasonal parameters
model_arima = sm.tsa.statespace.SARIMAX(train_ts, order=(1,1,1), seasonal_order=(1,1,1,7))
arima_result = model_arima.fit(disp=False)

# Make predictions for the test period
arima_preds = arima_result.predict(start=test.index[0], end=test.index[-1], dynamic=False)


#Step 4: Model Comparison
# Define a function to calculate evaluation metrics
def calculate_metrics(true, pred):
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    mape = np.mean(np.abs((true - pred) / true)) * 100
    return mae, rmse, mape

# Calculate metrics for Prophet
prophet_forecast = forecast.set_index('ds').loc[test['date'], 'yhat']
mae_prophet, rmse_prophet, mape_prophet = calculate_metrics(test['sales'], prophet_forecast)

# Calculate metrics for XGBoost
mae_xgb, rmse_xgb, mape_xgb = calculate_metrics(y_test, preds_xgb)

# Calculate metrics for ARIMA
arima_forecast = pd.Series(arima_preds, index=test['date'])
mae_arima, rmse_arima, mape_arima = calculate_metrics(test['sales'], arima_forecast)

# Print metrics for comparison
print("Prophet: MAE =", mae_prophet, " RMSE =", rmse_prophet, " MAPE =", mape_prophet)
print("XGBoost: MAE =", mae_xgb, " RMSE =", rmse_xgb, " MAPE =", mape_xgb)
print("ARIMA: MAE =", mae_arima, " RMSE =", rmse_arima, " MAPE =", mape_arima)

#Step 5: Model Saving
import joblib

# Save the XGBoost model and scaler
joblib.dump(xgb_model, 'models/xgb_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

# Save the Prophet model using pickle
import pickle
with open('models/prophet_model.pkl', 'wb') as f:
    pickle.dump(prophet_model, f)

#Step 6: Deployment (Inference)
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the saved model and scaler
model = joblib.load('models/xgb_model.pkl')
scaler = joblib.load('models/scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    json_data = request.get_json()
    # Preprocess the input data and make predictions
    # (Code for preprocessing and prediction would go here)
    return jsonify(predictions)

