
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import statsmodels.api as sm
from prophet import Prophet
import joblib
import pickle
import warnings
warnings.filterwarnings('ignore')

# Create directories for outputs
os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

#############################################
# Step 1: Data Collection and Preparation
#############################################

# Load data from a CSV file and parse the 'date' column as datetime
print("Loading and preparing data...")
data = pd.read_csv('sales_data.csv', parse_dates=['date'])

# Sort the data by date to maintain the temporal order
data.sort_values('date', inplace=True)

# Exploratory Data Analysis
print("Performing exploratory data analysis...")
print(f"Dataset shape: {data.shape}")
print("\nData overview:")
print(data.head())
print("\nData types:")
print(data.dtypes)
print("\nMissing values:")
print(data.isnull().sum())
print("\nDescriptive statistics:")
print(data.describe())

# Visualize the time series
plt.figure(figsize=(15, 7))
plt.plot(data['date'], data['sales'])
plt.title('Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.grid(True)
plt.savefig('plots/sales_over_time.png')
plt.close()

# Analyze seasonality
plt.figure(figsize=(15, 10))

# Daily seasonality
plt.subplot(3, 1, 1)
data.groupby(data['date'].dt.dayofweek)['sales'].mean().plot(kind='bar')
plt.title('Average Sales by Day of Week')
plt.xticks(ticks=range(7), labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
plt.grid(True, axis='y')

# Monthly seasonality
plt.subplot(3, 1, 2)
data.groupby(data['date'].dt.month)['sales'].mean().plot(kind='bar')
plt.title('Average Sales by Month')
plt.xticks(ticks=range(1, 13), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.grid(True, axis='y')

# Yearly trend
plt.subplot(3, 1, 3)
data.groupby(data['date'].dt.year)['sales'].mean().plot(kind='bar')
plt.title('Average Sales by Year')
plt.grid(True, axis='y')

plt.tight_layout()
plt.savefig('plots/seasonality_analysis.png')
plt.close()

# Data cleaning: Remove duplicates and handle missing values
print("Cleaning data...")
duplicate_count = data.duplicated().sum()
print(f"Found {duplicate_count} duplicate rows")
data.drop_duplicates(inplace=True)

missing_count = data.isnull().sum().sum()
print(f"Found {missing_count} missing values")
if missing_count > 0:
    # First try interpolation for missing values in time series
    data = data.set_index('date').interpolate(method='time').reset_index()
    # For any remaining NaN values, use forward fill
    data.fillna(method='ffill', inplace=True)
    # Last resort backward fill
    data.fillna(method='bfill', inplace=True)

# Feature engineering: Extract time-based features
print("Performing feature engineering...")
data['day_of_week'] = data['date'].dt.dayofweek
data['day_of_month'] = data['date'].dt.day
data['month'] = data['date'].dt.month
data['quarter'] = data['date'].dt.quarter
data['year'] = data['date'].dt.year
data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
data['is_month_start'] = data['date'].dt.is_month_start.astype(int)
data['is_month_end'] = data['date'].dt.is_month_end.astype(int)

# Create lag features (previous day, week, month)
for lag in [1, 7, 14, 30]:
    data[f'lag_{lag}'] = data['sales'].shift(lag)

# Create rolling window features (mean, std, min, max)
for window in [7, 14, 30]:
    data[f'rolling_mean_{window}'] = data['sales'].rolling(window=window).mean()
    data[f'rolling_std_{window}'] = data['sales'].rolling(window=window).std()
    data[f'rolling_min_{window}'] = data['sales'].rolling(window=window).min()
    data[f'rolling_max_{window}'] = data['sales'].rolling(window=window).max()

# Add sales momentum (percent change)
data['pct_change'] = data['sales'].pct_change()
data['pct_change_7'] = data['sales'].pct_change(periods=7)

# Drop rows with NaN values that resulted from creating lag features
print(f"Rows before dropping NaN: {len(data)}")
data.dropna(inplace=True)
print(f"Rows after dropping NaN: {len(data)}")

# Check correlation between features
plt.figure(figsize=(20, 16))
selected_features = ['sales', 'day_of_week', 'month', 'year', 
                    'lag_1', 'lag_7', 'rolling_mean_7', 'rolling_std_7',
                    'pct_change', 'is_weekend']
correlation = data[selected_features].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Matrix')
plt.savefig('plots/correlation_matrix.png')
plt.close()

#############################################
# Step 2: Data Splitting and Scaling
#############################################

print("Splitting data into training and test sets...")
# Define the cutoff date for train/test split (80/20 split based on time)
total_days = (data['date'].max() - data['date'].min()).days
cutoff_date = data['date'].min() + timedelta(days=int(total_days * 0.8))
print(f"Cutoff date for train/test split: {cutoff_date}")

# Split the data into training and testing sets
train = data[data['date'] < cutoff_date]
test = data[data['date'] >= cutoff_date]
print(f"Training set size: {len(train)}, Test set size: {len(test)}")

# Define features to use in machine learning models
features = [
    'day_of_week', 'day_of_month', 'month', 'quarter', 'year', 
    'is_weekend', 'is_month_start', 'is_month_end',
    'lag_1', 'lag_7', 'lag_14', 'lag_30',
    'rolling_mean_7', 'rolling_std_7', 'rolling_min_7', 'rolling_max_7',
    'rolling_mean_14', 'rolling_mean_30',
    'pct_change', 'pct_change_7'
]

# Separate features and target variable
X_train, y_train = train[features], train['sales']
X_test, y_test = test[features], test['sales']

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame to maintain feature names
X_train_scaled = pd.DataFrame(X_train_scaled, columns=features, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=features, index=X_test.index)

#############################################
# Step 3: Model Training and Comparison
#############################################

print("Training models...")

# 1. Prophet Model Training
print("Training Prophet model...")
# Prepare data for Prophet (requires columns 'ds' for date and 'y' for target)
prophet_df = train[['date', 'sales']].rename(columns={'date': 'ds', 'sales': 'y'})

# Initialize and train the Prophet model with seasonality components
prophet_model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode='multiplicative',  # Often better for retail sales
    changepoint_prior_scale=0.05,  # Controls flexibility of the trend
    seasonality_prior_scale=10.0    # Controls flexibility of the seasonality
)

# Add custom seasonality if needed (e.g., quarterly)
prophet_model.add_seasonality(name='quarterly', period=91.25, fourier_order=5)

# Add country-specific holidays if relevant
# prophet_model.add_country_holidays(country_name='US')

# Fit the model
prophet_model.fit(prophet_df)

# Make predictions for the test period
future = prophet_model.make_future_dataframe(periods=len(test), freq='D')
forecast = prophet_model.predict(future)

# Visualize Prophet components
fig1 = prophet_model.plot_components(forecast)
fig1.savefig('plots/prophet_components.png')
plt.close(fig1)

# 2. XGBoost Model Training
print("Training XGBoost model...")
# Create DMatrix objects for XGBoost
dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
dtest = xgb.DMatrix(X_test_scaled, label=y_test)

# Define a parameter grid for hyperparameter tuning
param_grid = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'eta': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2]
}

# Simplified hyperparameter tuning using a subset of parameters
best_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'seed': 42
}

# For a real implementation, consider:
# 1. Using cross-validation with TimeSeriesSplit
# 2. Grid search or Bayesian optimization for hyperparameter tuning

# Train the XGBoost model with early stopping
eval_set = [(dtrain, 'train'), (dtest, 'test')]
xgb_model = xgb.train(
    best_params, 
    dtrain, 
    num_boost_round=1000,
    evals=eval_set,
    early_stopping_rounds=50,
    verbose_eval=100
)

# Make predictions on the test set
preds_xgb = xgb_model.predict(dtest)

# 3. SARIMAX Model Training
print("Training SARIMAX model...")
# Prepare time series data for ARIMA
train_ts = train.set_index('date')['sales']

# Use auto_arima to find optimal parameters (simplified version)
# In a real implementation, you should use auto_arima from pmdarima
# Here we're using fixed parameters for demonstration
model_arima = sm.tsa.statespace.SARIMAX(
    train_ts,
    order=(1, 1, 1),             # ARIMA(p,d,q) parameters
    seasonal_order=(1, 1, 1, 7), # SARIMA(P,D,Q,s) seasonal parameters
    enforce_stationarity=False,
    enforce_invertibility=False
)

arima_result = model_arima.fit(disp=False)

# Summary of ARIMA model
arima_summary = arima_result.summary()
print("\nARIMA Model Summary:")
print(arima_summary)

# Make predictions for the test period
arima_preds = arima_result.predict(
    start=test.index[0],
    end=test.index[-1],
    dynamic=False
)

# Diagnostic plots for ARIMA
fig2 = arima_result.plot_diagnostics(figsize=(15, 12))
plt.savefig('plots/arima_diagnostics.png')
plt.close(fig2)

#############################################
# Step 4: Model Comparison and Analytics
#############################################

print("Evaluating and comparing models...")

# Define a function to calculate evaluation metrics
def calculate_metrics(true, pred):
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    r2 = r2_score(true, pred)
    
    # MAPE calculation with handling for zeros
    mask = true != 0
    mape = np.mean(np.abs((true[mask] - pred[mask]) / true[mask])) * 100
    
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R2': r2}

# Get forecasts for each model aligned with test dates
prophet_forecast = forecast[forecast['ds'].isin(test['date'])].set_index('ds')['yhat']
prophet_forecast.index = test.index

# Store results in a dictionary
results = {}

# Calculate metrics for Prophet
results['Prophet'] = calculate_metrics(test['sales'], prophet_forecast)

# Calculate metrics for XGBoost
results['XGBoost'] = calculate_metrics(y_test, preds_xgb)

# Calculate metrics for ARIMA
arima_forecast = pd.Series(arima_preds, index=test.index)
results['ARIMA'] = calculate_metrics(test['sales'], arima_forecast)

# Create a DataFrame for easy comparison
metrics_df = pd.DataFrame(results).T
print("\nModel Comparison Metrics:")
print(metrics_df)

# Save metrics to CSV
metrics_df.to_csv('models/model_metrics.csv')

# Visualize model predictions vs actual
plt.figure(figsize=(15, 7))
plt.plot(test['date'], test['sales'], label='Actual', color='black', linewidth=2)
plt.plot(test['date'], prophet_forecast, label='Prophet', linestyle='--')
plt.plot(test['date'], preds_xgb, label='XGBoost', linestyle='-.')
plt.plot(test['date'], arima_forecast, label='ARIMA', linestyle=':')
plt.title('Model Comparison: Actual vs Predicted Sales')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.savefig('plots/model_comparison.png')
plt.close()

# Analyze residuals for the best performing model (using XGBoost as example)
residuals_xgb = y_test - preds_xgb

# Plot residuals over time
plt.figure(figsize=(15, 7))
plt.plot(test['date'], residuals_xgb)
plt.axhline(y=0, color='r', linestyle='-')
plt.title('XGBoost Residuals Over Time')
plt.xlabel('Date')
plt.ylabel('Residual Value')
plt.grid(True)
plt.savefig('plots/residuals_time.png')
plt.close()

# Plot residuals vs predicted
plt.figure(figsize=(10, 6))
plt.scatter(preds_xgb, residuals_xgb, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='-')
plt.title('XGBoost Residual Plot')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.grid(True)
plt.savefig('plots/residuals_predicted.png')
plt.close()

# Residual distribution
plt.figure(figsize=(10, 6))
plt.hist(residuals_xgb, bins=30, alpha=0.7)
plt.title('Distribution of XGBoost Residuals')
plt.xlabel('Residual Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig('plots/residuals_distribution.png')
plt.close()

# Plot XGBoost feature importance
plt.figure(figsize=(12, 8))
xgb.plot_importance(xgb_model, max_num_features=20, importance_type='gain')
plt.title('XGBoost Feature Importance (Gain)')
plt.tight_layout()
plt.savefig('plots/feature_importance.png')
plt.close()

# Time series cross-validation (demonstration)
print("Performing time series cross-validation for XGBoost...")
tscv = TimeSeriesSplit(n_splits=5)
cv_scores = []

for train_idx, val_idx in tscv.split(X_train_scaled):
    X_train_cv, X_val_cv = X_train_scaled.iloc[train_idx], X_train_scaled.iloc[val_idx]
    y_train_cv, y_val_cv = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    dtrain_cv = xgb.DMatrix(X_train_cv, label=y_train_cv)
    dval_cv = xgb.DMatrix(X_val_cv, label=y_val_cv)
    
    model_cv = xgb.train(
        best_params,
        dtrain_cv,
        num_boost_round=100,
        evals=[(dval_cv, 'val')],
        verbose_eval=False
    )
    
    preds_cv = model_cv.predict(dval_cv)
    rmse = np.sqrt(mean_squared_error(y_val_cv, preds_cv))
    cv_scores.append(rmse)

print(f"XGBoost Cross-Validation RMSE scores: {cv_scores}")
print(f"Mean CV RMSE: {np.mean(cv_scores):.4f}, Std: {np.std(cv_scores):.4f}")

#############################################
# Step 5: Model Saving
#############################################

print("Saving models...")
# Save the XGBoost model and scaler
joblib.dump(xgb_model, 'models/xgb_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

# Save feature list
with open('models/features.pkl', 'wb') as f:
    pickle.dump(features, f)

# Save the Prophet model
with open('models/prophet_model.pkl', 'wb') as f:
    pickle.dump(prophet_model, f)

# Save the ARIMA model
with open('models/arima_model.pkl', 'wb') as f:
    pickle.dump(arima_result, f)

# Save model metadata
model_metadata = {
    'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'data_cutoff_date': cutoff_date.strftime('%Y-%m-%d'),
    'train_samples': len(train),
    'test_samples': len(test),
    'features': features,
    'best_model': metrics_df['RMSE'].idxmin(),
    'best_model_rmse': metrics_df['RMSE'].min(),
    'xgboost_params': best_params
}

with open('models/model_metadata.pkl', 'wb') as f:
    pickle.dump(model_metadata, f)

print("Models saved successfully!")

#############################################
# Step 6: Deployment (Inference API)
#############################################

from flask import Flask, request, jsonify
import logging
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Flask application
app = Flask(__name__)

# Load the best model (using XGBoost as example)
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data
        json_data = request.get_json(force=True)
        logger.info(f"Received prediction request: {json_data}")
        
        if not json_data or 'date' not in json_data:
            return jsonify({"error": "Invalid input. 'date' field is required."}), 400
        
        # Parse date
        try:
            prediction_date = datetime.strptime(json_data['date'], '%Y-%m-%d')
        except ValueError:
            return jsonify({"error": "Invalid date format. Use YYYY-MM-DD."}), 400
        
        # Load model and scaler
        try:
            model = joblib.load('models/xgb_model.pkl')
            scaler = joblib.load('models/scaler.pkl')
            with open('models/features.pkl', 'rb') as f:
                features = pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return jsonify({"error": "Model loading failed."}), 500
        
        # Create features for the input date
        input_data = pd.DataFrame({
            'date': [prediction_date],
            # You need historical data for lag features, this is a simplified example
            # In a real application, you would need to retrieve historical data from a database
            'lag_1': [json_data.get('previous_day_sales', 0)],
            'lag_7': [json_data.get('previous_week_sales', 0)],
            'lag_14': [json_data.get('previous_2week_sales', 0)],
            'lag_30': [json_data.get('previous_month_sales', 0)],
            'rolling_mean_7': [json_data.get('rolling_mean_7', 0)],
            'rolling_std_7': [json_data.get('rolling_std_7', 0)],
            'rolling_min_7': [json_data.get('rolling_min_7', 0)],
            'rolling_max_7': [json_data.get('rolling_max_7', 0)],
            'rolling_mean_14': [json_data.get('rolling_mean_14', 0)],
            'rolling_mean_30': [json_data.get('rolling_mean_30', 0)],
            'pct_change': [json_data.get('pct_change', 0)],
            'pct_change_7': [json_data.get('pct_change_7', 0)]
        })
        
        # Extract date features
        input_data['day_of_week'] = input_data['date'].dt.dayofweek
        input_data['day_of_month'] = input_data['date'].dt.day
        input_data['month'] = input_data['date'].dt.month
        input_data['quarter'] = input_data['date'].dt.quarter
        input_data['year'] = input_data['date'].dt.year
        input_data['is_weekend'] = (input_data['day_of_week'] >= 5).astype(int)
        input_data['is_month_start'] = input_data['date'].dt.is_month_start.astype(int)
        input_data['is_month_end'] = input_data['date'].dt.is_month_end.astype(int)
        
        # Prepare features for prediction
        X = input_data[features]
        X_scaled = scaler.transform(X)
        
        # Make prediction
        dtest = xgb.DMatrix(X_scaled)
        prediction = model.predict(dtest)[0]
        
        # Return prediction result
        result = {
            "date": json_data['date'],
            "predicted_sales": float(prediction),
            "model": "XGBoost"
        }
        
        logger.info(f"Prediction result: {result}")
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

# Endpoint to get model performance metrics
@app.route('/metrics', methods=['GET'])
def get_metrics():
    try:
        metrics_df = pd.read_csv('models/model_metrics.csv', index_col=0)
        return jsonify(metrics_df.to_dict())
    except Exception as e:
        logger.error(f"Error fetching metrics: {str(e)}")
        return jsonify({"error": f"Failed to retrieve metrics: {str(e)}"}), 500

# Run the application
if __name__ == '__main__':
    print("Starting Flask API for sales prediction...")
    app.run(host='0.0.0.0', port=5000, debug=False)
