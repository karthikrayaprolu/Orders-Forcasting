from fastapi import FastAPI, File, UploadFile, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from io import BytesIO
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
import pickle
import joblib
import warnings

# Forecasting libraries
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import lightgbm as lgb

warnings.filterwarnings("ignore")

app = FastAPI()

# Environment-based CORS configuration
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")
CORS_ORIGINS = [
    FRONTEND_URL,
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",  # Alternative frontend port
]

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,  # Specific origins for credentials
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition", "Content-Type"]
)

# Global cache
data_store = {}

# Utility function to save logs in Excel format
def save_forecast_logs(historical_data, forecast_data, forecast_params, model_metrics=None):
    """
    Save historical and forecast data to respective log folders with timestamps in Excel format
    
    Args:
        historical_data: DataFrame containing historical data
        forecast_data: DataFrame containing forecast predictions
        forecast_params: Dictionary containing forecast parameters (time_period, targets, models, etc.)
        model_metrics: Dictionary containing model performance metrics
    """
    try:
        # Create timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Extract parameters for filename
        time_period = forecast_params.get('time_period', 'day')
        targets = forecast_params.get('targets', [])
        models = forecast_params.get('models', {})
        horizon = forecast_params.get('horizon', 0)
        
        # Create descriptive filename components
        targets_str = "-".join(targets) if targets else "unknown"
        models_str = "-".join(set(models.values())) if models else "unknown"
        
        # Define folder paths (relative to main.py)
        historical_folder = "logs/historical"
        forecast_folder = "logs/forecast"
        
        # Create folders if they don't exist
        os.makedirs(historical_folder, exist_ok=True)
        os.makedirs(forecast_folder, exist_ok=True)
          # Create filenames (now with .xlsx extension)
        base_filename = f"{time_period}_{targets_str}_{models_str}_h{horizon}_{timestamp}"
        historical_filename = f"historical_{base_filename}.xlsx"
        forecast_filename = f"forecast_{base_filename}.xlsx"
        
        # Prepare and save historical data
        if not historical_data.empty:
            # Create metadata dataframe
            historical_metadata = pd.DataFrame([{
                "timestamp": timestamp,
                "forecast_type": time_period,
                "targets": ", ".join(targets) if targets else "unknown",
                "models_used": ", ".join(set(models.values())) if models else "unknown",
                "horizon": horizon,
                "aggregation_method": forecast_params.get('aggregation_method', 'mean'),
                "data_points": len(historical_data),
                "date_range_start": historical_data['date'].min() if 'date' in historical_data.columns else None,
                "date_range_end": historical_data['date'].max() if 'date' in historical_data.columns else None
            }])
            
            # Save historical data to Excel
            historical_path = os.path.join(historical_folder, historical_filename)
            with pd.ExcelWriter(historical_path, engine='openpyxl') as writer:
                # Save metadata in first sheet
                historical_metadata.to_excel(writer, sheet_name='Metadata', index=False)
                # Save actual data in second sheet
                historical_data.to_excel(writer, sheet_name='Historical_Data', index=False)
            
            print(f"Historical data saved to: {historical_path}")
        
        # Prepare and save forecast data
        if not forecast_data.empty:
            # Create metadata dataframe
            forecast_metadata = pd.DataFrame([{
                "timestamp": timestamp,
                "forecast_type": time_period,
                "targets": ", ".join(targets) if targets else "unknown", 
                "models_used": ", ".join(set(models.values())) if models else "unknown",
                "horizon": horizon,
                "aggregation_method": forecast_params.get('aggregation_method', 'mean'),
                "forecast_points": len(forecast_data),
                "forecast_date_range_start": forecast_data['date'].min() if 'date' in forecast_data.columns else None,
                "forecast_date_range_end": forecast_data['date'].max() if 'date' in forecast_data.columns else None
            }])
            
            # Save forecast data to Excel
            forecast_path = os.path.join(forecast_folder, forecast_filename)
            with pd.ExcelWriter(forecast_path, engine='openpyxl') as writer:
                # Save metadata in first sheet
                forecast_metadata.to_excel(writer, sheet_name='Metadata', index=False)
                # Save actual data in second sheet
                forecast_data.to_excel(writer, sheet_name='Forecast_Data', index=False)
                
                # Save model metrics if available
                if model_metrics:
                    metrics_df = pd.DataFrame([
                        {
                            'Target_Model': key,
                            'MAE': metrics.get('mae', 0),
                            'MAPE': metrics.get('mape', 0),
                            'RMSE': metrics.get('rmse', 0),
                            'Directional_Accuracy': metrics.get('directional_accuracy', 0),
                            'Data_Points': metrics.get('data_points', 0),
                            'Model_Type': metrics.get('model_type', 'Unknown'),
                            'Target': metrics.get('target', 'Unknown'),
                            'Notes': metrics.get('evaluation_note', '')
                        }
                        for key, metrics in model_metrics.items()
                    ])
                    metrics_df.to_excel(writer, sheet_name='Model_Performance', index=False)
            
            print(f"Forecast data saved to: {forecast_path}")
            
        return {
            "historical_file": historical_filename if not historical_data.empty else None,
            "forecast_file": forecast_filename if not forecast_data.empty else None,
            "timestamp": timestamp
        }
        
    except Exception as e:
        print(f"Error saving forecast logs: {str(e)}")
        return None
        with open(forecast_path, 'w') as f:
            json.dump(forecast_log, f, indent=2, default=str)
        print(f"Forecast data saved to: {forecast_path}")
            
        return {
            "historical_file": historical_filename if not historical_data.empty else None,
            "forecast_file": forecast_filename if not forecast_data.empty else None,
            "timestamp": timestamp
        }
        
    except Exception as e:
        print(f"Error saving forecast logs: {str(e)}")
        return None

# Utils
def clean_datetime(col):
    try:
        # Parse the datetime using the exact format
        datetime_col = pd.to_datetime(col, format='%Y-%m-%d-%H-%M-%S', errors='coerce')
        # Check if we have any valid dates
        if datetime_col.isna().all():
            print("No valid dates found in the column")
            return col
        return datetime_col
    except Exception as e:
        print(f"Error processing dates: {e}")
        print(f"Sample values causing error: {col.head()}")
        return col

# Utility function to save trained models
def save_trained_model(model, model_type, target, model_params, series_info):
    """
    Save trained model weights/parameters to models folder with identifiable names
    
    Args:
        model: The trained model object
        model_type: Type of model (ARIMA, Prophet, LSTM, etc.)
        target: Target variable name
        model_params: Dictionary containing model training parameters
        series_info: Information about the data series used for training
    """
    try:
        # Create timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Extract parameters for filename
        time_period = model_params.get('time_period', 'day')
        aggregation_method = model_params.get('aggregation_method', 'mean')
        data_points = series_info.get('data_points', 0)
        
        # Define models folder path
        models_folder = "models/trained_weights"
        os.makedirs(models_folder, exist_ok=True)
        
        # Create descriptive filename
        model_filename = f"{model_type}_{target}_{time_period}_{aggregation_method}_dp{data_points}_{timestamp}"
          # Save different model types with appropriate methods
        if model_type == 'LSTM':
            # Save Keras/TensorFlow model
            model_path = os.path.join(models_folder, f"{model_filename}.h5")
            model.save(model_path)
            
            # Also save scaler if exists
            if 'scaler' in model_params:
                scaler_path = os.path.join(models_folder, f"{model_filename}_scaler.pkl")
                joblib.dump(model_params['scaler'], scaler_path)
                
        elif model_type == 'RandomForest':
            # Save sklearn models
            model_path = os.path.join(models_folder, f"{model_filename}.pkl")
            joblib.dump(model, model_path)
            
        elif model_type == 'LightGBM':
            # Save LightGBM model
            model_path = os.path.join(models_folder, f"{model_filename}.pkl")
            joblib.dump(model, model_path)
            
        elif model_type == 'Prophet':
            # Save Prophet model
            model_path = os.path.join(models_folder, f"{model_filename}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
                
        elif model_type in ['ARIMA', 'HoltWinters']:
            # Save statsmodels fitted results
            model_path = os.path.join(models_folder, f"{model_filename}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
                
        else:
            # Generic pickle save for other models (EMA, etc.)
            model_path = os.path.join(models_folder, f"{model_filename}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        # Save model metadata
        metadata = {
            "timestamp": timestamp,
            "model_type": model_type,
            "target": target,
            "time_period": time_period,
            "aggregation_method": aggregation_method,
            "data_points": data_points,
            "series_info": series_info,
            "model_parameters": model_params,
            "model_file": os.path.basename(model_path),
            "date_range": series_info.get('date_range', {}),
            "performance_metrics": model_params.get('performance_metrics', {})
        }
        
        metadata_path = os.path.join(models_folder, f"{model_filename}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"✓ Model saved: {model_path}")
        print(f"✓ Metadata saved: {metadata_path}")
        
        return {
            "model_file": os.path.basename(model_path),
            "metadata_file": os.path.basename(metadata_path),
            "timestamp": timestamp,
            "model_type": model_type,
            "target": target
        }
        
    except Exception as e:
        print(f"⚠ Error saving model weights: {str(e)}")
        return None

def handle_nan_values(df):
    """Handle NaN values in the DataFrame."""
    # Fill NaN values with interpolation first
    df_filled = df.interpolate(method='linear')
    
    # For any remaining NaN values (at the start/end), use forward/backward fill
    df_filled = df_filled.fillna(method='ffill').fillna(method='bfill')
    
    # If there are still NaN values (empty series), replace with 0
    df_filled = df_filled.fillna(0)
    
    return df_filled

def aggregate_targets(df, time_period='day', agg_method='mean'):
    try:
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Ensure createdAt is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(df['createdAt']):
            df['createdAt'] = pd.to_datetime(df['createdAt'])
        
        # Get the last valid date from the data
        last_valid_date = df['createdAt'].max()
        
        # Remove any dates beyond the last valid date to prevent future data leakage
        df = df[df['createdAt'] <= last_valid_date]
          # Set up aggregation based on time period
        freq_map = {
            'day': ('D', 'D'),           # Daily at midnight
            'week': ('W-FRI', 'W'),      # Weekly, anchored to Friday (end of week)
            'month': ('M', 'M')          # Monthly, at month end
        }
        freq, period_type = freq_map.get(time_period, ('D', 'D'))
        
        # Set index for aggregation
        df.set_index('createdAt', inplace=True)
        
        # Define aggregation functions based on available columns
        agg_func = {
            'transformOrdNo': 'count',  # Always count unique orders
            'quantity': agg_method,     # Use selected method
            'woNumber': 'count'         # Always count work orders
        }
        
        # Add workers_needed aggregation if column exists
        if 'workers_needed' in df.columns:
            agg_func['workers_needed'] = agg_method
        
        # Add product-related aggregations if product column exists
        if 'product' in df.columns:
            # Products target: count of unique products per time period
            # This measures product variety/diversity in the business operations
            agg_func['product'] = 'nunique'  # Count unique products - this will be renamed later
            
            # Note: Alternative aggregation options for Products target:
            # 'count': Total product instances (volume) - typically 90-100 per day
            # 'nunique': Unique product types (variety) - typically 76-96 per day
            # Both metrics are valid but measure different business aspects
        
        # Perform aggregation
        result_df = df.resample(freq).agg(agg_func)
        
        # Rename product column to Products if it exists
        if 'product' in result_df.columns:
            result_df = result_df.rename(columns={'product': 'Products'})
        
        # Handle missing values
        result_df = result_df.fillna(method='ffill').fillna(0)
        
        # Reset index to get createdAt as column
        result_df = result_df.reset_index()
        
        return result_df
        
    except Exception as e:
        print(f"Aggregation error: {str(e)}")
        raise e

def forecast_arima(series, steps):
    """ARIMA model with proper training/testing split and parameter optimization"""
    try:
        # Ensure we have enough data
        min_samples = 30  # Minimum 30 data points needed
        if len(series) < min_samples:
            raise ValueError(f"Need at least {min_samples} data points, got {len(series)}")
        
        # Convert series to float and handle any missing values
        series = series.astype(float).fillna(method='ffill').fillna(method='bfill')
        
        # Determine seasonality
        freq = pd.infer_freq(series.index)
        seasonal_period = 12 if freq == 'M' else (52 if freq == 'W' else 7)
            
        # Split data into train/test
        train_size = int(len(series) * 0.8)
        train = series[:train_size]
        
        # Find best parameters
        best_aic = float('inf')
        best_params = None
        
        # Simplified parameter grid for faster processing
        for p in range(0, 3):
            for d in range(0, 2):
                for q in range(0, 3):
                    try:
                        # Handle non-stationary data
                        model = ARIMA(train, order=(p, d, q))
                        results = model.fit()
                        aic = results.aic
                        
                        if aic < best_aic:
                            best_aic = aic
                            best_params = (p, d, q)
                    except Exception as e:
                        print(f"ARIMA parameter search failed for order ({p},{d},{q}): {str(e)}")
                        continue
        
        if best_params is None:
            print("Using fallback ARIMA parameters")
            best_params = (1, 1, 1)
        
        print(f"Selected ARIMA parameters: {best_params}")
        
        # Train final model on full dataset
        final_model = ARIMA(series, order=best_params)
        final_results = final_model.fit()
        
        # Generate forecasts
        forecast = final_results.forecast(steps=steps)
        
        # Ensure non-negative values for counts
        forecast = np.maximum(forecast, 0)
        
        # Calculate performance metrics on validation data
        validation_metrics = None
        try:
            if len(series) > 20:  # Only if we have enough data
                # Use last 20% of data for validation
                val_size = max(5, int(len(series) * 0.2))
                train_data = series[:-val_size]
                val_actual = series[-val_size:]
                
                # Make predictions on validation set
                val_model = ARIMA(train_data, order=best_params)
                val_results = val_model.fit()
                val_forecast = val_results.forecast(steps=val_size)
                val_forecast = np.maximum(val_forecast, 0)
                
                validation_metrics = calculate_forecast_metrics(val_actual.values, val_forecast)
                validation_metrics['validation_period'] = f"Last {val_size} points"
        except Exception as val_error:
            print(f"⚠ Warning: Could not calculate ARIMA validation metrics: {str(val_error)}")
        
        # Save trained model (optional)
        try:
            model_params = {
                'arima_order': best_params,
                'seasonal_period': seasonal_period,
                'time_period': getattr(series, '_time_period', 'unknown'),
                'aggregation_method': getattr(series, '_agg_method', 'unknown'),
                'aic_score': best_aic,
                'performance_metrics': validation_metrics
            }
            series_info = {
                'data_points': len(series),
                'date_range': {
                    'start': str(series.index.min()) if len(series) > 0 else None,
                    'end': str(series.index.max()) if len(series) > 0 else None
                }
            }
            target_name = getattr(series, 'name', 'unknown_target')
            save_trained_model(final_results, 'ARIMA', target_name, model_params, series_info)
        except Exception as save_error:
            print(f"⚠ Warning: Could not save ARIMA model: {str(save_error)}")
        
        return forecast.tolist()
    except Exception as e:
        print(f"ARIMA Error for series: {str(e)}")
        print(f"Series head: {series.head()}")
        print(f"Series dtype: {series.dtype}")
        print(f"Series contains null: {series.isnull().any()}")
        raise e

def forecast_prophet(series, steps):
    """Prophet model with proper seasonality and validation"""
    try:
        # Prepare data
        df = series.reset_index()
        df.columns = ['ds', 'y']
        
        # Add extra seasonality features
        df['month'] = df['ds'].dt.month
        df['quarter'] = df['ds'].dt.quarter
        df['year'] = df['ds'].dt.year
        
        # Calculate trend indicators
        df['trend'] = np.arange(len(df))
        
        # Calculate seasonality parameters
        data_length = (df['ds'].max() - df['ds'].min()).days
        
        # Determine seasonality mode based on target characteristics
        # For count-based targets with relatively stable ranges, use additive
        # For highly variable targets like quantities, multiplicative may be better
        target_name = getattr(series, 'name', 'unknown_target')
        if target_name in ['Products', 'transformOrdNo', 'woNumber', 'workers_needed']:
            seasonality_mode = 'additive'  # Count-based targets with stable ranges
        else:
            seasonality_mode = 'multiplicative'  # Variable targets like quantities
        
        # Configure prophet based on data characteristics
        model = Prophet(
            yearly_seasonality='auto' if data_length > 365 else False,
            weekly_seasonality='auto' if data_length > 14 else False,
            daily_seasonality='auto' if data_length > 7 else False,
            seasonality_mode=seasonality_mode
        )
        
        # Add monthly seasonality if enough data
        if data_length > 60:
            model.add_seasonality(
                name='monthly',
                period=30.5,
                fourier_order=5
            )
        
        # Cross-validation
        model.fit(df)
          # Make future dataframe
        future = model.make_future_dataframe(periods=steps)
        forecast = model.predict(future)
        # Get forecasts and ensure non-negative values
        forecasted_values = forecast['yhat'].tail(steps)
        forecasted_values = forecasted_values.clip(lower=0)  # Ensure no negative values
        
        # Calculate performance metrics on validation data
        validation_metrics = None
        try:
            if len(df) > 20:  # Only if we have enough data
                # Use cross-validation approach for Prophet
                val_size = max(5, int(len(df) * 0.2))
                train_df = df[:-val_size].copy()
                val_actual = df[-val_size:]['y'].values
                
                # Train model on subset and predict validation period
                val_model = Prophet(
                    yearly_seasonality='auto' if data_length > 365 else False,
                    weekly_seasonality='auto' if data_length > 14 else False,
                    daily_seasonality='auto' if data_length > 7 else False,
                    seasonality_mode=seasonality_mode  # Use the same adaptive seasonality mode
                )
                val_model.fit(train_df)
                val_future = val_model.make_future_dataframe(periods=val_size)
                val_forecast = val_model.predict(val_future)
                val_predictions = val_forecast['yhat'].tail(val_size).clip(lower=0).values
                
                validation_metrics = calculate_forecast_metrics(val_actual, val_predictions)
                validation_metrics['validation_period'] = f"Last {val_size} points"
        except Exception as val_error:
            print(f"⚠ Warning: Could not calculate Prophet validation metrics: {str(val_error)}")
        
        # Save trained model (optional)
        try:
            model_params = {
                'yearly_seasonality': 'auto',
                'weekly_seasonality': 'auto', 
                'daily_seasonality': 'auto',
                'seasonality_mode': seasonality_mode,  # Use the adaptive mode
                'time_period': getattr(series, '_time_period', 'unknown'),
                'aggregation_method': getattr(series, '_agg_method', 'unknown'),
                'data_length_days': data_length,
                'performance_metrics': validation_metrics
            }
            series_info = {
                'data_points': len(series),
                'date_range': {
                    'start': str(series.index.min()) if len(series) > 0 else None,
                    'end': str(series.index.max()) if len(series) > 0 else None
                }
            }
            target_name = getattr(series, 'name', 'unknown_target')
            save_trained_model(model, 'Prophet', target_name, model_params, series_info)
        except Exception as save_error:
            print(f"⚠ Warning: Could not save Prophet model: {str(save_error)}")
        
        return forecasted_values.tolist()
    except Exception as e:
        print(f"Prophet Error: {str(e)}")
        raise e

def forecast_rf(series, steps):
    """Random Forest with proper feature engineering and validation"""
    try:
        # Create DataFrame with date features
        df = pd.DataFrame({'y': series})
        df.index = pd.to_datetime(df.index)
        
        # Time-based features
        df['dayofweek'] = df.index.dayofweek
        df['month'] = df.index.month
        df['year'] = df.index.year
        df['quarter'] = df.index.quarter
        df['time_idx'] = np.arange(len(df))  # Add time index for trend capture
        
        # Add cyclical encoding for better seasonality capture
        df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
        df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek']/7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek']/7)
        
        # Rolling statistics (multiple windows)
        for window in [7, 14, 30]:
            df[f'rolling_mean_{window}d'] = df['y'].rolling(window=window).mean()
            df[f'rolling_std_{window}d'] = df['y'].rolling(window=window).std()
        
        # Lag features (dynamic based on data size)
        max_lags = min(30, len(df) // 3)  # Use up to 30 lags or 1/3 of data length
        for lag in range(1, max_lags + 1):
            df[f'lag_{lag}'] = df['y'].shift(lag)
        
        # Drop rows with NaN from lag creation
        df.dropna(inplace=True)
        
        # Split features and target
        X = df.drop('y', axis=1)
        y = df['y']
          # Train model with improved hyperparameters
        model = RandomForestRegressor(
            n_estimators=200,           # Increased from 100
            max_depth=15,               # Increased from 10
            min_samples_split=3,        # Decreased from 5 for better splits
            min_samples_leaf=2,         # Added for better generalization
            max_features='sqrt',        # Added for better feature selection
            bootstrap=True,             # Ensure bootstrap sampling
            oob_score=True,             # Out-of-bag scoring for validation
            random_state=42,
            n_jobs=-1                   # Use all cores for faster training
        )
        model.fit(X, y)
        
        # Generate predictions
        last_row = X.iloc[-1].copy()
        predictions = []
        
        for i in range(steps):
            # Update time features
            next_date = df.index[-1] + pd.Timedelta(days=i+1)
            last_row['dayofweek'] = next_date.dayofweek
            last_row['month'] = next_date.month
            last_row['year'] = next_date.year
            last_row['quarter'] = next_date.quarter
            
            # Predict and update lags
            pred = model.predict([last_row])[0]
            predictions.append(pred)
            
            # Update lag features
            for lag in range(max_lags, 0, -1):
                if lag == 1:
                    last_row[f'lag_{lag}'] = pred
                else:
                    last_row[f'lag_{lag}'] = last_row[f'lag_{lag-1}']
        
        # Calculate performance metrics on validation data
        validation_metrics = None
        try:
            if len(X) > 20:  # Only if we have enough data
                # Use last 20% for validation
                val_size = max(5, int(len(X) * 0.2))
                X_val_test = X[-val_size:]
                y_val_test = y[-val_size:]
                
                # Train model on subset (excluding validation data)
                X_train_subset = X[:-val_size]
                y_train_subset = y[:-val_size]
                
                val_model = RandomForestRegressor(
                    n_estimators=100,  # Reduced for faster validation
                    max_depth=10,
                    min_samples_split=3,
                    min_samples_leaf=2,
                    max_features='sqrt',
                    bootstrap=True,
                    random_state=42,
                    n_jobs=-1
                )
                val_model.fit(X_train_subset, y_train_subset)
                val_predictions = val_model.predict(X_val_test)
                
                validation_metrics = calculate_forecast_metrics(y_val_test.values, val_predictions)
                validation_metrics['validation_period'] = f"Last {val_size} points"
        except Exception as val_error:
            print(f"⚠ Warning: Could not calculate RandomForest validation metrics: {str(val_error)}")
        
        # Save trained model (optional)
        try:
            model_params = {
                'n_estimators': 200,
                'max_depth': 15,
                'min_samples_split': 3,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'bootstrap': True,
                'oob_score': True,
                'oob_score_value': model.oob_score_ if hasattr(model, 'oob_score_') else None,
                'time_period': getattr(series, '_time_period', 'unknown'),
                'aggregation_method': getattr(series, '_agg_method', 'unknown'),
                'max_lags': max_lags,
                'performance_metrics': validation_metrics
            }
            series_info = {
                'data_points': len(series),
                'date_range': {
                    'start': str(series.index.min()) if len(series) > 0 else None,
                    'end': str(series.index.max()) if len(series) > 0 else None
                }
            }
            target_name = getattr(series, 'name', 'unknown_target')
            save_trained_model(model, 'RandomForest', target_name, model_params, series_info)
        except Exception as save_error:
            print(f"⚠ Warning: Could not save RandomForest model: {str(save_error)}")
        
        return predictions
    except Exception as e:
        print(f"Random Forest Error: {str(e)}")
        raise e

def forecast_lstm(series, steps):
    try:
        # Check if enough data points
        if len(series) < 30:
            raise ValueError("Need at least 30 data points for LSTM prediction")
            
        # Prepare data
        values = series.values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0.1, 0.9))  # Avoid extreme values
        scaled = scaler.fit_transform(values)
        
        # Add time features
        dates = pd.DataFrame(index=series.index)
        dates['month'] = series.index.month
        dates['quarter'] = series.index.quarter
        dates['year'] = series.index.year
        dates['dayofweek'] = series.index.dayofweek
        
        # Scale additional features
        feature_scaler = MinMaxScaler(feature_range=(0.1, 0.9))
        date_features = feature_scaler.fit_transform(dates)        # Create sequences with improved length
        sequence_length = min(15, len(scaled) // 4)  # Dynamic sequence length
        X = []
        y = []
        for i in range(sequence_length, len(scaled)):
            X.append(scaled[i-sequence_length:i])
            y.append(scaled[i])
        X = np.array(X)
        y = np.array(y)

        # Reshape for LSTM [samples, time steps, features]
        X = X.reshape((X.shape[0], X.shape[1], 1))

        # Split data for better training
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Build improved LSTM model
        model = Sequential([
            LSTM(128, activation='tanh', input_shape=(sequence_length, 1), 
                 return_sequences=True, dropout=0.2, recurrent_dropout=0.1),
            LSTM(64, activation='tanh', return_sequences=True, 
                 dropout=0.2, recurrent_dropout=0.1),
            LSTM(32, activation='tanh', dropout=0.2),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        # Improved training with validation
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if len(X_val) > 0 else None,
            epochs=100,
            batch_size=min(32, len(X_train)//3),
            verbose=0,
            callbacks=[
                EarlyStopping(monitor='val_loss' if len(X_val) > 0 else 'loss', 
                            patience=15, restore_best_weights=True),
            ]
        )

        # Generate predictions
        input_seq = scaled[-sequence_length:].reshape(1, sequence_length, 1)
        preds = []
        
        for _ in range(steps):
            pred = model.predict(input_seq, verbose=0)[0][0]
            preds.append(pred)
            # Update input sequence for next prediction
            input_seq = np.append(input_seq[:,1:,:], [[[pred]]], axis=1)
        
        # Inverse transform predictions
        forecast = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten().tolist()
        
        # Ensure non-negative values
        forecast = [max(0, x) for x in forecast]
        
        # Calculate performance metrics on validation data
        validation_metrics = None
        try:
            if len(X_val) > 0 and 'history' in locals():
                # Get validation loss from training history
                val_loss = history.history.get('val_loss', [])
                val_mae = history.history.get('val_mae', [])
                
                if val_loss and val_mae:
                    # Create validation metrics from training metrics
                    validation_metrics = {
                        'mae': round(float(val_mae[-1]), 4),
                        'rmse': round(float(np.sqrt(val_loss[-1])), 4),
                        'mape': 0.0,  # Not directly available from Keras
                        'directional_accuracy': 0.0,  # Would need actual vs predicted
                        'validation_loss': round(float(val_loss[-1]), 6),
                        'validation_mae': round(float(val_mae[-1]), 4),
                        'data_points': len(X_val),
                        'validation_period': f"Last {len(X_val)} sequences"
                    }
        except Exception as val_error:
            print(f"⚠ Warning: Could not calculate LSTM validation metrics: {str(val_error)}")
        
        # Save trained model (optional - can be enabled/disabled)
        try:
            model_params = {
                'scaler': scaler,
                'sequence_length': sequence_length,
                'architecture': '128-64-32-32-16-1',
                'dropout_rate': 0.2,
                'recurrent_dropout': 0.1,
                'learning_rate': 0.001,
                'validation_split': 0.8,
                'time_period': getattr(series, '_time_period', 'unknown'),
                'aggregation_method': getattr(series, '_agg_method', 'unknown'),
                'epochs': 100,
                'batch_size': min(32, len(X_train)//3) if 'X_train' in locals() else 32,
                'final_loss': history.history['loss'][-1] if 'history' in locals() else None,
                'final_val_loss': history.history['val_loss'][-1] if 'history' in locals() and 'val_loss' in history.history else None,
                'performance_metrics': validation_metrics
            }
            series_info = {
                'data_points': len(series),
                'date_range': {
                    'start': str(series.index.min()) if len(series) > 0 else None,
                    'end': str(series.index.max()) if len(series) > 0 else None
                }
            }
            target_name = getattr(series, 'name', 'unknown_target')
            save_trained_model(model, 'LSTM', target_name, model_params, series_info)
        except Exception as save_error:
            print(f"⚠ Warning: Could not save LSTM model: {str(save_error)}")
        
        return forecast
    except Exception as e:
        print(f"LSTM Error: {str(e)}")
        print(f"Series shape: {series.shape if hasattr(series, 'shape') else 'no shape'}")
        print(f"Series head: {series.head()}")
        raise e

def forecast_ema(series, steps):
    """Enhanced EMA model with proper error handling and validation"""
    try:
        # Ensure we have enough data
        min_samples = 10
        if len(series) < min_samples:
            raise ValueError(f"Need at least {min_samples} data points, got {len(series)}")
        
        # Convert series to float and handle missing values
        series = series.astype(float).fillna(method='ffill').fillna(method='bfill')
        
        # Calculate optimal span based on data characteristics
        volatility = series.std() / series.mean()
        # Adjust span based on volatility (more volatile = smaller span)
        span = max(5, min(30, int(1/volatility))) if volatility > 0 else 10
        
        # Calculate EMA with optimized span
        ema = series.ewm(span=span, adjust=False).mean()
        
        # Generate forecast
        last_value = ema.iloc[-1]
        forecast = [last_value] * steps
        
        # Add trend adjustment
        if len(series) >= 2:
            trend = (ema.iloc[-1] - ema.iloc[-2])
            forecast = [max(0, last_value + trend * i) for i in range(steps)]
        
        return forecast
        
    except Exception as e:
        print(f"EMA Error: {str(e)}")
        print(f"Series head: {series.head()}")
        print(f"Series dtype: {series.dtype}")
        print(f"Series contains null: {series.isnull().any()}")
        raise e

def forecast_holtwinters(series, steps):
    """Holt-Winters model with proper error handling and seasonality detection"""
    try:
        # Ensure we have enough data
        min_samples = 14  # Need at least 2 weeks of data
        if len(series) < min_samples:
            raise ValueError(f"Need at least {min_samples} data points, got {len(series)}")
        
        # Convert series to float and handle missing values
        series = series.astype(float).fillna(method='ffill').fillna(method='bfill')
        
        # Determine seasonal period based on data frequency
        freq = pd.infer_freq(series.index)
        seasonal_periods = 7  # Default to weekly seasonality
        if freq == 'M':
            seasonal_periods = 12  # Monthly data
        elif freq == 'Q':
            seasonal_periods = 4   # Quarterly data
        elif freq == 'W':
            seasonal_periods = 52  # Weekly data
        
        # Initialize model with automatic optimization
        model = ExponentialSmoothing(
            series,
            seasonal_periods=seasonal_periods,
            trend='add',
            seasonal='add',
            initialization_method='estimated'
        )
        
        # Fit model with optimized parameters
        try:
            fit = model.fit(optimized=True)
        except:
            # Fallback to simpler model if optimization fails
            model = ExponentialSmoothing(
                series,
                trend='add',
                seasonal=None,
                initialization_method='estimated'
            )
            fit = model.fit()
        
        # Generate forecast
        forecast = fit.forecast(steps)
          # Ensure non-negative values
        forecast = forecast.clip(lower=0)
        
        # Save trained model (optional)
        try:
            model_params = {
                'seasonal_periods': seasonal_periods,
                'trend': 'add',
                'seasonal': 'add' if hasattr(model, 'seasonal') else None,
                'time_period': getattr(series, '_time_period', 'unknown'),
                'aggregation_method': getattr(series, '_agg_method', 'unknown'),
                'initialization_method': 'estimated'
            }
            series_info = {
                'data_points': len(series),
                'date_range': {
                    'start': str(series.index.min()) if len(series) > 0 else None,
                    'end': str(series.index.max()) if len(series) > 0 else None
                }
            }
            target_name = getattr(series, 'name', 'unknown_target')
            save_trained_model(fit, 'HoltWinters', target_name, model_params, series_info)
        except Exception as save_error:
            print(f"⚠ Warning: Could not save HoltWinters model: {str(save_error)}")
        
        return forecast.tolist()
        
    except Exception as e:
        print(f"Holt-Winters Error: {str(e)}")
        print(f"Series head: {series.head()}")
        print(f"Series dtype: {series.dtype}")
        print(f"Series contains null: {series.isnull().any()}")
        raise e

def forecast_lightgbm(series, steps):
    """LightGBM model with horizon-aware hyperparameters and advanced feature engineering"""
    try:
        # Flexible minimum data requirement like Random Forest and LSTM
        min_samples = 20  # Reduced from 50 to match other models
        if len(series) < min_samples:
            # Fallback to simple forecast like EMA if not enough data
            print(f"LightGBM: Not enough data ({len(series)} < {min_samples}), using EMA fallback")
            return forecast_ema(series, steps)
        
        # Convert series to float and handle missing values
        series = series.astype(float).fillna(method='ffill').fillna(method='bfill')
        
        # Create DataFrame with advanced features
        df = pd.DataFrame({'y': series})
        df.index = pd.to_datetime(df.index)
        
        # Time-based features
        df['dayofweek'] = df.index.dayofweek
        df['month'] = df.index.month
        df['year'] = df.index.year
        df['quarter'] = df.index.quarter
        df['day_of_month'] = df.index.day
        df['week_of_year'] = df.index.isocalendar().week
        df['time_idx'] = np.arange(len(df))
        
        # Cyclical encoding for better seasonality capture
        df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
        df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek']/7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek']/7)
        df['quarter_sin'] = np.sin(2 * np.pi * df['quarter']/4)
        df['quarter_cos'] = np.cos(2 * np.pi * df['quarter']/4)
          # Rolling statistics with multiple windows (adaptive to data size)
        available_windows = []
        for window in [3, 7, 14, 30]:
            if len(df) > window * 2:  # Need at least 2x window size for meaningful rolling stats
                available_windows.append(window)
                df[f'rolling_mean_{window}d'] = df['y'].rolling(window=window, min_periods=1).mean()
                df[f'rolling_std_{window}d'] = df['y'].rolling(window=window, min_periods=1).std()
                df[f'rolling_min_{window}d'] = df['y'].rolling(window=window, min_periods=1).min()
                df[f'rolling_max_{window}d'] = df['y'].rolling(window=window, min_periods=1).max()
        
        # If no rolling windows are available, add basic statistics
        if not available_windows:
            df['expanding_mean'] = df['y'].expanding(min_periods=1).mean()
            df['expanding_std'] = df['y'].expanding(min_periods=1).std().fillna(0)
        
        # Lag features (adaptive like Random Forest - dynamic based on data size)
        max_lags = min(15, len(df) // 3, steps * 2)  # Reduced and more flexible
        if max_lags < 1:
            max_lags = 1  # At least 1 lag feature
        for lag in range(1, max_lags + 1):
            df[f'lag_{lag}'] = df['y'].shift(lag)
        
        # Statistical features
        df['expanding_mean'] = df['y'].expanding(min_periods=1).mean()
        df['expanding_std'] = df['y'].expanding(min_periods=1).std()
        
        # Trend features
        df['diff_1'] = df['y'].diff(1)
        df['diff_7'] = df['y'].diff(7) if len(df) > 7 else 0
        df['pct_change'] = df['y'].pct_change()
          # Drop rows with NaN from lag creation
        df.dropna(inplace=True)
        
        # More flexible minimum check like Random Forest
        if len(df) < 10:
            print(f"LightGBM: Very small dataset after feature engineering ({len(df)} points), using simple EMA fallback")
            return forecast_ema(series, steps)
        
        # Split features and target
        X = df.drop('y', axis=1)
        y = df['y']
        
        # Determine time period for horizon-aware hyperparameters
        time_period = getattr(series, '_time_period', 'day')
          # Horizon-aware hyperparameters (optimized for smaller datasets)
        if time_period == 'day':
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': min(15, len(df) // 2),  # Adaptive to dataset size
                'learning_rate': 0.1,  # Slightly higher for faster convergence
                'feature_fraction': 0.9,
                'bagging_fraction': 0.9,
                'bagging_freq': 3,
                'min_child_samples': max(5, len(df) // 10),  # Adaptive
                'reg_alpha': 0.05,
                'reg_lambda': 0.05,
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1
            }
            n_estimators = min(100, len(df) * 2)  # Adaptive
        elif time_period == 'week':
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': min(10, len(df) // 3),
                'learning_rate': 0.12,
                'feature_fraction': 0.95,
                'bagging_fraction': 0.95,
                'bagging_freq': 2,
                'min_child_samples': max(3, len(df) // 15),
                'reg_alpha': 0.03,
                'reg_lambda': 0.03,
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1
            }
            n_estimators = min(80, len(df) * 2)
        else:  # month
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': min(7, len(df) // 4),
                'learning_rate': 0.15,
                'feature_fraction': 1.0,
                'bagging_fraction': 1.0,
                'bagging_freq': 1,
                'min_child_samples': max(2, len(df) // 20),
                'reg_alpha': 0.01,
                'reg_lambda': 0.01,
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1
            }
            n_estimators = min(60, len(df) * 2)
          # Split data for validation (more flexible like LSTM)
        if len(X) > 15:  # Only split if we have enough data
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            # Create LightGBM datasets
            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            # Train model with early stopping
            model = lgb.train(
                params,
                train_data,
                valid_sets=[valid_data],
                num_boost_round=n_estimators,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=10),  # Reduced stopping rounds
                    lgb.log_evaluation(0)  # Silent training
                ]
            )
        else:
            # For very small datasets, train without validation split
            train_data = lgb.Dataset(X, label=y)
            model = lgb.train(
                params,
                train_data,
                num_boost_round=min(n_estimators, 30),  # Limit iterations for small data
                callbacks=[lgb.log_evaluation(0)]
            )
        
        # Generate predictions
        last_row = X.iloc[-1].copy()
        predictions = []
        
        for i in range(steps):
            # Update time features for future dates
            next_date = df.index[-1] + pd.Timedelta(days=i+1)
            last_row['dayofweek'] = next_date.dayofweek
            last_row['month'] = next_date.month
            last_row['year'] = next_date.year
            last_row['quarter'] = next_date.quarter
            last_row['day_of_month'] = next_date.day
            last_row['week_of_year'] = next_date.isocalendar().week
            last_row['time_idx'] = last_row['time_idx'] + 1
            
            # Update cyclical features
            last_row['month_sin'] = np.sin(2 * np.pi * last_row['month']/12)
            last_row['month_cos'] = np.cos(2 * np.pi * last_row['month']/12)
            last_row['dayofweek_sin'] = np.sin(2 * np.pi * last_row['dayofweek']/7)
            last_row['dayofweek_cos'] = np.cos(2 * np.pi * last_row['dayofweek']/7)
            last_row['quarter_sin'] = np.sin(2 * np.pi * last_row['quarter']/4)
            last_row['quarter_cos'] = np.cos(2 * np.pi * last_row['quarter']/4)
            
            # Predict
            pred = model.predict([last_row], num_iteration=model.best_iteration)[0]
            pred = max(0, pred)  # Ensure non-negative
            predictions.append(pred)
            
            # Update lag features for next iteration
            for lag in range(max_lags, 0, -1):
                if lag == 1:
                    last_row[f'lag_{lag}'] = pred
                else:
                    last_row[f'lag_{lag}'] = last_row[f'lag_{lag-1}']
        
        # Save trained model (optional)
        try:
            # Get feature importance
            feature_importance = dict(zip(X.columns, model.feature_importance()))
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Calculate performance metrics on validation data
            validation_metrics = None
            try:
                if len(X) > 15:  # Only if we used validation split
                    # Get validation predictions
                    val_predictions = model.predict(X_val, num_iteration=model.best_iteration)
                    validation_metrics = calculate_forecast_metrics(y_val.values, val_predictions)
                    validation_metrics['validation_period'] = f"Last {len(X_val)} points"
                    validation_metrics['lgb_train_rmse'] = model.best_score.get('valid_0', {}).get('rmse', None) if hasattr(model, 'best_score') else None
            except Exception as val_error:
                print(f"⚠ Warning: Could not calculate LightGBM validation metrics: {str(val_error)}")
            
            model_params = {
                'time_period': time_period,
                'aggregation_method': getattr(series, '_agg_method', 'unknown'),
                'lgb_params': params,
                'n_estimators': n_estimators,
                'best_iteration': model.best_iteration,
                'max_lags': max_lags,
                'feature_count': len(X.columns),
                'top_features': top_features,
                'performance_metrics': validation_metrics
            }
            series_info = {
                'data_points': len(series),
                'date_range': {
                    'start': str(series.index.min()) if len(series) > 0 else None,
                    'end': str(series.index.max()) if len(series) > 0 else None
                }
            }
            target_name = getattr(series, 'name', 'unknown_target')
            save_trained_model(model, 'LightGBM', target_name, model_params, series_info)
        except Exception as save_error:
            print(f"⚠ Warning: Could not save LightGBM model: {str(save_error)}")
        
        return predictions
        
    except Exception as e:
        print(f"LightGBM Error: {str(e)}")
        print(f"Series shape: {series.shape if hasattr(series, 'shape') else 'no shape'}")
        print(f"Series head: {series.head()}")
        raise e

# Forecasting Metrics Functions
def calculate_forecast_metrics(actual, predicted):
    """
    Calculate the 4 most important forecasting metrics:
    1. MAE (Mean Absolute Error)
    2. MAPE (Mean Absolute Percentage Error) 
    3. RMSE (Root Mean Square Error)
    4. Directional Accuracy
    """
    try:
        actual = np.array(actual)
        predicted = np.array(predicted)
        
        # Ensure arrays have the same length
        min_len = min(len(actual), len(predicted))
        actual = actual[:min_len]
        predicted = predicted[:min_len]
        
        # Remove any NaN or infinite values
        mask = ~(np.isnan(actual) | np.isnan(predicted) | np.isinf(actual) | np.isinf(predicted))
        actual = actual[mask]
        predicted = predicted[mask]
        
        if len(actual) == 0:
            return {
                'mae': 0.0,
                'mape': 0.0,
                'rmse': 0.0,
                'directional_accuracy': 0.0,
                'data_points': 0
            }
        
        # 1. MAE (Mean Absolute Error)
        mae = np.mean(np.abs(actual - predicted))
        
        # 2. MAPE (Mean Absolute Percentage Error)
        # Avoid division by zero by using a small epsilon
        epsilon = 1e-8
        mape = np.mean(np.abs((actual - predicted) / (actual + epsilon))) * 100
        
        # 3. RMSE (Root Mean Square Error)
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        
        # 4. Directional Accuracy
        if len(actual) > 1:
            # Calculate direction changes
            actual_direction = np.diff(actual)
            predicted_direction = np.diff(predicted)
            
            # Same direction: both positive, both negative, or both zero
            same_direction = (
                (actual_direction >= 0) & (predicted_direction >= 0) |
                (actual_direction < 0) & (predicted_direction < 0)
            )
            
            directional_accuracy = np.mean(same_direction) * 100
        else:
            directional_accuracy = 100.0  # Perfect if only one point
        
        return {
            'mae': round(float(mae), 4),
            'mape': round(float(mape), 2),
            'rmse': round(float(rmse), 4),
            'directional_accuracy': round(float(directional_accuracy), 2),
            'data_points': len(actual)
        }
        
    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        return {
            'mae': 0.0,
            'mape': 0.0,
            'rmse': 0.0,
            'directional_accuracy': 0.0,
            'data_points': 0,
            'error': str(e)
        }

def validate_model_performance(series, model_predictions, model_type):
    """
    Validate model performance using train/test split and calculate metrics
    """
    try:
        if len(series) < 10:
            return None
        
        # Use 80% for training, 20% for testing
        split_idx = int(len(series) * 0.8)
        
        # Get actual test values
        actual_test = series[split_idx:].values
        
        # Get predicted values for the test period
        if len(model_predictions) >= len(actual_test):
            predicted_test = model_predictions[:len(actual_test)]
        else:
            # If we don't have enough predictions, return None
            return None
        
        # Calculate metrics
        metrics = calculate_forecast_metrics(actual_test, predicted_test)
        metrics['model_type'] = model_type
        metrics['test_period_length'] = len(actual_test)
        
        return metrics
        
    except Exception as e:
        print(f"Error in model validation: {str(e)}")
        return None

# Routes
@app.post("/upload")
async def upload_files(
    header: UploadFile = File(...),
    items: UploadFile = File(...),
    workstation: UploadFile = File(...)
):
    try:
        # Read CSV files with explicit parsing for dates
        df_header = pd.read_csv(header.file)
        df_items = pd.read_csv(items.file)
        df_work = pd.read_csv(workstation.file)
        
        # Print debug information
        print("Header columns:", df_header.columns.tolist())
        print("Items columns:", df_items.columns.tolist())
        print("Workstation columns:", df_work.columns.tolist())
        print("Header data types:", df_header.dtypes)

        # Ensure createdAt column exists
        if 'createdAt' not in df_header.columns:
            date_columns = [col for col in df_header.columns if 'date' in col.lower()]
            if date_columns:
                df_header = df_header.rename(columns={date_columns[0]: 'createdAt'})
            else:
                return JSONResponse(
                    status_code=400, 
                    content={"error": "No date column found in header file"}
                )

        df_header['createdAt'] = clean_datetime(df_header['createdAt'])
        df_items['quantity'] = df_items['quantity'].astype(float)
        
        # Handle new workstation data structure with millisecond timestamps
        # Convert millisecond timestamps to datetime
        if 'toStartedAt' in df_work.columns and 'qaEndedAt' in df_work.columns:
            df_work['toStartedAt'] = pd.to_datetime(df_work['toStartedAt'], unit='ms')
            df_work['qaEndedAt'] = pd.to_datetime(df_work['qaEndedAt'], unit='ms')
            
            # Calculate time_taken_to_complete_order in minutes from millisecond timestamps
            df_work['time_taken_minutes'] = (df_work['qaEndedAt'] - df_work['toStartedAt']).dt.total_seconds() / 60
            
            # Calculate workers_needed based on work complexity and time
            # This is a business logic calculation - you may need to adjust based on your requirements
            df_work['workers_needed'] = df_work.apply(lambda row: 
                max(1, min(10, round(row['time_taken_minutes'] / 120, 1))), axis=1)  # 1-10 workers based on 2-hour chunks
        elif 'workers_needed' in df_work.columns:
            # Handle legacy data format
            df_work['workers_needed'] = df_work['workers_needed'].astype(float)
        else:
            return JSONResponse(
                status_code=400,
                content={"error": "Workstation file must contain either 'workers_needed' column or 'toStartedAt'/'qaEndedAt' timestamp columns"}
            )

        df_merged = pd.merge(df_header, df_items, left_on='transformOrdNo', right_on='transferOrdNo', how='inner')
        df_merged = pd.merge(df_merged, df_work, on=['transferOrdNo', 'woNumber'], how='inner')

        # Print debug information for merged dataframe
        print("Merged columns:", df_merged.columns.tolist())
        print("Sample merged data:", df_merged.head())

        data_store['merged'] = df_merged
        return {"message": "Files uploaded and merged successfully."}
    except Exception as e:
        print(f"Error during file upload: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

class ForecastRequest(BaseModel):
    targets: List[str]
    models: Dict[str, str]
    horizon: int
    time_period: str = 'day'  # day, week, month
    aggregation_method: str = 'mean'  # mean, sum, min, max
    output_format: str = 'json'  # json, csv, excel
    forDownload: bool = False  # Flag to indicate if this is a download request

@app.post("/forecast")
async def forecast(data: ForecastRequest):
    try:
        # Enhanced request validation
        if not data.targets:
            return JSONResponse(
                status_code=400,
                content={"error": "No targets specified for forecasting"}
            )
        
        if not data.models:
            return JSONResponse(
                status_code=400,
                content={"error": "No models specified for targets"}
            )
        
        if data.horizon <= 0:
            return JSONResponse(
                status_code=400,
                content={"error": "Horizon must be a positive number"}
            )
            
        # Validate time period
        valid_time_periods = ['day', 'week', 'month']
        if data.time_period not in valid_time_periods:
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid time period. Must be one of: {', '.join(valid_time_periods)}"}
            )
            
        # Validate aggregation method
        valid_agg_methods = ['mean', 'sum', 'min', 'max']
        if data.aggregation_method not in valid_agg_methods:
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid aggregation method. Must be one of: {', '.join(valid_agg_methods)}"}
            )
            
        # Get data from store with validation
        df = data_store.get('merged')
        if df is None:
            return JSONResponse(
                status_code=400, 
                content={"error": "No data found. Please upload data first."}
            )

        print("Starting forecast with parameters:", {
            "time_period": data.time_period,
            "aggregation_method": data.aggregation_method,
            "horizon": data.horizon,
            "targets": data.targets,
            "models": data.models
        })
        
        # Rest of the existing forecast endpoint code...
        try:
            result_df = aggregate_targets(
                df, 
                time_period=data.time_period, 
                agg_method=data.aggregation_method
            )
            
            if isinstance(result_df, JSONResponse):
                return result_df
                
            if result_df.empty:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Aggregation resulted in empty dataset"}
                )
                
            # Ensure createdAt is a column, not an index
            if result_df.index.name == 'createdAt':
                result_df = result_df.reset_index()
            
        except Exception as e:
            print(f"Error during aggregation: {str(e)}")
            print("DataFrame state:", df.info())
            return JSONResponse(
                status_code=500,
                content={"error": f"Error during data aggregation: {str(e)}"}
            )
        
        print("Aggregated data columns:", result_df.columns.tolist())
        print("Sample aggregated data:", result_df.head())
        
        results = {}
        model_metrics = {}  # Store metrics for each target-model combination
        
        for target in data.targets:
            if target not in result_df.columns:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Target '{target}' not found in aggregated data. Available targets: {result_df.columns.tolist()}"}
                )
            
            if target not in data.models:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"No model specified for target: {target}"}
                )
                
            try:
                series = result_df.set_index('createdAt')[target]
                
                # Add metadata to series for model saving
                series._time_period = data.time_period
                series._agg_method = data.aggregation_method
                
                # Store original series for metrics calculation
                original_series = series.copy()
                
                if data.models[target] == 'ARIMA':
                    forecasted = forecast_arima(series, data.horizon)
                elif data.models[target] == 'Prophet':
                    forecasted = forecast_prophet(series, data.horizon)
                elif data.models[target] == 'LSTM':
                    forecasted = forecast_lstm(series, data.horizon)
                elif data.models[target] == 'RandomForest':
                    forecasted = forecast_rf(series, data.horizon)
                elif data.models[target] == 'LightGBM':
                    forecasted = forecast_lightgbm(series, data.horizon)
                elif data.models[target] == 'EMA':
                    forecasted = forecast_ema(series, data.horizon)
                elif data.models[target] == 'HoltWinters':
                    forecasted = forecast_holtwinters(series, data.horizon)
                else:
                    return JSONResponse(
                        status_code=400,
                        content={"error": f"Invalid model type for {target}: {data.models[target]}"}
                    )
                
                # Calculate model performance metrics for frontend display
                try:
                    if len(original_series) > 10:
                        # Use last 20% of historical data for performance evaluation
                        eval_size = max(3, int(len(original_series) * 0.2))
                        train_data = original_series[:-eval_size]
                        test_actual = original_series[-eval_size:].values
                        
                        # For demonstration, use the first few forecast values as proxy
                        # (In production, you'd re-train the model on train_data and predict test period)
                        test_predictions = forecasted[:len(test_actual)] if len(forecasted) >= len(test_actual) else forecasted + [forecasted[-1]] * (len(test_actual) - len(forecasted))
                        
                        metrics = calculate_forecast_metrics(test_actual, test_predictions[:len(test_actual)])
                        metrics['model_type'] = data.models[target]
                        metrics['target'] = target
                        metrics['evaluation_note'] = f"Evaluated on last {eval_size} historical points"
                        
                        model_metrics[f"{target}_{data.models[target]}"] = metrics
                except Exception as metrics_error:
                    print(f"⚠ Warning: Could not calculate performance metrics for {target}: {str(metrics_error)}")
                    model_metrics[f"{target}_{data.models[target]}"] = {
                        'mae': 0.0,
                        'mape': 0.0,
                        'rmse': 0.0,
                        'directional_accuracy': 0.0,
                        'model_type': data.models[target],
                        'target': target,
                        'error': str(metrics_error)
                    }
                  # Apply smart rounding based on target type
                def smart_round_forecast(values, target_name):
                    """
                    Apply appropriate rounding based on target type:
                    - Orders (transformOrdNo): Always ceil (can't have partial orders)
                    - Work Orders (woNumber): Always ceil (can't have partial work orders)  
                    - Products: Always ceil (can't have partial unique products)
                    - Quantity: Floor (conservative estimate for inventory)
                    - Workers (workers_needed): Round to 1 decimal (allows part-time workers)
                    """
                    rounded_values = []
                    for val in values:
                        if pd.isna(val):
                            rounded_values.append(0)
                        elif target_name in ['transformOrdNo', 'woNumber', 'Products']:
                            # Orders, work orders, and unique products must be whole numbers, use ceiling
                            rounded_values.append(int(np.ceil(max(0, val))))
                        elif target_name == 'quantity':
                            # Quantities should be conservative (floor) for inventory planning
                            rounded_values.append(int(np.floor(max(0, val))))
                        elif target_name == 'workers_needed':
                            # Workers can be fractional (part-time), round to 1 decimal
                            rounded_values.append(round(max(0, val), 1))
                        else:
                            # Default: round to 2 decimals for other metrics
                            rounded_values.append(round(max(0, val), 2))
                    return rounded_values
                
                # Apply smart rounding instead of simple float conversion
                forecasted = smart_round_forecast(forecasted, target)
                
                # Add validation to detect scale issues
                if len(forecasted) > 0:
                    forecast_mean = np.mean(forecasted)
                    historical_mean = series.mean()
                    historical_std = series.std()
                    
                    # Check if forecast values are completely out of reasonable range
                    # Allow for some extrapolation but flag extreme deviations
                    reasonable_upper_bound = historical_mean + 10 * historical_std
                    reasonable_lower_bound = max(0, historical_mean - 10 * historical_std)
                    
                    if forecast_mean > reasonable_upper_bound or forecast_mean < reasonable_lower_bound:
                        print(f"⚠ WARNING: {target} forecast scale issue detected!")
                        print(f"   Historical mean: {historical_mean:.2f}, std: {historical_std:.2f}")
                        print(f"   Forecast mean: {forecast_mean:.2f}")
                        print(f"   Reasonable range: {reasonable_lower_bound:.2f} - {reasonable_upper_bound:.2f}")
                        print(f"   Model: {data.models[target]}")
                
                results[target] = forecasted
                
            except Exception as e:
                print(f"Error forecasting {target}: {str(e)}")
                print(f"Series for {target}:")
                print(series.head())
                print(f"Model type: {data.models[target]}")
                return JSONResponse(
                    status_code=500,
                    content={"error": f"Error forecasting {target}: {str(e)}"}
                )
          # Format outputs with appropriate date frequency
        # Get the last historical date
        last_date = result_df['createdAt'].max()
          # For weekly data, ensure we start from the next Friday (end of week)
        if data.time_period == 'week':
            # Calculate days until next Friday (4 = Friday, 0 = Monday)
            days_until_friday = (4 - last_date.weekday()) % 7
            if days_until_friday == 0:  # If today is Friday, go to next Friday
                days_until_friday = 7
            start_date = last_date + pd.Timedelta(days=days_until_friday)
            freq = 'W-FRI'  # Weekly ending on Friday (end of business week)
        else:
            start_date = last_date + pd.Timedelta(days=1)
            freq = {'day': 'D', 'month': 'M'}[data.time_period]

        # Generate forecast dates
        forecast_dates = pd.date_range(
            start=start_date,
            periods=data.horizon,
            freq=freq
        )

        # Create historical and forecast dataframes
        historical_df = result_df.copy()
        historical_df['type'] = 'historical'
        historical_df = historical_df.rename(columns={'createdAt': 'date'})

        # Create forecast DataFrame starting from current date
        forecast_df = pd.DataFrame({'date': forecast_dates})
        for target in results:
            forecast_df[target] = results[target]
        forecast_df['type'] = 'forecast'

        # Handle NaN and Inf values
        forecast_df = forecast_df.replace([np.inf, -np.inf], np.nan)
        historical_df = historical_df.replace([np.inf, -np.inf], np.nan)
        forecast_df = forecast_df.fillna(0)
        historical_df = historical_df.fillna(0)        # Smart rounding function for final data processing
        def apply_smart_rounding_to_dataframe(df, target_columns):
            """Apply target-specific rounding to dataframe columns"""
            df_copy = df.copy()
            for col in target_columns:
                if col in df_copy.columns:
                    if col in ['transformOrdNo', 'woNumber', 'Products']:
                        # Orders, work orders, and unique products: integers (ceiling for forecasts, keep historical as-is)
                        if 'type' in df_copy.columns:
                            forecast_mask = df_copy['type'] == 'forecast'
                            historical_mask = df_copy['type'] == 'historical'
                            df_copy.loc[forecast_mask, col] = df_copy.loc[forecast_mask, col].apply(lambda x: int(np.ceil(max(0, x))) if pd.notna(x) else 0)
                            df_copy.loc[historical_mask, col] = df_copy.loc[historical_mask, col].apply(lambda x: int(round(max(0, x))) if pd.notna(x) else 0)
                        else:
                            df_copy[col] = df_copy[col].apply(lambda x: int(np.ceil(max(0, x))) if pd.notna(x) else 0)
                    elif col == 'quantity':
                        # Quantities: integers (floor for conservative estimates)
                        if 'type' in df_copy.columns:
                            forecast_mask = df_copy['type'] == 'forecast'
                            historical_mask = df_copy['type'] == 'historical'
                            df_copy.loc[forecast_mask, col] = df_copy.loc[forecast_mask, col].apply(lambda x: int(np.floor(max(0, x))) if pd.notna(x) else 0)
                            df_copy.loc[historical_mask, col] = df_copy.loc[historical_mask, col].apply(lambda x: int(round(max(0, x))) if pd.notna(x) else 0)
                        else:
                            df_copy[col] = df_copy[col].apply(lambda x: int(np.floor(max(0, x))) if pd.notna(x) else 0)
                    elif col == 'workers_needed':
                        # Workers: 1 decimal place (allows part-time workers)
                        df_copy[col] = df_copy[col].apply(lambda x: round(max(0, x), 1) if pd.notna(x) else 0.0)
                    else:
                        # Other metrics: 2 decimal places
                        df_copy[col] = df_copy[col].apply(lambda x: round(max(0, x), 2) if pd.notna(x) else 0.0)
            return df_copy
        
        # Apply smart rounding to forecast and historical data separately
        forecast_df = apply_smart_rounding_to_dataframe(forecast_df, data.targets)
        historical_df = apply_smart_rounding_to_dataframe(historical_df, data.targets)
        
        # Combine historical and forecast data
        df_result = pd.concat([historical_df, forecast_df], axis=0, sort=False)
        
        # Convert dates to string format for JSON serialization
        df_result['date'] = df_result['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Apply final smart rounding and handle NaN/infinite values
        df_result = apply_smart_rounding_to_dataframe(df_result, data.targets)
        
        # Handle any remaining NaN or infinite values
        df_result = df_result.replace([np.inf, -np.inf], np.nan)
        for col in [c for c in df_result.columns if c not in ['date', 'type']]:
            if col in ['transformOrdNo', 'woNumber']:
                df_result[col] = df_result[col].fillna(0).astype(int)
            elif col == 'quantity':
                df_result[col] = df_result[col].fillna(0).astype(int) 
            elif col == 'workers_needed':
                df_result[col] = df_result[col].fillna(0.0).astype(float)
            else:
                df_result[col] = df_result[col].fillna(0.0).astype(float)
        
        # Filter to keep only relevant columns
        columns_to_keep = ['date', 'type'] + data.targets
        df_result = df_result[columns_to_keep]

        # Save logs (non-blocking - doesn't affect frontend response)
        try:
            # Separate historical and forecast data for logging
            historical_log_data = df_result[df_result['type'] == 'historical'].copy()
            forecast_log_data = df_result[df_result['type'] == 'forecast'].copy()
            
            # Prepare forecast parameters for logging
            forecast_params = {
                'time_period': data.time_period,
                'targets': data.targets,
                'models': data.models,
                'horizon': data.horizon,
                'aggregation_method': data.aggregation_method
            }
            
            # Save logs (this runs in background and doesn't affect response)
            log_result = save_forecast_logs(historical_log_data, forecast_log_data, forecast_params, model_metrics)
            if log_result:
                print(f"✓ Logs saved successfully at {log_result['timestamp']}")
        except Exception as log_error:
            print(f"⚠ Warning: Logging failed but continuing with response: {str(log_error)}")
            # Continue with response even if logging fails

        # Split into visualization and download paths
        if data.forDownload:
            # For downloads, filter to only forecast data
            df_export = df_result[df_result['type'] == 'forecast'].copy()
        else:
            # For visualization, use complete dataset
            df_export = df_result

        # Format output based on requested format
        if data.output_format == 'json':
            response_data = {
                'forecast_data': df_export.to_dict(orient='records'),
                'model_metrics': model_metrics,
                'forecast_summary': {
                    'targets': data.targets,
                    'models_used': data.models,
                    'time_period': data.time_period,
                    'horizon': data.horizon,
                    'aggregation_method': data.aggregation_method,
                    'historical_points': len(historical_df),
                    'forecast_points': len(forecast_df)
                }
            }
            return response_data
        elif data.output_format == 'csv':
            csv_content = df_export.to_csv(index=False)
            return Response(
                content=csv_content,
                media_type='text/csv',
                headers={
                    'Content-Disposition': f'attachment; filename=forecast_{"-".join(data.targets)}_{data.time_period}_{data.aggregation_method}.csv'
                }
            )       
        elif data.output_format == 'excel':
            filename = f'forecast_{"-".join(data.targets)}_{data.time_period}_{data.aggregation_method}.xlsx'
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                df_export.to_excel(writer, index=False, sheet_name='Forecast')
            excel_buffer.seek(0)
            
            headers = {
                'Content-Disposition': f'attachment; filename="{filename}"',
                'Access-Control-Expose-Headers': 'Content-Disposition',
                'Content-Type': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            }
            
            return Response(
                content=excel_buffer.getvalue(),
                media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                headers=headers
            )
        
        else:
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid output format: {data.output_format}. Must be one of: json, csv, excel"}
            )
    except Exception as e:
        print(f"Error in forecast endpoint: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Internal server error: {str(e)}"}
        )

# Endpoint to get model performance metrics
@app.get("/model-metrics")
async def get_model_metrics():
    """Get performance metrics for all recently trained models"""
    try:
        metrics_folder = "models/trained_weights"
        
        if not os.path.exists(metrics_folder):
            return {"message": "No trained models found", "metrics": []}
        
        all_metrics = []
        
        # Read all metadata files
        for filename in os.listdir(metrics_folder):
            if filename.endswith('_metadata.json'):
                try:
                    with open(os.path.join(metrics_folder, filename), 'r') as f:
                        metadata = json.load(f)
                        
                    # Extract performance metrics if available
                    if 'performance_metrics' in metadata and metadata['performance_metrics']:
                        metrics_data = {
                            'model_type': metadata.get('model_type', 'Unknown'),
                            'target': metadata.get('target', 'Unknown'),
                            'time_period': metadata.get('time_period', 'Unknown'),
                            'timestamp': metadata.get('timestamp', 'Unknown'),
                            'data_points': metadata.get('data_points', 0),
                            'metrics': metadata['performance_metrics']
                        }
                        all_metrics.append(metrics_data)
                        
                except Exception as file_error:
                    print(f"Error reading {filename}: {str(file_error)}")
                    continue
        
        # Sort by timestamp (most recent first)
        all_metrics.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return {
            "message": "Model metrics retrieved successfully",
            "total_models": len(all_metrics),
            "metrics": all_metrics[:20]  # Return last 20 models
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error retrieving model metrics: {str(e)}"}
        )

import uvicorn

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
