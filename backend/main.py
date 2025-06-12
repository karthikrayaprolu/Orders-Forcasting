from fastapi import FastAPI, File, UploadFile, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from io import BytesIO
import pandas as pd
import numpy as np
from datetime import datetime

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

import warnings
warnings.filterwarnings("ignore")

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition", "Content-Type"]
)

# Global cache
data_store = {}

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
            'week': (f"W-{last_valid_date.strftime('%a').upper()}", 'W'),  # Weekly, anchored to last date
            'month': ('M', 'M')          # Monthly, at month end
        }
        freq, period_type = freq_map.get(time_period, ('D', 'D'))
        
        # Set index for aggregation
        df.set_index('createdAt', inplace=True)
        
        # Define aggregation functions
        agg_func = {
            'transformOrdNo': 'count',  # Always count unique orders
            'quantity': agg_method,     # Use selected method
            'workers_needed': agg_method,
            'woNumber': 'count'         # Always count work orders
        }
        
        # Perform aggregation
        result_df = df.resample(freq).agg(agg_func)
        
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
        
        # Configure prophet based on data characteristics
        model = Prophet(
            yearly_seasonality='auto' if data_length > 365 else False,
            weekly_seasonality='auto' if data_length > 14 else False,
            daily_seasonality='auto' if data_length > 7 else False,
            seasonality_mode='multiplicative'
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
        
        # Train model with cross-validation
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
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
        date_features = feature_scaler.fit_transform(dates)

        # Create sequences
        sequence_length = 10
        X = []
        y = []
        for i in range(sequence_length, len(scaled)):
            X.append(scaled[i-sequence_length:i])
            y.append(scaled[i])
        X = np.array(X)
        y = np.array(y)

        # Reshape for LSTM [samples, time steps, features]
        X = X.reshape((X.shape[0], X.shape[1], 1))

        # Build and train model with improved architecture
        model = Sequential([
            LSTM(64, activation='relu', input_shape=(sequence_length, 1), return_sequences=True),
            LSTM(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
        
        # Train with early stopping
        model.fit(X, y, epochs=50, batch_size=32, verbose=0, 
                callbacks=[EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)])

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
        
        return forecast.tolist()
        
    except Exception as e:
        print(f"Holt-Winters Error: {str(e)}")
        print(f"Series head: {series.head()}")
        print(f"Series dtype: {series.dtype}")
        print(f"Series contains null: {series.isnull().any()}")
        raise e

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
        df_work['workers_needed'] = df_work['workers_needed'].astype(float)

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
                
                if data.models[target] == 'ARIMA':
                    forecasted = forecast_arima(series, data.horizon)
                elif data.models[target] == 'Prophet':
                    forecasted = forecast_prophet(series, data.horizon)
                elif data.models[target] == 'LSTM':
                    forecasted = forecast_lstm(series, data.horizon)
                elif data.models[target] == 'RandomForest':
                    forecasted = forecast_rf(series, data.horizon)
                elif data.models[target] == 'EMA':
                    forecasted = forecast_ema(series, data.horizon)
                elif data.models[target] == 'HoltWinters':
                    forecasted = forecast_holtwinters(series, data.horizon)
                else:
                    return JSONResponse(
                        status_code=400,
                        content={"error": f"Invalid model type for {target}: {data.models[target]}"}
                    )
                
                # Replace any NaN values with 0
                forecasted = [0 if pd.isna(x) else float(x) for x in forecasted]
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
        
        # For weekly data, ensure we start from the next occurrence of the same weekday
        if data.time_period == 'week':
            # Get the next occurrence of the same weekday
            start_date = last_date + pd.Timedelta(days=7)  # Next week same day
            freq = f"W-{last_date.strftime('%a').upper()}"  # Anchor to same weekday
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
        historical_df = historical_df.fillna(0)
        
        # Round numeric values to prevent float precision issues
        numeric_cols = forecast_df.select_dtypes(include=[np.number]).columns
        forecast_df[numeric_cols] = forecast_df[numeric_cols].round(2)
        numeric_cols = historical_df.select_dtypes(include=[np.number]).columns
        historical_df[numeric_cols] = historical_df[numeric_cols].round(2)
        
        # Combine historical and forecast data
        df_result = pd.concat([historical_df, forecast_df], axis=0, sort=False)        # Convert dates to string format for JSON serialization
        df_result['date'] = df_result['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Convert numeric columns to float with 2 decimal places
        numeric_cols = [col for col in df_result.columns if col not in ['date', 'type']]
        df_result[numeric_cols] = df_result[numeric_cols].astype(float).round(2)
        # Handle any NaN or infinite values before output
        df_result = df_result.replace([np.inf, -np.inf], np.nan)
        for col in [c for c in df_result.columns if c not in ['date', 'type']]:
            df_result[col] = df_result[col].fillna(0).astype(float).round(2)
          # Filter to keep only relevant columns
        columns_to_keep = ['date', 'type'] + data.targets
        df_result = df_result[columns_to_keep]

        # Split into visualization and download paths
        if data.forDownload:
            # For downloads, filter to only forecast data
            df_export = df_result[df_result['type'] == 'forecast'].copy()
        else:
            # For visualization, use complete dataset
            df_export = df_result

        # Format output based on requested format
        if data.output_format == 'json':
            return df_export.to_dict(orient='records')
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

import uvicorn

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
