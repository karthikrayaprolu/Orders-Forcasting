from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import pandas as pd
import io
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

import warnings
warnings.filterwarnings("ignore")

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
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

def aggregate_targets(df, time_period='day', agg_method='mean'):
    try:
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Check if createdAt is in the index
        if df.index.name == 'createdAt':
            # Keep a copy of the index as a column
            df['createdAt'] = df.index
        elif 'createdAt' not in df.columns:
            return JSONResponse(
                status_code=400,
                content={"error": "createdAt column not found"}
            )
        
        # Ensure createdAt is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(df['createdAt']):
            df['createdAt'] = pd.to_datetime(df['createdAt'])
        
        # Set index for resampling
        df.set_index('createdAt', inplace=True)
        
        # Define resample frequency based on time_period
        freq_map = {
            'day': 'D',
            'week': 'W',
            'month': 'M'
        }
        freq = freq_map.get(time_period, 'D')
        
        # Define aggregation method
        if agg_method == 'mean':
            agg_func = {
                'transformOrdNo': 'nunique',
                'quantity': 'mean',
                'workers_needed': 'mean',
                'woNumber': 'nunique'
            }
        elif agg_method == 'sum':
            agg_func = {
                'transformOrdNo': 'nunique',
                'quantity': 'sum',
                'workers_needed': 'sum',
                'woNumber': 'nunique'
            }
        elif agg_method == 'min':
            agg_func = {
                'transformOrdNo': 'nunique',
                'quantity': 'min',
                'workers_needed': 'min',
                'woNumber': 'nunique'
            }
        elif agg_method == 'max':
            agg_func = {
                'transformOrdNo': 'nunique',
                'quantity': 'max',
                'workers_needed': 'max',
                'woNumber': 'nunique'
            }
        else:
            raise ValueError(f"Invalid aggregation method: {agg_method}")

        # Perform resampling and aggregation
        result = df.resample(freq).agg(agg_func)
        
        # Rename columns for consistency
        result = result.rename(columns={
            'transformOrdNo': 'orders',
            'quantity': 'products',
            'workers_needed': 'employees',
            'woNumber': 'throughput'
        }).fillna(0)
        
        # Reset index to get createdAt as a column
        result = result.reset_index()
        
        return result
        
    except Exception as e:        
        print(f"Error in aggregate_targets: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

def forecast_arima(series, steps):
    """ARIMA model with proper training/testing split and parameter optimization"""
    try:
        # Ensure we have enough data
        min_samples = 30  # Minimum 30 data points needed
        if len(series) < min_samples:
            raise ValueError(f"Need at least {min_samples} data points, got {len(series)}")
        
        # Convert series to float and handle any missing values
        series = series.astype(float).fillna(method='ffill').fillna(method='bfill')
            
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
        
        return forecast['yhat'].tail(steps).tolist()
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
    values = series.values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)

    X, y = [], []
    for i in range(10, len(scaled)):
        X.append(scaled[i-10:i])
        y.append(scaled[i])
    X, y = np.array(X), np.array(y)

    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(X.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer=Adam(0.01), loss='mse')
    model.fit(X, y, epochs=10, verbose=0, callbacks=[EarlyStopping(patience=2)])

    input_seq = scaled[-10:].reshape(1, 10, 1)
    preds = []
    for _ in range(steps):
        pred = model.predict(input_seq)[0][0]
        preds.append(pred)
        input_seq = np.append(input_seq[:,1:,:], [[[pred]]], axis=1)
    return scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten().tolist()

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

@app.post("/forecast")
async def forecast(data: ForecastRequest):
    try:
        # Validate request
        if not data.targets:
            return JSONResponse(
                status_code=400,
                content={"error": "No targets specified"}
            )
        
        if not data.models:
            return JSONResponse(
                status_code=400,
                content={"error": "No models specified"}
            )
            
        # Get data from store
        df = data_store.get('merged')
        if df is None:
            return JSONResponse(
                status_code=400, 
                content={"error": "No data uploaded. Please upload data first."}
            )

        print("Dataframe columns before aggregation:", df.columns.tolist())
        print("Sample data before aggregation:", df.head())
        print("Forecast parameters:", {
            "time_period": data.time_period,
            "aggregation_method": data.aggregation_method,
            "horizon": data.horizon,
            "targets": data.targets,
            "models": data.models
        })
        
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
                else:
                    return JSONResponse(
                        status_code=400,
                        content={"error": f"Invalid model type for {target}: {data.models[target]}"}
                    )
                    
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
        freq_map = {'day': 'D', 'week': 'W', 'month': 'M'}
        dates = pd.date_range(
            start=result_df['createdAt'].max(),
            periods=data.horizon + 1,
            freq=freq_map[data.time_period]
        )[1:]  # Skip first date as it's the last historical date
        
        df_result = pd.DataFrame({'date': dates})
        for target in results:
            df_result[target] = results[target]

        # Format dates properly for export
        df_result['date'] = df_result['date'].dt.strftime('%Y-%m-%d %H:%M:%S')

        if data.output_format == 'json':
            return df_result.to_dict(orient="records")
            
        elif data.output_format == 'csv':
            buffer = io.StringIO()
            df_result.to_csv(buffer, index=False)
            buffer.seek(0)
            return StreamingResponse(
                iter([buffer.read()]),
                media_type="text/csv",
                headers={
                    "Content-Disposition": f'attachment; filename="forecast_{data.time_period}_{data.aggregation_method}.csv"',
                    "Access-Control-Expose-Headers": "Content-Disposition"
                }
            )
            
        elif data.output_format == 'excel':
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_result.to_excel(writer, index=False, sheet_name="Forecast")
            output.seek(0)
            return StreamingResponse(
                output,
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={
                    "Content-Disposition": f'attachment; filename="forecast_{data.time_period}_{data.aggregation_method}.xlsx"',
                    "Access-Control-Expose-Headers": "Content-Disposition"
                }
            )
    except Exception as e:
        print(f"Error during forecast: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

import uvicorn

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
