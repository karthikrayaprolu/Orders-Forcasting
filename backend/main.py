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
    allow_origins=["*"],
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

def aggregate_targets(df):
    # Ensure createdAt is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(df['createdAt']):
        df['createdAt'] = pd.to_datetime(df['createdAt'], format='%Y-%m-%d-%H-%M-%S')
    df.set_index('createdAt', inplace=True)
    daily = df.resample('D').agg({
        'transformOrdNo': 'nunique',
        'quantity': 'sum',
        'workers_needed': 'sum',
        'woNumber': 'nunique'
    }).rename(columns={
        'transformOrdNo': 'orders',
        'quantity': 'products',
        'workers_needed': 'employees',
        'woNumber': 'throughput'
    }).fillna(0)
    return daily

def forecast_arima(series, steps):
    model = ARIMA(series, order=(2,1,2))
    model_fit = model.fit()
    pred = model_fit.forecast(steps=steps)
    return pred.tolist()

def forecast_prophet(series, steps):
    df = series.reset_index()
    df.columns = ['ds', 'y']
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=steps)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']].tail(steps).yhat.tolist()

def forecast_rf(series, steps):
    df = pd.DataFrame({'y': series})
    for lag in range(1, 8):
        df[f'lag_{lag}'] = df['y'].shift(lag)
    df.dropna(inplace=True)
    X = df.drop('y', axis=1)
    y = df['y']
    model = RandomForestRegressor()
    model.fit(X, y)
    preds = []
    last_row = X.iloc[-1].values
    for _ in range(steps):
        next_val = model.predict([last_row])[0]
        preds.append(next_val)
        last_row = np.roll(last_row, -1)
        last_row[-1] = next_val
    return preds

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
    # Read CSV files with explicit parsing for dates
    df_header = pd.read_csv(header.file)
    df_items = pd.read_csv(items.file)
    df_work = pd.read_csv(workstation.file)
    
    # Print data types for debugging
    print("Header columns types:", df_header.dtypes)
    print("Sample createdAt values:", df_header['createdAt'].head())

    df_header['createdAt'] = clean_datetime(df_header['createdAt'])
    df_items['quantity'] = df_items['quantity'].astype(float)
    df_work['workers_needed'] = df_work['workers_needed'].astype(float)

    df_merged = pd.merge(df_header, df_items, left_on='transformOrdNo', right_on='transferOrdNo', how='inner')
    df_merged = pd.merge(df_merged, df_work, on=['transferOrdNo', 'woNumber'], how='inner')

    data_store['merged'] = df_merged

    return {"message": "Files uploaded and merged successfully."}

class ForecastRequest(BaseModel):
    targets: List[str]
    models: Dict[str, str]
    horizon: int
    output_format: str = 'json'  # json, csv, excel

@app.post("/forecast")
async def forecast(data: ForecastRequest):
    df = data_store.get('merged')
    if df is None:
        return JSONResponse(status_code=400, content={"error": "No data uploaded."})

    daily = aggregate_targets(df)
    results = {}
    for target in data.targets:
        series = daily[target]
        if data.models[target] == 'ARIMA':
            forecasted = forecast_arima(series, data.horizon)
        elif data.models[target] == 'Prophet':
            forecasted = forecast_prophet(series, data.horizon)
        elif data.models[target] == 'LSTM':
            forecasted = forecast_lstm(series, data.horizon)
        elif data.models[target] == 'RandomForest':
            forecasted = forecast_rf(series, data.horizon)
        else:
            forecasted = []
        results[target] = forecasted

    # Format outputs
    dates = pd.date_range(start=daily.index[-1] + pd.Timedelta(days=1), periods=data.horizon)
    df_result = pd.DataFrame({'date': dates})
    for target in results:
        df_result[target] = results[target]

    if data.output_format == 'json':
        return df_result.to_dict(orient="records")

    elif data.output_format == 'csv':
        buffer = io.StringIO()
        df_result.to_csv(buffer, index=False)
        buffer.seek(0)
        return StreamingResponse(iter([buffer.read()]),
                                 media_type="text/csv",
                                 headers={"Content-Disposition": "attachment; filename=forecast.csv"})

    elif data.output_format == 'excel':
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_result.to_excel(writer, index=False, sheet_name="Forecast")
        output.seek(0)
        return StreamingResponse(output,
                                 media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                 headers={"Content-Disposition": "attachment; filename=forecast.xlsx"})

    else:
        return JSONResponse(status_code=400, content={"error": "Invalid output format."})

import uvicorn

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
