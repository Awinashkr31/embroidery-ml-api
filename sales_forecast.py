import os
import random
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from datetime import datetime, timedelta
from dotenv import load_dotenv
from supabase import create_client, Client
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load environment variables
load_dotenv()

def get_supabase_client() -> Client:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("Missing Supabase credentials in .env")
    return create_client(url, key)

def fetch_or_generate_sales_data():
    supabase = get_supabase_client()
    try:
        response = supabase.table("orders").select("created_at, total").execute()
        data = response.data
    except Exception as e:
        print(f"Error fetching data from Supabase: {e}")
        data = []

    df = pd.DataFrame(data)
    
    if len(df) < 50:
        print("Not enough real sales data found (< 50 rows). Generating synthetic historical data for standard prediction model training...")
        synthetic_data = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)
        
        current_date = start_date
        while current_date < end_date:
            # Simulate daily orders with some noise and a minor upward trend
            num_orders_today = int(max(0, random.gauss(5, 2) + (current_date - start_date).days * 0.05))
            for _ in range(num_orders_today):
                synthetic_data.append({
                    "created_at": current_date.isoformat(),
                    "total": round(random.uniform(500, 3000), 2)
                })
            current_date += timedelta(days=1)
        
        df = pd.DataFrame(synthetic_data)
    else:
        print(f"Found {len(df)} real orders.")

    # Clean data for Prophet
    # Prophet requires 'ds' (datestamp) and 'y' (target)
    df['ds'] = pd.to_datetime(df['created_at']).dt.tz_localize(None).dt.date
    df['total'] = pd.to_numeric(df['total'], errors='coerce').fillna(0)
    
    # Aggregate by day
    df_daily = df.groupby('ds')['total'].sum().reset_index()
    df_daily.columns = ['ds', 'y']
    
    return df_daily

def train_and_forecast(df):
    print("Initializing Time Series Model (Prophet)...")
    
    # We leave out the last 15 days as a test set for evaluation
    test_days = 15
    train_df = df.iloc[:-test_days]
    test_df = df.iloc[-test_days:]

    # Fit Model
    model = Prophet(yearly_seasonality=False, daily_seasonality=False)
    model.fit(train_df)
    
    # Predict the test period to evaluate metrics
    future_test = model.make_future_dataframe(periods=test_days)
    forecast_test = model.predict(future_test)
    
    # Extract predicted yhat for test period
    forecast_test_only = forecast_test.iloc[-test_days:]['yhat'].values
    actual_test_only = test_df['y'].values
    
    mae = mean_absolute_error(actual_test_only, forecast_test_only)
    rmse = mean_squared_error(actual_test_only, forecast_test_only) ** 0.5
    
    print("\n--- MODEL EVALUATION METRICS ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print("--------------------------------\n")
    
    # Now retrain on ALL data to forecast the true future
    print("Retraining on all data for future 30 days forecast...")
    final_model = Prophet(yearly_seasonality=False, daily_seasonality=False)
    final_model.fit(df)
    
    future_30 = final_model.make_future_dataframe(periods=30)
    final_forecast = final_model.predict(future_30)
    
    # Generate Plots
    print("Generating forecast plots...")
    fig1 = final_model.plot(final_forecast)
    plt.title("Embroidery By Sana - 30 Day Sales Forecast")
    plt.xlabel("Date")
    plt.ylabel("Revenue (₹)")
    plt.savefig("sales_forecast.png", dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    print("Saved 'sales_forecast.png' successfully.")
def get_forecast_json():
    df = fetch_or_generate_sales_data()
    model = Prophet(yearly_seasonality=False, daily_seasonality=False)
    model.fit(df)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    
    actuals_dict = dict(zip(df['ds'].astype(str), df['y']))
    
    results = []
    for _, row in forecast.iterrows():
        date_str = str(row['ds'].date())
        yhat = round(row['yhat'], 2)
        actual = actuals_dict.get(date_str, None)
        
        row_type = "Historical" if actual is not None else "Forecast"
        
        results.append({
            "date": date_str[-5:], # e.g. "11-02" for UI friendliness
            "actual": actual,
            "forecast": yhat,
            "lower": round(row['yhat_lower'], 2),
            "upper": round(row['yhat_upper'], 2),
            "type": row_type
        })
        
    return results[-45:] # 15 days history + 30 days future

if __name__ == "__main__":
    df_sales = fetch_or_generate_sales_data()
    train_and_forecast(df_sales)
