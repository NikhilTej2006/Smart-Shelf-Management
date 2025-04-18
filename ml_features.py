import pandas as pd
import numpy as np
from prophet import Prophet

class MLForecasting:
    def __init__(self):
        pass  # No need for a CSV path, since data will be passed directly

    def stock_prediction(self, product_name, data):
        print("\n🔎 Starting Stock Prediction...")
        try:
            if 'ds' not in data.columns or 'y' not in data.columns:
                raise ValueError("Data must contain 'ds' (timestamp) and 'y' (sales quantity) columns.")

            m = Prophet()
            m.fit(data)
            future = m.make_future_dataframe(periods=30)
            forecast = m.predict(future)

            latest = forecast.tail(1)['yhat'].iloc[0]
            status = "🟢 Overstock" if latest > 80 else "🔴 Low Stock" if latest < 20 else "🟡 Normal"

            result = {
                "product": product_name,
                "predicted_quantity": int(latest),
                "status": status
            }

            print(f"[Stock Prediction] Product: {product_name}")
            print(f"  - Predicted Quantity: {int(latest)} units")
            print(f"  - Status: {status}")
            return result

        except Exception as e:
            print("❌ Stock Prediction Error:", e)
            return None

    def demand_forecasting(self, product_name, data):
        print("\n📈 Starting Demand Forecasting...")
        try:
            if 'ds' not in data.columns or 'y' not in data.columns:
                raise ValueError("Data must contain 'ds' (timestamp) and 'y' (sales quantity) columns.")

            # Add synthetic regressors
            data['seasonality'] = np.sin(np.linspace(0, 2 * np.pi * len(data) / 30, len(data)))
            data['promotions'] = np.random.choice([0, 1], len(data))

            m = Prophet()
            m.add_regressor('seasonality')
            m.add_regressor('promotions')
            m.fit(data)

            future = m.make_future_dataframe(periods=30)
            future['seasonality'] = np.sin(np.linspace(0, 2 * np.pi * (len(data) + 30) / 30, len(data) + 30))
            future['promotions'] = np.random.choice([0, 1], len(data) + 30)

            forecast = m.predict(future)
            avg = forecast['yhat'].tail(30).mean()

            result = {
                "product": product_name,
                "predicted_30_day_demand": int(avg)
            }

            print(f"[Demand Forecast] Product: {product_name}")
            print(f"  - Average Demand (next 30 days): {avg:.1f} units")
            return result

        except Exception as e:
            print("❌ Demand Forecasting Error:", e)
            return None


# --- Example Usage ---
if __name__ == "__main__":
    # Sample sales history (You can replace this with actual CSV data or database fetch)
    sales_data = {
        'ds': pd.date_range(start='2024-01-01', periods=10, freq='D'),
        'y': [20, 23, 19, 25, 22, 26, 30, 28, 35, 40]  # Sample bottle sales
    }
    df = pd.DataFrame(sales_data)

    # Create an instance of MLForecasting
    ml_forecaster = MLForecasting()

    # Stock Prediction for Soap
    stock_result = ml_forecaster.stock_prediction("Soap", df)

    # Demand Forecasting for Shampoo
    demand_result = ml_forecaster.demand_forecasting("Shampoo", df)

    # Final Output
    if stock_result:
        print(f"\n✅ Final Stock Forecast: {stock_result}")
    if demand_result:
        print(f"✅ Final Demand Forecast: {demand_result}")
