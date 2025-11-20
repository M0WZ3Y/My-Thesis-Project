"""
Test individual models with sample data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import models
from models.arima_model import ARIMAPricePredictor
from models.xgboost_model import XGBoostPricePredictor
from models.lstm_model import LSTMPricePredictor

def create_sample_data(n_records=1000):
    """
    Create sample PJM data for testing
    """
    print(f"Creating sample data with {n_records} records...")
    
    # Create datetime range
    start_date = datetime(2025, 1, 1)
    dates = [start_date + timedelta(hours=i) for i in range(n_records)]
    
    # Generate synthetic price data with patterns
    np.random.seed(42)
    base_price = 50  # Base price $/MWh
    
    prices = []
    for i, date in enumerate(dates):
        # Hourly pattern (higher during peak hours)
        hour_factor = 1.0
        if 8 <= date.hour <= 11 or 17 <= date.hour <= 21:
            hour_factor = 1.3
        elif 0 <= date.hour <= 5:
            hour_factor = 0.8
        
        # Weekly pattern (higher on weekdays)
        weekday_factor = 1.0
        if date.weekday() >= 5:  # Weekend
            weekday_factor = 0.9
        
        # Random noise
        noise = np.random.normal(0, 5)
        
        # Calculate price
        price = base_price * hour_factor * weekday_factor + noise
        prices.append(max(price, 10))  # Minimum price of $10/MWh
    
    # Create DataFrame
    data = pd.DataFrame({
        'datetime': dates,
        'total_lmp_da': prices,
        'system_energy_price_da': [p * 0.7 for p in prices],
        'congestion_price_da': [p * 0.2 for p in prices],
        'marginal_loss_price_da': [p * 0.1 for p in prices]
    })
    
    return data

def test_arima():
    """
    Test ARIMA model
    """
    print("\n" + "="*50)
    print("TESTING ARIMA MODEL")
    print("="*50)
    
    try:
        # Create sample data
        data = create_sample_data(500)  # Smaller dataset for ARIMA
        
        # Initialize and train model
        model = ARIMAPricePredictor(order=(1,1,1), seasonal_order=(1,1,1,24))
        price_series = model.prepare_data(data, 'total_lmp_da', 'datetime')
        success = model.fit(price_series)
        
        if success:
            print("ARIMA model test: PASSED")
            metrics = model.get_metrics()
            print(f"MAE: ${metrics['mae']:.2f}/MWh")
            print(f"MAPE: {metrics['mape']:.2f}%")
            
            # Test forecasting
            forecast = model.forecast(steps=24)
            print(f"24-hour forecast generated: {len(forecast['forecast'])} values")
            
            return True
        else:
            print("ARIMA model test: FAILED")
            return False
            
    except Exception as e:
        print(f"ARIMA model test: ERROR - {str(e)}")
        return False

def test_xgboost():
    """
    Test XGBoost model
    """
    print("\n" + "="*50)
    print("TESTING XGBOOST MODEL")
    print("="*50)
    
    try:
        # Create sample data
        data = create_sample_data(1000)
        
        # Initialize and train model
        model = XGBoostPricePredictor(n_estimators=50, max_depth=4)
        X, y, _ = model.prepare_data(data, 'total_lmp_da', 'datetime')
        success = model.fit(X, y, test_size=0.2)
        
        if success:
            print("XGBoost model test: PASSED")
            metrics = model.get_metrics()
            print(f"Test MAE: ${metrics['test_mae']:.2f}/MWh")
            print(f"Test MAPE: {metrics['test_mape']:.2f}%")
            
            # Test feature importance
            feature_imp = model.get_feature_importance(5)
            print("Top 5 features:")
            print(feature_imp[['feature', 'importance']].to_string(index=False))
            
            return True
        else:
            print("XGBoost model test: FAILED")
            return False
            
    except Exception as e:
        print(f"XGBoost model test: ERROR - {str(e)}")
        return False

def test_lstm():
    """
    Test LSTM model
    """
    print("\n" + "="*50)
    print("TESTING LSTM MODEL")
    print("="*50)
    
    try:
        # Create sample data
        data = create_sample_data(1000)
        
        # Initialize and train model
        model = LSTMPricePredictor(sequence_length=24, lstm_units=30)
        success = model.fit(data, 'total_lmp_da', 'datetime', test_size=0.2, epochs=10)
        
        if success:
            print("LSTM model test: PASSED")
            metrics = model.get_metrics()
            print(f"Test MAE: ${metrics['test_mae']:.2f}/MWh")
            print(f"Test MAPE: {metrics['test_mape']:.2f}%")
            
            # Test forecasting
            forecast = model.forecast_next_hours(data, hours=24)
            print(f"24-hour forecast generated: {len(forecast)} values")
            
            return True
        else:
            print("LSTM model test: FAILED")
            return False
            
    except Exception as e:
        print(f"LSTM model test: ERROR - {str(e)}")
        return False

def main():
    """
    Test all models
    """
    print("TESTING PJM PRICE PREDICTION MODELS")
    print("=" * 60)
    
    results = {}
    
    # Test each model
    results['arima'] = test_arima()
    results['xgboost'] = test_xgboost()
    results['lstm'] = test_lstm()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for model, success in results.items():
        status = "PASSED" if success else "FAILED"
        print(f"{model.upper()}: {status}")
    
    print(f"\nOverall: {passed}/{total} models passed")
    
    if passed == total:
        print("All models are working correctly!")
    elif passed > 0:
        print("Some models are working. Check failed models for issues.")
    else:
        print("All models failed. Check dependencies and data format.")

if __name__ == "__main__":
    main()