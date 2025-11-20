"""
ARIMA Model for PJM Electricity Price Prediction
Standalone implementation for thesis requirements
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class ARIMAPricePredictor:
    """
    ARIMA-based price prediction model
    """
    
    def __init__(self, order=(1,1,1), seasonal_order=(1,1,1,24)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.fitted_model = None
        self.predictions = None
        self.metrics = {}
    
    def prepare_data(self, df, price_column='total_lmp_da', datetime_column='datetime'):
        """
        Prepare time series data for ARIMA
        """
        # Create copy
        data = df.copy()
        
        # Ensure datetime column
        if datetime_column not in data.columns:
            if 'datetime_beginning_ept' in data.columns:
                datetime_column = 'datetime_beginning_ept'
        
        # Convert to datetime and set index
        data[datetime_column] = pd.to_datetime(data[datetime_column])
        data = data.set_index(datetime_column)
        
        # Extract price series
        price_series = data[price_column].dropna()
        
        return price_series
    
    def check_stationarity(self, series):
        """
        Check if time series is stationary using ADF test
        """
        result = adfuller(series.dropna())
        return {
            'adf_statistic': result[0],
            'p_value': result[1],
            'is_stationary': result[1] < 0.05,
            'critical_values': result[4]
        }
    
    def fit(self, price_series):
        """
        Fit ARIMA model to price series
        """
        print("Training ARIMA model...")
        
        try:
            # Check stationarity
            stationarity = self.check_stationarity(price_series)
            print(f"Stationarity check: p-value = {stationarity['p_value']:.6f}")
            
            # Fit model
            self.model = ARIMA(price_series, order=self.order, seasonal_order=self.seasonal_order)
            self.fitted_model = self.model.fit()
            
            # Generate predictions
            self.predictions = self.fitted_model.fittedvalues
            
            # Calculate metrics
            actual = price_series.dropna()
            pred = self.predictions.dropna()
            
            # Align actual and predictions
            common_index = actual.index.intersection(pred.index)
            actual_aligned = actual.loc[common_index]
            pred_aligned = pred.loc[common_index]
            
            self.metrics = {
                'mae': mean_absolute_error(actual_aligned, pred_aligned),
                'rmse': np.sqrt(mean_squared_error(actual_aligned, pred_aligned)),
                'mape': np.mean(np.abs((actual_aligned - pred_aligned) / actual_aligned)) * 100,
                'aic': self.fitted_model.aic,
                'bic': self.fitted_model.bic
            }
            
            print(f"ARIMA model trained successfully")
            print(f"MAE: ${self.metrics['mae']:.2f}/MWh")
            print(f"RMSE: ${self.metrics['rmse']:.2f}/MWh")
            print(f"MAPE: {self.metrics['mape']:.2f}%")
            
            return True
            
        except Exception as e:
            print(f"ARIMA training failed: {str(e)}")
            return False
    
    def forecast(self, steps=24):
        """
        Generate forecasts for future periods
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before forecasting")
        
        forecast = self.fitted_model.forecast(steps=steps)
        confidence_intervals = self.fitted_model.get_forecast(steps=steps).conf_int()
        
        return {
            'forecast': forecast,
            'confidence_intervals': confidence_intervals
        }
    
    def get_model_summary(self):
        """
        Get model summary statistics
        """
        if self.fitted_model is None:
            return "Model not fitted yet"
        
        return self.fitted_model.summary()
    
    def get_metrics(self):
        """
        Get performance metrics
        """
        return self.metrics


def main():
    """
    Example usage of ARIMA model
    """
    print("ARIMA Price Prediction Model")
    print("=" * 40)
    
    # This would be used with actual data
    print("Model ready for training with PJM price data")
    print("Required columns: datetime, total_lmp_da")
    print("\nUsage:")
    print("1. Load your PJM data")
    print("2. Initialize ARIMAPricePredictor()")
    print("3. Call fit() with price series")
    print("4. Use forecast() for predictions")


if __name__ == "__main__":
    main()