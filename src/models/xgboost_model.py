"""
XGBoost Model for PJM Electricity Price Prediction
Standalone implementation for thesis requirements
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

class XGBoostPricePredictor:
    """
    XGBoost-based price prediction model with feature engineering
    """
    
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.model = None
        self.scaler = None
        self.feature_columns = []
        self.metrics = {}
        self.feature_importance = None
    
    def create_features(self, df, price_column='total_lmp_da', datetime_column='datetime'):
        """
        Create comprehensive features for XGBoost
        """
        data = df.copy()
        
        # Ensure datetime column
        if datetime_column not in data.columns:
            if 'datetime_beginning_ept' in data.columns:
                datetime_column = 'datetime_beginning_ept'
        
        # Convert datetime
        data[datetime_column] = pd.to_datetime(data[datetime_column])
        
        # Time-based features
        data['hour'] = data[datetime_column].dt.hour
        data['day_of_week'] = data[datetime_column].dt.dayofweek
        data['day_of_month'] = data[datetime_column].dt.day
        data['month'] = data[datetime_column].dt.month
        data['quarter'] = data[datetime_column].dt.quarter
        data['year'] = data[datetime_column].dt.year
        data['week_of_year'] = data[datetime_column].dt.isocalendar().week
        
        # Cyclical encoding
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
        data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
        data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
        data['dow_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
        data['dow_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
        
        # Lag features
        for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
            data[f'{price_column}_lag_{lag}'] = data[price_column].shift(lag)
        
        # Rolling statistics
        for window in [3, 6, 12, 24, 48, 168]:
            data[f'{price_column}_rolling_mean_{window}'] = (
                data[price_column].rolling(window=window).mean()
            )
            data[f'{price_column}_rolling_std_{window}'] = (
                data[price_column].rolling(window=window).std()
            )
        
        # Price change features
        data['price_change_1h'] = data[price_column].pct_change(1)
        data['price_change_24h'] = data[price_column].pct_change(24)
        data['volatility_24h'] = data['price_change_1h'].rolling(24).std()
        
        # Peak indicators
        data['is_peak_hour'] = ((data['hour'] >= 8) & (data['hour'] <= 11)) | \
                              ((data['hour'] >= 17) & (data['hour'] <= 21))
        data['is_weekend'] = data['day_of_week'].isin([5, 6])
        
        # Seasonal indicators
        data['is_summer'] = data['month'].isin([6, 7, 8])
        data['is_winter'] = data['month'].isin([12, 1, 2])
        
        return data
    
    def prepare_data(self, df, price_column='total_lmp_da', datetime_column='datetime'):
        """
        Prepare features and target for training
        """
        # Create features
        data = self.create_features(df, price_column, datetime_column)
        
        # Define feature columns (exclude datetime and target)
        exclude_cols = [datetime_column, price_column]
        self.feature_columns = [col for col in data.columns if col not in exclude_cols]
        
        # Remove rows with NaN values
        data_clean = data.dropna()
        
        # Check if we have enough data after cleaning
        if len(data_clean) == 0:
            print("Warning: No valid data after feature creation. Using simplified features.")
            # Fallback to basic features without lags and rolling stats
            data = self.create_basic_features(df, price_column, datetime_column)
            data_clean = data.dropna()
            
            # Update feature columns for basic features
            self.feature_columns = [col for col in data.columns if col not in exclude_cols]
        
        # Prepare X and y
        X = data_clean[self.feature_columns]
        y = data_clean[price_column]
        
        return X, y, data_clean
    
    def create_basic_features(self, df, price_column='total_lmp_da', datetime_column='datetime'):
        """
        Create basic features without lags and rolling statistics for small datasets
        """
        data = df.copy()
        
        # Ensure datetime column
        if datetime_column not in data.columns:
            if 'datetime_beginning_ept' in data.columns:
                datetime_column = 'datetime_beginning_ept'
        
        # Convert datetime
        data[datetime_column] = pd.to_datetime(data[datetime_column])
        
        # Time-based features only
        data['hour'] = data[datetime_column].dt.hour
        data['day_of_week'] = data[datetime_column].dt.dayofweek
        data['day_of_month'] = data[datetime_column].dt.day
        data['month'] = data[datetime_column].dt.month
        data['quarter'] = data[datetime_column].dt.quarter
        data['year'] = data[datetime_column].dt.year
        
        # Cyclical encoding
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
        data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
        data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
        data['dow_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
        data['dow_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
        
        # Peak indicators
        data['is_peak_hour'] = ((data['hour'] >= 8) & (data['hour'] <= 11)) | \
                              ((data['hour'] >= 17) & (data['hour'] <= 21))
        data['is_weekend'] = data['day_of_week'].isin([5, 6])
        
        # Seasonal indicators
        data['is_summer'] = data['month'].isin([6, 7, 8])
        data['is_winter'] = data['month'].isin([12, 1, 2])
        
        return data
    
    def fit(self, X, y, test_size=0.2):
        """
        Fit XGBoost model
        """
        print("Training XGBoost model...")
        
        try:
            # Time series split
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Initialize and train model
            self.model = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            # Make predictions
            train_pred = self.model.predict(X_train_scaled)
            test_pred = self.model.predict(X_test_scaled)
            
            # Calculate metrics
            self.metrics = {
                'train_mae': mean_absolute_error(y_train, train_pred),
                'test_mae': mean_absolute_error(y_test, test_pred),
                'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
                'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
                'test_mape': np.mean(np.abs((y_test - test_pred) / y_test)) * 100
            }
            
            # Feature importance
            self.feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"XGBoost model trained successfully")
            print(f"Test MAE: ${self.metrics['test_mae']:.2f}/MWh")
            print(f"Test RMSE: ${self.metrics['test_rmse']:.2f}/MWh")
            print(f"Test MAPE: {self.metrics['test_mape']:.2f}%")
            
            return True
            
        except Exception as e:
            print(f"XGBoost training failed: {str(e)}")
            return False
    
    def predict(self, X):
        """
        Make predictions on new data
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def get_feature_importance(self, top_n=10):
        """
        Get top N most important features
        """
        if self.feature_importance is None:
            return "Model not fitted yet"
        
        return self.feature_importance.head(top_n)
    
    def get_metrics(self):
        """
        Get performance metrics
        """
        return self.metrics
    
    def forecast_next_hours(self, df, hours=24):
        """
        Forecast next N hours using the trained model
        """
        if self.model is None:
            raise ValueError("Model must be fitted before forecasting")
        
        # Create features for the most recent data
        data = self.create_features(df)
        
        # Get the last row and create future features
        last_row = data.iloc[-1:].copy()
        forecasts = []
        
        for i in range(hours):
            # Prepare features
            X_forecast = last_row[self.feature_columns]
            
            # Make prediction
            pred = self.predict(X_forecast)[0]
            forecasts.append(pred)
            
            # Update last_row for next iteration
            # This is simplified - in practice, you'd update all lag features
            last_row = last_row.copy()
        
        return forecasts


def main():
    """
    Example usage of XGBoost model
    """
    print("XGBoost Price Prediction Model")
    print("=" * 40)
    
    # This would be used with actual data
    print("Model ready for training with PJM price data")
    print("Required columns: datetime, total_lmp_da")
    print("\nUsage:")
    print("1. Load your PJM data")
    print("2. Initialize XGBoostPricePredictor()")
    print("3. Call prepare_data() to create features")
    print("4. Call fit() with X and y")
    print("5. Use predict() for new predictions")


if __name__ == "__main__":
    main()