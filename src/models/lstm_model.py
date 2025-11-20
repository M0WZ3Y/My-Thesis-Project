"""
LSTM Model for PJM Electricity Price Prediction
TensorFlow-based sequential model for thesis requirements
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Try to import TensorFlow, provide fallback if not available
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. LSTM model will use fallback implementation.")

class LSTMPricePredictor:
    """
    LSTM-based price prediction model for time series forecasting
    """
    
    def __init__(self, sequence_length=24, lstm_units=50, dropout_rate=0.2):
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = None
        self.scaler = None
        self.feature_columns = []
        self.metrics = {}
        self.use_tensorflow = TENSORFLOW_AVAILABLE
    
    def create_features(self, df, price_column='total_lmp_da', datetime_column='datetime'):
        """
        Create features suitable for LSTM
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
        data['month'] = data[datetime_column].dt.month
        
        # Cyclical encoding
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
        data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
        data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
        
        # Lag features
        for lag in [1, 2, 3, 6, 12, 24]:
            data[f'{price_column}_lag_{lag}'] = data[price_column].shift(lag)
        
        # Rolling statistics
        for window in [3, 6, 12, 24]:
            data[f'{price_column}_rolling_mean_{window}'] = (
                data[price_column].rolling(window=window).mean()
            )
            data[f'{price_column}_rolling_std_{window}'] = (
                data[price_column].rolling(window=window).std()
            )
        
        return data
    
    def create_sequences(self, data, target_column='total_lmp_da'):
        """
        Create sequences for LSTM training
        """
        # Define feature columns
        exclude_cols = ['datetime', 'datetime_beginning_ept', target_column]
        self.feature_columns = [col for col in data.columns if col not in exclude_cols]
        
        # Remove rows with NaN values
        data_clean = data.dropna()
        
        # Create sequences
        X_sequences = []
        y_sequences = []
        
        for i in range(self.sequence_length, len(data_clean)):
            # Sequence of features
            seq_X = data_clean.iloc[i-self.sequence_length:i][self.feature_columns].values
            # Target value
            seq_y = data_clean.iloc[i][target_column]
            
            X_sequences.append(seq_X)
            y_sequences.append(seq_y)
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def build_tensorflow_model(self, n_features):
        """
        Build TensorFlow LSTM model
        """
        model = Sequential([
            LSTM(self.lstm_units, return_sequences=True, input_shape=(self.sequence_length, n_features)),
            Dropout(self.dropout_rate),
            LSTM(self.lstm_units, return_sequences=False),
            Dropout(self.dropout_rate),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    
    def build_fallback_model(self):
        """
        Build fallback model using sklearn (Gradient Boosting)
        """
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        )
    
    def fit(self, df, price_column='total_lmp_da', datetime_column='datetime', 
            test_size=0.2, epochs=50, batch_size=32):
        """
        Fit LSTM model
        """
        print(f"Training LSTM model (TensorFlow: {self.use_tensorflow})...")
        
        try:
            # Create features
            data = self.create_features(df, price_column, datetime_column)
            
            # Create sequences
            X, y = self.create_sequences(data, price_column)
            
            if len(X) == 0:
                raise ValueError("Not enough data to create sequences")
            
            # Split data
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            if self.use_tensorflow:
                # Scale features for TensorFlow
                self.scaler = StandardScaler()
                # Reshape for scaling (combine sequence and feature dimensions)
                n_samples, n_timesteps, n_features = X_train.shape
                X_train_reshaped = X_train.reshape(-1, n_features)
                X_test_reshaped = X_test.reshape(-1, n_features)
                
                X_train_scaled = self.scaler.fit_transform(X_train_reshaped)
                X_test_scaled = self.scaler.transform(X_test_reshaped)
                
                # Reshape back
                X_train_scaled = X_train_scaled.reshape(n_samples, n_timesteps, n_features)
                X_test_scaled = X_test_scaled.reshape(X_test.shape[0], n_timesteps, n_features)
                
                # Build and train model
                self.model = self.build_tensorflow_model(n_features)
                
                history = self.model.fit(
                    X_train_scaled, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=0.1,
                    verbose=0
                )
                
                # Make predictions
                train_pred = self.model.predict(X_train_scaled).flatten()
                test_pred = self.model.predict(X_test_scaled).flatten()
                
            else:
                # Fallback: Flatten sequences for Gradient Boosting
                X_train_flat = X_train.reshape(X_train.shape[0], -1)
                X_test_flat = X_test.reshape(X_test.shape[0], -1)
                
                # Scale features
                self.scaler = StandardScaler()
                X_train_scaled = self.scaler.fit_transform(X_train_flat)
                X_test_scaled = self.scaler.transform(X_test_flat)
                
                # Build and train model
                self.model = self.build_fallback_model()
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
            
            print(f"LSTM model trained successfully")
            print(f"Test MAE: ${self.metrics['test_mae']:.2f}/MWh")
            print(f"Test RMSE: ${self.metrics['test_rmse']:.2f}/MWh")
            print(f"Test MAPE: {self.metrics['test_mape']:.2f}%")
            
            return True
            
        except Exception as e:
            print(f"LSTM training failed: {str(e)}")
            return False
    
    def predict(self, df, price_column='total_lmp_da', datetime_column='datetime'):
        """
        Make predictions on new data
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model must be fitted before prediction")
        
        # Create features and sequences
        data = self.create_features(df, price_column, datetime_column)
        X, _ = self.create_sequences(data, price_column)
        
        if len(X) == 0:
            raise ValueError("Not enough data to create sequences for prediction")
        
        if self.use_tensorflow:
            # Scale features
            n_samples, n_timesteps, n_features = X.shape
            X_reshaped = X.reshape(-1, n_features)
            X_scaled = self.scaler.transform(X_reshaped)
            X_scaled = X_scaled.reshape(n_samples, n_timesteps, n_features)
            
            # Make predictions
            predictions = self.model.predict(X_scaled).flatten()
        else:
            # Flatten for fallback model
            X_flat = X.reshape(X.shape[0], -1)
            X_scaled = self.scaler.transform(X_flat)
            predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def forecast_next_hours(self, df, hours=24, price_column='total_lmp_da', datetime_column='datetime'):
        """
        Forecast next N hours
        """
        if self.model is None:
            raise ValueError("Model must be fitted before forecasting")
        
        # This is a simplified forecast - in practice, you'd implement
        # recursive forecasting with proper feature updates
        data = self.create_features(df, price_column, datetime_column)
        
        # Get the last sequence
        if len(data) < self.sequence_length:
            raise ValueError(f"Not enough data for forecasting (need {self.sequence_length} points)")
        
        last_sequence = data.iloc[-self.sequence_length:][self.feature_columns].values
        
        forecasts = []
        current_sequence = last_sequence.copy()
        
        for _ in range(hours):
            if self.use_tensorflow:
                # Prepare sequence
                seq_input = current_sequence.reshape(1, self.sequence_length, len(self.feature_columns))
                # Scale
                seq_reshaped = seq_input.reshape(-1, len(self.feature_columns))
                seq_scaled = self.scaler.transform(seq_reshaped)
                seq_scaled = seq_scaled.reshape(1, self.sequence_length, len(self.feature_columns))
                # Predict
                pred = self.model.predict(seq_scaled, verbose=0)[0][0]
            else:
                # Flatten for fallback
                seq_flat = current_sequence.reshape(1, -1)
                seq_scaled = self.scaler.transform(seq_flat)
                pred = self.model.predict(seq_scaled)[0]
            
            forecasts.append(pred)
            
            # Update sequence (simplified - would need proper feature engineering)
            # This is a placeholder for proper recursive forecasting
            current_sequence = np.roll(current_sequence, -1, axis=0)
            # In practice, you'd update all features based on the new prediction
        
        return forecasts
    
    def get_metrics(self):
        """
        Get performance metrics
        """
        return self.metrics


def main():
    """
    Example usage of LSTM model
    """
    print("LSTM Price Prediction Model")
    print("=" * 40)
    print(f"TensorFlow available: {TENSORFLOW_AVAILABLE}")
    
    if not TENSORFLOW_AVAILABLE:
        print("Note: Using Gradient Boosting fallback instead of LSTM")
    
    print("\nModel ready for training with PJM price data")
    print("Required columns: datetime, total_lmp_da")
    print("\nUsage:")
    print("1. Load your PJM data")
    print("2. Initialize LSTMPricePredictor()")
    print("3. Call fit() with your data")
    print("4. Use predict() or forecast_next_hours() for predictions")


if __name__ == "__main__":
    main()