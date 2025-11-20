import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class PJMPricePredictor:
    def __init__(self, data_file):
        self.data_file = data_file
        self.data = None
        self.models = {}
        self.scalers = {}
        
    def load_data(self):
        """Load and preprocess the PJM data"""
        print("Loading PJM data...")
        
        # Load data in chunks to handle large file
        chunks = []
        for chunk in pd.read_csv(self.data_file, chunksize=100000):
            chunks.append(chunk)
        
        self.data = pd.concat(chunks, ignore_index=True)
        
        # Convert datetime columns
        self.data['datetime_beginning_utc'] = pd.to_datetime(self.data['datetime_beginning_utc'])
        self.data['datetime_beginning_ept'] = pd.to_datetime(self.data['datetime_beginning_ept'])
        
        print(f"Data loaded: {len(self.data)} records")
        print(f"Date range: {self.data['datetime_beginning_utc'].min()} to {self.data['datetime_beginning_utc'].max()}")
        print(f"Number of unique nodes: {self.data['pnode_id'].nunique()}")
        print(f"Number of zones: {self.data['zone'].nunique()}")
        
        return self.data
    
    def analyze_data_structure(self):
        """Analyze the data structure and provide insights"""
        print("\n=== DATA ANALYSIS ===")
        
        # Basic statistics
        print(f"\nPrice Statistics (Total LMP):")
        print(self.data['total_lmp_da'].describe())
        
        # Zone analysis
        zone_stats = self.data.groupby('zone')['total_lmp_da'].agg(['mean', 'std', 'count']).round(2)
        print(f"\nPrice Statistics by Zone:")
        print(zone_stats)
        
        # Time patterns
        self.data['hour'] = self.data['datetime_beginning_utc'].dt.hour
        self.data['day_of_week'] = self.data['datetime_beginning_utc'].dt.dayofweek
        self.data['month'] = self.data['datetime_beginning_utc'].dt.month
        
        hourly_avg = self.data.groupby('hour')['total_lmp_da'].mean()
        daily_avg = self.data.groupby('day_of_week')['total_lmp_da'].mean()
        
        print(f"\nAverage Price by Hour:")
        for hour, price in hourly_avg.items():
            print(f"Hour {hour:2d}: ${price:.2f}")
            
        print(f"\nAverage Price by Day of Week:")
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        for day, price in daily_avg.items():
            print(f"{days[day]}: ${price:.2f}")
    
    def create_features(self, target_zone=None):
        """Create features for machine learning"""
        print("\n=== FEATURE ENGINEERING ===")
        
        # Filter for specific zone if provided
        if target_zone:
            df = self.data[self.data['zone'] == target_zone].copy()
            print(f"Analyzing zone: {target_zone}")
        else:
            # Use system average if no specific zone
            df = self.data.groupby('datetime_beginning_utc').agg({
                'total_lmp_da': 'mean',
                'system_energy_price_da': 'mean',
                'congestion_price_da': 'mean',
                'marginal_loss_price_da': 'mean'
            }).reset_index()
            print("Using system average prices")
        
        # Time-based features
        df['hour'] = df['datetime_beginning_utc'].dt.hour
        df['day_of_week'] = df['datetime_beginning_utc'].dt.dayofweek
        df['month'] = df['datetime_beginning_utc'].dt.month
        df['quarter'] = df['datetime_beginning_utc'].dt.quarter
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Cyclical features for time
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Lag features (previous hours) - adjust for short time periods
        df.sort_values('datetime_beginning_utc', inplace=True)
        
        # Check data length and adjust lag features accordingly
        data_hours = len(df)
        print(f"Data has {data_hours} records")
        
        if data_hours >= 48:
            lags = [1, 2, 3, 6, 12, 24, 48]
            windows = [6, 12, 24]
        elif data_hours >= 24:
            lags = [1, 2, 3, 6, 12]
            windows = [6, 12]
        elif data_hours >= 12:
            lags = [1, 2, 3, 6]
            windows = [6]
        else:
            lags = [1, 2, 3]
            windows = []
        
        for lag in lags:
            df[f'price_lag_{lag}h'] = df['total_lmp_da'].shift(lag)
        
        # Rolling statistics
        for window in windows:
            df[f'price_rolling_mean_{window}h'] = df['total_lmp_da'].rolling(window=window).mean()
            df[f'price_rolling_std_{window}h'] = df['total_lmp_da'].rolling(window=window).std()
        
        # Price components as features
        if 'system_energy_price_da' in df.columns:
            df['energy_component_ratio'] = df['system_energy_price_da'] / df['total_lmp_da']
            df['congestion_component_ratio'] = df['congestion_price_da'] / df['total_lmp_da']
            df['loss_component_ratio'] = df['marginal_loss_price_da'] / df['total_lmp_da']
        
        # Remove rows with NaN values (from lag features)
        initial_rows = len(df)
        df.dropna(inplace=True)
        final_rows = len(df)
        
        print(f"Features created. Dataset shape: {df.shape}")
        print(f"Removed {initial_rows - final_rows} rows due to NaN values from lag features")
        
        # If we have too few rows after feature engineering, use simpler features
        if final_rows < 10:
            print("Warning: Too few rows after feature engineering. Using basic features only.")
            # Reload data with basic features only
            if target_zone:
                df = self.data[self.data['zone'] == target_zone].copy()
            else:
                df = self.data.groupby('datetime_beginning_utc').agg({
                    'total_lmp_da': 'mean',
                    'system_energy_price_da': 'mean',
                    'congestion_price_da': 'mean',
                    'marginal_loss_price_da': 'mean'
                }).reset_index()
            
            # Only basic time features
            df['hour'] = df['datetime_beginning_utc'].dt.hour
            df['day_of_week'] = df['datetime_beginning_utc'].dt.dayofweek
            df['month'] = df['datetime_beginning_utc'].dt.month
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            
            # Only 1-hour lag
            df.sort_values('datetime_beginning_utc', inplace=True)
            df['price_lag_1h'] = df['total_lmp_da'].shift(1)
            df.dropna(inplace=True)
            
            print(f"Using simplified features. Dataset shape: {df.shape}")
        
        return df
    
    def prepare_train_test(self, df, test_days=7):
        """Split data into training and testing sets"""
        # Sort by datetime
        df = df.sort_values('datetime_beginning_utc')
        
        # For short time periods, use smaller test set
        total_hours = len(df)
        if total_hours < 24:
            test_hours = max(1, total_hours // 4)  # 25% for testing
        elif total_hours < 48:
            test_hours = 6  # 6 hours for testing
        else:
            test_hours = min(24, total_hours // 3)  # 1 day or 33% for testing
        
        # Calculate split point
        split_index = len(df) - test_hours
        
        train_df = df.iloc[:split_index].copy()
        test_df = df.iloc[split_index:].copy()
        
        print(f"Training data: {len(train_df)} records ({train_df['datetime_beginning_utc'].min()} to {train_df['datetime_beginning_utc'].max()})")
        print(f"Testing data: {len(test_df)} records ({test_df['datetime_beginning_utc'].min()} to {test_df['datetime_beginning_utc'].max()})")
        
        # Ensure we have data in both sets
        if len(train_df) == 0 or len(test_df) == 0:
            raise ValueError("Not enough data for train/test split. Consider getting more historical data.")
        
        return train_df, test_df
    
    def train_models(self, train_df, test_df):
        """Train multiple prediction models"""
        print("\n=== MODEL TRAINING ===")
        
        # Feature columns (exclude datetime and target)
        feature_cols = [col for col in train_df.columns if col not in [
            'datetime_beginning_utc', 'total_lmp_da', 'system_energy_price_da',
            'congestion_price_da', 'marginal_loss_price_da'
        ]]
        
        X_train = train_df[feature_cols]
        y_train = train_df['total_lmp_da']
        X_test = test_df[feature_cols]
        y_test = test_df['total_lmp_da']
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Models to train
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            if name == 'Linear Regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'predictions': y_pred,
                'mae': mae,
                'rmse': rmse,
                'r2': r2
            }
            
            print(f"  MAE: ${mae:.2f}")
            print(f"  RMSE: ${rmse:.2f}")
            print(f"  R²: {r2:.3f}")
        
        # Store the best model
        best_model = min(results.keys(), key=lambda x: results[x]['mae'])
        print(f"\nBest model: {best_model} (MAE: ${results[best_model]['mae']:.2f})")
        
        self.models = results
        self.scalers = {'scaler': scaler, 'feature_cols': feature_cols}
        
        return results, best_model
    
    def create_visualizations(self, test_df, results):
        """Create visualization plots"""
        print("\n=== CREATING VISUALIZATIONS ===")
        
        # Plot 1: Actual vs Predicted for best model
        best_model = min(results.keys(), key=lambda x: results[x]['mae'])
        best_predictions = results[best_model]['predictions']
        
        plt.figure(figsize=(15, 10))
        
        # Subplot 1: Time series comparison
        plt.subplot(2, 2, 1)
        plt.plot(test_df['datetime_beginning_utc'], test_df['total_lmp_da'], 
                label='Actual', alpha=0.7, linewidth=1)
        plt.plot(test_df['datetime_beginning_utc'], best_predictions, 
                label=f'Predicted ({best_model})', alpha=0.7, linewidth=1)
        plt.title('Actual vs Predicted Prices')
        plt.xlabel('Date')
        plt.ylabel('Price ($/MWh)')
        plt.legend()
        plt.xticks(rotation=45)
        
        # Subplot 2: Scatter plot
        plt.subplot(2, 2, 2)
        plt.scatter(test_df['total_lmp_da'], best_predictions, alpha=0.5)
        plt.plot([test_df['total_lmp_da'].min(), test_df['total_lmp_da'].max()],
                [test_df['total_lmp_da'].min(), test_df['total_lmp_da'].max()], 'r--', lw=2)
        plt.xlabel('Actual Price ($/MWh)')
        plt.ylabel('Predicted Price ($/MWh)')
        plt.title('Actual vs Predicted Scatter')
        
        # Subplot 3: Residuals
        plt.subplot(2, 2, 3)
        residuals = test_df['total_lmp_da'] - best_predictions
        plt.scatter(best_predictions, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Price ($/MWh)')
        plt.ylabel('Residuals ($/MWh)')
        plt.title('Residual Plot')
        
        # Subplot 4: Model comparison
        plt.subplot(2, 2, 4)
        model_names = list(results.keys())
        mae_values = [results[name]['mae'] for name in model_names]
        bars = plt.bar(model_names, mae_values)
        plt.ylabel('Mean Absolute Error ($/MWh)')
        plt.title('Model Performance Comparison')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, mae in zip(bars, mae_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'${mae:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('pjm_price_prediction_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Visualizations saved as 'pjm_price_prediction_results.png'")
    
    def predict_future(self, hours_ahead=24):
        """Make predictions for future hours"""
        print(f"\n=== FUTURE PREDICTION ({hours_ahead} hours ahead) ===")
        
        if not self.models:
            print("No trained models available. Please train models first.")
            return None
        
        # Get the best model
        best_model_name = min(self.models.keys(), key=lambda x: self.models[x]['mae'])
        best_model = self.models[best_model_name]['model']
        
        # Get the most recent data point
        latest_data = self.data.sort_values('datetime_beginning_utc').tail(1)
        
        print(f"Using {best_model_name} for predictions")
        print(f"Latest data point: {latest_data['datetime_beginning_utc'].iloc[0]}")
        print(f"Latest price: ${latest_data['total_lmp_da'].iloc[0]:.2f}")
        
        # Note: In a real implementation, you would need additional data (weather, load, etc.)
        # for accurate future predictions. This is a simplified example.
        print("\nNote: For accurate future predictions, additional data sources needed:")
        print("- Weather forecasts (temperature, wind, solar)")
        print("- Load forecasts")
        print("- Generation availability")
        print("- Fuel prices")
        
        return None

def main():
    """Main execution function"""
    print("=== PJM ELECTRICITY PRICE PREDICTION ===\n")
    
    # Initialize predictor
    predictor = PJMPricePredictor('da_hrl_lmps (1).csv')
    
    # Load and analyze data
    predictor.load_data()
    predictor.analyze_data_structure()
    
    # Create features
    df = predictor.create_features(target_zone=None)  # Use None for system average
    
    # Prepare train/test split
    train_df, test_df = predictor.prepare_train_test(df, test_days=7)
    
    # Train models
    results, best_model = predictor.train_models(train_df, test_df)
    
    # Create visualizations
    predictor.create_visualizations(test_df, results)
    
    # Future predictions (placeholder)
    predictor.predict_future(hours_ahead=24)
    
    print("\n=== ANALYSIS COMPLETE ===")
    print("Check the generated visualization for detailed results.")

if __name__ == "__main__":
    main()