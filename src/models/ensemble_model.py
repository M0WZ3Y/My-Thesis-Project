"""
Ensemble Model for PJM Electricity Price Prediction
Combines multiple models for robust forecasting
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Import individual models
try:
    from .arima_model import ARIMAPricePredictor
    from .xgboost_model import XGBoostPricePredictor
    from .lstm_model import LSTMPricePredictor
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from arima_model import ARIMAPricePredictor
    from xgboost_model import XGBoostPricePredictor
    from lstm_model import LSTMPricePredictor

class EnsemblePricePredictor:
    """
    Ensemble model combining ARIMA, XGBoost, and LSTM predictions
    """
    
    def __init__(self, weights=None):
        """
        Initialize ensemble model
        
        Args:
            weights: Dictionary of model weights, e.g., {'arima': 0.3, 'xgboost': 0.4, 'lstm': 0.3}
                     If None, uses equal weights
        """
        self.weights = weights or {'arima': 0.33, 'xgboost': 0.33, 'lstm': 0.34}
        self.models = {}
        self.predictions = {}
        self.metrics = {}
        self.meta_learner = None
        
        # Initialize individual models
        self.models['arima'] = ARIMAPricePredictor()
        self.models['xgboost'] = XGBoostPricePredictor()
        self.models['lstm'] = LSTMPricePredictor()
    
    def fit(self, df, price_column='total_lmp_da', datetime_column='datetime', test_size=0.2):
        """
        Fit all models in the ensemble
        """
        print("Training ensemble models...")
        
        try:
            # Store original data for metrics calculation
            self.original_data = df.copy()
            self.price_column = price_column
            self.datetime_column = datetime_column
            
            # Fit ARIMA model
            print("\n1. Training ARIMA model...")
            arima_series = self.models['arima'].prepare_data(df, price_column, datetime_column)
            arima_success = self.models['arima'].fit(arima_series)
            
            # Fit XGBoost model
            print("\n2. Training XGBoost model...")
            X, y, _ = self.models['xgboost'].prepare_data(df, price_column, datetime_column)
            xgb_success = self.models['xgboost'].fit(X, y, test_size)
            
            # Fit LSTM model
            print("\n3. Training LSTM model...")
            lstm_success = self.models['lstm'].fit(df, price_column, datetime_column, test_size)
            
            # Check which models were successfully trained
            successful_models = []
            for model_name, success in [('arima', arima_success), 
                                       ('xgboost', xgb_success), 
                                       ('lstm', lstm_success)]:
                if success:
                    successful_models.append(model_name)
                    print(f"[OK] {model_name.upper()} model trained successfully")
                else:
                    print(f"[FAIL] {model_name.upper()} model training failed")
            
            if not successful_models:
                raise ValueError("No models were successfully trained")
            
            # Adjust weights if some models failed
            if len(successful_models) < len(self.models):
                self._adjust_weights(successful_models)
            
            # Generate predictions for ensemble training
            self._generate_ensemble_predictions(df, price_column, datetime_column)
            
            # Train meta-learner for weighted combination
            self._train_meta_learner()
            
            # Calculate ensemble metrics
            self._calculate_ensemble_metrics(price_column)
            
            print(f"\nEnsemble training completed with {len(successful_models)} models")
            print(f"Final weights: {self.weights}")
            
            return True
            
        except Exception as e:
            print(f"Ensemble training failed: {str(e)}")
            return False
    
    def _adjust_weights(self, successful_models):
        """
        Adjust weights when some models fail to train
        """
        # Keep only successful models
        new_weights = {}
        total_weight = sum(self.weights[model] for model in successful_models)
        
        for model in successful_models:
            new_weights[model] = self.weights[model] / total_weight
        
        self.weights = new_weights
        
        # Remove failed models
        failed_models = [model for model in self.models.keys() if model not in successful_models]
        for model in failed_models:
            del self.models[model]
    
    def _generate_ensemble_predictions(self, df, price_column, datetime_column):
        """
        Generate predictions from all models for ensemble training
        """
        print("Generating predictions for ensemble...")
        
        # Get predictions from each model
        for model_name, model in self.models.items():
            try:
                if model_name == 'arima':
                    # ARIMA predictions are stored in the model
                    if hasattr(model, 'predictions') and model.predictions is not None:
                        self.predictions[model_name] = model.predictions
                    else:
                        # Generate predictions if not available
                        arima_series = model.prepare_data(df, price_column, datetime_column)
                        model.fit(arima_series)
                        self.predictions[model_name] = model.predictions
                        
                elif model_name == 'xgboost':
                    # Generate XGBoost predictions
                    X, _, data_clean = model.prepare_data(df, price_column, datetime_column)
                    xgb_pred = model.predict(X)
                    self.predictions[model_name] = pd.Series(xgb_pred, index=data_clean.index)
                    
                elif model_name == 'lstm':
                    # Generate LSTM predictions on training data
                    try:
                        lstm_pred = model.predict(df, price_column, datetime_column)
                        # Create appropriate index for LSTM predictions
                        data = model.create_features(df, price_column, datetime_column)
                        data_clean = data.dropna()
                        if len(data_clean) > model.sequence_length:
                            pred_index = data_clean.index[model.sequence_length:]
                            self.predictions[model_name] = pd.Series(lstm_pred, index=pred_index)
                        else:
                            print(f"[WARN] LSTM: Not enough data for predictions")
                            continue
                    except Exception as lstm_error:
                        print(f"[WARN] LSTM prediction failed: {str(lstm_error)}")
                        continue
                
                print(f"[OK] Generated {model_name.upper()} predictions")
                
            except Exception as e:
                print(f"[FAIL] Failed to generate {model_name.upper()} predictions: {str(e)}")
    
    def _train_meta_learner(self):
        """
        Train meta-learner for optimal weight combination
        """
        print("Training meta-learner...")
        
        try:
            # Align all predictions
            common_index = None
            aligned_predictions = {}
            
            for model_name, pred in self.predictions.items():
                if common_index is None:
                    common_index = pred.index
                else:
                    common_index = common_index.intersection(pred.index)
            
            if common_index is None or len(common_index) == 0:
                print("No common predictions found for meta-learner training")
                return
            
            # Align predictions to common index
            for model_name, pred in self.predictions.items():
                aligned_predictions[model_name] = pred.loc[common_index]
            
            # Prepare meta-features
            meta_X = pd.DataFrame(aligned_predictions)
            
            # Get actual values for meta-training
            if hasattr(self, 'original_data'):
                # Extract actual values for the common index
                data_copy = self.original_data.copy()
                data_copy[self.datetime_column] = pd.to_datetime(data_copy[self.datetime_column])
                data_copy = data_copy.set_index(self.datetime_column)
                actual_values = data_copy[self.price_column].loc[common_index]
                
                # Remove any NaN values
                valid_mask = ~(actual_values.isna() | meta_X.isna().any(axis=1))
                if valid_mask.sum() > 0:
                    meta_X_clean = meta_X[valid_mask]
                    meta_y_clean = actual_values[valid_mask]
                    
                    # Train meta-learner
                    self.meta_learner = LinearRegression()
                    self.meta_learner.fit(meta_X_clean, meta_y_clean)
                    
                    # Update weights based on meta-learner coefficients
                    if hasattr(self.meta_learner, 'coef_'):
                        # Normalize coefficients to sum to 1
                        coef = np.abs(self.meta_learner.coef_)
                        if coef.sum() > 0:
                            new_weights = {}
                            for i, model_name in enumerate(meta_X_clean.columns):
                                new_weights[model_name] = coef[i] / coef.sum()
                            self.weights.update(new_weights)
                    
                    print("[OK] Meta-learner training completed")
                    print(f"[OK] Updated weights: {self.weights}")
                else:
                    print("No valid data for meta-learner training")
            else:
                print("No original data available for meta-learner training")
            
        except Exception as e:
            print(f"Meta-learner training failed: {str(e)}")
    
    def _calculate_ensemble_metrics(self, price_column):
        """
        Calculate ensemble performance metrics
        """
        try:
            # Align all predictions
            common_index = None
            aligned_predictions = {}
            
            for model_name, pred in self.predictions.items():
                if common_index is None:
                    common_index = pred.index
                else:
                    common_index = common_index.intersection(pred.index)
            
            if common_index is None or len(common_index) == 0:
                print("No common predictions found for metrics calculation")
                return
            
            # Align predictions and calculate weighted ensemble
            ensemble_pred = pd.Series(0, index=common_index)
            
            for model_name, pred in self.predictions.items():
                if model_name in self.weights:
                    aligned_pred = pred.loc[common_index]
                    ensemble_pred += aligned_pred * self.weights[model_name]
            
            # Get actual values
            if hasattr(self, 'original_data'):
                data_copy = self.original_data.copy()
                data_copy[self.datetime_column] = pd.to_datetime(data_copy[self.datetime_column])
                data_copy = data_copy.set_index(self.datetime_column)
                actual_values = data_copy[price_column].loc[common_index]
                
                # Remove NaN values
                valid_mask = ~(actual_values.isna() | ensemble_pred.isna())
                if valid_mask.sum() > 0:
                    actual_clean = actual_values[valid_mask]
                    ensemble_clean = ensemble_pred[valid_mask]
                    
                    # Calculate metrics
                    self.metrics['ensemble'] = {
                        'mae': mean_absolute_error(actual_clean, ensemble_clean),
                        'rmse': np.sqrt(mean_squared_error(actual_clean, ensemble_clean)),
                        'mape': np.mean(np.abs((actual_clean - ensemble_clean) / actual_clean)) * 100
                    }
                    
                    print(f"Ensemble MAE: ${self.metrics['ensemble']['mae']:.2f}/MWh")
                    print(f"Ensemble RMSE: ${self.metrics['ensemble']['rmse']:.2f}/MWh")
                    print(f"Ensemble MAPE: {self.metrics['ensemble']['mape']:.2f}%")
            
            # Store ensemble predictions
            self.predictions['ensemble'] = ensemble_pred
            
            print("[OK] Ensemble predictions and metrics calculated")
            
        except Exception as e:
            print(f"Ensemble metrics calculation failed: {str(e)}")
    
    def predict(self, df, price_column='total_lmp_da', datetime_column='datetime'):
        """
        Generate ensemble predictions
        """
        if not self.models:
            raise ValueError("Models must be fitted before prediction")
        
        print("Generating ensemble predictions...")
        
        ensemble_predictions = {}
        
        for model_name, model in self.models.items():
            try:
                if model_name == 'arima':
                    # ARIMA forecasting
                    forecast_result = model.forecast(steps=len(df))
                    if isinstance(forecast_result, dict) and 'forecast' in forecast_result:
                        ensemble_predictions[model_name] = forecast_result['forecast']
                    else:
                        ensemble_predictions[model_name] = forecast_result
                elif model_name == 'xgboost':
                    # XGBoost predictions
                    X, _, _ = model.prepare_data(df, price_column, datetime_column)
                    if len(X) > 0:
                        pred = model.predict(X)
                        ensemble_predictions[model_name] = pred
                elif model_name == 'lstm':
                    # LSTM predictions
                    try:
                        pred = model.predict(df, price_column, datetime_column)
                        if len(pred) > 0:
                            ensemble_predictions[model_name] = pred
                    except Exception as lstm_pred_error:
                        print(f"[WARN] LSTM prediction issue: {str(lstm_pred_error)}")
                        # Skip LSTM for this prediction if it fails
                        continue
                
            except Exception as e:
                print(f"[FAIL] Failed to generate {model_name.upper()} predictions: {str(e)}")
                continue
        
        # Calculate weighted ensemble
        if ensemble_predictions:
            # Find minimum length to align predictions
            min_length = min(len(pred) for pred in ensemble_predictions.values())
            
            ensemble_final = np.zeros(min_length)
            total_weight = 0
            
            for model_name, pred in ensemble_predictions.items():
                if model_name in self.weights:
                    # Ensure pred is numpy array and handle different types
                    if isinstance(pred, pd.Series):
                        pred_array = pred.values
                    else:
                        pred_array = np.array(pred)
                    
                    ensemble_final += pred_array[:min_length] * self.weights[model_name]
                    total_weight += self.weights[model_name]
            
            # Normalize by total weight
            if total_weight > 0:
                ensemble_final /= total_weight
            
            return ensemble_final
        else:
            raise ValueError("No predictions could be generated")
    
    def forecast_next_hours(self, df, hours=24, price_column='total_lmp_da', datetime_column='datetime'):
        """
        Forecast next N hours using ensemble
        """
        print(f"Forecasting next {hours} hours with ensemble...")
        
        forecasts = {}
        
        for model_name, model in self.models.items():
            try:
                if model_name == 'arima':
                    forecast_result = model.forecast(steps=hours)
                    if isinstance(forecast_result, dict) and 'forecast' in forecast_result:
                        forecasts[model_name] = np.array(forecast_result['forecast'])
                    else:
                        forecasts[model_name] = np.array(forecast_result)
                elif model_name == 'xgboost':
                    forecast = model.forecast_next_hours(df, hours)
                    forecasts[model_name] = forecast
                elif model_name == 'lstm':
                    try:
                        forecast = model.forecast_next_hours(df, hours, price_column, datetime_column)
                        forecasts[model_name] = forecast
                    except Exception as lstm_forecast_error:
                        print(f"[WARN] LSTM forecast issue: {str(lstm_forecast_error)}")
                        # Skip LSTM for this forecast if it fails
                        continue
                
            except Exception as e:
                print(f"[FAIL] Failed to generate {model_name.upper()} forecast: {str(e)}")
                continue
        
        # Calculate weighted ensemble forecast
        if forecasts:
            min_length = min(len(forecast) for forecast in forecasts.values())
            
            ensemble_forecast = np.zeros(min_length)
            total_weight = 0
            
            for model_name, forecast in forecasts.items():
                if model_name in self.weights:
                    # Ensure forecast is numpy array
                    if isinstance(forecast, list):
                        forecast_array = np.array(forecast)
                    else:
                        forecast_array = forecast
                    
                    ensemble_forecast += forecast_array[:min_length] * self.weights[model_name]
                    total_weight += self.weights[model_name]
            
            if total_weight > 0:
                ensemble_forecast /= total_weight
            
            return ensemble_forecast
        else:
            raise ValueError("No forecasts could be generated")
    
    def get_model_metrics(self):
        """
        Get individual model metrics
        """
        metrics = {}
        for model_name, model in self.models.items():
            try:
                metrics[model_name] = model.get_metrics()
            except Exception as e:
                print(f"Failed to get metrics for {model_name}: {str(e)}")
                metrics[model_name] = {}
        
        # Add ensemble metrics if available
        if hasattr(self, 'metrics') and 'ensemble' in self.metrics:
            metrics['ensemble'] = self.metrics['ensemble']
        
        return metrics
    
    def get_ensemble_weights(self):
        """
        Get current ensemble weights
        """
        return self.weights.copy()
    
    def update_weights(self, new_weights):
        """
        Update ensemble weights
        """
        # Validate weights
        if not isinstance(new_weights, dict):
            raise ValueError("Weights must be a dictionary")
        
        # Check if all models exist
        for model_name in new_weights:
            if model_name not in self.models:
                print(f"Warning: Model {model_name} not found in ensemble, skipping")
        
        # Filter weights to only include existing models
        valid_weights = {k: v for k, v in new_weights.items() if k in self.models}
        
        if not valid_weights:
            raise ValueError("No valid models found in weights")
        
        # Normalize weights
        total_weight = sum(valid_weights.values())
        if total_weight == 0:
            raise ValueError("Sum of weights cannot be zero")
        
        self.weights = {k: v/total_weight for k, v in valid_weights.items()}
        print(f"Updated ensemble weights: {self.weights}")


def main():
    """
    Example usage of Ensemble model
    """
    print("Ensemble Price Prediction Model")
    print("=" * 40)
    
    print("Combining ARIMA, XGBoost, and LSTM models")
    print("\nModel ready for training with PJM price data")
    print("Required columns: datetime, total_lmp_da")
    print("\nUsage:")
    print("1. Load your PJM data")
    print("2. Initialize EnsemblePricePredictor()")
    print("3. Call fit() with your data")
    print("4. Use predict() or forecast_next_hours() for ensemble predictions")
    print("\nFeatures:")
    print("- Automatic weight adjustment if models fail")
    print("- Meta-learner for optimal combination")
    print("- Robust forecasting with multiple models")


if __name__ == "__main__":
    main()