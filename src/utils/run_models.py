"""
Main script to run PJM price prediction models
Simplified interface for thesis work
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import models
from models import ARIMAPricePredictor, XGBoostPricePredictor, LSTMPricePredictor, EnsemblePricePredictor

class PJMModelRunner:
    """
    Main runner for PJM price prediction models
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.data = None
    
    def load_data(self, file_path='da_hrl_lmps (1).csv'):
        """
        Load PJM data
        """
        try:
            print(f"Loading data from {file_path}...")
            self.data = pd.read_csv(file_path)
            print(f"Data loaded: {len(self.data)} records")
            return True
        except FileNotFoundError:
            print(f"Data file {file_path} not found")
            return False
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False
    
    def run_arima(self, price_column='total_lmp_da', datetime_column='datetime_beginning_ept'):
        """
        Run ARIMA model
        """
        print("\n" + "="*50)
        print("RUNNING ARIMA MODEL")
        print("="*50)
        
        if self.data is None:
            print("No data loaded")
            return False
        
        try:
            # Initialize and train model
            model = ARIMAPricePredictor()
            price_series = model.prepare_data(self.data, price_column, datetime_column)
            success = model.fit(price_series)
            
            if success:
                self.models['arima'] = model
                self.results['arima'] = {
                    'metrics': model.get_metrics(),
                    'predictions': model.predictions
                }
                print("ARIMA model completed successfully")
                return True
            else:
                print("ARIMA model failed")
                return False
                
        except Exception as e:
            print(f"ARIMA model error: {str(e)}")
            return False
    
    def run_xgboost(self, price_column='total_lmp_da', datetime_column='datetime_beginning_ept'):
        """
        Run XGBoost model
        """
        print("\n" + "="*50)
        print("RUNNING XGBOOST MODEL")
        print("="*50)
        
        if self.data is None:
            print("No data loaded")
            return False
        
        try:
            # Initialize and train model
            model = XGBoostPricePredictor()
            X, y, _ = model.prepare_data(self.data, price_column, datetime_column)
            success = model.fit(X, y)
            
            if success:
                self.models['xgboost'] = model
                self.results['xgboost'] = {
                    'metrics': model.get_metrics(),
                    'feature_importance': model.get_feature_importance()
                }
                print("XGBoost model completed successfully")
                return True
            else:
                print("XGBoost model failed")
                return False
                
        except Exception as e:
            print(f"XGBoost model error: {str(e)}")
            return False
    
    def run_lstm(self, price_column='total_lmp_da', datetime_column='datetime_beginning_ept'):
        """
        Run LSTM model
        """
        print("\n" + "="*50)
        print("RUNNING LSTM MODEL")
        print("="*50)
        
        if self.data is None:
            print("No data loaded")
            return False
        
        try:
            # Initialize and train model
            model = LSTMPricePredictor()
            success = model.fit(self.data, price_column, datetime_column)
            
            if success:
                self.models['lstm'] = model
                self.results['lstm'] = {
                    'metrics': model.get_metrics()
                }
                print("LSTM model completed successfully")
                return True
            else:
                print("LSTM model failed")
                return False
                
        except Exception as e:
            print(f"LSTM model error: {str(e)}")
            return False
    
    def run_ensemble(self, price_column='total_lmp_da', datetime_column='datetime_beginning_ept'):
        """
        Run Ensemble model
        """
        print("\n" + "="*50)
        print("RUNNING ENSEMBLE MODEL")
        print("="*50)
        
        if self.data is None:
            print("No data loaded")
            return False
        
        try:
            # Initialize and train model
            model = EnsemblePricePredictor()
            success = model.fit(self.data, price_column, datetime_column)
            
            if success:
                self.models['ensemble'] = model
                self.results['ensemble'] = {
                    'metrics': model.get_model_metrics(),
                    'weights': model.get_ensemble_weights()
                }
                print("Ensemble model completed successfully")
                return True
            else:
                print("Ensemble model failed")
                return False
                
        except Exception as e:
            print(f"Ensemble model error: {str(e)}")
            return False
    
    def compare_models(self):
        """
        Compare all trained models
        """
        print("\n" + "="*50)
        print("MODEL COMPARISON")
        print("="*50)
        
        if not self.results:
            print("No models trained yet")
            return
        
        # Create comparison table
        comparison_data = []
        
        for model_name, result in self.results.items():
            if 'metrics' in result and result['metrics']:
                metrics = result['metrics']
                
                # Handle different metric formats
                if isinstance(metrics, dict):
                    row = {
                        'Model': model_name.upper(),
                        'MAE': metrics.get('test_mae', metrics.get('mae', 'N/A')),
                        'RMSE': metrics.get('test_rmse', metrics.get('rmse', 'N/A')),
                        'MAPE': metrics.get('test_mape', metrics.get('mape', 'N/A'))
                    }
                    
                    # Add model-specific metrics
                    if model_name == 'arima':
                        row['AIC'] = metrics.get('aic', 'N/A')
                    elif model_name == 'xgboost':
                        row['Train_MAE'] = metrics.get('train_mae', 'N/A')
                    
                    comparison_data.append(row)
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            print("\nModel Performance Comparison:")
            print(comparison_df.to_string(index=False))
            
            # Find best model
            if 'MAE' in comparison_df.columns:
                best_mae_idx = comparison_df['MAE'].idxmin()
                best_model = comparison_df.loc[best_mae_idx, 'Model']
                best_mae = comparison_df.loc[best_mae_idx, 'MAE']
                print(f"\nBest performing model: {best_model} (MAE: ${best_mae:.2f}/MWh)")
        else:
            print("No performance metrics available for comparison")
    
    def generate_forecast(self, model_name='ensemble', hours=24):
        """
        Generate forecast for next N hours
        """
        print(f"\nGenerating {hours}-hour forecast using {model_name.upper()} model...")
        
        if model_name not in self.models:
            print(f"Model {model_name} not available")
            return None
        
        try:
            model = self.models[model_name]
            
            if model_name == 'ensemble':
                forecast = model.forecast_next_hours(self.data, hours)
            elif model_name == 'arima':
                forecast_result = model.forecast(steps=hours)
                forecast = forecast_result['forecast'].values
            elif model_name == 'xgboost':
                forecast = model.forecast_next_hours(self.data, hours)
            elif model_name == 'lstm':
                forecast = model.forecast_next_hours(self.data, hours)
            
            print(f"Forecast generated for next {hours} hours")
            print(f"Average predicted price: ${np.mean(forecast):.2f}/MWh")
            print(f"Price range: ${np.min(forecast):.2f} - ${np.max(forecast):.2f}/MWh")
            
            return forecast
            
        except Exception as e:
            print(f"Forecast generation failed: {str(e)}")
            return None
    
    def save_results(self, filename='model_results.txt'):
        """
        Save results to file
        """
        try:
            with open(filename, 'w') as f:
                f.write("PJM ELECTRICITY PRICE PREDICTION RESULTS\n")
                f.write("=" * 50 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Model results
                for model_name, result in self.results.items():
                    f.write(f"{model_name.upper()} MODEL RESULTS\n")
                    f.write("-" * 30 + "\n")
                    
                    if 'metrics' in result and result['metrics']:
                        for metric, value in result['metrics'].items():
                            f.write(f"{metric}: {value}\n")
                    
                    if 'weights' in result:
                        f.write(f"\nEnsemble weights: {result['weights']}\n")
                    
                    f.write("\n")
            
            print(f"Results saved to {filename}")
            
        except Exception as e:
            print(f"Failed to save results: {str(e)}")
    
    def run_all_models(self):
        """
        Run all available models
        """
        print("RUNNING ALL PJM PRICE PREDICTION MODELS")
        print("=" * 60)
        
        if not self.load_data():
            return False
        
        # Run individual models
        models_to_run = [
            ('arima', self.run_arima),
            ('xgboost', self.run_xgboost),
            ('lstm', self.run_lstm),
            ('ensemble', self.run_ensemble)
        ]
        
        successful_models = []
        
        for model_name, run_function in models_to_run:
            try:
                success = run_function()
                if success:
                    successful_models.append(model_name)
            except Exception as e:
                print(f"Error running {model_name}: {str(e)}")
        
        # Compare models
        if successful_models:
            self.compare_models()
            
            # Generate sample forecast
            if 'ensemble' in successful_models:
                self.generate_forecast('ensemble', 24)
            elif successful_models:
                self.generate_forecast(successful_models[0], 24)
            
            # Save results
            self.save_results()
            
            print(f"\nCompleted analysis with {len(successful_models)} models")
            print(f"Successful models: {', '.join(successful_models)}")
        else:
            print("No models were successfully trained")
        
        return len(successful_models) > 0


def main():
    """
    Main function
    """
    print("PJM ELECTRICITY PRICE PREDICTION SYSTEM")
    print("=" * 60)
    
    runner = PJMModelRunner()
    
    # Option 1: Run all models
    runner.run_all_models()
    
    # Option 2: Run individual models (uncomment to use)
    # runner.load_data()
    # runner.run_arima()
    # runner.run_xgboost()
    # runner.run_lstm()
    # runner.run_ensemble()
    # runner.compare_models()


if __name__ == "__main__":
    main()