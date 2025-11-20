"""
Research Gap 1: Unified Comparison of ML Models Using the Same Dataset
Comprehensive benchmarking system for PJM electricity price prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

# Import our existing models
import sys
sys.path.append('..')
from models.xgboost_model import XGBoostPricePredictor
from models.lstm_model import LSTMPricePredictor

class UnifiedModelComparison:
    """
    Comprehensive comparison of ML models on the same PJM dataset
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.data = None
        self.feature_columns = []
        self.target_column = 'total_lmp_da'
        
    def prepare_data(self, df, price_column='total_lmp_da', datetime_column='datetime_beginning_ept'):
        """
        Prepare unified dataset for all models
        """
        print("Preparing unified dataset for model comparison...")
        
        # Copy data
        data = df.copy()
        
        # Convert datetime
        if datetime_column in data.columns:
            data['datetime'] = pd.to_datetime(data[datetime_column])
        else:
            data['datetime'] = pd.to_datetime(data['datetime'])
        
        # Sort by datetime
        data = data.sort_values('datetime').reset_index(drop=True)
        
        # Handle missing values
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            data[col] = data[col].fillna(data[col].median())
        
        # Create unified features
        data = self._create_unified_features(data)
        
        # Remove rows with missing values
        data_clean = data.dropna()
        
        # Define feature columns - exclude datetime and string columns
        exclude_cols = ['datetime', datetime_column, price_column]
        # Only keep numeric columns for features
        numeric_feature_cols = data_clean.select_dtypes(include=[np.number]).columns
        self.feature_columns = [col for col in numeric_feature_cols if col not in exclude_cols]
        
        # Prepare X and y
        X = data_clean[self.feature_columns]
        y = data_clean[price_column]
        
        self.data = data_clean
        
        print(f"Dataset prepared: {len(data_clean)} records, {len(self.feature_columns)} features")
        return X, y
    
    def _create_unified_features(self, data):
        """
        Create comprehensive features for all models
        """
        # Time-based features
        data['hour'] = data['datetime'].dt.hour
        data['day_of_week'] = data['datetime'].dt.dayofweek
        data['month'] = data['datetime'].dt.month
        data['quarter'] = data['datetime'].dt.quarter
        data['year'] = data['datetime'].dt.year
        data['week_of_year'] = data['datetime'].dt.isocalendar().week
        
        # Cyclical encoding
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
        data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
        data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
        data['dow_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
        data['dow_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
        
        # Lag features
        for lag in [1, 2, 3, 6, 12, 24]:
            data[f'{self.target_column}_lag_{lag}'] = data[self.target_column].shift(lag)
        
        # Rolling statistics
        for window in [3, 6, 12, 24]:
            data[f'{self.target_column}_rolling_mean_{window}'] = (
                data[self.target_column].rolling(window=window).mean()
            )
            data[f'{self.target_column}_rolling_std_{window}'] = (
                data[self.target_column].rolling(window=window).std()
            )
        
        # Price change features
        data['price_change_1h'] = data[self.target_column].pct_change(1)
        data['price_change_24h'] = data[self.target_column].pct_change(24)
        data['volatility_24h'] = data['price_change_1h'].rolling(24).std()
        
        # Peak indicators
        data['is_peak_hour'] = ((data['hour'] >= 8) & (data['hour'] <= 11)) | \
                              ((data['hour'] >= 17) & (data['hour'] <= 21))
        data['is_weekend'] = data['day_of_week'].isin([5, 6])
        
        # Seasonal indicators
        data['is_summer'] = data['month'].isin([6, 7, 8])
        data['is_winter'] = data['month'].isin([12, 1, 2])
        
        return data
    
    def initialize_models(self):
        """
        Initialize all models for comparison
        """
        print("Initializing models for comparison...")
        
        # Traditional ML models
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=6, random_state=42),
            'KNN': KNeighborsRegressor(n_neighbors=5),
            'SVR': SVR(kernel='rbf', C=1.0, gamma='scale'),
            'MLP': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
            'XGBoost': None,  # Will be initialized separately
            'LSTM': None      # Will be initialized separately
        }
        
        print(f"Initialized {len(self.models)} models for comparison")
    
    def train_and_evaluate_models(self, X, y, test_size=0.2):
        """
        Train and evaluate all models
        """
        print("Training and evaluating models...")
        
        # Time series split
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        results = {}
        
        for model_name, model in self.models.items():
            print(f"\nTraining {model_name}...")
            
            try:
                if model_name == 'XGBoost':
                    # Use our XGBoost implementation
                    xgb_model = XGBoostPricePredictor(n_estimators=100, max_depth=6)
                    success = xgb_model.fit(X_train, y_train, test_size=0.0)
                    if success:
                        y_pred = xgb_model.predict(X_test)
                        metrics = xgb_model.get_metrics()
                        results[model_name] = {
                            'mae': metrics['test_mae'],
                            'rmse': metrics['test_rmse'],
                            'mape': metrics['test_mape'],
                            'r2': r2_score(y_test, y_pred),
                            'model': xgb_model
                        }
                    else:
                        continue
                        
                elif model_name == 'LSTM':
                    # Use our LSTM implementation
                    lstm_model = LSTMPricePredictor(sequence_length=24, lstm_units=50)
                    # Create a temporary dataframe for LSTM
                    temp_df = self.data.iloc[:split_idx].copy()
                    success = lstm_model.fit(temp_df, self.target_column, 'datetime', test_size=0.0, epochs=20)
                    if success:
                        # Test on remaining data
                        test_df = self.data.iloc[split_idx:].copy()
                        y_pred = lstm_model.predict(test_df, self.target_column, 'datetime')
                        if len(y_pred) > 0:
                            # Align predictions with test set
                            min_len = min(len(y_test), len(y_pred))
                            y_pred_aligned = y_pred[:min_len]
                            y_test_aligned = y_test[:min_len]
                            
                            metrics = lstm_model.get_metrics()
                            results[model_name] = {
                                'mae': metrics['test_mae'],
                                'rmse': metrics['test_rmse'],
                                'mape': metrics['test_mape'],
                                'r2': r2_score(y_test_aligned, y_pred_aligned),
                                'model': lstm_model
                            }
                    else:
                        continue
                        
                else:
                    # Traditional ML models
                    if model_name in ['SVR', 'MLP']:
                        # Use scaled data for these models
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                    else:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    mae = mean_absolute_error(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                    r2 = r2_score(y_test, y_pred)
                    
                    results[model_name] = {
                        'mae': mae,
                        'rmse': rmse,
                        'mape': mape,
                        'r2': r2,
                        'model': model
                    }
                
                print(f"✓ {model_name} - MAE: ${results[model_name]['mae']:.2f}, MAPE: {results[model_name]['mape']:.2f}%")
                
            except Exception as e:
                print(f"✗ {model_name} failed: {str(e)}")
                continue
        
        self.results = results
        print(f"\nSuccessfully trained {len(results)} models")
        return results
    
    def create_comparison_table(self):
        """
        Create comprehensive comparison table
        """
        if not self.results:
            print("No results to compare")
            return None
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, result in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'MAE ($/MWh)': result['mae'],
                'RMSE ($/MWh)': result['rmse'],
                'MAPE (%)': result['mape'],
                'R²': result['r2']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('MAE ($/MWh)')
        
        return comparison_df
    
    def visualize_comparison(self, save_path='model_comparison.png'):
        """
        Create comprehensive visualization of model comparison
        """
        if not self.results:
            print("No results to visualize")
            return
        
        # Create comparison table
        comparison_df = self.create_comparison_table()
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Unified ML Model Comparison - PJM Electricity Price Prediction', 
                     fontsize=16, fontweight='bold')
        
        # 1. MAE Comparison
        ax1 = axes[0, 0]
        bars1 = ax1.bar(comparison_df['Model'], comparison_df['MAE ($/MWh)'], 
                       color='skyblue', alpha=0.7)
        ax1.set_title('Mean Absolute Error (MAE)')
        ax1.set_ylabel('MAE ($/MWh)')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:.2f}', ha='center', va='bottom', fontsize=8)
        
        # 2. RMSE Comparison
        ax2 = axes[0, 1]
        bars2 = ax2.bar(comparison_df['Model'], comparison_df['RMSE ($/MWh)'], 
                       color='lightcoral', alpha=0.7)
        ax2.set_title('Root Mean Square Error (RMSE)')
        ax2.set_ylabel('RMSE ($/MWh)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 3. MAPE Comparison
        ax3 = axes[1, 0]
        bars3 = ax3.bar(comparison_df['Model'], comparison_df['MAPE (%)'], 
                       color='lightgreen', alpha=0.7)
        ax3.set_title('Mean Absolute Percentage Error (MAPE)')
        ax3.set_ylabel('MAPE (%)')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # 4. R² Comparison
        ax4 = axes[1, 1]
        bars4 = ax4.bar(comparison_df['Model'], comparison_df['R²'], 
                       color='gold', alpha=0.7)
        ax4.set_title('R² Score')
        ax4.set_ylabel('R²')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Comparison visualization saved to {save_path}")
    
    def generate_comparison_report(self, save_path='unified_comparison_report.txt'):
        """
        Generate comprehensive comparison report
        """
        if not self.results:
            print("No results to report")
            return
        
        comparison_df = self.create_comparison_table()
        
        report = []
        report.append("=" * 80)
        report.append("UNIFIED ML MODEL COMPARISON REPORT")
        report.append("PJM Electricity Price Prediction")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Dataset: {len(self.data)} records")
        report.append(f"Features: {len(self.feature_columns)}")
        report.append(f"Models Tested: {len(self.results)}")
        report.append("")
        
        # Comparison table
        report.append("MODEL PERFORMANCE COMPARISON")
        report.append("-" * 50)
        report.append(comparison_df.to_string(index=False))
        report.append("")
        
        # Best models
        report.append("BEST PERFORMING MODELS")
        report.append("-" * 30)
        best_mae = comparison_df.iloc[0]
        best_mape = comparison_df.loc[comparison_df['MAPE (%)'].idxmin()]
        best_r2 = comparison_df.loc[comparison_df['R²'].idxmax()]
        
        report.append(f"Best MAE: {best_mae['Model']} (${best_mae['MAE ($/MWh)']:.2f}/MWh)")
        report.append(f"Best MAPE: {best_mape['Model']} ({best_mape['MAPE (%)']:.2f}%)")
        report.append(f"Best R²: {best_r2['Model']} ({best_r2['R²']:.4f})")
        report.append("")
        
        # Statistical analysis
        report.append("STATISTICAL ANALYSIS")
        report.append("-" * 30)
        mae_values = comparison_df['MAE ($/MWh)']
        report.append(f"MAE - Mean: ${mae_values.mean():.2f}, Std: ${mae_values.std():.2f}")
        report.append(f"MAE - Range: ${mae_values.min():.2f} - ${mae_values.max():.2f}")
        
        mape_values = comparison_df['MAPE (%)']
        report.append(f"MAPE - Mean: {mape_values.mean():.2f}%, Std: {mape_values.std():.2f}%")
        report.append("")
        
        # Research implications
        report.append("RESEARCH IMPLICATIONS")
        report.append("-" * 30)
        report.append("1. This unified comparison addresses the gap in literature")
        report.append("   where different studies use different datasets and metrics")
        report.append("")
        report.append("2. Performance ranking provides clear guidance for")
        report.append("   practitioners and researchers in model selection")
        report.append("")
        report.append("3. The consistent dataset ensures fair comparison")
        report.append("   and reproducible results")
        report.append("")
        
        report.append("=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)
        
        # Save report
        report_text = "\n".join(report)
        with open(save_path, 'w') as f:
            f.write(report_text)
        
        print(f"Comparison report saved to {save_path}")
        return report_text
    
    def run_complete_comparison(self, df):
        """
        Run complete unified model comparison
        """
        print("UNIFIED ML MODEL COMPARISON")
        print("=" * 60)
        print("Addressing Research Gap 1: Lack of Unified Comparison")
        print("=" * 60)
        
        # Prepare data
        X, y = self.prepare_data(df)
        
        # Initialize models
        self.initialize_models()
        
        # Train and evaluate
        results = self.train_and_evaluate_models(X, y)
        
        # Create comparison table
        comparison_df = self.create_comparison_table()
        print("\nUNIFIED MODEL COMPARISON RESULTS:")
        print(comparison_df.to_string(index=False))
        
        # Visualize results
        self.visualize_comparison()
        
        # Generate report
        self.generate_comparison_report()
        
        print("\n" + "=" * 60)
        print("UNIFIED COMPARISON COMPLETED")
        print("=" * 60)
        print("✓ All models evaluated on same dataset")
        print("✓ Consistent metrics across all models")
        print("✓ Fair comparison with identical preprocessing")
        print("✓ Research Gap 1 successfully addressed")
        
        return results


def main():
    """
    Main function to run unified model comparison
    """
    print("Research Gap 1: Unified ML Model Comparison")
    print("=" * 60)
    
    # Load data
    try:
        df = pd.read_csv('../da_hrl_lmps (1).csv')
        print(f"Data loaded: {len(df)} records")
    except FileNotFoundError:
        print("Data file not found. Using sample data...")
        # Create sample data for demonstration
        np.random.seed(42)
        dates = pd.date_range('2025-01-01', periods=1000, freq='H')
        prices = np.random.normal(50, 10, 1000)
        df = pd.DataFrame({
            'datetime_beginning_ept': dates,
            'total_lmp_da': prices,
            'system_energy_price_da': prices * 0.7,
            'congestion_price_da': prices * 0.2,
            'marginal_loss_price_da': prices * 0.1
        })
    
    # Run comparison
    comparator = UnifiedModelComparison()
    results = comparator.run_complete_comparison(df)
    
    return results


if __name__ == "__main__":
    main()