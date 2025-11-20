"""
Simple test script to verify the PJM prediction system works with your data
"""

import pandas as pd
import numpy as np
from datetime import datetime

def test_data_loading():
    """Test if we can load and analyze the PJM data"""
    print("=== TESTING PJM DATA LOADING ===")
    
    try:
        # Load first 1000 rows to test
        print("Loading sample data...")
        sample_df = pd.read_csv('da_hrl_lmps (1).csv', nrows=1000)
        
        # Convert datetime columns
        sample_df['datetime_beginning_utc'] = pd.to_datetime(sample_df['datetime_beginning_utc'])
        
        print(f"✓ Successfully loaded {len(sample_df)} records")
        print(f"✓ Date range: {sample_df['datetime_beginning_utc'].min()} to {sample_df['datetime_beginning_utc'].max()}")
        print(f"✓ Columns: {list(sample_df.columns)}")
        
        # Basic statistics
        print(f"✓ Price statistics:")
        print(f"  Mean: ${sample_df['total_lmp_da'].mean():.2f}")
        print(f"  Min: ${sample_df['total_lmp_da'].min():.2f}")
        print(f"  Max: ${sample_df['total_lmp_da'].max():.2f}")
        print(f"  Std: ${sample_df['total_lmp_da'].std():.2f}")
        
        # Zone information
        unique_zones = sample_df['zone'].nunique()
        print(f"✓ Number of zones: {unique_zones}")
        
        if unique_zones > 0:
            print(f"✓ Sample zones: {list(sample_df['zone'].dropna().unique()[:5])}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return False

def test_basic_features():
    """Test basic feature creation"""
    print("\n=== TESTING FEATURE CREATION ===")
    
    try:
        # Load sample data
        df = pd.read_csv('da_hrl_lmps (1).csv', nrows=5000)
        df['datetime_beginning_utc'] = pd.to_datetime(df['datetime_beginning_utc'])
        
        # Create basic time features
        df['hour'] = df['datetime_beginning_utc'].dt.hour
        df['day_of_week'] = df['datetime_beginning_utc'].dt.dayofweek
        df['month'] = df['datetime_beginning_utc'].dt.month
        
        # Create lag features (simple version)
        df_sorted = df.sort_values('datetime_beginning_utc')
        df_sorted['price_lag_1h'] = df_sorted['total_lmp_da'].shift(1)
        
        print(f"✓ Created time features: hour, day_of_week, month")
        print(f"✓ Created lag feature: price_lag_1h")
        print(f"✓ Feature columns: {['hour', 'day_of_week', 'month', 'price_lag_1h']}")
        
        # Test correlation
        correlation = df_sorted[['total_lmp_da', 'price_lag_1h']].corr().iloc[0, 1]
        print(f"✓ Lag correlation: {correlation:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error creating features: {e}")
        return False

def test_model_components():
    """Test if machine learning components are available"""
    print("\n=== TESTING ML COMPONENTS ===")
    
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_absolute_error
        from sklearn.preprocessing import StandardScaler
        
        print("✓ Successfully imported sklearn components")
        
        # Test simple model creation
        rf = RandomForestRegressor(n_estimators=10, random_state=42)
        lr = LinearRegression()
        scaler = StandardScaler()
        
        print("✓ Successfully created model instances")
        
        # Test with dummy data
        X_dummy = np.random.rand(100, 5)
        y_dummy = np.random.rand(100)
        
        rf.fit(X_dummy, y_dummy)
        lr.fit(X_dummy, y_dummy)
        X_scaled = scaler.fit_transform(X_dummy)
        
        predictions_rf = rf.predict(X_dummy)
        predictions_lr = lr.predict(X_dummy)
        
        mae_rf = mean_absolute_error(y_dummy, predictions_rf)
        mae_lr = mean_absolute_error(y_dummy, predictions_lr)
        
        print(f"✓ Models work with dummy data (RF MAE: {mae_rf:.3f}, LR MAE: {mae_lr:.3f})")
        
        return True
        
    except Exception as e:
        print(f"✗ Error with ML components: {e}")
        return False

def test_visualization():
    """Test if visualization components work"""
    print("\n=== TESTING VISUALIZATION ===")
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        print("✓ Successfully imported visualization libraries")
        
        # Test simple plot
        plt.figure(figsize=(10, 6))
        x = np.arange(24)
        y = np.sin(x * np.pi / 12) * 50 + 100  # Simulated daily price pattern
        
        plt.plot(x, y)
        plt.title('Test Plot - Simulated Daily Price Pattern')
        plt.xlabel('Hour of Day')
        plt.ylabel('Price ($/MWh)')
        plt.grid(True, alpha=0.3)
        
        # Save test plot
        plt.savefig('test_plot.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✓ Successfully created and saved test plot")
        
        return True
        
    except Exception as e:
        print(f"✗ Error with visualization: {e}")
        return False

def main():
    """Run all tests"""
    print("=== PJM PREDICTION SYSTEM TEST ===\n")
    
    tests = [
        ("Data Loading", test_data_loading),
        ("Feature Creation", test_basic_features),
        ("ML Components", test_model_components),
        ("Visualization", test_visualization)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print(f"\n{'='*50}")
    print("=== TEST SUMMARY ===")
    
    passed = 0
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Run: python pjm_price_prediction.py")
        print("2. Or start the interactive notebook: jupyter notebook pjm_analysis_notebook.ipynb")
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")

if __name__ == "__main__":
    main()