"""
Comprehensive test script to verify all required modules and run the PJM prediction system
"""

import sys
import importlib

def test_module(module_name, min_version=None):
    """Test if a module is installed and optionally check version"""
    try:
        module = importlib.import_module(module_name)
        if hasattr(module, '__version__'):
            version = module.__version__
            print(f"[OK] {module_name}: {version}")
            if min_version and version < min_version:
                print(f"  [WARN] Version {version} is older than recommended {min_version}")
        else:
            print(f"[OK] {module_name}: Installed (version unknown)")
        return True
    except ImportError as e:
        print(f"[FAIL] {module_name}: Not installed - {e}")
        return False

def test_all_modules():
    """Test all required modules"""
    print("=== TESTING REQUIRED MODULES ===\n")
    
    required_modules = {
        'pandas': '1.3.0',
        'numpy': '1.20.0', 
        'matplotlib': '3.5.0',
        'sklearn': None,  # scikit-learn imports as sklearn
        'seaborn': '0.11.0',
        'jupyter': None
    }
    
    results = {}
    
    for module, min_version in required_modules.items():
        results[module] = test_module(module, min_version)
    
    # Special check for scikit-learn
    try:
        import sklearn
        print(f"[OK] scikit-learn: {sklearn.__version__}")
        results['scikit-learn'] = True
    except ImportError:
        print("[FAIL] scikit-learn: Not installed")
        results['scikit-learn'] = False
    
    return results

def test_data_access():
    """Test if we can access the PJM data file"""
    print("\n=== TESTING DATA ACCESS ===\n")
    
    try:
        import pandas as pd
        
        # Test reading a small sample of the data
        sample_df = pd.read_csv('da_hrl_lmps (1).csv', nrows=5)
        print(f"[OK] Data file accessible: {len(sample_df)} sample rows loaded")
        print(f"[OK] Columns: {list(sample_df.columns)}")
        
        # Test datetime conversion
        sample_df['datetime_beginning_utc'] = pd.to_datetime(sample_df['datetime_beginning_utc'])
        print(f"[OK] Datetime conversion works")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Data access failed: {e}")
        return False

def test_basic_functionality():
    """Test basic ML functionality"""
    print("\n=== TESTING BASIC FUNCTIONALITY ===\n")
    
    try:
        import pandas as pd
        import numpy as np
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_absolute_error
        
        # Create dummy data
        X = np.random.rand(100, 5)
        y = np.random.rand(100)
        
        # Test models
        rf = RandomForestRegressor(n_estimators=10, random_state=42)
        lr = LinearRegression()
        
        rf.fit(X, y)
        lr.fit(X, y)
        
        predictions_rf = rf.predict(X)
        predictions_lr = lr.predict(X)
        
        mae_rf = mean_absolute_error(y, predictions_rf)
        mae_lr = mean_absolute_error(y, predictions_lr)
        
        print(f"[OK] Random Forest works (MAE: {mae_rf:.3f})")
        print(f"[OK] Linear Regression works (MAE: {mae_lr:.3f})")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Basic functionality failed: {e}")
        return False

def test_visualization():
    """Test visualization capabilities"""
    print("\n=== TESTING VISUALIZATION ===\n")
    
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Create a simple test plot
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        
        plt.figure(figsize=(8, 4))
        plt.plot(x, y)
        plt.title('Test Plot - Sine Wave')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True, alpha=0.3)
        
        # Save test plot
        plt.savefig('test_visualization.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("[OK] Matplotlib visualization works")
        print("[OK] Test plot saved as 'test_visualization.png'")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Visualization failed: {e}")
        return False

def run_pjm_system_test():
    """Test the actual PJM prediction system"""
    print("\n=== TESTING PJM PREDICTION SYSTEM ===\n")
    
    try:
        # Import the PJM system
        import sys
        sys.path.append('.')
        
        # Test if we can import the main class
        from pjm_price_prediction import PJMPricePredictor
        
        print("[OK] PJM prediction system imports successfully")
        
        # Test initialization
        predictor = PJMPricePredictor('da_hrl_lmps (1).csv')
        print("[OK] PJM predictor initializes successfully")
        
        # Test data loading (with small sample)
        print("Testing data loading with sample...")
        predictor.load_data()
        print("[OK] Data loading works")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] PJM system test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=== COMPREHENSIVE PJM SYSTEM TEST ===\n")
    
    # Test all modules
    module_results = test_all_modules()
    
    # Check if all required modules are available
    required_available = all([
        module_results.get('pandas', False),
        module_results.get('numpy', False),
        module_results.get('matplotlib', False),
        module_results.get('scikit-learn', False)
    ])
    
    if not required_available:
        print("\n[ERROR] Some required modules are missing. Please install them first:")
        print("pip install pandas numpy matplotlib scikit-learn seaborn jupyter")
        return False
    
    # Test data access
    data_ok = test_data_access()
    
    # Test basic functionality
    functionality_ok = test_basic_functionality()
    
    # Test visualization
    viz_ok = test_visualization()
    
    # Test PJM system
    pjm_ok = run_pjm_system_test()
    
    # Summary
    print("\n" + "="*60)
    print("=== TEST SUMMARY ===")
    print(f"Modules: {'[OK] PASS' if required_available else '[FAIL] FAIL'}")
    print(f"Data Access: {'[OK] PASS' if data_ok else '[FAIL] FAIL'}")
    print(f"Basic Functionality: {'[OK] PASS' if functionality_ok else '[FAIL] FAIL'}")
    print(f"Visualization: {'[OK] PASS' if viz_ok else '[FAIL] FAIL'}")
    print(f"PJM System: {'[OK] PASS' if pjm_ok else '[FAIL] FAIL'}")
    
    all_passed = all([required_available, data_ok, functionality_ok, viz_ok, pjm_ok])
    
    if all_passed:
        print(f"\n[SUCCESS] ALL TESTS PASSED! The system is ready to use.")
        print("\nNext steps:")
        print("1. Run: python pjm_price_prediction.py")
        print("2. Or start: jupyter notebook pjm_analysis_notebook.ipynb")
    else:
        print(f"\n[WARNING] Some tests failed. Check the errors above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)