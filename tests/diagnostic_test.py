#!/usr/bin/env python3
"""
Diagnostic test to identify the specific error
"""

import sys
import os
import traceback

def test_imports():
    """Test all required imports"""
    print("=== DIAGNOSTIC TEST ===")
    print(f"Python version: {sys.version}")
    print(f"Current directory: {os.getcwd()}")
    
    print("\n1. Testing basic imports...")
    try:
        import pandas as pd
        print("   [OK] pandas imported successfully")
    except Exception as e:
        print(f"   [ERROR] pandas failed: {e}")
        return False
    
    try:
        import numpy as np
        print("   [OK] numpy imported successfully")
    except Exception as e:
        print(f"   [ERROR] numpy failed: {e}")
        return False
    
    try:
        import sklearn
        print("   [OK] sklearn imported successfully")
    except Exception as e:
        print(f"   [ERROR] sklearn failed: {e}")
        return False
    
    print("\n2. Testing ML model imports...")
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        print("   [OK] sklearn models imported successfully")
    except Exception as e:
        print(f"   [ERROR] sklearn models failed: {e}")
        return False
    
    print("\n3. Testing time series imports...")
    try:
        from statsmodels.tsa.arima.model import ARIMA
        print("   [OK] ARIMA imported successfully")
    except Exception as e:
        print(f"   [ERROR] ARIMA failed: {e}")
        return False
    
    try:
        import xgboost as xgb
        print("   [OK] XGBoost imported successfully")
    except Exception as e:
        print(f"   [ERROR] XGBoost failed: {e}")
        return False
    
    print("\n4. Testing TensorFlow (should fail gracefully)...")
    try:
        import tensorflow as tf
        print("   [WARNING] TensorFlow imported (unexpected)")
    except ImportError:
        print("   [OK] TensorFlow not available (expected)")
    except Exception as e:
        print(f"   [ERROR] TensorFlow error: {e}")
    
    print("\n5. Testing our main modules...")
    try:
        from enhanced_pjm_models_clean import EnhancedPJMPricePredictor
        print("   [OK] enhanced_pjm_models_clean imported successfully")
    except Exception as e:
        print(f"   [ERROR] enhanced_pjm_models_clean failed: {e}")
        traceback.print_exc()
        return False
    
    try:
        from enhanced_pjm_models_sklearn_only import EnhancedPJMSklearnPredictor
        print("   [OK] enhanced_pjm_models_sklearn_only imported successfully")
    except Exception as e:
        print(f"   [ERROR] enhanced_pjm_models_sklearn_only failed: {e}")
        traceback.print_exc()
        return False
    
    print("\n=== ALL TESTS PASSED ===")
    return True

def test_basic_functionality():
    """Test basic functionality"""
    print("\n=== BASIC FUNCTIONALITY TEST ===")
    
    try:
        from enhanced_pjm_models_clean import EnhancedPJMPricePredictor
        
        # Test initialization
        predictor = EnhancedPJMPricePredictor('da_hrl_lmps (1).csv')
        print("   [OK] Predictor initialized successfully")
        
        # Test data loading
        predictor.load_data()
        print("   [OK] Data loaded successfully")
        
        # Test feature engineering
        data = predictor.enhanced_feature_engineering()
        print("   [OK] Feature engineering completed successfully")
        
        print("\n=== BASIC FUNCTIONALITY TEST PASSED ===")
        return True
        
    except Exception as e:
        print(f"   [ERROR] Basic functionality failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Main diagnostic function"""
    print("PJM Prediction System - Diagnostic Tool")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n[FAILED] IMPORT TESTS FAILED")
        return False
    
    # Test basic functionality
    if not test_basic_functionality():
        print("\n[FAILED] FUNCTIONALITY TESTS FAILED")
        return False
    
    print("\n[SUCCESS] ALL DIAGNOSTIC TESTS PASSED!")
    print("The system is working correctly.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)