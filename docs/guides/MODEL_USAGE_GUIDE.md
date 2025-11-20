# PJM Electricity Price Prediction Models - Usage Guide

## 🎯 Overview

This guide provides comprehensive instructions for using the PJM electricity price prediction models developed for thesis research. The system includes individual models (ARIMA, XGBoost, LSTM) and an ensemble approach.

## 📁 Project Structure

```
├── models/
│   ├── __init__.py              # Package initialization
│   ├── arima_model.py           # ARIMA time series model
│   ├── xgboost_model.py         # XGBoost gradient boosting model
│   ├── lstm_model.py            # LSTM neural network (with fallback)
│   └── ensemble_model.py        # Ensemble combination of models
├── run_models.py                # Main execution script
├── test_individual_models.py    # Model testing with sample data
├── requirements.txt             # Python dependencies
└── MODEL_USAGE_GUIDE.md         # This guide
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Test Models with Sample Data

```bash
python test_individual_models.py
```

This will test all models with synthetic data to verify installation.

### 3. Run with Your PJM Data

```bash
python run_models.py
```

## 📊 Data Requirements

### Required Columns
- `datetime` or `datetime_beginning_ept`: Timestamp
- `total_lmp_da`: Day-ahead total locational marginal price ($/MWh)

### Optional Columns (for enhanced features)
- `system_energy_price_da`: System energy component
- `congestion_price_da`: Congestion component  
- `marginal_loss_price_da`: Marginal loss component

### Data Format
```csv
datetime_beginning_ept,total_lmp_da,system_energy_price_da,congestion_price_da,marginal_loss_price_da
1/1/2025 5:00:00 AM,45.23,31.67,8.45,5.11
1/1/2025 6:00:00 AM,47.89,33.52,9.12,5.25
...
```

## 🔧 Individual Model Usage

### ARIMA Model

```python
from models import ARIMAPricePredictor

# Initialize model
model = ARIMAPricePredictor(order=(1,1,1), seasonal_order=(1,1,1,24))

# Prepare data
price_series = model.prepare_data(df, 'total_lmp_da', 'datetime')

# Train model
success = model.fit(price_series)

# Generate forecast
forecast = model.forecast(steps=24)
```

### XGBoost Model

```python
from models import XGBoostPricePredictor

# Initialize model
model = XGBoostPricePredictor(n_estimators=100, max_depth=6)

# Prepare data
X, y, _ = model.prepare_data(df, 'total_lmp_da', 'datetime')

# Train model
success = model.fit(X, y, test_size=0.2)

# Make predictions
predictions = model.predict(X_test)

# Get feature importance
feature_importance = model.get_feature_importance(10)
```

### LSTM Model

```python
from models import LSTMPricePredictor

# Initialize model
model = LSTMPricePredictor(sequence_length=24, lstm_units=50)

# Train model
success = model.fit(df, 'total_lmp_da', 'datetime', epochs=50)

# Generate forecast
forecast = model.forecast_next_hours(df, hours=24)
```

## 🎯 Ensemble Model

```python
from models import EnsemblePricePredictor

# Initialize ensemble
ensemble = EnsemblePricePredictor(weights={'arima': 0.3, 'xgboost': 0.4, 'lstm': 0.3})

# Train all models
success = ensemble.fit(df, 'total_lmp_da', 'datetime')

# Generate ensemble forecast
forecast = ensemble.forecast_next_hours(df, hours=24)

# Get individual model metrics
metrics = ensemble.get_model_metrics()

# Update weights
ensemble.update_weights({'arima': 0.25, 'xgboost': 0.5, 'lstm': 0.25})
```

## 📈 Model Performance Metrics

All models provide the following metrics:

- **MAE**: Mean Absolute Error ($/MWh)
- **RMSE**: Root Mean Square Error ($/MWh)
- **MAPE**: Mean Absolute Percentage Error (%)
- **AIC/BIC**: Information criteria (ARIMA only)

## 🛠️ Advanced Configuration

### ARIMA Parameters
```python
# Custom ARIMA orders
model = ARIMAPricePredictor(
    order=(2,1,2),           # Non-seasonal (p,d,q)
    seasonal_order=(1,1,1,24) # Seasonal (P,D,Q,s)
)
```

### XGBoost Parameters
```python
# Custom XGBoost settings
model = XGBoostPricePredictor(
    n_estimators=200,        # Number of trees
    max_depth=8,             # Tree depth
    learning_rate=0.05       # Learning rate
)
```

### LSTM Parameters
```python
# Custom LSTM settings
model = LSTMPricePredictor(
    sequence_length=48,      # Input sequence length
    lstm_units=100,          # LSTM units
    dropout_rate=0.3         # Dropout rate
)
```

## 🔍 Troubleshooting

### Common Issues

1. **Memory Error with ARIMA**
   - Reduce dataset size or use simpler ARIMA orders
   - Use `order=(1,1,1)` instead of complex orders

2. **TensorFlow Not Available**
   - LSTM automatically falls back to Gradient Boosting
   - Install TensorFlow: `pip install tensorflow`

3. **Data Format Issues**
   - Ensure datetime column is properly formatted
   - Check for missing values in price columns

4. **Feature Engineering Errors**
   - Models automatically handle missing features
   - Check data types are numeric for price columns

### Performance Optimization

1. **For Large Datasets**
   ```python
   # Use subset for initial testing
   sample_data = df.head(10000)
   ```

2. **For Faster Training**
   ```python
   # Reduce model complexity
   model = XGBoostPricePredictor(n_estimators=50, max_depth=4)
   ```

## 📊 Thesis Integration

### Required Models for Thesis
✅ **ARIMA**: Classical time series approach  
✅ **XGBoost**: Machine learning ensemble method  
✅ **LSTM**: Deep learning sequential model  
✅ **Ensemble**: Combined approach for robustness  

### Key Features for Research
- **Multi-resolution analysis**: Hourly and daily predictions
- **Volatility analysis**: Price spike detection
- **Component analysis**: Energy, congestion, loss components
- **Feature importance**: Model interpretability
- **Cross-validation**: Time series appropriate splitting

### Academic Contributions
1. **Adaptive feature engineering** for limited data scenarios
2. **Component-based price analysis** for market understanding
3. **Volatility-focused analysis** for price spike prediction
4. **Multi-model ensemble** for robust forecasting

## 📝 Example Results

### Sample Model Performance
```
Model Performance Comparison:
    Model    MAE    RMSE    MAPE
   ARIMA   4.73    6.27   9.81%
 XGBOOST   0.10    0.15   0.19%
     LSTM   4.42    5.47   8.61%
```

### Feature Importance (XGBoost)
```
                feature  importance
     congestion_price_da    0.497
  system_energy_price_da    0.461
           is_peak_hour    0.019
total_lmp_da_rolling_std_3    0.005
```

## 🚀 Next Steps

1. **Load your PJM dataset** into the working directory
2. **Run the models** using `python run_models.py`
3. **Analyze results** in the generated output files
4. **Fine-tune parameters** based on your specific data characteristics
5. **Generate forecasts** for your required time horizon

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify data format matches requirements
3. Test with sample data first
4. Review model-specific documentation in source files

---

**Note**: This system is designed for academic research and thesis work. All models include appropriate validation and evaluation metrics for publication-quality results.