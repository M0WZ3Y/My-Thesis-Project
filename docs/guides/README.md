# PJM Electricity Price Prediction System

A comprehensive system for predicting PJM electricity prices on daily and hourly horizons using machine learning and time series analysis.

## 📊 Overview

This system provides:
- **Hourly and daily price predictions** for PJM electricity markets
- **Multiple machine learning models** (Linear Regression, Random Forest, Gradient Boosting)
- **Comprehensive feature engineering** for time series data
- **Data acquisition guidance** for enhancing predictions
- **Interactive analysis tools** via Jupyter notebook
- **Visualization and reporting** capabilities

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Basic Prediction Model

```python
python pjm_price_prediction.py
```

### 3. Interactive Analysis (Recommended)

```bash
jupyter notebook pjm_analysis_notebook.ipynb
```

### 4. Data Acquisition Guidance

```python
python data_acquisition.py
```

## 📁 Project Structure

```
├── pjm_price_prediction.py      # Main prediction engine
├── pjm_analysis_notebook.ipynb  # Interactive analysis notebook
├── data_acquisition.py          # Data acquisition guidance
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── da_hrl_lmps (1).csv         # Your PJM price data
└── output/                      # Generated results (auto-created)
    ├── pjm_price_prediction_results.png
    ├── data_integration_template.json
    └── data_acquisition_plan_*.json
```

## 📈 Data Requirements

### Primary Data (You Already Have)
- **File**: `da_hrl_lmps (1).csv`
- **Content**: Day-Ahead Hourly Locational Marginal Prices (LMPs)
- **Key Columns**:
  - `datetime_beginning_utc/ept`: Timestamps
  - `total_lmp_da`: Target variable (Day-ahead total LMP)
  - `system_energy_price_da`: Energy component
  - `congestion_price_da`: Congestion component
  - `marginal_loss_price_da`: Loss component

### Recommended Additional Data

#### High Priority (High Impact, Easy Acquisition)
1. **Load Forecasts**
   - Source: [PJM Data Miner](https://data.pjm.com/)
   - Files: `rt_hrl_load_metered.csv`, `da_hrl_load_forecast.csv`

2. **Weather Data**
   - Source: NOAA API (free) or OpenWeatherMap
   - Parameters: Temperature, wind speed, solar irradiance, cloud cover

3. **Fuel Prices**
   - Source: [EIA API](https://www.eia.gov/opendata/)
   - Data: Natural gas prices, coal prices

#### Medium Priority
4. **Generation Outages**
   - Source: PJM Data Miner
   - Impact: High (affects supply constraints)

5. **Transmission Constraints**
   - Source: PJM Data Miner
   - Impact: High (causes congestion pricing)

## 🤖 Machine Learning Models

### Available Models
1. **Linear Regression** - Baseline model
2. **Random Forest** - Non-linear relationships
3. **Gradient Boosting** - Advanced ensemble method

### Feature Engineering
- **Time-based features**: Hour, day of week, month, cyclical encoding
- **Lag features**: Previous 1, 2, 3, 6, 12, 24, 48 hours
- **Rolling statistics**: Moving averages and standard deviations
- **Price components**: Energy, congestion, loss ratios
- **Calendar effects**: Weekends, holidays, seasonal patterns

### Evaluation Metrics
- **Mean Absolute Error (MAE)**: Primary metric
- **Root Mean Square Error (RMSE)**: Penalizes large errors
- **R² Score**: Explained variance

## 📊 Usage Examples

### Basic Prediction

```python
from pjm_price_prediction import PJMPricePredictor

# Initialize predictor
predictor = PJMPricePredictor('da_hrl_lmps (1).csv')

# Load and analyze data
predictor.load_data()
predictor.analyze_data_structure()

# Create features and train models
df = predictor.create_features(target_zone=None)  # System average
train_df, test_df = predictor.prepare_train_test(df, test_days=7)
results, best_model = predictor.train_models(train_df, test_df)

# Generate visualizations
predictor.create_visualizations(test_df, results)
```

### Zone-Specific Analysis

```python
# Analyze specific PJM zone
df = predictor.create_features(target_zone='PSEG')  # PSEG zone
# ... rest of the analysis
```

### Future Predictions

```python
# Make predictions for next 24 hours
predictions = predictor.predict_future(hours_ahead=24)
```

## 🔧 Data Acquisition

### Automated Guidance

Run the data acquisition helper to get personalized recommendations:

```python
python data_acquisition.py
```

This will generate:
- **Data acquisition plan** for your target zone
- **API integration instructions** for weather, fuel prices
- **Data integration template** for structuring your dataset

### Manual Data Sources

#### PJM Data Miner
1. Visit: https://data.pjm.com/
2. Navigate to desired data categories
3. Select date range and download CSV files
4. Recommended files:
   - Load forecasts
   - Generation outages
   - Transmission constraints

#### Weather Data APIs
- **NOAA**: Free, US-only, rate limited
- **OpenWeatherMap**: Free tier (1000 calls/day)
- **Weather Underground**: Paid, comprehensive

#### Economic Data
- **FRED API**: Free economic indicators
- **EIA API**: Free energy data

## 📈 Model Performance

### Expected Accuracy Ranges
- **Good MAE**: $2-5/MWh for hourly predictions
- **Excellent MAE**: <$2/MWh with additional data
- **Daily predictions**: Typically more accurate than hourly

### Performance Factors
- **Data quality**: Clean, complete historical data
- **Feature richness**: Weather, load, fuel prices
- **Market volatility**: Higher during extreme weather or outages
- **Zone characteristics**: Some zones more predictable than others

## 🎯 Advanced Features

### Time Series Cross-Validation
```python
# Implement time series split for more robust validation
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
```

### Hyperparameter Tuning
```python
# Optimize model parameters
from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}
```

### Ensemble Methods
```python
# Combine multiple models for better predictions
from sklearn.ensemble import VotingRegressor
ensemble = VotingRegressor([
    ('rf', RandomForestRegressor()),
    ('gb', GradientBoostingRegressor()),
    ('lr', LinearRegression())
])
```

## 📊 Visualization and Reporting

### Generated Outputs
- **Time series plots**: Actual vs predicted prices
- **Scatter plots**: Prediction accuracy analysis
- **Residual plots**: Error pattern analysis
- **Feature importance**: Most influential factors
- **Model comparison**: Performance across different models

### Custom Visualizations
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Custom price pattern analysis
plt.figure(figsize=(15, 6))
sns.boxplot(data=df, x='hour', y='total_lmp_da')
plt.title('Price Distribution by Hour')
plt.show()
```

## 🚀 Deployment Considerations

### Production Requirements
1. **Data Pipeline**: Automated data acquisition and preprocessing
2. **Model Retraining**: Regular updates with new data
3. **Monitoring**: Track prediction accuracy over time
4. **API Integration**: Serve predictions via REST API

### Scaling Considerations
- **Database**: Use time-series database for large datasets
- **Caching**: Cache predictions for repeated requests
- **Parallel Processing**: Handle multiple zones simultaneously

## 🔍 Troubleshooting

### Common Issues

#### Memory Errors
```python
# Process data in chunks for large files
for chunk in pd.read_csv('large_file.csv', chunksize=100000):
    process_chunk(chunk)
```

#### Time Zone Issues
```python
# Ensure consistent timezone handling
df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_localize('UTC')
```

#### Missing Data
```python
# Handle missing values appropriately
df['price'].fillna(method='forward', inplace=True)
df['price'].fillna(df['price'].mean(), inplace=True)
```

## 📚 References and Resources

### PJM Market Information
- [PJM Learning Center](https://learn.pjm.com/)
- [PJM Data Miner](https://data.pjm.com/)
- [PJM Manuals](https://www.pjm.com/documents/manuals.aspx)

### Data Sources
- [EIA API Documentation](https://www.eia.gov/opendata/)
- [NOAA Weather API](https://www.weather.gov/documentation/services-web-api)
- [FRED API](https://fred.stlouisfed.org/docs/api/api_key.html)

### Machine Learning Resources
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Time Series Analysis](https://otexts.com/fpp3/)
- [Feature Engineering for Time Series](https://github.com/blue-yonder/tsfresh)

## 📞 Support

For questions or issues:
1. Check the troubleshooting section above
2. Review the Jupyter notebook for detailed examples
3. Run `data_acquisition.py` for data source guidance
4. Ensure all dependencies are installed correctly

## 📄 License

This project is provided as-is for educational and research purposes. Please ensure compliance with data usage terms from PJM and other data providers.

---

**Next Steps:**
1. Install dependencies and run the basic model
2. Explore the interactive notebook
3. Acquire additional data sources using the guidance
4. Enhance the model with new features
5. Deploy for regular predictions