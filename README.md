# PJM Energy Market Price Prediction Project

This project focuses on predicting energy prices in the PJM market using various machine learning and deep learning models.

## Project Structure

```
├── config/                 # Configuration files
│   └── requirements.txt    # Python dependencies
├── data/                   # Data files
│   ├── raw/               # Raw data files
│   └── processed/         # Processed data files
├── docs/                   # Documentation
│   ├── guides/            # User guides and documentation
│   ├── proposals/         # Project proposals
│   └── research/          # Research analysis documents
├── notebooks/              # Jupyter notebooks
│   └── pjm_analysis_notebook.ipynb
├── results/                # Results and outputs
│   ├── plots/             # Visualization plots
│   └── reports/           # Analysis reports and summaries
├── src/                    # Source code
│   ├── main/              # Main application files
│   ├── models/            # Model implementations
│   └── utils/             # Utility scripts
└── tests/                  # Test files
    ├── unit/              # Unit tests
    └── integration/       # Integration tests
```

## Key Components

- **Models**: Implementation of ARIMA, LSTM, XGBoost, and ensemble models
- **Data Processing**: Scripts for data acquisition and preprocessing
- **Analysis**: Comprehensive research gap analysis and model comparisons
- **Visualization**: Price prediction results and model performance plots

## Getting Started

1. Install dependencies: `pip install -r config/requirements.txt`
2. Run main models: `python src/main/enhanced_pjm_models.py`
3. View analysis results: Check `results/reports/` directory

## Research Focus

This project addresses several research gaps in energy market prediction:
- Model explainability and interpretability
- Feature fusion techniques
- Ensemble model optimization
- Real-time prediction capabilities

For detailed documentation, see the `docs/` directory.