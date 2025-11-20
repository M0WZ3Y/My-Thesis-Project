"""
Data Acquisition Script for PJM Electricity Price Prediction

This script helps acquire additional data sources that complement the PJM price data
to improve prediction accuracy for daily and hourly forecasts.
"""

import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

class PJMDataAcquisition:
    def __init__(self):
        self.base_urls = {
            'pjm': 'https://api.pjm.com/api/v1/',
            'eia': 'https://api.eia.gov/v2/',
            'noaa': 'https://api.weather.gov/'
        }
        
    def get_pjm_load_data(self, start_date, end_date, api_key=None):
        """
        Acquire PJM load forecast data
        Note: Requires PJM API key for official access
        """
        print("=== PJM Load Data Acquisition ===")
        
        if not api_key:
            print("PJM API key required for load data access")
            print("Alternative: Use publicly available data from PJM Data Miner")
            return self._get_pjm_public_load_data(start_date, end_date)
        
        # Implementation with API key would go here
        print("PJM API integration not implemented - requires API key")
        return None
    
    def _get_pjm_public_load_data(self, start_date, end_date):
        """
        Guide for acquiring PJM load data from public sources
        """
        print("To acquire PJM load data:")
        print("1. Visit: https://data.pjm.com/")
        print("2. Navigate to 'Load Forecasts' section")
        print("3. Download 'Real-Time Load' or 'Day-Ahead Load' data")
        print("4. Select date range and format (CSV recommended)")
        print(f"5. Request data from {start_date} to {end_date}")
        
        return {
            'instructions': {
                'source': 'PJM Data Miner',
                'url': 'https://data.pjm.com/',
                'data_type': 'Load Forecasts',
                'recommended_files': [
                    'rt_hrl_load_metered.csv',
                    'da_hrl_load_forecast.csv'
                ],
                'date_range': f"{start_date} to {end_date}"
            }
        }
    
    def get_weather_data(self, location, start_date, end_date):
        """
        Acquire weather data for PJM regions
        """
        print("=== Weather Data Acquisition ===")
        
        # For demonstration, we'll provide guidance on acquiring weather data
        print("Weather data sources for PJM regions:")
        print("1. NOAA Weather Service API (free)")
        print("2. OpenWeatherMap API (free tier available)")
        print("3. Weather Underground API")
        print("4. Commercial weather data providers")
        
        # Major PJM regions and approximate weather stations
        pjm_regions = {
            'PJM-RTO': {'lat': 39.95, 'lon': -75.16, 'city': 'Philadelphia'},
            'PSEG': {'lat': 40.71, 'lon': -74.01, 'city': 'Newark'},
            'COMED': {'lat': 41.88, 'lon': -87.63, 'city': 'Chicago'},
            'AEP': {'lat': 39.96, 'lon': -82.99, 'city': 'Columbus'},
            'PEPCO': {'lat': 38.91, 'lon': -77.03, 'city': 'Washington DC'},
            'DOM': {'lat': 37.54, 'lon': -77.46, 'city': 'Richmond'},
            'DUK': {'lat': 35.23, 'lon': -80.84, 'city': 'Charlotte'},
            'Progress': {'lat': 35.78, 'lon': -78.64, 'city': 'Raleigh'}
        }
        
        if location in pjm_regions:
            region_info = pjm_regions[location]
            print(f"\nFor {location} region ({region_info['city']}):")
            print(f"Coordinates: {region_info['lat']}, {region_info['lon']}")
            print(f"Weather parameters needed:")
            print("- Temperature (2m above ground)")
            print("- Dew point temperature")
            print("- Wind speed and direction")
            print("- Solar irradiance (GHI, DNI)")
            print("- Cloud cover")
            print("- Precipitation")
        
        return {
            'weather_parameters': [
                'temperature_2m',
                'dewpoint_2m', 
                'wind_speed_10m',
                'wind_direction_10m',
                'shortwave_radiation',
                'cloud_cover',
                'precipitation'
            ],
            'pjm_regions': pjm_regions,
            'api_options': {
                'noaa': {
                    'url': 'https://www.weather.gov/documentation/services-web-api',
                    'cost': 'Free',
                    'limitations': 'US only, rate limited'
                },
                'openweathermap': {
                    'url': 'https://openweathermap.org/api',
                    'cost': 'Free tier (1000 calls/day)',
                    'limitations': 'Historical data requires paid plan'
                }
            }
        }
    
    def get_fuel_price_data(self, start_date, end_date):
        """
        Acquire fuel price data (natural gas, coal)
        """
        print("=== Fuel Price Data Acquisition ===")
        
        print("Fuel price data sources:")
        print("1. EIA (Energy Information Administration) - Free")
        print("2. CME Group - Paid")
        print("3. ICE (Intercontinental Exchange) - Paid")
        print("4. Platts - Paid")
        
        print("\nEIA API Instructions:")
        print("1. Register for free API key at: https://www.eia.gov/opendata/register.php")
        print("2. Use the following endpoints:")
        print("   - Natural Gas: https://api.eia.gov/v2/natural-gas/price/data/")
        print("   - Coal: https://api.eia.gov/v2/coal/data/")
        print("3. Key data series:")
        print("   - Henry Hub Natural Gas Spot Price")
        print("   - Natural Gas Futures")
        print("   - App Coal Prices")
        
        return {
            'eia_api_endpoints': {
                'natural_gas': {
                    'endpoint': 'natural-gas/price/data/',
                    'series_id': 'NG.RNGWHHD.D',  # Henry Hub
                    'description': 'Henry Hub Natural Gas Spot Price'
                },
                'coal': {
                    'endpoint': 'coal/data/',
                    'series_id': 'COAL.US-APP.A',  # Appalachian Coal
                    'description': 'Appalachian Coal Price'
                }
            },
            'alternative_sources': [
                'CME Group Historical Data',
                'ICE Data Services',
                'Bloomberg Terminal'
            ]
        }
    
    def get_economic_data(self, start_date, end_date):
        """
        Acquire economic indicators that affect electricity demand
        """
        print("=== Economic Data Acquisition ===")
        
        print("Economic data sources:")
        print("1. FRED (Federal Reserve Economic Data) - Free")
        print("2. Bureau of Labor Statistics - Free")
        print("3. Bureau of Economic Analysis - Free")
        
        print("\nKey economic indicators for electricity demand:")
        print("- Industrial Production Index")
        print("- GDP Growth Rate")
        print("- Manufacturing PMI")
        print("- Employment Data")
        print("- Retail Sales")
        
        return {
            'fred_api_series': {
                'industrial_production': 'IPMAN',
                'gdp': 'GDP',
                'manufacturing_pmi': 'NAPM',
                'employment': 'PAYEMS',
                'retail_sales': 'RSXFS'
            },
            'api_url': 'https://api.stlouisfed.org/fred/series/observations',
            'instructions': {
                'registration': 'Free API key at: https://fred.stlouisfed.org/docs/api/api_key.html',
                'usage': 'Use series IDs above with FRED API'
            }
        }
    
    def create_data_integration_template(self):
        """
        Create a template for integrating all data sources
        """
        print("=== Data Integration Template ===")
        
        template = {
            'datetime': 'timestamp',
            'pjm_data': {
                'total_lmp_da': 'float',
                'system_energy_price_da': 'float',
                'congestion_price_da': 'float',
                'marginal_loss_price_da': 'float',
                'zone': 'string',
                'pnode_id': 'string'
            },
            'load_data': {
                'system_load': 'float',
                'load_forecast_da': 'float',
                'load_forecast_rt': 'float'
            },
            'weather_data': {
                'temperature': 'float',
                'dewpoint': 'float',
                'wind_speed': 'float',
                'wind_direction': 'float',
                'solar_irradiance': 'float',
                'cloud_cover': 'float',
                'precipitation': 'float'
            },
            'fuel_prices': {
                'natural_gas_price': 'float',
                'coal_price': 'float'
            },
            'economic_indicators': {
                'industrial_production': 'float',
                'manufacturing_pmi': 'float',
                'employment_index': 'float'
            },
            'calendar_features': {
                'hour': 'int',
                'day_of_week': 'int',
                'month': 'int',
                'is_holiday': 'boolean',
                'is_weekend': 'boolean'
            }
        }
        
        # Save template to file
        with open('data_integration_template.json', 'w') as f:
            json.dump(template, f, indent=2)
        
        print("Data integration template saved to 'data_integration_template.json'")
        return template
    
    def generate_acquisition_plan(self, target_zone, prediction_horizon):
        """
        Generate a customized data acquisition plan
        """
        print(f"=== DATA ACQUISITION PLAN FOR {target_zone} ===")
        print(f"Prediction Horizon: {prediction_horizon}")
        
        plan = {
            'primary_data': {
                'source': 'Your existing PJM price data',
                'file': 'da_hrl_lmps (1).csv',
                'status': 'Available',
                'priority': 'High'
            },
            'recommended_additions': [
                {
                    'data_type': 'Load Forecasts',
                    'source': 'PJM Data Miner',
                    'priority': 'High',
                    'impact': 'Very High',
                    'acquisition_difficulty': 'Easy',
                    'description': 'System load and load forecasts are primary drivers of electricity prices'
                },
                {
                    'data_type': 'Weather Data',
                    'source': 'NOAA/OpenWeatherMap',
                    'priority': 'High',
                    'impact': 'High',
                    'acquisition_difficulty': 'Medium',
                    'description': 'Temperature and weather conditions significantly affect demand and prices'
                },
                {
                    'data_type': 'Fuel Prices',
                    'source': 'EIA API',
                    'priority': 'Medium',
                    'impact': 'Medium',
                    'acquisition_difficulty': 'Easy',
                    'description': 'Natural gas prices influence marginal generation costs'
                },
                {
                    'data_type': 'Generation Outages',
                    'source': 'PJM Data Miner',
                    'priority': 'Medium',
                    'impact': 'High',
                    'acquisition_difficulty': 'Medium',
                    'description': 'Power plant outages affect supply and cause price spikes'
                },
                {
                    'data_type': 'Economic Indicators',
                    'source': 'FRED API',
                    'priority': 'Low',
                    'impact': 'Low',
                    'acquisition_difficulty': 'Easy',
                    'description': 'Useful for long-term trends, less important for daily/hourly forecasts'
                }
            ]
        }
        
        # Save plan to file
        with open(f'data_acquisition_plan_{target_zone}.json', 'w') as f:
            json.dump(plan, f, indent=2)
        
        print(f"\nData acquisition plan saved to 'data_acquisition_plan_{target_zone}.json'")
        
        print("\n=== PRIORITY RECOMMENDATIONS ===")
        print("1. IMMEDIATE (High Impact, Easy Acquisition):")
        print("   - PJM Load Forecasts from Data Miner")
        print("   - Weather data from NOAA")
        print("   - Natural gas prices from EIA")
        
        print("\n2. SHORT TERM (High Impact, Medium Difficulty):")
        print("   - Generation outage schedules")
        print("   - Transmission constraint data")
        
        print("\n3. LONG TERM (Lower Priority):")
        print("   - Economic indicators")
        print("   - Fuel transportation costs")
        
        return plan

def main():
    """Main execution function"""
    print("=== PJM DATA ACQUISITION HELPER ===\n")
    
    # Initialize data acquisition helper
    acquirer = PJMDataAcquisition()
    
    # Example usage
    target_zone = "PJM-RTO"  # You can change this
    prediction_horizon = "24 hours ahead"
    
    # Generate acquisition plan
    plan = acquirer.generate_acquisition_plan(target_zone, prediction_horizon)
    
    # Get specific data acquisition guidance
    print("\n" + "="*50)
    weather_info = acquirer.get_weather_data(target_zone, "2024-01-01", "2024-12-31")
    
    print("\n" + "="*50)
    fuel_info = acquirer.get_fuel_price_data("2024-01-01", "2024-12-31")
    
    print("\n" + "="*50)
    economic_info = acquirer.get_economic_data("2024-01-01", "2024-12-31")
    
    print("\n" + "="*50)
    template = acquirer.create_data_integration_template()
    
    print("\n=== NEXT STEPS ===")
    print("1. Review the generated acquisition plan")
    print("2. Acquire the high-priority data sources first")
    print("3. Use the integration template to structure your data")
    print("4. Update the prediction model to include new features")
    print("5. Retrain models with enhanced dataset")

if __name__ == "__main__":
    main()