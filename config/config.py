import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY', '')
NOAA_API_TOKEN = os.getenv('NOAA_API_TOKEN', '')

# Model Configuration
MODEL_CONFIG = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'random_state': 42
    },
    'gradient_boost': {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 6,
        'random_state': 42
    }
}

# Application Configuration
APP_CONFIG = {
    'title': 'Climate-Smart Crop Risk Predictor',
    'version': '1.0.0',
    'author': 'Agricultural AI Team',
    'description': 'Predict crop risks using weather forecasts and machine learning'
}

# Data Configuration
DATA_CONFIG = {
    'synthetic_samples': 2000,
    'test_size': 0.2,
    'validation_size': 0.15,
    'random_seed': 42
}

# Deployment Configuration
DEPLOY_CONFIG = {
    'streamlit_port': 8501,
    'host': '0.0.0.0',
    'debug': False
}
