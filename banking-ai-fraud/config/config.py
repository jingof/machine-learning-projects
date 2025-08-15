import os
from pathlib import Path
from typing import Dict, Any
import yaml

class Config:
    """Configuration management for the banking AI system"""
    
    # Base directories
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    MODELS_DIR = BASE_DIR / "models"
    LOGS_DIR = BASE_DIR / "logs"
    
    # Database configuration
    DATABASE_CONFIG = {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": os.getenv("DB_PORT", 5432),
        "database": os.getenv("DB_NAME", "banking_ai"),
        "username": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", "password"),
    }
    
    # Redis configuration for caching
    REDIS_CONFIG = {
        "host": os.getenv("REDIS_HOST", "localhost"),
        "port": os.getenv("REDIS_PORT", 6379),
        "db": os.getenv("REDIS_DB", 0),
    }
    
    # Model configuration
    MODEL_CONFIG = {
        "credit_risk": {
            "model_type": "ensemble",
            "algorithms": ["xgboost", "lightgbm", "neural_network"],
            "hyperparameters": {
                "xgb_n_estimators": 1000,
                "xgb_learning_rate": 0.01,
                "lgb_n_estimators": 1000,
                "lgb_learning_rate": 0.01,
                "nn_hidden_layers": [512, 256, 128],
                "nn_dropout": 0.3,
            }
        },
        "fraud_detection": {
            "model_type": "anomaly_ensemble",
            "algorithms": ["isolation_forest", "autoencoder", "one_class_svm"],
            "threshold": 0.1,
        }
    }
    
    # API configuration
    API_CONFIG = {
        "host": "0.0.0.0",
        "port": 8000,
        "workers": 4,
        "timeout": 30,
    }
    
    # Monitoring configuration
    MONITORING_CONFIG = {
        "model_drift_threshold": 0.05,
        "performance_threshold": 0.8,
        "alert_email": os.getenv("ALERT_EMAIL", "admin@bank.com"),
    }
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        for directory in [cls.DATA_DIR, cls.RAW_DATA_DIR, cls.PROCESSED_DATA_DIR, 
                         cls.MODELS_DIR, cls.LOGS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
