"""
Configuration module for Market Sentiment Analysis System.
Centralizes all configuration parameters for maintainability.
"""
import os
from datetime import timedelta
from typing import Dict, Any
from dataclasses import dataclass
from enum import Enum

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

class DataSource(Enum):
    NEWS_API = "news"
    TWITTER = "twitter"
    REDDIT = "reddit"
    FINANCIAL_DATA = "financial"
    ALTERNATIVE_DATA = "alternative"

@dataclass
class APIConfig:
    """API Configuration DataClass"""
    NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")
    TWITTER_BEARER_TOKEN: str = os.getenv("TWITTER_BEARER_TOKEN", "")
    ALPHA_VANTAGE_KEY: str = os.getenv("ALPHA_VANTAGE_KEY", "")
    FIREBASE_CREDENTIALS_PATH: str = os.getenv("FIREBASE_CREDENTIALS_PATH", "./firebase-credentials.json")
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")

@dataclass
class ModelConfig:
    """Machine Learning Model Configuration"""
    TRAINING_INTERVAL_HOURS: int = 24
    PREDICTION_INTERVAL_MINUTES: int = 15
    TRAINING_WINDOW_DAYS: int = 30
    TEST_SPLIT_RATIO: float = 0.2
    VALIDATION_SPLIT_RATIO: float = 0.15
    SEQUENCE_LENGTH: int = 50
    BATCH_SIZE: int = 32
    EMBEDDING_DIM: int = 128
    LSTM_UNITS: int = 64
    DROPOUT_RATE: float = 0.3
    LEARNING_RATE: float = 0.001
    PATIENCE_EPOCHS: int = 10

@dataclass
class CollectionConfig:
    """Firebase Collection Configuration"""
    SENTIMENT_DATA: str = "sentiment_data"
    RAW_ARTICLES: str = "raw_articles"
    PROCESSED_FEATURES: str = "processed_features"
    MODEL_METRICS: str = "model_metrics"
    PREDICTIONS: str = "predictions"
    SYSTEM_STATE: str = "system_state"
    ALERTS: str = "alerts"

@dataclass
class ThresholdConfig:
    """Alert and Threshold Configuration"""
    SENTIMENT_SPIKE_THRESHOLD: float = 2.5
    VOLUME_THRESHOLD_PERCENTILE: float = 95.0
    CONFIDENCE_THRESHOLD: float = 0.7
    MAX_FAILED_ATTEMPTS: int = 3
    RETRY_DELAY_SECONDS: int = 5

class SystemConfig:
    """Main System Configuration"""
    def __init__(self):
        self.api = APIConfig()
        self.model = ModelConfig()
        self.collections = CollectionConfig()
        self.thresholds = ThresholdConfig()
        self.log_level: str = os.getenv("LOG_LEVEL", "INFO")
        self.data_retention_days: int = 90
        self.backup_enabled: bool = True
        self.telegram_alerts_enabled: bool = bool(self.api.TELEGRAM_BOT_TOKEN and self.api.TELEGRAM_CHAT_ID)
    
    def validate_config(self) -> bool:
        """Validate critical configuration parameters"""
        required_env_vars = [
            (self.api.FIREBASE_CREDENTIALS_PATH, "FIREBASE_CREDENTIALS_PATH"),
            (self.api.NEWS_API_KEY, "NEWS_API_KEY")
        ]
        
        for value, name in required_env_vars:
            if not value:
                print(f"❌ Critical configuration missing: {name}")
                return False
        
        # Validate file existence for Firebase credentials
        if not os.path.exists(self.api.FIREBASE_CREDENTIALS_PATH):
            print(f"❌ Firebase credentials file not found: {self.api.FIREBASE_CREDENTIALS_PATH}")
            return False
        
        print("✅ Configuration validated successfully")
        return True

# Global configuration instance
config = SystemConfig()