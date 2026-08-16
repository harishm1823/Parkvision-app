"""
Configuration file for ParkVision
"""
import os

class Config:
    """Base configuration"""
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'parkvision-secret-key-change-in-production'
    DEBUG = False
    TESTING = False
    
    # Application settings
    APP_NAME = 'ParkVision'
    APP_VERSION = '1.0.0'
    
    # Model settings
    MODEL_PATH = os.environ.get('MODEL_PATH') or 'models/best.pt'
    CONFIDENCE_THRESHOLD = float(os.environ.get('CONFIDENCE_THRESHOLD') or 0.5)
    
    # Camera settings
    CAMERA_SOURCE = os.environ.get('CAMERA_SOURCE') or '0'  # 0 for webcam, or RTSP URL
    CAMERA_WIDTH = int(os.environ.get('CAMERA_WIDTH') or 640)
    CAMERA_HEIGHT = int(os.environ.get('CAMERA_HEIGHT') or 480)
    CAMERA_FPS = int(os.environ.get('CAMERA_FPS') or 30)
    
    # Detection settings
    IMAGE_SIZE = int(os.environ.get('IMAGE_SIZE') or 640)
    BATCH_SIZE = int(os.environ.get('BATCH_SIZE') or 1)
    
    # Performance settings
    USE_GPU = os.environ.get('USE_GPU', 'False').lower() == 'true'
    HALF_PRECISION = os.environ.get('HALF_PRECISION', 'False').lower() == 'true'
    
    # Logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL') or 'INFO'
    LOG_FILE = os.environ.get('LOG_FILE') or 'parkvision.log'
    
    # Database (optional)
    DATABASE_URI = os.environ.get('DATABASE_URI') or 'sqlite:///parkvision.db'
    
    # API settings
    API_ENABLED = os.environ.get('API_ENABLED', 'True').lower() == 'true'
    API_KEY = os.environ.get('API_KEY') or None


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    CAMERA_SOURCE = '0'  # Use webcam for development


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    # Override with environment variables in production


class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    CAMERA_SOURCE = 'test_video.mp4'


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}


def get_config(env=None):
    """Get configuration based on environment"""
    env = env or os.environ.get('FLASK_ENV', 'development')
    return config.get(env, config['default'])
