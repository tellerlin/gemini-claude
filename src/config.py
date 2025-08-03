"""
Enhanced configuration management for Gemini Claude Adapter - Flat Structure
"""

import os
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class Environment(str, Enum):
    """Environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class LogLevel(str, Enum):
    """Log levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class AppConfig(BaseSettings):
    """Main application configuration - Flat Structure"""
    
    # =============================================
    # Service Configuration
    # =============================================
    SERVICE_ENVIRONMENT: Environment = Field(
        Environment.DEVELOPMENT, 
        description="Runtime environment"
    )
    SERVICE_HOST: str = Field("0.0.0.0", description="Service host")
    SERVICE_PORT: int = Field(8000, description="Service port")
    SERVICE_WORKERS: int = Field(1, description="Number of workers")
    SERVICE_LOG_LEVEL: LogLevel = Field(LogLevel.INFO, description="Log level")
    SERVICE_ENABLE_METRICS: bool = Field(True, description="Enable metrics collection")
    SERVICE_ENABLE_HEALTH_CHECK: bool = Field(True, description="Enable health check endpoint")
    
    SERVICE_CORS_ORIGINS: Union[str, List[str]] = Field(default="*", description="CORS allowed origins")
    
    # =============================================
    # Gemini API Configuration - [REQUIRED]
    # =============================================
    GEMINI_API_KEYS: Union[str, List[str]] = Field(default="", description="Gemini API keys")
    GEMINI_PROXY_URL: Optional[str] = Field(None, description="Proxy URL for API calls")
    GEMINI_MAX_FAILURES: int = Field(3, description="Maximum failures before cooling", ge=1)
    GEMINI_COOLING_PERIOD: int = Field(300, description="Cooling period in seconds", ge=60)
    GEMINI_HEALTH_CHECK_INTERVAL: int = Field(60, description="Health check interval", ge=10)
    GEMINI_REQUEST_TIMEOUT: int = Field(45, description="Request timeout in seconds", ge=10)
    GEMINI_MAX_RETRIES: int = Field(2, description="Maximum retry attempts", ge=0)
    
    # =============================================
    # Security Configuration - [REQUIRED]
    # =============================================
    SECURITY_ADAPTER_API_KEYS: Union[str, List[str]] = Field(default="", description="Client API keys")
    SECURITY_ADMIN_API_KEYS: Union[str, List[str]] = Field(default="", description="Admin API keys")
    SECURITY_ENABLE_IP_BLOCKING: bool = Field(True, description="Enable IP blocking")
    SECURITY_MAX_FAILED_ATTEMPTS: int = Field(5, description="Maximum failed attempts before blocking")
    SECURITY_BLOCK_DURATION: int = Field(300, description="IP block duration in seconds")
    SECURITY_ENABLE_RATE_LIMITING: bool = Field(True, description="Enable rate limiting")
    SECURITY_RATE_LIMIT_REQUESTS: int = Field(100, description="Rate limit requests per window")
    SECURITY_RATE_LIMIT_WINDOW: int = Field(60, description="Rate limit window in seconds")
    
    # =============================================
    # Cache Configuration
    # =============================================
    CACHE_ENABLED: bool = Field(True, description="Enable response caching")
    CACHE_MAX_SIZE: int = Field(1000, description="Maximum cache size")
    CACHE_TTL: int = Field(300, description="Cache TTL in seconds")
    CACHE_KEY_PREFIX: str = Field("gemini_adapter", description="Cache key prefix")
    
    # =============================================
    # Database Configuration (Optional)
    # =============================================
    DATABASE_REDIS_URL: Optional[str] = Field(None, description="Redis URL for caching")
    DATABASE_REDIS_PASSWORD: Optional[SecretStr] = Field(None, description="Redis password")
    DATABASE_REDIS_DB: int = Field(0, description="Redis database number")
    DATABASE_REDIS_MAX_CONNECTIONS: int = Field(10, description="Maximum Redis connections")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore"
    )
    
    @field_validator('SECURITY_ADAPTER_API_KEYS', 'SECURITY_ADMIN_API_KEYS', 'GEMINI_API_KEYS', mode='before')
    @classmethod
    def validate_str_to_list(cls, v):
        """Validate and clean comma-separated strings into lists."""
        if v is None or v == "":
            return []
        if isinstance(v, str):
            if not v.strip():
                return []
            return [key.strip() for key in v.split(',') if key.strip()]
        elif isinstance(v, list):
            return [str(key).strip() for key in v if str(key).strip()]
        return v

    @field_validator('SERVICE_CORS_ORIGINS', mode='before')
    @classmethod
    def validate_cors_origins(cls, v):
        """Validate CORS origins from a comma-separated string."""
        if v is None or v == "":
            return ["*"]
        if isinstance(v, str):
            if not v.strip():
                return ["*"]
            return [origin.strip() for origin in v.split(',') if origin.strip()]
        elif isinstance(v, list):
            return v
        return ["*"]

    def model_post_init(self, __context):
        """Post-initialization validation and setup"""
        
        if isinstance(self.SERVICE_CORS_ORIGINS, str):
            self.SERVICE_CORS_ORIGINS = [origin.strip() for origin in self.SERVICE_CORS_ORIGINS.split(',') if origin.strip()]
        
        if isinstance(self.GEMINI_API_KEYS, str):
            self.GEMINI_API_KEYS = [key.strip() for key in self.GEMINI_API_KEYS.split(',') if key.strip()]
            
        if isinstance(self.SECURITY_ADAPTER_API_KEYS, str):
            self.SECURITY_ADAPTER_API_KEYS = [key.strip() for key in self.SECURITY_ADAPTER_API_KEYS.split(',') if key.strip()]
            
        if isinstance(self.SECURITY_ADMIN_API_KEYS, str):
            self.SECURITY_ADMIN_API_KEYS = [key.strip() for key in self.SECURITY_ADMIN_API_KEYS.split(',') if key.strip()]
        
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration consistency"""
        if self.SERVICE_ENVIRONMENT == Environment.PRODUCTION:
            if not self.SECURITY_ADAPTER_API_KEYS:
                logger.warning("Production environment without adapter API keys - service will be unsecured")
            
            if not self.GEMINI_API_KEYS:
                raise ValueError("Production environment requires Gemini API keys")
        
        if self.CACHE_ENABLED and self.CACHE_MAX_SIZE <= 0:
            raise ValueError("Cache max_size must be positive when caching is enabled")
        
        logger.info(f"Configuration validated for {self.SERVICE_ENVIRONMENT.value} environment")

    def log_configuration(self):
        """Log current configuration (without sensitive data)"""
        logger.info("=== Application Configuration ===")
        logger.info(f"Environment: {self.SERVICE_ENVIRONMENT.value}")
        logger.info(f"Host: {self.SERVICE_HOST}:{self.SERVICE_PORT}")
        logger.info(f"Workers: {self.SERVICE_WORKERS}")
        logger.info(f"Log Level: {self.SERVICE_LOG_LEVEL.value}")
        logger.info(f"Security Enabled: {bool(self.SECURITY_ADAPTER_API_KEYS)}")
        logger.info(f"Admin Keys: {len(self.SECURITY_ADMIN_API_KEYS)} configured")
        logger.info(f"Gemini Keys: {len(self.GEMINI_API_KEYS)} configured")
        logger.info(f"CORS Origins: {self.SERVICE_CORS_ORIGINS}")
        logger.info(f"Caching: {'Enabled' if self.CACHE_ENABLED else 'Disabled'}")
        logger.info(f"Metrics: {'Enabled' if self.SERVICE_ENABLE_METRICS else 'Disabled'}")
        logger.info("=================================")

_config: Optional[AppConfig] = None

def load_configuration() -> AppConfig:
    """Load and validate configuration"""
    global _config
    try:
        _config = AppConfig()
        _config.log_configuration()
        return _config
    except Exception as e:
        print(f"Failed to load configuration: {e}")
        raise

def get_config() -> AppConfig:
    """Get the global configuration instance, loading it if it doesn't exist."""
    global _config
    if _config is None:
        _config = load_configuration()
    return _config

def reload_configuration():
    """Reload configuration (useful for runtime updates)"""
    global _config
    _config = load_configuration()
    logger.info("Configuration reloaded")
