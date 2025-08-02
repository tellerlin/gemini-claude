"""
Enhanced configuration management for Gemini Claude Adapter - Flat Structure
"""

import os
from typing import List, Optional, Dict, Any
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
    SERVICE_CORS_ORIGINS: List[str] = Field(["*"], description="CORS allowed origins")
    
    # =============================================
    # Gemini API Configuration - [REQUIRED]
    # =============================================
    GEMINI_API_KEYS: List[str] = Field(default_factory=list, description="Gemini API keys")
    GEMINI_PROXY_URL: Optional[str] = Field(None, description="Proxy URL for API calls")
    GEMINI_MAX_FAILURES: int = Field(3, description="Maximum failures before cooling", ge=1)
    GEMINI_COOLING_PERIOD: int = Field(300, description="Cooling period in seconds", ge=60)
    GEMINI_HEALTH_CHECK_INTERVAL: int = Field(60, description="Health check interval", ge=10)
    GEMINI_REQUEST_TIMEOUT: int = Field(45, description="Request timeout in seconds", ge=10)
    GEMINI_MAX_RETRIES: int = Field(2, description="Maximum retry attempts", ge=0)
    
    # =============================================
    # Security Configuration - [REQUIRED]
    # =============================================
    SECURITY_ADAPTER_API_KEYS: List[str] = Field(default_factory=list, description="Client API keys")
    SECURITY_ADMIN_API_KEYS: List[str] = Field(default_factory=list, description="Admin API keys")
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
    # Performance Configuration
    # =============================================
    PERFORMANCE_MAX_KEEPALIVE_CONNECTIONS: int = Field(20, description="Max keepalive connections")
    PERFORMANCE_MAX_CONNECTIONS: int = Field(100, description="Max total connections")
    PERFORMANCE_KEEPALIVE_EXPIRY: float = Field(30.0, description="Keepalive expiry time")
    PERFORMANCE_CONNECT_TIMEOUT: float = Field(10.0, description="Connection timeout")
    PERFORMANCE_READ_TIMEOUT: float = Field(45.0, description="Read timeout")
    PERFORMANCE_WRITE_TIMEOUT: float = Field(10.0, description="Write timeout")
    PERFORMANCE_POOL_TIMEOUT: float = Field(5.0, description="Pool timeout")
    PERFORMANCE_HTTP2_ENABLED: bool = Field(True, description="Enable HTTP/2")
    PERFORMANCE_TRUST_ENV: bool = Field(True, description="Trust environment for proxy support")
    PERFORMANCE_VERIFY_SSL: bool = Field(True, description="Verify SSL certificates")
    
    # =============================================
    # Database Configuration (Optional)
    # =============================================
    DATABASE_REDIS_URL: Optional[str] = Field(None, description="Redis URL for caching")
    DATABASE_REDIS_PASSWORD: Optional[SecretStr] = Field(None, description="Redis password")
    DATABASE_REDIS_DB: int = Field(0, description="Redis database number")
    DATABASE_REDIS_MAX_CONNECTIONS: int = Field(10, description="Maximum Redis connections")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        # 移除了 env_nested_delimiter，直接映射环境变量
        case_sensitive=False,
        extra="ignore"  # Ignore extra fields to prevent parsing errors
    )
    
    # =============================================
    # Field Validators
    # =============================================
    @field_validator('SECURITY_ADAPTER_API_KEYS', mode='before')
    @classmethod
    def validate_adapter_keys(cls, v):
        """Validate and clean adapter API keys"""
        if v is None:
            return []
        if isinstance(v, str):
            if not v.strip():
                return []
            v = [key.strip() for key in v.split(',') if key.strip()]
        elif isinstance(v, list):
            v = [str(key).strip() for key in v if str(key).strip()]
        return [key.strip() for key in v if key.strip()]
    
    @field_validator('SECURITY_ADMIN_API_KEYS', mode='before')
    @classmethod
    def validate_admin_keys(cls, v):
        """Validate and clean admin API keys"""
        if v is None:
            return []
        if isinstance(v, str):
            if not v.strip():
                return []
            v = [key.strip() for key in v.split(',') if key.strip()]
        elif isinstance(v, list):
            v = [str(key).strip() for key in v if str(key).strip()]
        return [key.strip() for key in v if key.strip()]

    @field_validator('GEMINI_API_KEYS', mode='before')
    @classmethod
    def validate_gemini_api_keys(cls, v):
        """Validate and clean Gemini API keys"""
        if v is None:
            return []
        
        if isinstance(v, str):
            if not v.strip():
                return []
            v = [key.strip() for key in v.split(',') if key.strip()]
        elif isinstance(v, list):
            v = [str(key).strip() for key in v if str(key).strip()]
        
        valid_keys = []
        for key in v:
            if key and str(key).strip():
                cleaned_key = str(key).strip().strip('"\'').strip()
                if cleaned_key:
                    valid_keys.append(cleaned_key)
        
        if valid_keys:
            invalid_keys = [key for key in valid_keys if not key.startswith('AIza')]
            if invalid_keys:
                logger.warning(f"Potentially invalid API keys detected: {len(invalid_keys)} keys don't start with 'AIza'")
        
        return valid_keys

    @field_validator('SERVICE_CORS_ORIGINS', mode='before')
    @classmethod
    def validate_cors_origins(cls, v):
        """Validate CORS origins"""
        if v is None:
            return ["*"]
        if isinstance(v, str):
            if not v.strip():
                return ["*"]
            return [origin.strip() for origin in v.split(',') if origin.strip()]
        return v

    # =============================================
    # Post-init Validation
    # =============================================
    def model_post_init(self, __context):
        """Post-initialization validation and setup"""
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
        
        if self.PERFORMANCE_MAX_CONNECTIONS <= 0:
            raise ValueError("Max connections must be positive")
        
        logger.info(f"Configuration validated for {self.SERVICE_ENVIRONMENT.value} environment")
    
    # =============================================
    # Helper Methods (for backward compatibility)
    # =============================================
    def get_security_status(self) -> Dict[str, Any]:
        """Get security configuration status"""
        return {
            "security_enabled": bool(self.SECURITY_ADAPTER_API_KEYS),
            "admin_keys_configured": bool(self.SECURITY_ADMIN_API_KEYS),
            "ip_blocking_enabled": self.SECURITY_ENABLE_IP_BLOCKING,
            "rate_limiting_enabled": self.SECURITY_ENABLE_RATE_LIMITING,
            "environment": self.SERVICE_ENVIRONMENT.value
        }
    
    def get_cache_config_dict(self) -> Dict[str, Any]:
        """Get cache configuration as dictionary"""
        return {
            "enabled": self.CACHE_ENABLED,
            "max_size": self.CACHE_MAX_SIZE,
            "ttl": self.CACHE_TTL,
            "key_prefix": self.CACHE_KEY_PREFIX
        }
    
    def get_performance_config_dict(self) -> Dict[str, Any]:
        """Get performance configuration as dictionary"""
        return {
            "max_keepalive_connections": self.PERFORMANCE_MAX_KEEPALIVE_CONNECTIONS,
            "max_connections": self.PERFORMANCE_MAX_CONNECTIONS,
            "keepalive_expiry": self.PERFORMANCE_KEEPALIVE_EXPIRY,
            "connect_timeout": self.PERFORMANCE_CONNECT_TIMEOUT,
            "read_timeout": self.PERFORMANCE_READ_TIMEOUT,
            "write_timeout": self.PERFORMANCE_WRITE_TIMEOUT,
            "pool_timeout": self.PERFORMANCE_POOL_TIMEOUT,
            "http2_enabled": self.PERFORMANCE_HTTP2_ENABLED
        }
    
    def get_gemini_config_dict(self) -> Dict[str, Any]:
        """Get Gemini configuration as dictionary"""
        return {
            "api_keys": self.GEMINI_API_KEYS,
            "proxy_url": self.GEMINI_PROXY_URL,
            "max_failures": self.GEMINI_MAX_FAILURES,
            "cooling_period": self.GEMINI_COOLING_PERIOD,
            "health_check_interval": self.GEMINI_HEALTH_CHECK_INTERVAL,
            "request_timeout": self.GEMINI_REQUEST_TIMEOUT,
            "max_retries": self.GEMINI_MAX_RETRIES
        }
    
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
        logger.info(f"Caching: {'Enabled' if self.CACHE_ENABLED else 'Disabled'}")
        logger.info(f"Metrics: {'Enabled' if self.SERVICE_ENABLE_METRICS else 'Disabled'}")
        logger.info("=================================")

# =============================================
# Global Configuration Management
# =============================================
_config: Optional[AppConfig] = None

def load_configuration() -> AppConfig:
    """Load and validate configuration"""
    global _config
    try:
        _config = AppConfig()
        _config.log_configuration()
        return _config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
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
