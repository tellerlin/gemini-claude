"""
Enhanced configuration management for Gemini Claude Adapter
"""

import os
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseSettings, Field, validator, SecretStr
from enum import Enum
import json
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

class DatabaseConfig(BaseSettings):
    """Database configuration"""
    redis_url: Optional[str] = Field(None, description="Redis URL for caching")
    redis_password: Optional[SecretStr] = Field(None, description="Redis password")
    redis_db: int = Field(0, description="Redis database number")
    redis_max_connections: int = Field(10, description="Maximum Redis connections")
    
    class Config:
        env_prefix = "REDIS_"

class SecurityConfig(BaseSettings):
    """Security configuration"""
    adapter_api_keys: List[str] = Field([], description="Client API keys")
    admin_api_keys: List[str] = Field([], description="Admin API keys")
    enable_ip_blocking: bool = Field(True, description="Enable IP blocking")
    max_failed_attempts: int = Field(5, description="Maximum failed attempts before blocking")
    block_duration: int = Field(300, description="IP block duration in seconds")
    enable_rate_limiting: bool = Field(True, description="Enable rate limiting")
    rate_limit_requests: int = Field(100, description="Rate limit requests per window")
    rate_limit_window: int = Field(60, description="Rate limit window in seconds")
    
    @validator('adapter_api_keys')
    def validate_adapter_keys(cls, v):
        """Validate and clean adapter API keys"""
        if isinstance(v, str):
            # Support comma-separated string
            v = [key.strip() for key in v.split(',') if key.strip()]
        return [key.strip() for key in v if key.strip()]
    
    @validator('admin_api_keys')
    def validate_admin_keys(cls, v):
        """Validate and clean admin API keys"""
        if isinstance(v, str):
            # Support comma-separated string
            v = [key.strip() for key in v.split(',') if key.strip()]
        return [key.strip() for key in v if key.strip()]
    
    class Config:
        env_prefix = "SECURITY_"

class GeminiConfig(BaseSettings):
    """Gemini API configuration"""
    api_keys: List[str] = Field(..., description="Gemini API keys")
    proxy_url: Optional[str] = Field(None, description="Proxy URL for API calls")
    max_failures: int = Field(3, description="Maximum failures before cooling", ge=1)
    cooling_period: int = Field(300, description="Cooling period in seconds", ge=60)
    health_check_interval: int = Field(60, description="Health check interval", ge=10)
    request_timeout: int = Field(45, description="Request timeout in seconds", ge=10)
    max_retries: int = Field(2, description="Maximum retry attempts", ge=0)
    
    @validator('api_keys')
    def validate_api_keys(cls, v):
        """Validate and clean Gemini API keys"""
        if isinstance(v, str):
            # Support comma-separated string
            v = [key.strip() for key in v.split(',') if key.strip()]
        
        valid_keys = []
        for key in v:
            if key and key.strip():
                cleaned_key = key.strip().strip('"\'').strip()
                if cleaned_key:
                    valid_keys.append(cleaned_key)
        
        if not valid_keys:
            raise ValueError("No valid API keys provided")
        
        # Warn about potentially invalid keys
        invalid_keys = [key for key in valid_keys if not key.startswith('AIza')]
        if invalid_keys:
            logger.warning(f"Potentially invalid API keys detected: {len(invalid_keys)} keys don't start with 'AIza'")
        
        return valid_keys
    
    class Config:
        env_prefix = "GEMINI_"

class CacheConfig(BaseSettings):
    """Cache configuration"""
    enabled: bool = Field(True, description="Enable response caching")
    max_size: int = Field(1000, description="Maximum cache size")
    ttl: int = Field(300, description="Cache TTL in seconds")
    key_prefix: str = Field("gemini_adapter", description="Cache key prefix")
    
    class Config:
        env_prefix = "CACHE_"

class PerformanceConfig(BaseSettings):
    """Performance configuration"""
    max_keepalive_connections: int = Field(20, description="Max keepalive connections")
    max_connections: int = Field(100, description="Max total connections")
    keepalive_expiry: float = Field(30.0, description="Keepalive expiry time")
    connect_timeout: float = Field(10.0, description="Connection timeout")
    read_timeout: float = Field(45.0, description="Read timeout")
    write_timeout: float = Field(10.0, description="Write timeout")
    pool_timeout: float = Field(5.0, description="Pool timeout")
    http2_enabled: bool = Field(True, description="Enable HTTP/2")
    
    class Config:
        env_prefix = "PERF_"

class ServiceConfig(BaseSettings):
    """Service configuration"""
    environment: Environment = Field(Environment.DEVELOPMENT, description="Runtime environment")
    host: str = Field("0.0.0.0", description="Service host")
    port: int = Field(8000, description="Service port")
    workers: int = Field(1, description="Number of workers")
    log_level: LogLevel = Field(LogLevel.INFO, description="Log level")
    enable_metrics: bool = Field(True, description="Enable metrics collection")
    enable_health_check: bool = Field(True, description="Enable health check endpoint")
    cors_origins: List[str] = Field(["*"], description="CORS allowed origins")
    
    class Config:
        env_prefix = "SERVICE_"

class AppConfig(BaseSettings):
    """Main application configuration"""
    gemini: GeminiConfig
    security: SecurityConfig
    cache: CacheConfig
    performance: PerformanceConfig
    service: ServiceConfig
    database: DatabaseConfig = DatabaseConfig()
    
    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"
        case_sensitive = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration consistency"""
        # Check if security is properly configured
        if self.service.environment == Environment.PRODUCTION:
            if not self.security.adapter_api_keys:
                logger.warning("Production environment without adapter API keys - service will be unsecured")
            
            if not self.gemini.api_keys:
                raise ValueError("Production environment requires Gemini API keys")
        
        # Validate cache configuration
        if self.cache.enabled and self.cache.max_size <= 0:
            raise ValueError("Cache max_size must be positive when caching is enabled")
        
        # Validate performance configuration
        if self.performance.max_connections <= 0:
            raise ValueError("Max connections must be positive")
        
        logger.info(f"Configuration validated for {self.service.environment.value} environment")
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get security configuration status"""
        return {
            "security_enabled": bool(self.security.adapter_api_keys),
            "admin_keys_configured": bool(self.security.admin_api_keys),
            "ip_blocking_enabled": self.security.enable_ip_blocking,
            "rate_limiting_enabled": self.security.enable_rate_limiting,
            "environment": self.service.environment.value
        }
    
    def get_cache_config_dict(self) -> Dict[str, Any]:
        """Get cache configuration as dictionary"""
        return {
            "enabled": self.cache.enabled,
            "max_size": self.cache.max_size,
            "ttl": self.cache.ttl,
            "key_prefix": self.cache.key_prefix
        }
    
    def get_performance_config_dict(self) -> Dict[str, Any]:
        """Get performance configuration as dictionary"""
        return {
            "max_keepalive_connections": self.performance.max_keepalive_connections,
            "max_connections": self.performance.max_connections,
            "keepalive_expiry": self.performance.keepalive_expiry,
            "connect_timeout": self.performance.connect_timeout,
            "read_timeout": self.performance.read_timeout,
            "write_timeout": self.performance.write_timeout,
            "pool_timeout": self.performance.pool_timeout,
            "http2_enabled": self.performance.http2_enabled
        }
    
    def log_configuration(self):
        """Log current configuration (without sensitive data)"""
        logger.info("=== Application Configuration ===")
        logger.info(f"Environment: {self.service.environment.value}")
        logger.info(f"Host: {self.service.host}:{self.service.port}")
        logger.info(f"Workers: {self.service.workers}")
        logger.info(f"Log Level: {self.service.log_level.value}")
        logger.info(f"Security Enabled: {bool(self.security.adapter_api_keys)}")
        logger.info(f"Admin Keys: {len(self.security.admin_api_keys)} configured")
        logger.info(f"Gemini Keys: {len(self.gemini.api_keys)} configured")
        logger.info(f"Caching: {'Enabled' if self.cache.enabled else 'Disabled'}")
        logger.info(f"Metrics: {'Enabled' if self.service.enable_metrics else 'Disabled'}")
        logger.info("=================================")

def load_configuration() -> AppConfig:
    """Load and validate configuration"""
    try:
        config = AppConfig()
        config.log_configuration()
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise

def get_environment_config() -> Dict[str, Any]:
    """Get environment-specific configuration"""
    env = os.getenv("SERVICE_ENVIRONMENT", "development").lower()
    
    base_config = {
        "service": {
            "environment": env,
            "log_level": "DEBUG" if env == "development" else "INFO",
            "enable_metrics": env != "development"
        },
        "cache": {
            "enabled": env != "development",
            "ttl": 60 if env == "development" else 300
        },
        "performance": {
            "max_connections": 50 if env == "development" else 100,
            "http2_enabled": env != "development"
        }
    }
    
    return base_config

# Global configuration instance
config: Optional[AppConfig] = None

def get_config() -> AppConfig:
    """Get the global configuration instance"""
    global config
    if config is None:
        config = load_configuration()
    return config

def reload_configuration():
    """Reload configuration (useful for runtime updates)"""
    global config
    config = load_configuration()
    logger.info("Configuration reloaded")