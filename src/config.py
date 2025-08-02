"""
Enhanced configuration management for Gemini Claude Adapter
"""

import os
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
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

class DatabaseConfig(BaseModel):
    """Database configuration"""
    redis_url: Optional[str] = Field(None, description="Redis URL for caching")
    redis_password: Optional[SecretStr] = Field(None, description="Redis password")
    redis_db: int = Field(0, description="Redis database number")
    redis_max_connections: int = Field(10, description="Maximum Redis connections")

class SecurityConfig(BaseModel):
    """Security configuration"""
    adapter_api_keys: List[str] = Field(default_factory=list, description="Client API keys")
    admin_api_keys: List[str] = Field(default_factory=list, description="Admin API keys")
    enable_ip_blocking: bool = Field(True, description="Enable IP blocking")
    max_failed_attempts: int = Field(5, description="Maximum failed attempts before blocking")
    block_duration: int = Field(300, description="IP block duration in seconds")
    enable_rate_limiting: bool = Field(True, description="Enable rate limiting")
    rate_limit_requests: int = Field(100, description="Rate limit requests per window")
    rate_limit_window: int = Field(60, description="Rate limit window in seconds")
    
    @field_validator('adapter_api_keys', mode='before')
    @classmethod
    def validate_adapter_keys(cls, v):
        """Validate and clean adapter API keys"""
        if v is None:
            return []
        if isinstance(v, str):
            # Handle empty string
            if not v.strip():
                return []
            # Handle comma-separated string
            v = [key.strip() for key in v.split(',') if key.strip()]
        elif isinstance(v, list):
            # Handle list (from JSON parsing)
            v = [str(key).strip() for key in v if str(key).strip()]
        return [key.strip() for key in v if key.strip()]
    
    @field_validator('admin_api_keys', mode='before')
    @classmethod
    def validate_admin_keys(cls, v):
        """Validate and clean admin API keys"""
        if v is None:
            return []
        if isinstance(v, str):
            # Handle empty string
            if not v.strip():
                return []
            # Handle comma-separated string
            v = [key.strip() for key in v.split(',') if key.strip()]
        elif isinstance(v, list):
            # Handle list (from JSON parsing)
            v = [str(key).strip() for key in v if str(key).strip()]
        return [key.strip() for key in v if key.strip()]

class GeminiConfig(BaseModel):
    """Gemini API configuration"""
    api_keys: List[str] = Field(default_factory=list, description="Gemini API keys")
    proxy_url: Optional[str] = Field(None, description="Proxy URL for API calls")
    max_failures: int = Field(3, description="Maximum failures before cooling", ge=1)
    cooling_period: int = Field(300, description="Cooling period in seconds", ge=60)
    health_check_interval: int = Field(60, description="Health check interval", ge=10)
    request_timeout: int = Field(45, description="Request timeout in seconds", ge=10)
    max_retries: int = Field(2, description="Maximum retry attempts", ge=0)
    
    @field_validator('api_keys', mode='before')
    @classmethod
    def validate_api_keys(cls, v):
        """Validate and clean Gemini API keys"""
        if v is None:
            raise ValueError("Gemini API keys are required")
        
        if isinstance(v, str):
            # Handle empty string
            if not v.strip():
                raise ValueError("No valid API keys provided")
            # Handle comma-separated string
            v = [key.strip() for key in v.split(',') if key.strip()]
        elif isinstance(v, list):
            # Handle list (from JSON parsing)
            v = [str(key).strip() for key in v if str(key).strip()]
        
        valid_keys = []
        for key in v:
            if key and str(key).strip():
                cleaned_key = str(key).strip().strip('"\'').strip()
                if cleaned_key:
                    valid_keys.append(cleaned_key)
        
        if not valid_keys:
            raise ValueError("No valid API keys provided")
        
        invalid_keys = [key for key in valid_keys if not key.startswith('AIza')]
        if invalid_keys:
            logger.warning(f"Potentially invalid API keys detected: {len(invalid_keys)} keys don't start with 'AIza'")
        
        return valid_keys

class CacheConfig(BaseModel):
    """Cache configuration"""
    enabled: bool = Field(True, description="Enable response caching")
    max_size: int = Field(1000, description="Maximum cache size")
    ttl: int = Field(300, description="Cache TTL in seconds")
    key_prefix: str = Field("gemini_adapter", description="Cache key prefix")

class PerformanceConfig(BaseModel):
    """Performance configuration"""
    max_keepalive_connections: int = Field(20, description="Max keepalive connections")
    max_connections: int = Field(100, description="Max total connections")
    keepalive_expiry: float = Field(30.0, description="Keepalive expiry time")
    connect_timeout: float = Field(10.0, description="Connection timeout")
    read_timeout: float = Field(45.0, description="Read timeout")
    write_timeout: float = Field(10.0, description="Write timeout")
    pool_timeout: float = Field(5.0, description="Pool timeout")
    http2_enabled: bool = Field(True, description="Enable HTTP/2")
    trust_env: bool = Field(True, description="Trust environment for proxy support")
    verify_ssl: bool = Field(True, description="Verify SSL certificates")

class ServiceConfig(BaseModel):
    """Service configuration"""
    environment: Environment = Field(Environment.DEVELOPMENT, description="Runtime environment")
    host: str = Field("0.0.0.0", description="Service host")
    port: int = Field(8000, description="Service port")
    workers: int = Field(1, description="Number of workers")
    log_level: LogLevel = Field(LogLevel.INFO, description="Log level")
    enable_metrics: bool = Field(True, description="Enable metrics collection")
    enable_health_check: bool = Field(True, description="Enable health check endpoint")
    cors_origins: List[str] = Field(["*"], description="CORS allowed origins")

class AppConfig(BaseSettings):
    """Main application configuration"""
    # Initialize with default factories to avoid parsing issues
    gemini: GeminiConfig = Field(default_factory=GeminiConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    service: ServiceConfig = Field(default_factory=ServiceConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore"  # Ignore extra fields to prevent parsing errors
    )
    
    def model_post_init(self, __context):
        """Post-initialization validation and setup"""
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration consistency"""
        if self.service.environment == Environment.PRODUCTION:
            if not self.security.adapter_api_keys:
                logger.warning("Production environment without adapter API keys - service will be unsecured")
            
            if not self.gemini.api_keys:
                raise ValueError("Production environment requires Gemini API keys")
        
        if self.cache.enabled and self.cache.max_size <= 0:
            raise ValueError("Cache max_size must be positive when caching is enabled")
        
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

# Alternative configuration loader that handles env vars manually if needed
def load_config_from_env() -> AppConfig:
    """Load configuration with manual environment variable handling as fallback"""
    try:
        # Try standard pydantic-settings loading first
        return AppConfig()
    except Exception as e:
        logger.warning(f"Standard config loading failed: {e}. Trying manual env loading...")
        
        # Manual environment variable loading as fallback
        gemini_config = GeminiConfig(
            api_keys=os.getenv('GEMINI__API_KEYS', '').split(',') if os.getenv('GEMINI__API_KEYS') else [],
            proxy_url=os.getenv('GEMINI__PROXY_URL'),
            max_failures=int(os.getenv('GEMINI__MAX_FAILURES', '3')),
            cooling_period=int(os.getenv('GEMINI__COOLING_PERIOD', '300')),
            health_check_interval=int(os.getenv('GEMINI__HEALTH_CHECK_INTERVAL', '60')),
            request_timeout=int(os.getenv('GEMINI__REQUEST_TIMEOUT', '45')),
            max_retries=int(os.getenv('GEMINI__MAX_RETRIES', '2'))
        )
        
        security_config = SecurityConfig(
            adapter_api_keys=os.getenv('SECURITY__ADAPTER_API_KEYS', '').split(',') if os.getenv('SECURITY__ADAPTER_API_KEYS') else [],
            admin_api_keys=os.getenv('SECURITY__ADMIN_API_KEYS', '').split(',') if os.getenv('SECURITY__ADMIN_API_KEYS') else [],
            enable_ip_blocking=os.getenv('SECURITY__ENABLE_IP_BLOCKING', 'true').lower() == 'true',
            max_failed_attempts=int(os.getenv('SECURITY__MAX_FAILED_ATTEMPTS', '5')),
            block_duration=int(os.getenv('SECURITY__BLOCK_DURATION', '300')),
            enable_rate_limiting=os.getenv('SECURITY__ENABLE_RATE_LIMITING', 'true').lower() == 'true',
            rate_limit_requests=int(os.getenv('SECURITY__RATE_LIMIT_REQUESTS', '100')),
            rate_limit_window=int(os.getenv('SECURITY__RATE_LIMIT_WINDOW', '60'))
        )
        
        cache_config = CacheConfig(
            enabled=os.getenv('CACHE__ENABLED', 'true').lower() == 'true',
            max_size=int(os.getenv('CACHE__MAX_SIZE', '1000')),
            ttl=int(os.getenv('CACHE__TTL', '300')),
            key_prefix=os.getenv('CACHE__KEY_PREFIX', 'gemini_adapter')
        )
        
        performance_config = PerformanceConfig(
            max_keepalive_connections=int(os.getenv('PERFORMANCE__MAX_KEEPALIVE_CONNECTIONS', '20')),
            max_connections=int(os.getenv('PERFORMANCE__MAX_CONNECTIONS', '100')),
            keepalive_expiry=float(os.getenv('PERFORMANCE__KEEPALIVE_EXPIRY', '30.0')),
            connect_timeout=float(os.getenv('PERFORMANCE__CONNECT_TIMEOUT', '10.0')),
            read_timeout=float(os.getenv('PERFORMANCE__READ_TIMEOUT', '45.0')),
            write_timeout=float(os.getenv('PERFORMANCE__WRITE_TIMEOUT', '10.0')),
            pool_timeout=float(os.getenv('PERFORMANCE__POOL_TIMEOUT', '5.0')),
            http2_enabled=os.getenv('PERFORMANCE__HTTP2_ENABLED', 'true').lower() == 'true',
            trust_env=os.getenv('PERFORMANCE__TRUST_ENV', 'true').lower() == 'true',
            verify_ssl=os.getenv('PERFORMANCE__VERIFY_SSL', 'true').lower() == 'true'
        )
        
        service_config = ServiceConfig(
            environment=Environment(os.getenv('SERVICE__ENVIRONMENT', Environment.DEVELOPMENT.value)),
            host=os.getenv('SERVICE__HOST', '0.0.0.0'),
            port=int(os.getenv('SERVICE__PORT', '8000')),
            workers=int(os.getenv('SERVICE__WORKERS', '1')),
            log_level=LogLevel(os.getenv('SERVICE__LOG_LEVEL', LogLevel.INFO.value)),
            enable_metrics=os.getenv('SERVICE__ENABLE_METRICS', 'true').lower() == 'true',
            enable_health_check=os.getenv('SERVICE__ENABLE_HEALTH_CHECK', 'true').lower() == 'true',
            cors_origins=os.getenv('SERVICE__CORS_ORIGINS', '*').split(',')
        )
        
        database_config = DatabaseConfig(
            redis_url=os.getenv('DATABASE__REDIS_URL'),
            redis_password=SecretStr(os.getenv('DATABASE__REDIS_PASSWORD', '')) if os.getenv('DATABASE__REDIS_PASSWORD') else None,
            redis_db=int(os.getenv('DATABASE__REDIS_DB', '0')),
            redis_max_connections=int(os.getenv('DATABASE__REDIS_MAX_CONNECTIONS', '10'))
        )
        
        # Create config manually
        config = AppConfig.model_construct(
            gemini=gemini_config,
            security=security_config,
            cache=cache_config,
            performance=performance_config,
            service=service_config,
            database=database_config
        )
        
        # Manually trigger post-init validation
        config._validate_config()
        return config

# Global configuration instance
_config: Optional[AppConfig] = None

def load_configuration() -> AppConfig:
    """Load and validate configuration"""
    global _config
    try:
        _config = load_config_from_env()
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
