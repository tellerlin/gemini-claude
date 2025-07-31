"""
Enhanced error handling and monitoring for Gemini Claude Adapter
"""

import time
import json
import traceback
from enum import Enum
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from collections import defaultdict, deque
import asyncio
import logging

logger = logging.getLogger(__name__)

class ErrorType(Enum):
    """Error type classification for better handling and monitoring"""
    RATE_LIMIT = "rate_limit"
    AUTH_ERROR = "auth_error"
    QUOTA_EXCEEDED = "quota_exceeded"
    SERVER_ERROR = "server_error"
    NETWORK_ERROR = "network_error"
    TIMEOUT = "timeout"
    VALIDATION_ERROR = "validation_error"
    INTERNAL_ERROR = "internal_error"
    KEY_FAILURE = "key_failure"

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ErrorContext:
    """Context information for errors"""
    timestamp: float
    error_type: ErrorType
    severity: ErrorSeverity
    message: str
    stack_trace: Optional[str] = None
    request_id: Optional[str] = None
    client_key: Optional[str] = None
    endpoint: Optional[str] = None
    gemini_key: Optional[str] = None
    retry_after: Optional[int] = None
    additional_info: Optional[Dict[str, Any]] = None

class ErrorClassifier:
    """Intelligent error classification system"""
    
    @staticmethod
    def classify_error(error: Exception, context: Dict[str, Any] = None) -> ErrorContext:
        """Classify an error with appropriate type and severity"""
        error_msg = str(error).lower()
        error_type = ErrorType.INTERNAL_ERROR
        severity = ErrorSeverity.MEDIUM
        retry_after = None
        
        # Import here to avoid circular imports
        import re
        
        # Extract HTTP status code if available
        status_code = 0
        status_patterns = [
            r'status code (\d{3})',
            r'HTTP (\d{3})',
            r'Error (\d{3})',
            r'(\d{3})'
        ]
        
        for pattern in status_patterns:
            match = re.search(pattern, error_msg)
            if match:
                status_code = int(match.group(1))
                break
        
        # Classify error based on patterns and status codes
        if status_code == 429 or 'rate limit' in error_msg or 'too many requests' in error_msg:
            error_type = ErrorType.RATE_LIMIT
            severity = ErrorSeverity.MEDIUM
            retry_after = 60  # Default retry after 60 seconds
        elif status_code == 401 or 'unauthorized' in error_msg or 'invalid api key' in error_msg:
            error_type = ErrorType.AUTH_ERROR
            severity = ErrorSeverity.HIGH
        elif status_code == 403 or 'forbidden' in error_msg or 'access denied' in error_msg:
            error_type = ErrorType.AUTH_ERROR
            severity = ErrorSeverity.HIGH
        elif 'quota' in error_msg or 'limit exceeded' in error_msg or 'billing' in error_msg:
            error_type = ErrorType.QUOTA_EXCEEDED
            severity = ErrorSeverity.HIGH
            retry_after = 300  # 5 minutes for quota issues
        elif status_code >= 500:
            error_type = ErrorType.SERVER_ERROR
            severity = ErrorSeverity.MEDIUM
        elif 'timeout' in error_msg or 'connection' in error_msg or 'network' in error_msg:
            error_type = ErrorType.NETWORK_ERROR
            severity = ErrorSeverity.LOW
        elif 'validation' in error_msg or 'invalid' in error_msg:
            error_type = ErrorType.VALIDATION_ERROR
            severity = ErrorSeverity.LOW
        elif 'api key' in error_msg or 'key' in error_msg:
            error_type = ErrorType.KEY_FAILURE
            severity = ErrorSeverity.HIGH
        
        # Adjust severity based on context
        if context:
            if context.get('is_streaming', False):
                severity = ErrorSeverity.LOW  # Streaming errors are less critical
            
            if context.get('is_admin_endpoint', False):
                severity = ErrorSeverity.HIGH  # Admin endpoint errors are more critical
        
        return ErrorContext(
            timestamp=time.time(),
            error_type=error_type,
            severity=severity,
            message=str(error),
            stack_trace=traceback.format_exc(),
            retry_after=retry_after,
            additional_info=context or {}
        )

class ErrorMonitor:
    """Error monitoring and aggregation system"""
    
    def __init__(self, max_errors: int = 1000, retention_hours: int = 24):
        self.max_errors = max_errors
        self.retention_hours = retention_hours
        self.errors: deque = deque(maxlen=max_errors)
        self.error_counts = defaultdict(int)
        self.error_by_type = defaultdict(list)
        self.error_by_client = defaultdict(list)
        self.lock = asyncio.Lock()
        
    async def record_error(self, error_context: ErrorContext):
        """Record an error for monitoring"""
        async with self.lock:
            # Clean old errors
            self._cleanup_old_errors()
            
            # Add error to collections
            self.errors.append(error_context)
            self.error_counts[error_context.error_type.value] += 1
            self.error_by_type[error_context.error_type.value].append(error_context)
            
            if error_context.client_key:
                self.error_by_client[error_context.client_key].append(error_context)
            
            # Log the error
            self._log_error(error_context)
    
    def _cleanup_old_errors(self):
        """Remove errors older than retention period"""
        cutoff_time = time.time() - (self.retention_hours * 3600)
        
        # Clean main errors deque
        while self.errors and self.errors[0].timestamp < cutoff_time:
            self.errors.popleft()
        
        # Clean error_by_type
        for error_type in self.error_by_type:
            self.error_by_type[error_type] = [
                error for error in self.error_by_type[error_type]
                if error.timestamp >= cutoff_time
            ]
        
        # Clean error_by_client
        for client_key in self.error_by_client:
            self.error_by_client[client_key] = [
                error for error in self.error_by_client[client_key]
                if error.timestamp >= cutoff_time
            ]
    
    def _log_error(self, error_context: ErrorContext):
        """Log error with appropriate level"""
        log_message = f"[{error_context.error_type.value.upper()}] {error_context.message}"
        
        if error_context.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message, extra={"error_context": asdict(error_context)})
        elif error_context.severity == ErrorSeverity.HIGH:
            logger.error(log_message, extra={"error_context": asdict(error_context)})
        elif error_context.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message, extra={"error_context": asdict(error_context)})
        else:
            logger.info(log_message, extra={"error_context": asdict(error_context)})
    
    async def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics"""
        async with self.lock:
            total_errors = len(self.errors)
            recent_errors = sum(1 for error in self.errors 
                              if error.timestamp > time.time() - 3600)  # Last hour
            
            error_by_type = {}
            for error_type, count in self.error_counts.items():
                error_by_type[error_type] = {
                    "count": count,
                    "percentage": (count / total_errors * 100) if total_errors > 0 else 0
                }
            
            # Top error clients
            top_clients = {}
            for client_key, errors in self.error_by_client.items():
                if client_key:
                    top_clients[client_key[:8] + "..."] = len(errors)
            
            return {
                "total_errors": total_errors,
                "recent_errors_1h": recent_errors,
                "error_by_type": error_by_type,
                "top_error_clients": dict(sorted(top_clients.items(), key=lambda x: x[1], reverse=True)[:10]),
                "last_error": self.errors[-1].message if self.errors else None,
                "last_error_time": datetime.fromtimestamp(self.errors[-1].timestamp).isoformat() if self.errors else None
            }
    
    async def get_recent_errors(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent errors"""
        async with self.lock:
            recent_errors = list(self.errors)[-limit:]
            return [
                {
                    "timestamp": error.timestamp,
                    "type": error.error_type.value,
                    "severity": error.severity.value,
                    "message": error.message,
                    "client_key": error.client_key[:8] + "..." if error.client_key else None,
                    "endpoint": error.endpoint,
                    "gemini_key": error.gemini_key[:8] + "..." if error.gemini_key else None
                }
                for error in reversed(recent_errors)
            ]

class CircuitBreaker:
    """Circuit breaker for preventing cascading failures"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.lock = asyncio.Lock()
    
    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        async with self.lock:
            if self.state == "OPEN":
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = "HALF_OPEN"
                    self.failure_count = 0
                else:
                    raise Exception("Circuit breaker is OPEN")
            
            try:
                result = await func(*args, **kwargs)
                if self.state == "HALF_OPEN":
                    self.state = "CLOSED"
                    self.failure_count = 0
                return result
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"
                
                raise e
    
    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state"""
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time,
            "recovery_timeout": self.recovery_timeout
        }

# Global error monitor instance
error_monitor = ErrorMonitor()

# Circuit breakers for different operations
key_manager_circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)
api_call_circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)

async def monitor_errors(func):
    """Decorator to monitor errors in functions"""
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            # Extract context from function call
            context = {
                "function": func.__name__,
                "args_count": len(args),
                "kwargs_keys": list(kwargs.keys())
            }
            
            # Try to extract client key and endpoint from args/kwargs
            if args and hasattr(args[0], '__dict__'):
                # Method call, try to get request info
                for arg in args:
                    if hasattr(arg, 'client_key'):
                        context['client_key'] = arg.client_key
                    if hasattr(arg, 'url'):
                        context['endpoint'] = str(arg.url.path)
            
            error_context = ErrorClassifier.classify_error(e, context)
            await error_monitor.record_error(error_context)
            
            raise e
    return wrapper