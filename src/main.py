import asyncio
import time
import random
from typing import List, Dict, Optional, Any, Union, Set, Tuple, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
from itertools import cycle
import json
import httpx
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, field_validator
from loguru import logger
import litellm
from contextlib import asynccontextmanager
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from starlette.status import HTTP_401_UNAUTHORIZED, HTTP_403_FORBIDDEN
from collections import defaultdict, deque
import hashlib

# Correct relative imports for the new 'src' structure
try:
    from .config import get_config, AppConfig, GeminiConfig
    from .error_handling import error_monitor, monitor_errors, ErrorClassifier
    from .performance import response_cache, http_client, performance_monitor, monitor_performance, get_performance_stats
    from .anthropic_api import (
        MessagesRequest, MessagesResponse, TokenCountRequest, TokenCountResponse,
        AnthropicToGeminiConverter, GeminiToAnthropicConverter, 
        StreamingResponseGenerator, ToolConverter, log_request_beautifully
    )
except ImportError:
    # Fallback for environments where relative imports might fail
    from config import get_config, AppConfig, GeminiConfig
    from error_handling import error_monitor, monitor_errors, ErrorClassifier
    from performance import response_cache, http_client, performance_monitor, monitor_performance, get_performance_stats
    from anthropic_api import (
        MessagesRequest, MessagesResponse, TokenCountRequest, TokenCountResponse,
        AnthropicToGeminiConverter, GeminiToAnthropicConverter, 
        StreamingResponseGenerator, ToolConverter, log_request_beautifully
    )


# Load environment variables from .env file
load_dotenv()

# Configure logging with improved format
os.makedirs("logs", exist_ok=True)
logger.add(
    "logs/gemini_adapter_{time}.log", 
    rotation="1 day", 
    retention="7 days", 
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
)

class KeyStatus(Enum):
    ACTIVE = "active"
    COOLING = "cooling"
    FAILED = "failed"

@dataclass
class APIKeyInfo:
    key: str
    status: KeyStatus = KeyStatus.ACTIVE
    failure_count: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    cooling_until: Optional[float] = None
    total_requests: int = 0
    successful_requests: int = 0

# GeminiConfig is now handled by AppConfig.gemini in config module

class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    model: str = "gemini-2.5-pro"
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, ge=1, le=8192)
    stream: bool = False
    
    # CORRECTED: Use field_validator and fix the logic bug
    @field_validator('messages')
    @classmethod
    def validate_messages(cls, v):
        if not v:
            raise ValueError("Messages cannot be empty")
        for msg in v:
            # CORRECTED: Fixed 'not not in' typo
            if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                raise ValueError("Each message must have 'role' and 'content' fields")
        return v

# Security Configuration (enhanced with new config system)
class SecurityConfig:
    def __init__(self, app_config: AppConfig):
        # Use new AppConfig only
        self.valid_api_keys: Set[str] = set(app_config.security.adapter_api_keys)
        self.admin_keys: Set[str] = set(app_config.security.admin_api_keys)
        self.enable_ip_blocking = app_config.security.enable_ip_blocking
        self.max_failed_attempts = app_config.security.max_failed_attempts
        self.block_duration = app_config.security.block_duration
        self.enable_rate_limiting = app_config.security.enable_rate_limiting
        self.rate_limit_requests = app_config.security.rate_limit_requests
        self.rate_limit_window = app_config.security.rate_limit_window
        
        self.security_enabled = bool(self.valid_api_keys)
        
        if self.security_enabled:
            logger.info(f"Security enabled with {len(self.valid_api_keys)} client keys")
        else:
            logger.warning("Security disabled - no SECURITY_ADAPTER_API_KEYS configured")
            
        if self.admin_keys:
            logger.info(f"Admin access enabled with {len(self.admin_keys)} admin keys")
        else:
            logger.info("No admin keys configured - client keys will have admin access")

# Global security config (will be initialized with AppConfig)
security_config: Optional[SecurityConfig] = None

# Authentication schemes
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)

async def verify_api_key(
    api_key: Optional[str] = Depends(api_key_header),
    bearer_token: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme)
) -> str:
    """
    Verify API key from either X-API-Key header or Bearer token
    Returns the validated key or raises HTTPException
    """
    if not security_config.security_enabled:
        logger.debug("Security disabled, allowing access")
        return "insecure_mode"
    
    # Try X-API-Key header first
    if api_key and api_key in security_config.valid_api_keys:
        return api_key
    
    # Try Bearer token
    if bearer_token and bearer_token.credentials in security_config.valid_api_keys:
        return bearer_token.credentials
    
    # Check if any key was provided
    if not api_key and not bearer_token:
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="API key required. Use X-API-Key header or Authorization: Bearer <key>",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    # Invalid key provided
    raise HTTPException(
        status_code=HTTP_403_FORBIDDEN,
        detail="Invalid API key"
    )

async def verify_admin_key(
    api_key: Optional[str] = Depends(api_key_header),
    bearer_token: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme)
) -> str:
    """
    Verify admin API key for management endpoints
    """
    if not security_config.admin_keys:
        # If no admin keys configured, fall back to regular API key verification
        return await verify_api_key(api_key, bearer_token)
    
    # Try X-API-Key header first
    if api_key and api_key in security_config.admin_keys:
        return api_key
    
    # Try Bearer token
    if bearer_token and bearer_token.credentials in security_config.admin_keys:
        return bearer_token.credentials
    
    # Check if any key was provided
    if not api_key and not bearer_token:
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Admin API key required",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    # Invalid admin key
    raise HTTPException(
        status_code=HTTP_403_FORBIDDEN,
        detail="Invalid admin API key"
    )

class GeminiKeyManager:
    def __init__(self, config):  # config is now AppConfig.gemini
        self.config = config
        self.keys: Dict[str, APIKeyInfo] = {}
        self.key_cycle = None
        self.lock = asyncio.Lock()
        self.last_key_used = None  # Add this attribute to avoid consecutive use of the same key
        # Add key performance tracking
        self.key_performance = defaultdict(lambda: {"response_times": deque(maxlen=100), "errors": 0})

        for key in config.api_keys:
            if key and key.strip():
                self.keys[key] = APIKeyInfo(key=key)

        if not self.keys:
            raise ValueError("No valid API keys provided to key manager")

        logger.info(f"Initialized {len(self.keys)} API keys with performance tracking.")

    async def get_available_key(self) -> Optional[APIKeyInfo]:
        async with self.lock:
            # Check if there are keys that can be recovered
            recovered_count = await self._check_and_recover_keys_internal()
            if recovered_count > 0:
                logger.info(f"Recovered {recovered_count} keys during get_available_key")
            
            active_keys = [k for k in self.keys.values() if k.status == KeyStatus.ACTIVE]
            
            if not active_keys:
                logger.warning("No available API keys.")
                return None

            # Intelligent key selection: choose best key based on performance metrics
            selected_key = self._select_best_key(active_keys)
            
            self.last_key_used = selected_key.key
            return selected_key
    
    async def _check_and_recover_keys_internal(self) -> int:
        """Internal method: check and recover keys in cooling, returns the number of recovered keys"""
        current_time = time.time()
        recovered_count = 0
        
        for key_info in self.keys.values():
            if (key_info.status == KeyStatus.COOLING and 
                key_info.cooling_until and 
                current_time > key_info.cooling_until):
                
                old_status = key_info.status
                key_info.status = KeyStatus.ACTIVE
                key_info.failure_count = 0
                key_info.cooling_until = None
                recovered_count += 1
                
                logger.info(f"API key {key_info.key[:8]}... has cooled down and recovered from {old_status.value} to active")
        
        # If any key is recovered, reset the cycle
        if recovered_count > 0:
            self.key_cycle = None
        
        return recovered_count

    async def mark_key_failed(self, key: str, error: str):
        async with self.lock:
            key_info = self.keys.get(key)
            if not key_info:
                logger.warning(f"Attempted to mark unknown key as failed: {key[:8]}...")
                return

            # Enhanced error classification system
            error_type, cooling_time = self._classify_error(error)
            
            if error_type == 'PERMANENT':
                key_info.status = KeyStatus.FAILED
                logger.error(f"API key {key[:8]}... permanently failed due to {error_type}: {error}")
                return
            
            key_info.status = KeyStatus.COOLING
            key_info.failure_count += 1
            key_info.last_failure_time = time.time()
            key_info.cooling_until = time.time() + cooling_time
            logger.warning(f"API key {key[:8]}... failed ({error_type}), cooling for {cooling_time}s. Error: {error}")
    
    def _classify_error(self, error: str) -> Tuple[str, int]:
        """
        Classify error type and determine appropriate cooling time
        Returns: (error_type, cooling_time_seconds)
        """
        import re
        
        error_lower = error.lower()
        
        # Extract HTTP status code from error message if available
        status_code = 0
        status_patterns = [
            r'status code (\d{3})',
            r'HTTP (\d{3})',
            r'Error (\d{3})',
            r'(\d{3})'
        ]
        
        for pattern in status_patterns:
            match = re.search(pattern, error)
            if match:
                status_code = int(match.group(1))
                break
        
        # PERMANENT FAILURES - Key should be disabled permanently
        permanent_patterns = [
            'invalid api key', 'api key not found', 'api key disabled',
            'account disabled', 'account suspended', 'account terminated',
            'unauthorized', 'authentication failed', 'access denied',
            'billing disabled', 'payment required', 'payment failed',
            'quota exceeded permanently', 'api key revoked',
            'project not found', 'project deleted', 'service disabled',
            'forbidden', 'permission denied'
        ]
        
        # HTTP status codes that indicate permanent failures
        permanent_status_codes = [401, 402, 403, 404]
        
        if any(pattern in error_lower for pattern in permanent_patterns) or status_code in permanent_status_codes:
            return 'PERMANENT', -1  # -1 means permanent disable
        
        # EXTENDED COOLING - Quota/billing related but potentially recoverable
        extended_patterns = [
            'quota', 'rate limit', 'rate_limit', 'too many requests',
            'resource exhausted', 'limit exceeded', 'usage limit',
            'billing quota', 'daily limit', 'monthly limit'
        ]
        
        if any(pattern in error_lower for pattern in extended_patterns) or status_code == 429:
            return 'EXTENDED_COOLING', 1800  # 30 minutes
        
        # SERVER ERRORS - Google's side issues
        if status_code >= 500:
            return 'SERVER_ERROR', 300  # 5 minutes
        
        # NETWORK/TIMEOUT ERRORS
        timeout_patterns = [
            'timeout', 'connection', 'network', 'dns', 'unreachable',
            'read timeout', 'connect timeout', 'request timeout',
            'connection reset', 'connection refused'
        ]
        
        if any(pattern in error_lower for pattern in timeout_patterns):
            return 'NETWORK_ERROR', 600  # 10 minutes
        
        # DEFAULT - Standard cooling
        return 'DEFAULT', 300  # 5 minutes

    def _select_best_key(self, active_keys: List[APIKeyInfo]) -> APIKeyInfo:
        """Select the best key based on performance metrics"""
        def key_score(key_info: APIKeyInfo) -> float:
            # Success rate weight 60%
            success_rate = key_info.successful_requests / max(key_info.total_requests, 1)
            
            # Response time weight 30%
            perf_data = self.key_performance[key_info.key]
            avg_response_time = sum(perf_data["response_times"]) / max(len(perf_data["response_times"]), 1)
            response_score = 1.0 / (1.0 + avg_response_time)  # Lower response time gets a higher score
            
            # Recent usage weight 10% (to avoid always using the same key)
            recent_use_penalty = 0.9 if key_info.key == self.last_key_used else 1.0
            
            return (success_rate * 0.6 + response_score * 0.3) * recent_use_penalty
        
        return max(active_keys, key=key_score)
    
    async def record_key_performance(self, key: str, response_time: float, success: bool):
        """Record key performance data"""
        async with self.lock:
            perf_data = self.key_performance[key]
            perf_data["response_times"].append(response_time)
            if not success:
                perf_data["errors"] += 1
    
    async def mark_key_success(self, key: str):
        async with self.lock:
            key_info = self.keys.get(key)
            if not key_info:
                logger.warning(f"Attempted to mark unknown key as successful: {key[:8]}...")
                return
            
            # Reset failure count on success
            key_info.failure_count = 0
            key_info.last_success_time = time.time()
            key_info.successful_requests += 1
            key_info.total_requests += 1

    async def get_stats(self) -> Dict[str, Any]:
        async with self.lock:
            return {
                "total_keys": len(self.keys),
                "active_keys": sum(1 for k in self.keys.values() if k.status == KeyStatus.ACTIVE),
                "cooling_keys": sum(1 for k in self.keys.values() if k.status == KeyStatus.COOLING),
                "failed_keys": sum(1 for k in self.keys.values() if k.status == KeyStatus.FAILED),
                "keys_detail": [
                    {
                        "key": f"{k.key[:8]}...",
                        "status": k.status.value,
                        "failure_count": k.failure_count,
                        "total_requests": k.total_requests,
                        "successful_requests": k.successful_requests,
                        "success_rate": (k.successful_requests / k.total_requests * 100) if k.total_requests > 0 else 0,
                        "last_success": datetime.fromtimestamp(k.last_success_time).isoformat() if k.last_success_time else None,
                        "last_failure": datetime.fromtimestamp(k.last_failure_time).isoformat() if k.last_failure_time else None,
                        "cooling_until": datetime.fromtimestamp(k.cooling_until).isoformat() if k.cooling_until else None,
                    }
                    for k in self.keys.values()
                ],
            }

    async def _check_and_recover_keys(self) -> int:
        """Check and recover keys in cooling, returns the number of recovered keys"""
        async with self.lock:
            return await self._check_and_recover_keys_internal()
    
    async def reset_key(self, key_prefix: str) -> Dict[str, Any]:
        """Reset a key's status by key prefix"""
        async with self.lock:
            if len(key_prefix) < 4:
                return {"error": "Key prefix must be at least 4 characters long"}
            
            matched_keys = [key for key in self.keys.keys() if key.startswith(key_prefix)]
            
            if not matched_keys:
                return {"error": f"No key found with prefix '{key_prefix}'"}
            
            if len(matched_keys) > 1:
                return {"error": f"Multiple keys found with prefix '{key_prefix}'. Please use a more specific prefix."}
            
            matched_key = matched_keys[0]
            key_info = self.keys[matched_key]
            old_status = key_info.status
            
            key_info.status = KeyStatus.ACTIVE
            key_info.failure_count = 0
            key_info.cooling_until = None
            key_info.last_failure_time = None
            
            # Reset cycle to include the newly activated key
            self.key_cycle = None
            
            logger.info(f"Reset API key {matched_key[:8]}... from {old_status.value} to active status")
            return {"message": f"Successfully reset key {matched_key[:8]}... from {old_status.value} to active"}
    
    async def attempt_key_recovery(self, key_prefix: str) -> Dict[str, Any]:
        """Attempt to recover a permanently failed key"""
        async with self.lock:
            if len(key_prefix) < 4:
                return {"error": "Key prefix must be at least 4 characters long"}
            
            matched_keys = [key for key in self.keys.keys() if key.startswith(key_prefix)]
            
            if not matched_keys:
                return {"error": f"No key found with prefix '{key_prefix}'"}
            
            if len(matched_keys) > 1:
                return {"error": f"Multiple keys found with prefix '{key_prefix}'. Please use a more specific prefix."}
            
            matched_key = matched_keys[0]
            key_info = self.keys[matched_key]
            
            # Only attempt recovery for failed keys
            if key_info.status != KeyStatus.FAILED:
                return {"error": f"Key {matched_key[:8]}... is not in FAILED status (current: {key_info.status.value})"}
            
            old_status = key_info.status
            key_info.status = KeyStatus.ACTIVE
            key_info.failure_count = 0
            key_info.cooling_until = None
            key_info.last_failure_time = None
            
            # Reset cycle to include the recovered key
            self.key_cycle = None
            
            logger.info(f"Recovered API key {matched_key[:8]}... from {old_status.value} to active status")
            return {"message": f"Successfully recovered key {matched_key[:8]}... from {old_status.value} to active"}

class LiteLLMAdapter:
    def __init__(self, config: GeminiConfig, key_manager: GeminiKeyManager):
        self.config = config
        self.key_manager = key_manager
        self.anthropic_to_gemini = AnthropicToGeminiConverter()
        self.gemini_to_anthropic = GeminiToAnthropicConverter()
        self.tool_converter = ToolConverter()
        
        # Add request deduplication
        self._request_deduplicator: Dict[str, asyncio.Future] = {}
        self._dedup_lock = asyncio.Lock()
        
        if config.proxy_url:
            os.environ['HTTPS_PROXY'] = config.proxy_url
            os.environ['HTTP_PROXY'] = config.proxy_url
            logger.info(f"Using proxy: {config.proxy_url}")
        
        # Optimize litellm configuration
        litellm.request_timeout = 30  # Reduce timeout
        litellm.max_retries = 0
        litellm.set_verbose = False
        litellm.drop_params = True
        litellm.num_retries = 0  # Disable internal retries, use our retry logic

    @monitor_errors
    async def chat_completion(self, request: ChatRequest) -> Union[Dict, Any]:
        # CORRECTED: Safe and correct request deduplication logic
        request_hash = hashlib.md5(
            json.dumps({
                "model": request.model,
                "messages": request.messages,
                "temperature": request.temperature,
                "stream": request.stream
            }, sort_keys=True).encode()
        ).hexdigest()

        if not request.stream:
            # Check if a request is already in progress
            async with self._dedup_lock:
                if request_hash in self._request_deduplicator:
                    # Wait for the result of the existing request
                    return await self._request_deduplicator[request_hash]
                
                # This is a new request, create a future for it
                future = asyncio.Future()
                self._request_deduplicator[request_hash] = future
        else:
            future = None

        try:
            # The rest of the logic proceeds here.
            # If this coroutine completes, it will set the future's result.
            response = await self._execute_chat_completion(request)

            if future:
                future.set_result(response)
            
            return response

        except Exception as e:
            if future:
                future.set_exception(e)
            raise e # Re-raise the exception for the original caller
        
        finally:
            # Clean up the future from the map once it's resolved
            if future:
                async with self._dedup_lock:
                    if request_hash in self._request_deduplicator:
                        del self._request_deduplicator[request_hash]
    
    async def _execute_chat_completion(self, request: ChatRequest) -> Union[Dict, Any]:
        last_error = None
        attempted_keys = set()
        
        # Check cache
        if not request.stream:
            cache_key = {
                "model": request.model,
                "messages": request.messages,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens
            }
            cached_response = await response_cache.get(cache_key)
            if cached_response:
                return cached_response
        
        max_concurrent = min(3, len(self.key_manager.keys))
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def try_key_request(key_info: APIKeyInfo):
            async with semaphore:
                if key_info.key in attempted_keys:
                    return None
                    
                attempted_keys.add(key_info.key)
                start_time = time.time()
                
                try:
                    async with self.key_manager.lock:
                        key_info.total_requests += 1
                    
                    # CORRECTED: Ensured http_client.client is passed to litellm
                    kwargs = {
                        "model": f"gemini/{request.model}",
                        "messages": request.messages,
                        "api_key": key_info.key,
                        "temperature": request.temperature,
                        "stream": request.stream,
                        "client": http_client.client, # Use the optimized client
                    }
                    
                    if request.max_tokens:
                        kwargs["max_tokens"] = request.max_tokens
                    
                    response = await asyncio.wait_for(
                        litellm.acompletion(**kwargs),
                        timeout=self.config.request_timeout + 5.0
                    )
                    
                    response_time = time.time() - start_time
                    await self.key_manager.mark_key_success(key_info.key)
                    await self.key_manager.record_key_performance(key_info.key, response_time, True)
                    
                    return response
                    
                except Exception as e:
                    response_time = time.time() - start_time
                    await self.key_manager.mark_key_failed(key_info.key, str(e))
                    await self.key_manager.record_key_performance(key_info.key, response_time, False)
                    raise e
        
        active_keys = [k for k in self.key_manager.keys.values() if k.status == KeyStatus.ACTIVE]
        if not active_keys:
            raise HTTPException(status_code=502, detail="No available API keys")
        
        keys_to_try = active_keys[:max_concurrent]
        
        try:
            tasks = [try_key_request(key_info) for key_info in keys_to_try]
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            
            for task in pending:
                task.cancel()
            
            for task in done:
                try:
                    result = await task
                    if result:
                        if not request.stream:
                            await response_cache.set(cache_key, result)
                        return result
                except Exception as e:
                    last_error = str(e)
                    continue
                    
        except Exception as e:
            last_error = str(e)
        
        raise HTTPException(status_code=502, detail=f"All available keys failed. Last error: {last_error}")
    
    async def anthropic_messages_completion(self, request: MessagesRequest) -> Union[MessagesResponse, AsyncGenerator[str, None]]:
        """Handle Anthropic Messages API requests"""
        gemini_request_dict = self.anthropic_to_gemini.convert_request(request)
        
        if request.tools:
            gemini_tools = self.tool_converter.convert_tools_to_gemini(request.tools)
            gemini_request_dict["tools"] = gemini_tools
            
            if request.tool_choice:
                tool_config = self.tool_converter.convert_tool_choice_to_gemini(request.tool_choice)
                gemini_request_dict["tool_config"] = {"function_calling_config": {"mode": tool_config}}
        
        chat_request = ChatRequest(
            messages=gemini_request_dict["messages"],
            model=gemini_request_dict["model"].replace("gemini/", ""),
            temperature=gemini_request_dict["temperature"],
            max_tokens=gemini_request_dict.get("max_tokens"),
            stream=gemini_request_dict["stream"]
        )
        
        if request.stream:
            gemini_stream = await self.chat_completion(chat_request)
            streaming_generator = StreamingResponseGenerator(request)
            return streaming_generator.generate_sse_events(gemini_stream)
        else:
            gemini_response = await self.chat_completion(chat_request)
            return self.gemini_to_anthropic.convert_response(gemini_response, request)

# CORRECTED: Added missing health check task function
async def optimized_health_check_task():
    """Optimized health check task that runs periodically"""
    while True:
        try:
            # Check and recover keys
            if 'key_manager' in globals() and key_manager is not None:
                recovered_count = await key_manager._check_and_recover_keys()
                if recovered_count > 0:
                    logger.info(f"Health check recovered {recovered_count} keys")
            
            # Wait for next health check
            app_config = get_config()
            await asyncio.sleep(app_config.gemini.health_check_interval)
            
        except asyncio.CancelledError:
            logger.info("Health check task cancelled.")
            break
        except Exception as e:
            logger.error(f"Health check error: {e}")
            await asyncio.sleep(60)  # Fallback interval on error

@asynccontextmanager
async def lifespan(app: FastAPI):
    global key_manager, adapter, security_config
    
    os.makedirs("logs", exist_ok=True)
    logger.add(
        "logs/gemini_adapter_{time}.log", 
        rotation="1 day", 
        retention="7 days", 
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        enqueue=True, # Important for multiprocessing
        catch=True
    )

    try:
        app_config = get_config()
        
        security_config = SecurityConfig(app_config)
        
        key_manager = GeminiKeyManager(app_config.gemini)
        adapter = LiteLLMAdapter(app_config.gemini, key_manager)
        
        await http_client.initialize()
        
        health_task = asyncio.create_task(optimized_health_check_task())
        
        logger.info("Gemini Claude Adapter v2.1.0 started successfully with optimizations.")
        logger.info(f"Environment: {app_config.service.environment.value}")
        logger.info(f"Caching: {'Enabled' if app_config.cache.enabled else 'Disabled'}")
        logger.info(f"Performance monitoring: {'Enabled' if app_config.service.enable_metrics else 'Disabled'}")
        
        if security_config.security_enabled:
            logger.info("API key authentication is ENABLED")
        else:
            logger.warning("API key authentication is DISABLED - service is unsecured!")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise
    finally:
        if 'health_task' in locals():
            health_task.cancel()
            try:
                await health_task
            except asyncio.CancelledError:
                pass
        
        await http_client.close()
        
        logger.info("Gemini Claude Adapter shutting down.")

app = FastAPI(
    title="Gemini Claude Adapter v2.1.0",
    description="A high-performance, secure Gemini adapter with complete Anthropic API compatibility.",
    version="2.1.0",
    lifespan=lifespan
)

# ... (all endpoints remain the same) ...
