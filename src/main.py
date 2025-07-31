import asyncio
import time
import random
from typing import List, Dict, Optional, Any, Union, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from itertools import cycle
import json
import httpx
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from loguru import logger
import litellm
from contextlib import asynccontextmanager
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from starlette.status import HTTP_401_UNAUTHORIZED, HTTP_403_FORBIDDEN
from collections import defaultdict, deque

# Import new optimization modules
from .config import get_config, AppConfig, GeminiConfig
from .error_handling import error_monitor, monitor_errors, ErrorClassifier
from .performance import response_cache, http_client, performance_monitor, monitor_performance, get_performance_stats

# Import Anthropic API compatibility layer
try:
    from .anthropic_api import (
        MessagesRequest, MessagesResponse, TokenCountRequest, TokenCountResponse,
        AnthropicToGeminiConverter, GeminiToAnthropicConverter, 
        StreamingResponseGenerator, ToolConverter, log_request_beautifully
    )
except ImportError:
    # Fallback for different import contexts
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
    
    @validator('messages')
    def validate_messages(cls, v):
        if not v:
            raise ValueError("Messages cannot be empty")
        for msg in v:
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
        self.last_key_used = None  # 添加这个属性来避免连续使用同一个key
        # 添加密钥性能跟踪
        self.key_performance = defaultdict(lambda: {"response_times": deque(maxlen=100), "errors": 0})

        for key in config.api_keys:
            if key and key.strip():
                self.keys[key] = APIKeyInfo(key=key)

        if not self.keys:
            raise ValueError("No valid API keys provided to key manager")

        logger.info(f"Initialized {len(self.keys)} API keys with performance tracking.")

    async def get_available_key(self) -> Optional[APIKeyInfo]:
        async with self.lock:
            # 检查是否有可以恢复的keys
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
        """内部方法：检查并恢复冷却中的keys，返回恢复的key数量"""
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
        
        # 如果有key被恢复，重置cycle
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
        """基于性能指标选择最佳密钥"""
        def key_score(key_info: APIKeyInfo) -> float:
            # 成功率权重 60%
            success_rate = key_info.successful_requests / max(key_info.total_requests, 1)
            
            # 响应时间权重 30%
            perf_data = self.key_performance[key_info.key]
            avg_response_time = sum(perf_data["response_times"]) / max(len(perf_data["response_times"]), 1)
            response_score = 1.0 / (1.0 + avg_response_time)  # 响应时间越低分数越高
            
            # 最近使用权重 10%（避免总是使用同一个key）
            recent_use_penalty = 0.9 if key_info.key == self.last_key_used else 1.0
            
            return (success_rate * 0.6 + response_score * 0.3) * recent_use_penalty
        
        return max(active_keys, key=key_score)
    
    async def record_key_performance(self, key: str, response_time: float, success: bool):
        """记录密钥性能数据"""
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
        """检查并恢复冷却中的keys，返回恢复的key数量"""
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
        
        # 添加请求去重
        self._request_deduplicator = {}
        self._dedup_lock = asyncio.Lock()
        
        if config.proxy_url:
            os.environ['HTTPS_PROXY'] = config.proxy_url
            os.environ['HTTP_PROXY'] = config.proxy_url
            logger.info(f"Using proxy: {config.proxy_url}")
        
        # 优化 litellm 配置
        litellm.request_timeout = 30  # 减少超时时间
        litellm.max_retries = 0
        litellm.set_verbose = False
        litellm.drop_params = True
        litellm.num_retries = 0  # 禁用内部重试，使用我们的重试逻辑

    async def _deduplicate_request(self, request_hash: str):
        """请求去重：避免相同请求同时处理"""
        async with self._dedup_lock:
            if request_hash in self._request_deduplicator:
                # 等待已有请求完成
                return await self._request_deduplicator[request_hash]
            else:
                # 创建新的Future
                future = asyncio.Future()
                self._request_deduplicator[request_hash] = future
                return future

    @monitor_errors
    async def chat_completion(self, request: ChatRequest) -> Union[Dict, Any]:
        # 生成请求哈希用于去重
        import hashlib
        request_hash = hashlib.md5(
            json.dumps({
                "model": request.model,
                "messages": request.messages,
                "temperature": request.temperature,
                "stream": request.stream
            }, sort_keys=True).encode()
        ).hexdigest()
        
        # 检查去重
        if not request.stream:  # 只对非流式请求去重
            existing_future = await self._deduplicate_request(request_hash)
            if not existing_future.done():
                try:
                    return await existing_future
                except:
                    pass  # 如果等待失败，继续处理新请求
        
        last_error = None
        attempted_keys = set()
        
        # 检查缓存
        if not request.stream:
            cache_key = {
                "model": request.model,
                "messages": request.messages,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens
            }
            cached_response = await response_cache.get(cache_key)
            if cached_response:
                # 更新去重状态
                if request_hash in self._request_deduplicator:
                    self._request_deduplicator[request_hash].set_result(cached_response)
                    del self._request_deduplicator[request_hash]
                return cached_response
        
        # 并发尝试多个密钥（限制并发数）
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
                    
                    kwargs = {
                        "model": f"gemini/{request.model}",
                        "messages": request.messages,
                        "api_key": key_info.key,
                        "temperature": request.temperature,
                        "stream": request.stream,
                        "timeout": self.config.request_timeout,
                    }
                    
                    if request.max_tokens:
                        kwargs["max_tokens"] = request.max_tokens
                    
                    response = await asyncio.wait_for(
                        litellm.acompletion(**kwargs),
                        timeout=self.config.request_timeout
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
        
        # 获取可用密钥并并发尝试
        active_keys = [k for k in self.key_manager.keys.values() if k.status == KeyStatus.ACTIVE]
        if not active_keys:
            raise HTTPException(status_code=502, detail="No available API keys")
        
        # 限制尝试的密钥数量
        keys_to_try = active_keys[:max_concurrent]
        
        try:
            # 并发执行请求
            tasks = [try_key_request(key_info) for key_info in keys_to_try]
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            
            # 取消剩余任务
            for task in pending:
                task.cancel()
            
            # 获取第一个成功的结果
            for task in done:
                try:
                    result = await task
                    if result:
                        # 缓存成功的响应
                        if not request.stream:
                            await response_cache.set(cache_key, result)
                            # 更新去重状态
                            if request_hash in self._request_deduplicator:
                                self._request_deduplicator[request_hash].set_result(result)
                                del self._request_deduplicator[request_hash]
                        return result
                except Exception as e:
                    last_error = str(e)
                    continue
                    
        except Exception as e:
            last_error = str(e)
        
        # 清理去重状态
        if request_hash in self._request_deduplicator:
            self._request_deduplicator[request_hash].set_exception(
                HTTPException(status_code=502, detail=f"All keys failed. Last error: {last_error}")
            )
            del self._request_deduplicator[request_hash]
        
        raise HTTPException(status_code=502, detail=f"All available keys failed. Last error: {last_error}")
    
    async def anthropic_messages_completion(self, request: MessagesRequest) -> Union[MessagesResponse, AsyncGenerator[str, None]]:
        """Handle Anthropic Messages API requests"""
        # Convert Anthropic request to Gemini format
        gemini_request_dict = self.anthropic_to_gemini.convert_request(request)
        
        # Add tools if present
        if request.tools:
            gemini_tools = self.tool_converter.convert_tools_to_gemini(request.tools)
            gemini_request_dict["tools"] = gemini_tools
            
            if request.tool_choice:
                tool_config = self.tool_converter.convert_tool_choice_to_gemini(request.tool_choice)
                gemini_request_dict["tool_config"] = {"function_calling_config": {"mode": tool_config}}
        
        # Create ChatRequest object
        chat_request = ChatRequest(
            messages=gemini_request_dict["messages"],
            model=gemini_request_dict["model"].replace("gemini/", ""),
            temperature=gemini_request_dict["temperature"],
            max_tokens=gemini_request_dict.get("max_tokens"),
            stream=gemini_request_dict["stream"]
        )
        
        if request.stream:
            # Handle streaming response
            gemini_stream = await self.chat_completion(chat_request)
            streaming_generator = StreamingResponseGenerator(request)
            return streaming_generator.generate_sse_events(gemini_stream)
        else:
            # Handle regular response
            gemini_response = await self.chat_completion(chat_request)
            return self.gemini_to_anthropic.convert_response(gemini_response, request)

# Global state
key_manager: Optional[GeminiKeyManager] = None
adapter: Optional[LiteLLMAdapter] = None

# 添加请求限流中间件
class RateLimiter:
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(deque)
        self.lock = asyncio.Lock()  # 添加锁以支持并发
        
    async def is_allowed(self, client_id: str) -> bool:
        async with self.lock:
            now = time.time()
            client_requests = self.requests[client_id]
            
            # 清理过期请求（批量处理）
            cutoff_time = now - self.window_seconds
            while client_requests and client_requests[0] < cutoff_time:
                client_requests.popleft()
            
            # 检查限制
            if len(client_requests) >= self.max_requests:
                return False
            
            # 记录新请求
            client_requests.append(now)
            return True
    
    def get_remaining_requests(self, client_id: str) -> int:
        """获取剩余请求数"""
        client_requests = self.requests[client_id]
        return max(0, self.max_requests - len(client_requests))

# 创建全局限流器实例
rate_limiter = RateLimiter(max_requests=100, window_seconds=60)

# 在需要限流的endpoint上添加依赖
async def check_rate_limit(client_key: str = Depends(verify_api_key)):
    if not rate_limiter.is_allowed(client_key):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later.",
            headers={"Retry-After": "60"}
        )
    return client_key

# 优化的健康检查任务
async def optimized_health_check_task():
    """优化的健康检查任务"""
    check_interval = 30
    last_recovery_check = 0
    consecutive_failures = 0
    
    while True:
        try:
            if not key_manager:
                await asyncio.sleep(check_interval)
                continue
                
            current_time = time.time()
            stats = await key_manager.get_stats()
            
            total_keys = stats.get("total_keys", 0)
            active_keys = stats.get("active_keys", 0)
            cooling_keys = stats.get("cooling_keys", 0)
            failed_keys = stats.get("failed_keys", 0)
            
            # 动态调整检查间隔
            if active_keys == 0:
                check_interval = 10  # 紧急情况下更频繁检查
            elif active_keys < total_keys * 0.3:
                check_interval = 15  # 密钥不足时更频繁检查
            else:
                check_interval = 30  # 正常间隔
            
            # 智能恢复检查条件
            should_check = False
            check_reason = ""
            
            # 条件1: 无可用密钥（紧急）
            if active_keys == 0:
                should_check = True
                check_reason = "CRITICAL: No active keys available"
            # 条件2: 可用密钥严重不足
            elif active_keys < max(1, total_keys * 0.2):
                should_check = True
                check_reason = f"URGENT: Very low active keys: {active_keys}/{total_keys}"
            # 条件3: 有冷却密钥且时间间隔足够
            elif cooling_keys > 0 and (current_time - last_recovery_check) > 120:
                should_check = True
                check_reason = f"ROUTINE: Check cooling keys ({cooling_keys})"
            
            if should_check:
                logger.info(f"Health check triggered: {check_reason}")
                recovered_count = await key_manager._check_and_recover_keys()
                last_recovery_check = current_time
                
                if recovered_count > 0:
                    logger.info(f"Recovery successful: {recovered_count} keys restored")
                    consecutive_failures = 0
                elif active_keys == 0:
                    consecutive_failures += 1
                    logger.error(f"Recovery failed, consecutive failures: {consecutive_failures}")
                    
                    # 如果连续失败太多次，发送警报
                    if consecutive_failures >= 5:
                        logger.critical("ALERT: Service degraded - all keys failed multiple times")
                        # 这里可以添加外部通知逻辑
            
        except Exception as e:
            logger.error(f"Health check error: {e}")
            consecutive_failures += 1
        
        await asyncio.sleep(check_interval)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global key_manager, adapter, security_config
    
    try:
        # Load new configuration system
        app_config = get_config()
        
        # Initialize security config with new AppConfig
        security_config = SecurityConfig(app_config)
        
        # Initialize key manager and adapter using AppConfig directly
        key_manager = GeminiKeyManager(app_config.gemini)
        adapter = LiteLLMAdapter(app_config.gemini, key_manager)
        
        # Initialize HTTP client
        await http_client.initialize()
        
        # 启动优化的健康检查任务
        health_task = asyncio.create_task(optimized_health_check_task())
        
        logger.info("Gemini Claude Adapter v2.1.0 started successfully with optimizations.")
        logger.info(f"Environment: {app_config.service.environment.value}")
        logger.info(f"Caching: {'Enabled' if app_config.cache.enabled else 'Disabled'}")
        logger.info(f"Performance monitoring: {'Enabled' if app_config.service.enable_metrics else 'Disabled'}")
        
        # Log security status
        if security_config.security_enabled:
            logger.info("API key authentication is ENABLED")
        else:
            logger.warning("API key authentication is DISABLED - service is unsecured!")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise
    finally:
        # 取消健康检查任务
        if 'health_task' in locals():
            health_task.cancel()
            try:
                await health_task
            except asyncio.CancelledError:
                pass
        
        # Close HTTP client
        await http_client.close()
        
        logger.info("Gemini Claude Adapter shutting down.")

app = FastAPI(
    title="Gemini Claude Code Adapter",
    description="An adapter for Claude Code to use Gemini API with key rotation and fault tolerance.",
    version="2.0.0",
    lifespan=lifespan
)

async def stream_generator(response_stream):
    """Generate streaming response chunks"""
    try:
        async for chunk in response_stream:
            try:
                # Handle different chunk formats
                if hasattr(chunk, 'dict'):
                    chunk_data = chunk.dict()
                elif hasattr(chunk, 'model_dump'):
                    chunk_data = chunk.model_dump()
                else:
                    chunk_data = chunk
                
                yield f"data: {json.dumps(chunk_data)}\n\n"
            except Exception as e:
                logger.error(f"Error serializing chunk: {e}")
                continue
                
        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error(f"Error during streaming: {e}")
        error_payload = {"error": {"message": str(e), "type": "stream_error"}}
        yield f"data: {json.dumps(error_payload)}\n\n"

# Public endpoints (no authentication required)
@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint with basic info - no authentication required"""
    return {
        "name": "Gemini Claude Adapter",
        "version": "1.3.0",
        "status": "running",
        "security_enabled": security_config.security_enabled,
        "endpoints": {
            "chat": "/v1/chat/completions",
            "messages": "/v1/messages",
            "messages_tokens": "/v1/messages/count_tokens",
            "models": "/v1/models",
            "health": "/health",
            "stats": "/stats",
            "admin": "/admin/reset-key/{key_prefix}, /admin/recover-key/{key_prefix}"
        },
        "authentication": {
            "required": security_config.security_enabled,
            "methods": ["X-API-Key header", "Authorization Bearer token"] if security_config.security_enabled else []
        },
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint - no authentication required for monitoring"""
    if not key_manager:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        stats = await key_manager.get_stats()
        status_code = 200 if stats["active_keys"] > 0 else 503
        
        return {
            "status": "healthy" if stats["active_keys"] > 0 else "degraded",
            "timestamp": datetime.now().isoformat(),
            "service_version": "2.0.0",
            "security_enabled": security_config.security_enabled,
            **stats
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

# 修改主要endpoint以包含限流
@app.post("/v1/messages", dependencies=[Depends(check_rate_limit)])
async def create_message(
    request: MessagesRequest,
    raw_request: Request,
    client_key: str = Depends(verify_api_key)
):
    """Create a message - Anthropic Messages API"""
    if not adapter:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        # Get request details for logging
        anthropic_model = request.model
        gemini_model = adapter.anthropic_to_gemini.convert_model(anthropic_model)
        num_messages = len(request.messages)
        num_tools = len(request.tools) if request.tools else 0
        
        logger.info(f"Anthropic Messages API request from client: {client_key[:8] if client_key != 'insecure_mode' else 'insecure_mode'}...")
        logger.info(f"Request model: {anthropic_model}, stream: {request.stream}")
        
        # Log the request beautifully
        log_request_beautifully(
            method="POST",
            path=str(raw_request.url.path),
            anthropic_model=anthropic_model,
            gemini_model=gemini_model,
            num_messages=num_messages,
            num_tools=num_tools
        )
        
        response = await adapter.anthropic_messages_completion(request)
        
        if request.stream:
            return StreamingResponse(
                response,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "POST, OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type, Authorization, X-API-Key, Anthropic-Version",
                    "Anthropic-Version": "2023-06-01"
                }
            )
        else:
            return response
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in Anthropic Messages API: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.options("/v1/messages")
async def options_messages():
    """Handle OPTIONS requests for CORS preflight"""
    return {"status": "ok"}

@app.post("/v1/messages/count_tokens", dependencies=[Depends(verify_api_key)])
async def count_tokens(
    request: TokenCountRequest,
    client_key: str = Depends(verify_api_key)
):
    """Count tokens for a message"""
    if not adapter:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        logger.info(f"Token count request from client: {client_key[:8] if client_key != 'insecure_mode' else 'insecure_mode'}...")
        
        # Convert request for token counting
        gemini_request_dict = adapter.anthropic_to_gemini.convert_request(
            MessagesRequest(
                model=request.model,
                max_tokens=100,  # Arbitrary value for token counting
                messages=request.messages,
                system=request.system,
                tools=request.tools,
                tool_choice=request.tool_choice
            )
        )
        
        # Use LiteLLM's token counter
        try:
            from litellm import token_counter
            token_count = token_counter(
                model=gemini_request_dict["model"],
                messages=gemini_request_dict["messages"]
            )
            return TokenCountResponse(input_tokens=token_count)
        except ImportError:
            logger.error("Could not import token_counter from litellm")
            # Fallback estimation
            text_content = ""
            for msg in gemini_request_dict["messages"]:
                if isinstance(msg.get("content"), str):
                    text_content += msg["content"] + " "
            estimated_tokens = len(text_content.split()) * 1.3  # Rough estimate
            return TokenCountResponse(input_tokens=int(estimated_tokens))
            
    except Exception as e:
        logger.error(f"Error counting tokens: {e}")
        raise HTTPException(status_code=500, detail="Token counting failed")

# Legacy OpenAI-compatible endpoint (maintained for backward compatibility)
@app.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
async def chat_completions(request: ChatRequest, client_key: str = Depends(verify_api_key)):
    if not adapter:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        logger.info(f"Legacy chat completion request from client: {client_key[:8] if client_key != 'insecure_mode' else 'insecure_mode'}...")
        response = await adapter.chat_completion(request)
        if request.stream:
            return StreamingResponse(
                stream_generator(response), 
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "POST, OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type, Authorization, X-API-Key",
                }
            )
        else:
            return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in chat completion: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.options("/v1/chat/completions")
async def options_chat_completions():
    """Handle OPTIONS requests for CORS preflight - no authentication required"""
    return {
        "status": "ok"
    }

@app.get("/stats", dependencies=[Depends(verify_api_key)])
async def get_stats(client_key: str = Depends(verify_api_key)):
    if not key_manager:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        return await key_manager.get_stats()
    except Exception as e:
        logger.error(f"Stats retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Stats retrieval failed")

@app.get("/v1/models", dependencies=[Depends(verify_api_key)])
async def get_models(client_key: str = Depends(verify_api_key)):
    """Get available models"""
    return {
        "object": "list",
        "data": [
            {
                "id": "claude-3-5-sonnet",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "anthropic"
            },
            {
                "id": "claude-3-5-haiku",
                "object": "model", 
                "created": int(time.time()),
                "owned_by": "anthropic"
            },
            {
                "id": "claude-3-opus",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "anthropic"
            },
            {
                "id": "claude-3-sonnet",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "anthropic"
            },
            {
                "id": "claude-3-haiku",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "anthropic"
            }
        ]
    }

# Admin endpoints (require admin authentication)
@app.post("/admin/reset-key/{key_prefix}", dependencies=[Depends(verify_admin_key)])
async def reset_key(key_prefix: str, admin_key: str = Depends(verify_admin_key)):
    """Reset a key's status by prefix - requires admin authentication"""
    if not key_manager:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        logger.info(f"Admin key reset requested by: {admin_key[:8] if admin_key != 'insecure_mode' else 'insecure_mode'}...")
        result = await key_manager.reset_key(key_prefix)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Key reset failed: {e}")
        raise HTTPException(status_code=500, detail="Key reset failed")

@app.post("/admin/recover-key/{key_prefix}", dependencies=[Depends(verify_admin_key)])
async def recover_key(key_prefix: str, admin_key: str = Depends(verify_admin_key)):
    """Attempt to recover a permanently failed key - requires admin authentication"""
    if not key_manager:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        logger.info(f"Admin key recovery requested by: {admin_key[:8] if admin_key != 'insecure_mode' else 'insecure_mode'}...")
        result = await key_manager.attempt_key_recovery(key_prefix)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Key recovery failed: {e}")
        raise HTTPException(status_code=500, detail="Key recovery failed")

# 添加更详细的错误处理和日志记录
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理器"""
    logger.error(f"Unhandled exception: {type(exc).__name__}: {str(exc)}")
    logger.error(f"Request: {request.method} {request.url}")
    
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"message": exc.detail, "type": "http_exception"}}
        )
    
    return JSONResponse(
        status_code=500,
        content={"error": {"message": "Internal server error", "type": "internal_error"}}
    )

# Enhanced monitoring endpoint with new optimization metrics
@app.get("/metrics", dependencies=[Depends(verify_api_key)])
async def get_metrics(client_key: str = Depends(verify_api_key)):
    """获取详细的服务监控指标"""
    if not key_manager:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        # Get key manager stats
        stats = await key_manager.get_stats()
        
        # Calculate additional metrics
        total_requests = sum(k.total_requests for k in key_manager.keys.values())
        total_successes = sum(k.successful_requests for k in key_manager.keys.values())
        overall_success_rate = (total_successes / total_requests * 100) if total_requests > 0 else 0
        
        # Get performance metrics
        perf_stats = get_performance_stats()
        
        # Get error monitoring stats
        error_stats = await error_monitor.get_error_stats()
        
        return {
            "key_manager_stats": stats,
            "overall_success_rate": round(overall_success_rate, 2),
            "total_requests": total_requests,
            "total_successes": total_successes,
            "service_uptime": time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0,
            "timestamp": datetime.now().isoformat(),
            "performance_metrics": perf_stats,
            "error_metrics": error_stats
        }
    except Exception as e:
        logger.error(f"Metrics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Metrics retrieval failed")

@app.get("/cache/stats", dependencies=[Depends(verify_api_key)])
async def get_cache_stats(client_key: str = Depends(verify_api_key)):
    """Get cache statistics"""
    try:
        return response_cache.get_stats()
    except Exception as e:
        logger.error(f"Cache stats retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Cache stats retrieval failed")

@app.post("/cache/clear", dependencies=[Depends(verify_admin_key)])
async def clear_cache(admin_key: str = Depends(verify_admin_key)):
    """Clear cache"""
    try:
        response_cache.clear()
        return {"message": "Cache cleared successfully"}
    except Exception as e:
        logger.error(f"Cache clear failed: {e}")
        raise HTTPException(status_code=500, detail="Cache clear failed")

@app.get("/errors/recent", dependencies=[Depends(verify_admin_key)])
async def get_recent_errors(admin_key: str = Depends(verify_admin_key), limit: int = 50):
    """Get recent errors for admin monitoring"""
    try:
        return await error_monitor.get_recent_errors(limit)
    except Exception as e:
        logger.error(f"Recent errors retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Recent errors retrieval failed")

@app.get("/health/detailed", dependencies=[Depends(verify_api_key)])
async def detailed_health_check(client_key: str = Depends(verify_api_key)):
    """Detailed health check with optimization metrics"""
    if not key_manager:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        # Basic health check
        stats = await key_manager.get_stats()
        basic_healthy = stats["active_keys"] > 0
        
        # Check cache health
        cache_stats = response_cache.get_stats()
        cache_healthy = cache_stats["hit_rate"] > 0.1  # At least 10% hit rate
        
        # Check HTTP client health
        http_stats = http_client.get_stats()
        http_healthy = http_stats["error_rate"] < 5  # Less than 5% error rate
        
        # Overall health status
        overall_healthy = basic_healthy and cache_healthy and http_healthy
        
        return {
            "status": "healthy" if overall_healthy else "degraded",
            "components": {
                "key_manager": {"healthy": basic_healthy, "active_keys": stats["active_keys"]},
                "cache": {"healthy": cache_healthy, "hit_rate": cache_stats["hit_rate"]},
                "http_client": {"healthy": http_healthy, "error_rate": http_stats["error_rate"]}
            },
            "timestamp": datetime.now().isoformat(),
            "version": "2.1.0"
        }
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

# 在应用启动时记录启动时间
@app.on_event("startup")
async def startup_event():
    app.state.start_time = time.time()

@app.get("/admin/security-status", dependencies=[Depends(verify_admin_key)])
async def get_security_status(admin_key: str = Depends(verify_admin_key)):
    """Get security configuration status - admin only"""
    return {
        "security_enabled": security_config.security_enabled,
        "client_keys_count": len(security_config.valid_api_keys),
        "admin_keys_count": len(security_config.admin_keys),
        "has_admin_keys": bool(security_config.admin_keys),
        "authentication_methods": ["X-API-Key header", "Authorization Bearer token"],
        "admin_endpoints": ["/admin/reset-key/{key_prefix}", "/admin/recover-key/{key_prefix}", "/admin/security-status"]
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("main:app", host=host, port=port, log_level="info")