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

# Import Anthropic API compatibility layer
from .anthropic_api import (
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

class GeminiConfig(BaseModel):
    api_keys: List[str] = Field(..., description="List of Gemini API keys")
    proxy_url: Optional[str] = Field(None, description="Proxy URL")
    max_failures: int = Field(3, description="Maximum number of failures before cooling", ge=1)
    cooling_period: int = Field(300, description="Cooling period in seconds", ge=60)
    health_check_interval: int = Field(60, description="Health check interval in seconds", ge=10)
    request_timeout: int = Field(45, description="Request timeout in seconds", ge=10)
    max_retries: int = Field(2, description="Maximum retry attempts for a request", ge=0)
    
    @validator('api_keys')
    def validate_api_keys(cls, v):
        if not v:
            raise ValueError("At least one API key is required")
        # Filter out empty keys and clean quotes
        valid_keys = []
        for key in v:
            if key and key.strip():
                cleaned_key = key.strip().strip('"\'').strip()
                if cleaned_key:
                    valid_keys.append(cleaned_key)
        if not valid_keys:
            raise ValueError("No valid API keys provided")
        return valid_keys

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

# Security Configuration
class SecurityConfig:
    def __init__(self):
        # Load adapter API keys from environment
        adapter_keys_str = os.getenv("ADAPTER_API_KEYS", "")
        self.valid_api_keys: Set[str] = set(
            key.strip() for key in adapter_keys_str.split(',') 
            if key.strip()
        )
        self.security_enabled = bool(self.valid_api_keys)
        
        # Admin keys for management endpoints
        admin_keys_str = os.getenv("ADMIN_API_KEYS", "")
        self.admin_keys: Set[str] = set(
            key.strip() for key in admin_keys_str.split(',')
            if key.strip()
        )
        
        if self.security_enabled:
            logger.info(f"Security enabled with {len(self.valid_api_keys)} client keys")
        else:
            logger.warning("Security disabled - no ADAPTER_API_KEYS configured")
            
        if self.admin_keys:
            logger.info(f"Admin access enabled with {len(self.admin_keys)} admin keys")

# Initialize security config
security_config = SecurityConfig()

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
    def __init__(self, config: GeminiConfig):
        self.config = config
        self.keys: Dict[str, APIKeyInfo] = {}
        self.key_cycle = None
        self.lock = asyncio.Lock()
        self.last_key_used = None  # 添加这个属性来避免连续使用同一个key

        for key in config.api_keys:
            if key and key.strip():
                self.keys[key] = APIKeyInfo(key=key)

        if not self.keys:
            raise ValueError("No valid API keys provided to key manager")

        logger.info(f"Initialized {len(self.keys)} API keys.")

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

            # 智能选择key，避免连续使用同一个key
            if len(active_keys) > 1 and self.last_key_used:
                available_keys = [k for k in active_keys if k.key != self.last_key_used]
                if available_keys:
                    # 选择成功率最高的key
                    selected_key = max(available_keys, key=lambda k: k.successful_requests / max(k.total_requests, 1))
                else:
                    selected_key = active_keys[0]
            else:
                # 选择成功率最高的key
                selected_key = max(active_keys, key=lambda k: k.successful_requests / max(k.total_requests, 1))
            
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
        
        # Initialize converters
        self.anthropic_to_gemini = AnthropicToGeminiConverter()
        self.gemini_to_anthropic = GeminiToAnthropicConverter()
        self.tool_converter = ToolConverter()
        
        # Configure proxy if provided
        if config.proxy_url:
            os.environ['HTTPS_PROXY'] = config.proxy_url
            os.environ['HTTP_PROXY'] = config.proxy_url
            logger.info(f"Using proxy: {config.proxy_url}")
        
        # Configure litellm for optimal performance
        litellm.request_timeout = config.request_timeout
        litellm.max_retries = 0  # We handle our own retries
        litellm.set_verbose = False  # Reduce noise in logs
        litellm.drop_params = True  # Drop unsupported parameters for better compatibility

    async def chat_completion(self, request: ChatRequest) -> Union[Dict, Any]:
        last_error = None
        attempted_keys = set()
        
        # 最多尝试所有可用的key
        max_attempts = len(self.key_manager.keys)
        
        for attempt in range(max_attempts):
            key_info = await self.key_manager.get_available_key()
            if not key_info or key_info.key in attempted_keys:
                break
                
            attempted_keys.add(key_info.key)
            
            try:
                # Increment total requests counter
                async with self.key_manager.lock:
                    key_info.total_requests += 1
                
                kwargs = {
                    "model": f"gemini/{request.model}",
                    "messages": request.messages,
                    "api_key": key_info.key,
                    "temperature": request.temperature,
                    "stream": request.stream,
                    "timeout": self.config.request_timeout,  # 显式设置超时
                }
                
                if request.max_tokens:
                    kwargs["max_tokens"] = request.max_tokens
                
                logger.info(f"Attempting request with key {key_info.key[:8]}... (attempt {attempt + 1}/{max_attempts})")
                
                # 使用 asyncio.wait_for 添加额外的超时保护
                response = await asyncio.wait_for(
                    litellm.acompletion(**kwargs),
                    timeout=self.config.request_timeout + 5  # 比litellm的超时多5秒
                )
                
                await self.key_manager.mark_key_success(key_info.key)
                logger.info(f"Request successful with key {key_info.key[:8]}...")
                return response

            except asyncio.TimeoutError:
                last_error = "Request timeout"
                await self.key_manager.mark_key_failed(key_info.key, last_error)
                logger.warning(f"Key {key_info.key[:8]}... timed out, trying next key...")
            except Exception as e:
                last_error = str(e)
                await self.key_manager.mark_key_failed(key_info.key, last_error)
                logger.warning(f"Key {key_info.key[:8]}... failed: {last_error}, trying next key...")
        
        # 所有key都失败了
        error_msg = f"Failed to process request with all available keys. Last error: {last_error}"
        logger.error(error_msg)
        raise HTTPException(status_code=502, detail=error_msg)
    
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
    
    def is_allowed(self, client_id: str) -> bool:
        now = time.time()
        client_requests = self.requests[client_id]
        
        # 清理过期的请求记录
        while client_requests and client_requests[0] < now - self.window_seconds:
            client_requests.popleft()
        
        # 检查是否超过限制
        if len(client_requests) >= self.max_requests:
            return False
        
        # 记录新请求
        client_requests.append(now)
        return True

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

# 条件触发的key恢复检查任务
async def health_check_task():
    """条件触发检查API keys的健康状态"""
    while True:
        if key_manager:
            try:
                stats = await key_manager.get_stats()
                total_keys = stats.get("total_keys", 0)
                active_keys = stats.get("active_keys", 0)
                cooling_keys = stats.get("cooling_keys", 0)
                
                # 检查是否需要触发key恢复检查
                should_check = False
                check_reason = ""
                
                # 条件1: 可用key数量少于3个
                if active_keys < 3:
                    should_check = True
                    check_reason = f"Low active keys: {active_keys} < 3"
                
                # 条件2: 可用key数量少于总数的10%
                elif total_keys > 0 and (active_keys / total_keys) < 0.1:
                    should_check = True
                    check_reason = f"Low active key ratio: {active_keys}/{total_keys} ({active_keys/total_keys*100:.1f}%)"
                
                # 条件3: 有冷却中的key且距离上次检查超过5分钟
                elif cooling_keys > 0:
                    current_time = time.time()
                    last_check_time = getattr(key_manager, 'last_recovery_check', 0)
                    if current_time - last_check_time > 300:  # 5分钟
                        should_check = True
                        check_reason = f"Cooling keys present ({cooling_keys}) and check interval reached"
                
                if should_check:
                    logger.info(f"Triggering key recovery check: {check_reason}")
                    
                    # 手动触发一次key状态检查
                    await key_manager._check_and_recover_keys()
                    
                    # 更新检查时间
                    key_manager.last_recovery_check = time.time()
                    
                    # 获取更新后的统计信息
                    new_stats = await key_manager.get_stats()
                    recovered_keys = new_stats.get("active_keys", 0) - active_keys
                    
                    if recovered_keys > 0:
                        logger.info(f"Key recovery successful: {recovered_keys} keys recovered")
                    else:
                        logger.info(f"No keys recovered in this check")
                
            except Exception as e:
                logger.error(f"Health check error: {e}")
        
        # 降低检查频率，每30秒检查一次条件
        await asyncio.sleep(30)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global key_manager, adapter
    
    try:
        # 原有的初始化代码...
        # Load and validate environment variables
        api_keys_str = os.getenv("GEMINI_API_KEYS", "")
        if not api_keys_str:
            logger.error("GEMINI_API_KEYS environment variable is required!")
            raise ValueError("GEMINI_API_KEYS environment variable is required!")
        
        # Parse and validate API keys with flexible format support
        api_keys = []
        for key in api_keys_str.split(","):
            if key and key.strip():
                cleaned_key = key.strip().strip('"\'').strip()
                if cleaned_key:
                    api_keys.append(cleaned_key)
        if not api_keys:
            logger.error("No valid API keys provided!")
            raise ValueError("No valid API keys provided!")
        
        # Validate key format (Gemini keys typically start with 'AIza')
        invalid_keys = [key for key in api_keys if not key.startswith('AIza')]
        if invalid_keys:
            logger.warning(f"Potentially invalid API keys detected: {len(invalid_keys)} keys don't start with 'AIza'")
        
        config = GeminiConfig(
            api_keys=api_keys,
            proxy_url=os.getenv("PROXY_URL"),
            max_failures=int(os.getenv("MAX_FAILURES", "1")),
            cooling_period=int(os.getenv("COOLING_PERIOD", "300")),
            health_check_interval=int(os.getenv("HEALTH_CHECK_INTERVAL", "60")),
            request_timeout=int(os.getenv("REQUEST_TIMEOUT", "45")),
            max_retries=int(os.getenv("MAX_RETRIES", "0"))
        )
        
        key_manager = GeminiKeyManager(config)
        adapter = LiteLLMAdapter(config, key_manager)
        
        # 启动健康检查任务
        health_task = asyncio.create_task(health_check_task())
        
        logger.info("Gemini Claude Adapter started successfully.")
        
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

# 添加更好的监控endpoint
@app.get("/metrics", dependencies=[Depends(verify_api_key)])
async def get_metrics(client_key: str = Depends(verify_api_key)):
    """获取详细的服务监控指标"""
    if not key_manager:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        stats = await key_manager.get_stats()
        
        # 计算额外的指标
        total_requests = sum(k.total_requests for k in key_manager.keys.values())
        total_successes = sum(k.successful_requests for k in key_manager.keys.values())
        overall_success_rate = (total_successes / total_requests * 100) if total_requests > 0 else 0
        
        return {
            **stats,
            "overall_success_rate": round(overall_success_rate, 2),
            "total_requests": total_requests,
            "total_successes": total_successes,
            "service_uptime": time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Metrics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Metrics retrieval failed")

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