import asyncio
import time
import random
from typing import List, Dict, Optional, Any, Union, Set, Tuple, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
from itertools import cycle
import json
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

# --- MODIFIED: Imports adjusted for flat structure ---
from config import get_config, AppConfig
from error_handling import error_monitor, monitor_errors, ErrorClassifier
import performance
# Models are now imported from anthropic_api
from anthropic_api import (
    MessagesRequest, MessagesResponse, TokenCountRequest, TokenCountResponse,
    AnthropicToGeminiConverter, GeminiToAnthropicConverter,
    StreamingResponseGenerator, ToolConverter, log_request_beautifully,
    AnthropicAPIConfig
)

load_dotenv()

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

class ChatRequest(BaseModel):
    messages: List[Dict[str, Any]]
    model: str = "gemini-2.5-pro"
    temperature: Optional[float] = Field(default=0.1, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.95, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    stop_sequences: Optional[List[str]] = Field(default=None, max_length=5)
    stream: bool = Field(default=False)

    @field_validator('messages')
    @classmethod
    def validate_messages(cls, v):
        if not v:
            raise ValueError("Messages cannot be empty")
        for msg in v:
            if not isinstance(msg, dict) or 'role' not in msg:
                raise ValueError("Each message must have 'role' field")
            # More flexible validation - allow either content or parts
            if 'content' not in msg and 'parts' not in msg and 'text' not in msg:
                logger.warning(f"Message may be missing content: {msg}")
        return v

class SecurityConfig:
    def __init__(self, app_config: AppConfig):
        self.valid_api_keys: Set[str] = set(app_config.SECURITY_ADAPTER_API_KEYS)
        self.admin_keys: Set[str] = set(app_config.SECURITY_ADMIN_API_KEYS)
        self.enable_ip_blocking = app_config.SECURITY_ENABLE_IP_BLOCKING
        self.max_failed_attempts = app_config.SECURITY_MAX_FAILED_ATTEMPTS
        self.block_duration = app_config.SECURITY_BLOCK_DURATION
        self.enable_rate_limiting = app_config.SECURITY_ENABLE_RATE_LIMITING
        self.rate_limit_requests = app_config.SECURITY_RATE_LIMIT_REQUESTS
        self.rate_limit_window = app_config.SECURITY_RATE_LIMIT_WINDOW
        self.security_enabled = bool(self.valid_api_keys)
        if self.security_enabled:
            logger.info(f"Security enabled with {len(self.valid_api_keys)} client keys")
        else:
            logger.warning("Security disabled - no SECURITY_ADAPTER_API_KEYS configured")
        if self.admin_keys:
            logger.info(f"Admin access enabled with {len(self.admin_keys)} admin keys")
        else:
            logger.info("No admin keys configured - client keys will have admin access")

security_config: Optional[SecurityConfig] = None
api_config: Optional[AnthropicAPIConfig] = None

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)

async def verify_api_key(
    api_key: Optional[str] = Depends(api_key_header),
    bearer_token: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme)
) -> str:
    if not security_config.security_enabled:
        logger.debug("Security disabled, allowing access")
        return "insecure_mode"
    key_to_check = api_key or (bearer_token.credentials if bearer_token else None)
    if key_to_check and key_to_check in security_config.valid_api_keys:
        return key_to_check
    if not key_to_check:
        raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="API key required.")
    raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Invalid API key")

async def verify_admin_key(
    api_key: Optional[str] = Depends(api_key_header),
    bearer_token: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme)
) -> str:
    if not security_config.admin_keys:
        return await verify_api_key(api_key, bearer_token)
    key_to_check = api_key or (bearer_token.credentials if bearer_token else None)
    if key_to_check and key_to_check in security_config.admin_keys:
        return key_to_check
    if not key_to_check:
        raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Admin API key required")
    raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Invalid admin API key")

class GeminiKeyManager:
    def __init__(self, config: AppConfig):
        self.config = config
        self.keys: Dict[str, APIKeyInfo] = {}
        self.key_cycle = None
        self.lock = asyncio.Lock()
        self.last_key_used = None
        self.key_performance = defaultdict(lambda: {"response_times": deque(maxlen=100), "errors": 0})

        for key in config.GEMINI_API_KEYS:
            if key and key.strip():
                self.keys[key] = APIKeyInfo(key=key)

        if not self.keys:
            raise ValueError("No valid API keys provided to key manager")

        logger.info(f"Initialized {len(self.keys)} API keys with performance tracking.")

    async def get_available_key(self) -> Optional[APIKeyInfo]:
        async with self.lock:
            recovered_count = await self._check_and_recover_keys_internal()
            if recovered_count > 0:
                logger.info(f"Recovered {recovered_count} keys during get_available_key")
            active_keys = [k for k in self.keys.values() if k.status == KeyStatus.ACTIVE]
            if not active_keys:
                logger.warning("No available API keys.")
                return None
            selected_key = self._select_best_key(active_keys)
            self.last_key_used = selected_key.key
            return selected_key

    async def _check_and_recover_keys_internal(self) -> int:
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
        if recovered_count > 0:
            self.key_cycle = None
        return recovered_count

    async def mark_key_failed(self, key: str, error: str):
        async with self.lock:
            key_info = self.keys.get(key)
            if not key_info:
                logger.warning(f"Attempted to mark unknown key as failed: {key[:8]}...")
                return
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
        import re
        error_lower = error.lower()
        status_code = 0
        status_patterns = [r'status code (\d{3})', r'HTTP (\d{3})', r'Error (\d{3})', r'(\d{3})']
        for pattern in status_patterns:
            match = re.search(pattern, error_lower)
            if match:
                status_code = int(match.group(1))
                break
        permanent_patterns = [
            'invalid api key', 'api key not found', 'api key disabled', 'account disabled',
            'account suspended', 'account terminated', 'unauthorized', 'authentication failed',
            'access denied', 'billing disabled', 'payment required', 'payment failed',
            'quota exceeded permanently', 'api key revoked', 'project not found',
            'project deleted', 'service disabled', 'permission denied'
        ]
        if any(pattern in error_lower for pattern in permanent_patterns) or status_code in [401, 402, 403, 404]:
            return 'PERMANENT', -1
        extended_patterns = [
            'quota', 'rate limit', 'rate_limit', 'too many requests', 'resource exhausted',
            'limit exceeded', 'usage limit', 'billing quota', 'daily limit', 'monthly limit'
        ]
        if any(pattern in error_lower for pattern in extended_patterns) or status_code == 429:
            return 'EXTENDED_COOLING', 1800
        if status_code >= 500:
            return 'SERVER_ERROR', 300
        timeout_patterns = [
            'timeout', 'connection', 'network', 'dns', 'unreachable', 'read timeout',
            'connect timeout', 'request timeout', 'connection reset', 'connection refused'
        ]
        if any(pattern in error_lower for pattern in timeout_patterns):
            return 'NETWORK_ERROR', 600
        return 'DEFAULT', self.config.GEMINI_COOLING_PERIOD

    def _select_best_key(self, active_keys: List[APIKeyInfo]) -> APIKeyInfo:
        def key_score(key_info: APIKeyInfo) -> float:
            success_rate = key_info.successful_requests / max(key_info.total_requests, 1)
            perf_data = self.key_performance[key_info.key]
            avg_response_time = sum(perf_data["response_times"]) / max(len(perf_data["response_times"]), 1)
            response_score = 1.0 / (1.0 + avg_response_time)
            recent_use_penalty = 0.9 if key_info.key == self.last_key_used else 1.0
            return (success_rate * 0.6 + response_score * 0.3) * recent_use_penalty
        return max(active_keys, key=key_score, default=random.choice(active_keys))

    async def record_key_performance(self, key: str, response_time: float, success: bool):
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
            key_info.failure_count = 0
            key_info.last_success_time = time.time()
            if key_info.status == KeyStatus.ACTIVE:
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
                        "key": f"{k.key[:8]}...", "status": k.status.value, "failure_count": k.failure_count,
                        "total_requests": k.total_requests, "successful_requests": k.successful_requests,
                        "success_rate": (k.successful_requests / k.total_requests * 100) if k.total_requests > 0 else 0,
                        "last_success": datetime.fromtimestamp(k.last_success_time).isoformat() if k.last_success_time else None,
                        "last_failure": datetime.fromtimestamp(k.last_failure_time).isoformat() if k.last_failure_time else None,
                        "cooling_until": datetime.fromtimestamp(k.cooling_until).isoformat() if k.cooling_until else None,
                    } for k in self.keys.values()
                ],
            }

    async def _check_and_recover_keys(self) -> int:
        async with self.lock:
            return await self._check_and_recover_keys_internal()

    async def reset_key(self, key_prefix: str) -> Dict[str, Any]:
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
            self.key_cycle = None
            logger.info(f"Admin reset API key {matched_key[:8]}... from {old_status.value} to active status")
            return {"message": f"Successfully reset key {matched_key[:8]}... from {old_status.value} to active"}

    async def attempt_key_recovery(self, key_prefix: str) -> Dict[str, Any]:
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
            if key_info.status != KeyStatus.FAILED:
                return {"error": f"Key {matched_key[:8]}... is not in FAILED status (current: {key_info.status.value})"}
            old_status = key_info.status
            key_info.status = KeyStatus.ACTIVE
            key_info.failure_count = 0
            key_info.cooling_until = None
            key_info.last_failure_time = None
            self.key_cycle = None
            logger.info(f"Admin recovered API key {matched_key[:8]}... from {old_status.value} to active status")
            return {"message": f"Successfully recovered key {matched_key[:8]}... from {old_status.value} to active"}

class LiteLLMAdapter:
    def __init__(self, config: AppConfig, key_manager: GeminiKeyManager, api_config: AnthropicAPIConfig):
        self.config = config
        self.key_manager = key_manager
        self.anthropic_to_gemini = api_config.anthropic_to_gemini
        self.gemini_to_anthropic = api_config.gemini_to_anthropic
        self.tool_converter = api_config.tool_converter
        self.claude_code_simulator = api_config.claude_code_simulator
        
        self._request_deduplicator: Dict[str, asyncio.Future] = {}
        self._dedup_lock = asyncio.Lock()
        if config.GEMINI_PROXY_URL:
            os.environ['HTTPS_PROXY'] = config.GEMINI_PROXY_URL
            os.environ['HTTP_PROXY'] = config.GEMINI_PROXY_URL
            logger.info(f"Using proxy: {config.GEMINI_PROXY_URL}")
        litellm.request_timeout = 30
        litellm.max_retries = 0
        litellm.set_verbose = False
        litellm.drop_params = True
        litellm.num_retries = 0

    @monitor_errors
    async def chat_completion(self, request: ChatRequest) -> Union[Dict, Any]:
        request_hash = hashlib.md5(json.dumps({
            "model": request.model, "messages": request.messages,
            "temperature": request.temperature, "stream": request.stream
        }, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()
        
        if not request.stream:
            async with self._dedup_lock:
                if request_hash in self._request_deduplicator:
                    logger.debug(f"Request de-duplication hit for hash: {request_hash}")
                    return await self._request_deduplicator[request_hash]
                future = asyncio.Future()
                self._request_deduplicator[request_hash] = future
        else:
            future = None
        try:
            response = await self._execute_chat_completion(request)
            if future:
                future.set_result(response)
            return response
        except Exception as e:
            if future:
                future.set_exception(e)
            raise e
        finally:
            if future:
                async with self._dedup_lock:
                    if request_hash in self._request_deduplicator:
                        del self._request_deduplicator[request_hash]
    
    def _safe_validate_messages(self, messages: List) -> bool:
        """Safe message validation that doesn't raise exceptions"""
        try:
            if not messages:
                logger.warning("Messages list is empty")
                return False
            
            for i, msg in enumerate(messages):
                if not isinstance(msg, dict):
                    logger.warning(f"Message {i} is not a dict: {type(msg)}")
                    return False
                
                if 'role' not in msg:
                    logger.warning(f"Message {i} missing 'role': {msg}")
                    return False
                
                # Check for any content field
                has_content = any(field in msg for field in ['content', 'parts', 'text'])
                if not has_content:
                    logger.warning(f"Message {i} has no content field: {msg}")
                    # Don't fail validation, just warn
                    
            return True
        except Exception as e:
            logger.error(f"Error validating messages: {e}")
            return False

    async def _execute_chat_completion(self, request: ChatRequest) -> Union[Dict, Any]:
        last_error = "No active keys to attempt request."
        attempted_keys = set()
        
        cache_key = None

        if not request.stream and self.config.CACHE_ENABLED and performance.response_cache:
            cache_key = {
                "model": request.model, "messages": request.messages,
                "temperature": request.temperature, "max_tokens": request.max_tokens
            }
            cached_response = await performance.response_cache.get(cache_key)
            if cached_response:
                logger.debug("Cache hit for chat completion request.")
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
                    kwargs = {
                        "model": f"gemini/{request.model}", 
                        "messages": request.messages,
                        "api_key": key_info.key, 
                        "temperature": request.temperature,
                        "stream": request.stream,
                    }
                    
                    # Safe validation
                    if not self._safe_validate_messages(kwargs["messages"]):
                        raise ValueError("Message validation failed")
                    
                    if request.max_tokens:
                        kwargs["max_tokens"] = request.max_tokens
                    if request.stop_sequences:
                        kwargs["stop"] = request.stop_sequences

                    response = await asyncio.wait_for(
                        litellm.acompletion(**kwargs), timeout=self.config.GEMINI_REQUEST_TIMEOUT
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
            raise HTTPException(status_code=503, detail="No available API keys")
        
        def _select_best_key(k_info):
            return random.random()

        keys_to_try = sorted(active_keys, key=_select_best_key, reverse=True)[:max_concurrent]

        try:
            tasks = [asyncio.create_task(try_key_request(key_info)) for key_info in keys_to_try]
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            
            for task in pending:
                task.cancel()
                
            for task in done:
                try:
                    result = task.result()
                    if result:
                        if not request.stream and cache_key and self.config.CACHE_ENABLED and performance.response_cache:
                            await performance.response_cache.set(cache_key, result.model_dump() if hasattr(result, 'model_dump') else result)
                        return result
                except Exception as e:
                    last_error = str(e)
                    continue
        except Exception as e:
            last_error = str(e)
            
        raise HTTPException(status_code=502, detail=f"All attempted keys failed. Last error: {last_error}")

    async def anthropic_messages_completion(self, request: MessagesRequest) -> Union[MessagesResponse, AsyncGenerator[str, None]]:
        try:
            # Log original request for debugging
            logger.info(f"Original Anthropic request - Model: {request.model}, Messages: {len(request.messages)}")
            logger.debug(f"Original messages preview: {json.dumps(request.model_dump().get('messages', [])[:2], indent=2)}")
            
            # Convert Anthropic request to Gemini format
            gemini_request_dict = self.anthropic_to_gemini.convert_request(request)
            
            # Log converted request
            logger.info(f"Converted Gemini request - Model: {gemini_request_dict.get('model', 'N/A')}")
            logger.info(f"Converted messages count: {len(gemini_request_dict.get('messages', []))}")
            logger.debug(f"Converted messages preview: {json.dumps(gemini_request_dict.get('messages', [])[:2], indent=2)}")
            
            # Validate that messages exist after conversion
            if not gemini_request_dict.get("messages"):
                logger.error("Empty messages after conversion!")
                logger.error(f"Original request: {request.model_dump()}")
                logger.error(f"Converted request: {gemini_request_dict}")
                raise HTTPException(status_code=400, detail="Messages cannot be empty after conversion")
            
            # Create ChatRequest with proper validation
            chat_request_data = {
                "messages": gemini_request_dict["messages"],
                "model": gemini_request_dict["model"],
                "temperature": gemini_request_dict.get("temperature"),
                "max_tokens": gemini_request_dict.get("max_tokens"),
                "stream": gemini_request_dict["stream"]
            }
            
            # Clean up model name
            if chat_request_data["model"].startswith("gemini/"):
                chat_request_data["model"] = chat_request_data["model"].replace("gemini/", "")

            logger.info(f"Creating ChatRequest with model: {chat_request_data['model']}")
            chat_request = ChatRequest(**chat_request_data)
            
            # Prepare LiteLLM kwargs
            litellm_kwargs = {}
            if "tools" in gemini_request_dict:
                litellm_kwargs["tools"] = gemini_request_dict["tools"]
            if "tool_config" in gemini_request_dict:
                 litellm_kwargs["tool_choice"] = gemini_request_dict["tool_config"]["function_calling_config"]["mode"]

            if "system_instruction" in gemini_request_dict:
                system_content = gemini_request_dict["system_instruction"]["parts"][0]["text"]
                litellm_kwargs["system_message"] = system_content

            litellm_kwargs.update(chat_request.model_dump())
            
            # Final validation before calling LiteLLM
            if not litellm_kwargs.get("messages"):
                logger.error("Final litellm_kwargs has empty messages!")
                logger.error(f"Final kwargs: {json.dumps({k: v for k, v in litellm_kwargs.items() if k != 'api_key'}, indent=2)}")
                raise HTTPException(status_code=400, detail="Final request has empty messages")

            logger.info(f"Calling LiteLLM with {len(litellm_kwargs['messages'])} messages")

            if request.stream:
                gemini_stream = await self.chat_completion_with_litellm(litellm_kwargs)
                streaming_generator = StreamingResponseGenerator(request, self.claude_code_simulator)
                return streaming_generator.generate_sse_events(gemini_stream)
            else:
                gemini_response_model = await self.chat_completion_with_litellm(litellm_kwargs)
                gemini_response_dict = gemini_response_model.model_dump()
                return await self.gemini_to_anthropic.convert_response(gemini_response_dict, request)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in anthropic_messages_completion: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    async def chat_completion_with_litellm(self, litellm_kwargs: Dict) -> Any:
        key_info = await self.key_manager.get_available_key()
        if not key_info:
            raise HTTPException(status_code=503, detail="No available API keys")
        
        litellm_kwargs["api_key"] = key_info.key
        litellm_kwargs["model"] = f"gemini/{litellm_kwargs['model']}"
        
        # Log request details
        logger.info(f"LiteLLM call - Model: {litellm_kwargs['model']}, Key: {key_info.key[:8]}...")
        logger.info(f"Messages count: {len(litellm_kwargs.get('messages', []))}")
        
        # Safe validation
        messages = litellm_kwargs.get('messages', [])
        if not self._safe_validate_messages(messages):
            raise HTTPException(status_code=400, detail="Message validation failed")

        try:
            start_time = time.time()
            response = await litellm.acompletion(**litellm_kwargs)
            response_time = time.time() - start_time
            await self.key_manager.mark_key_success(key_info.key)
            await self.key_manager.record_key_performance(key_info.key, response_time, True)
            return response
        except Exception as e:
            response_time = time.time() - start_time
            await self.key_manager.mark_key_failed(key_info.key, str(e))
            await self.key_manager.record_key_performance(key_info.key, response_time, False)
            logger.error(f"LiteLLM call failed with key {key_info.key[:8]}: {e}")
            # Log the failed request arguments for easier debugging, masking the api_key
            failed_kwargs = {k: v for k, v in litellm_kwargs.items() if k != 'api_key'}
            logger.error(f"Failed request kwargs: {json.dumps(failed_kwargs, indent=2)}")
            raise HTTPException(status_code=502, detail=f"Model provider error: {e}")


key_manager: Optional[GeminiKeyManager] = None
adapter: Optional[LiteLLMAdapter] = None

class RateLimiter:
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(deque)
        self.lock = asyncio.Lock()
    async def is_allowed(self, client_id: str) -> bool:
        async with self.lock:
            now = time.time()
            client_requests = self.requests[client_id]
            cutoff_time = now - self.window_seconds
            while client_requests and client_requests[0] < cutoff_time:
                client_requests.popleft()
            if len(client_requests) >= self.max_requests:
                return False
            client_requests.append(now)
            return True
    def get_remaining_requests(self, client_id: str) -> int:
        client_requests = self.requests[client_id]
        return max(0, self.max_requests - len(client_requests))


rate_limiter = RateLimiter(max_requests=100, window_seconds=60)

async def check_rate_limit(client_key: str = Depends(verify_api_key)):
    if security_config and security_config.enable_rate_limiting:
        if not await rate_limiter.is_allowed(client_key):
            raise HTTPException(status_code=429, detail="Rate limit exceeded.")
    return client_key

async def optimized_health_check_task():
    while True:
        try:
            if key_manager is not None:
                recovered_count = await key_manager._check_and_recover_keys()
                if recovered_count > 0:
                    logger.info(f"Health check recovered {recovered_count} keys")
            app_config = get_config()
            await asyncio.sleep(app_config.GEMINI_HEALTH_CHECK_INTERVAL)
        except asyncio.CancelledError:
            logger.info("Health check task cancelled.")
            break
        except Exception as e:
            logger.error(f"Health check error: {e}")
            await asyncio.sleep(60)

def custom_convert_model(anthropic_model: str) -> str:
    """
    Converts an Anthropic model name to a corresponding Gemini model name
    based on the user's explicit requirements.

    - "sonnet" and "opus" variants map to "gemini-2.5-pro".
    - "haiku" variants map to "gemini-2.5-flash".
    """
    anthropic_model_lower = anthropic_model.lower()
    
    if "sonnet" in anthropic_model_lower or "opus" in anthropic_model_lower:
        return "gemini-2.5-pro"
    elif "haiku" in anthropic_model_lower:
        return "gemini-2.5-flash"
    else:
        logger.warning(f"Model '{anthropic_model}' not in custom mappings, falling back to 'gemini-2.5-pro'.")
        return "gemini-2.5-pro"

@asynccontextmanager
async def lifespan(app: FastAPI):
    global key_manager, adapter, security_config, api_config
    
    os.makedirs("logs", exist_ok=True)
    logger.add("logs/gemini_adapter_{time}.log", rotation="1 day", retention="7 days", level="INFO", enqueue=True, catch=True)
    try:
        app.state.start_time = time.time()
        app_config = get_config()
        
        performance.initialize_performance_modules(
            cache_enabled=app_config.CACHE_ENABLED,
            cache_max_size=app_config.CACHE_MAX_SIZE,
            cache_ttl=app_config.CACHE_TTL,
            cache_key_prefix=app_config.CACHE_KEY_PREFIX
        )
        security_config = SecurityConfig(app_config)
        
        working_dir = os.getenv("CLAUDE_CODE_WORKING_DIR", ".")
        api_config = AnthropicAPIConfig(working_directory=working_dir)
        
        api_config.anthropic_to_gemini.convert_model = custom_convert_model
        logger.info("Applied custom model mapping: 'opus'/'sonnet' -> 'gemini-2.5-pro', 'haiku' -> 'gemini-2.5-flash'.")

        logger.info(f"Claude Code support enabled. Working directory: {api_config.working_directory}")
        
        key_manager = GeminiKeyManager(app_config)
        adapter = LiteLLMAdapter(app_config, key_manager, api_config)
        
        health_task = asyncio.create_task(optimized_health_check_task())
        logger.info("Gemini Claude Adapter v2.1.0 (Claude Code Enabled) started successfully.")
        yield
    except Exception as e:
        logger.critical(f"Failed to start application: {e}", exc_info=True)
        raise
    finally:
        if 'health_task' in locals() and not health_task.done():
            health_task.cancel()
        logger.info("Gemini Claude Adapter shutting down.")

app = FastAPI(
    title="Gemini Claude Adapter v2.1.0 (Claude Code Enabled)",
    description="High-performance adapter with Anthropic API compatibility and Claude Code support.",
    version="2.1.0-claude",
    lifespan=lifespan
)

async def stream_generator(response_stream):
    try:
        async for chunk in response_stream:
            try:
                if hasattr(chunk, 'model_dump'):
                    chunk_data = chunk.model_dump()
                elif hasattr(chunk, 'dict'):
                    chunk_data = chunk.dict()
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

@app.get("/", include_in_schema=False)
async def root():
    return {"name": "Gemini Claude Adapter", "version": "2.1.0-claude", "status": "running"}

@app.get("/health")
async def health_check():
    if not key_manager:
        raise HTTPException(status_code=503, detail="Service not initialized")
    stats = await key_manager.get_stats()
    is_healthy = stats["active_keys"] > 0
    return JSONResponse(status_code=200 if is_healthy else 503, content={"status": "healthy" if is_healthy else "degraded", **stats})


@app.post("/v1/messages")
async def create_message(request: MessagesRequest, raw_request: Request, api_key: str = Depends(check_rate_limit)):
    if not adapter:
        raise HTTPException(status_code=503, detail="Service not initialized")
    try:
        gemini_model = adapter.anthropic_to_gemini.convert_model(request.model)
        log_request_beautifully(
            method="POST", path=str(raw_request.url.path), anthropic_model=request.model,
            gemini_model=gemini_model, num_messages=len(request.messages), 
            num_tools=len(request.tools) if request.tools else 0
        )
        response = await adapter.anthropic_messages_completion(request)
        if request.stream:
            return StreamingResponse(response, media_type="text/event-stream")
        else:
            return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /v1/messages: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.options("/v1/messages")
async def options_messages():
    return JSONResponse(content={"status": "ok"}, headers={"Access-Control-Allow-Origin": "*", "Access-Control-Allow-Methods": "POST, OPTIONS", "Access-Control-Allow-Headers": "Content-Type, Authorization, X-API-Key, Anthropic-Version"})

@app.post("/v1/messages/count_tokens", response_model=TokenCountResponse)
async def count_tokens(request: TokenCountRequest, api_key: str = Depends(verify_api_key)):
    if not adapter:
        raise HTTPException(status_code=503, detail="Service not initialized")
    try:
        gemini_request_dict = adapter.anthropic_to_gemini.convert_request(
            MessagesRequest(
                model=request.model, max_tokens=1, messages=request.messages,
                system=request.system, tools=request.tools
            )
        )
        if "system_instruction" in gemini_request_dict:
            gemini_request_dict["messages"].insert(0, {"role": "system", "content": gemini_request_dict["system_instruction"]["parts"][0]["text"]})
        
        token_count = litellm.token_counter(
            model=gemini_request_dict["model"], 
            messages=gemini_request_dict["messages"]
        )
        return TokenCountResponse(input_tokens=token_count)
    except Exception as e:
        logger.error(f"Error counting tokens: {e}")
        raise HTTPException(status_code=500, detail="Token counting failed")

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest, api_key: str = Depends(verify_api_key)):
    if not adapter:
        raise HTTPException(status_code=503, detail="Service not initialized")
    try:
        response = await adapter.chat_completion(request)
        if request.stream:
            return StreamingResponse(stream_generator(response), media_type="text/event-stream")
        else:
            return response.model_dump()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in chat completion: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/stats")
async def get_stats_endpoint(api_key: str = Depends(verify_api_key)):
    if not key_manager:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return await key_manager.get_stats()

@app.get("/v1/models")
async def get_models(api_key: str = Depends(verify_api_key)):
    return {
        "object": "list",
        "data": [
            {"id": "claude-3-sonnet", "object": "model", "owned_by": "anthropic"},
            {"id": "claude-3-opus", "object": "model", "owned_by": "anthropic"},
            {"id": "claude-3-haiku", "object": "model", "owned_by": "anthropic"}
        ]
    }

@app.post("/admin/reset-key/{key_prefix}")
async def reset_key_endpoint(key_prefix: str, api_key: str = Depends(verify_admin_key)):
    if not key_manager:
        raise HTTPException(status_code=503, detail="Service not initialized")
    result = await key_manager.reset_key(key_prefix)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result

@app.post("/admin/recover-key/{key_prefix}")
async def recover_key_endpoint(key_prefix: str, api_key: str = Depends(verify_admin_key)):
    if not key_manager:
        raise HTTPException(status_code=503, detail="Service not initialized")
    result = await key_manager.attempt_key_recovery(key_prefix)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception for request {request.method} {request.url}: {exc}", exc_info=True)
    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"error": {"message": exc.detail, "type": "http_exception"}})
    return JSONResponse(status_code=500, content={"error": {"message": "An internal server error occurred.", "type": "internal_error"}})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info", reload=True)

