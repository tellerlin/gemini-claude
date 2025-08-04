# main.py
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
import copy

# 导入您本地的 anthropic_api.py 文件
from anthropic_api import (
    MessagesRequest, MessagesResponse, TokenCountRequest, TokenCountResponse,
    AnthropicToGeminiConverter, GeminiToAnthropicConverter,
    ToolConverter, log_request_beautifully,
    AnthropicAPIConfig
)

# --- MOCK AppConfig for standalone execution ---
# 在实际部署中，您可能会从一个单独的 config.py 文件导入这些
class AppConfig:
    SECURITY_ADAPTER_API_KEYS = os.getenv("SECURITY_ADAPTER_API_KEYS", "test-key").split(",")
    SECURITY_ADMIN_API_KEYS = os.getenv("SECURITY_ADMIN_API_KEYS", "").split(",")
    SECURITY_ENABLE_IP_BLOCKING = False
    SECURITY_MAX_FAILED_ATTEMPTS = 5
    SECURITY_BLOCK_DURATION = 300
    SECURITY_ENABLE_RATE_LIMITING = False
    SECURITY_RATE_LIMIT_REQUESTS = 100
    SECURITY_RATE_LIMIT_WINDOW = 60
    GEMINI_API_KEYS = os.getenv("GEMINI_API_KEYS", "your-gemini-key-1,your-gemini-key-2").split(",")
    GEMINI_COOLING_PERIOD = 300
    GEMINI_PROXY_URL = os.getenv("GEMINI_PROXY_URL", None)
    GEMINI_REQUEST_TIMEOUT = 120
    CACHE_ENABLED = False
    CACHE_MAX_SIZE = 100
    CACHE_TTL = 3600
    CACHE_KEY_PREFIX = "gemini_adapter"
    GEMINI_HEALTH_CHECK_INTERVAL = 60
    MAX_RETRIES_PER_REQUEST = 3

def get_config():
    return AppConfig()
# ---------------------------------

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
    model: str = "gemini-1.5-pro-latest"
    temperature: Optional[float] = Field(default=0.1, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.95, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    stream: bool = Field(default=False)
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[str] = None

class SecurityConfig:
    def __init__(self, app_config: AppConfig):
        self.valid_api_keys: Set[str] = {k for k in app_config.SECURITY_ADAPTER_API_KEYS if k}
        self.admin_keys: Set[str] = {k for k in app_config.SECURITY_ADMIN_API_KEYS if k}
        self.security_enabled = bool(self.valid_api_keys)
        if self.security_enabled:
            logger.info(f"Security enabled with {len(self.valid_api_keys)} client keys")
        else:
            logger.warning("Security disabled - no SECURITY_ADAPTER_API_KEYS configured")
        if self.admin_keys:
            logger.info(f"Admin access enabled with {len(self.admin_keys)} admin keys")
        else:
            logger.info("No admin keys configured")

security_config: Optional[SecurityConfig] = None
api_config: Optional[AnthropicAPIConfig] = None

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)

async def verify_api_key(
    api_key: Optional[str] = Depends(api_key_header),
    bearer_token: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme)
) -> str:
    if not security_config or not security_config.security_enabled:
        logger.debug("Security disabled, allowing access")
        return "insecure_mode"
    key_to_check = api_key or (bearer_token.credentials if bearer_token else None)
    if key_to_check and key_to_check in security_config.valid_api_keys:
        return key_to_check
    if not key_to_check:
        raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="API key required.")
    raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Invalid API key")

class GeminiKeyManager:
    def __init__(self, config: AppConfig):
        self.config = config
        self.keys: Dict[str, APIKeyInfo] = {}
        self.lock = asyncio.Lock()
        self.last_key_used: Optional[str] = None

        for key in config.GEMINI_API_KEYS:
            if key and key.strip():
                self.keys[key] = APIKeyInfo(key=key)

        if not self.keys:
            raise ValueError("No valid GEMINI_API_KEYS provided to key manager")

        logger.info(f"Initialized {len(self.keys)} Gemini API keys.")

    async def get_available_key(self) -> Optional[APIKeyInfo]:
        async with self.lock:
            await self._check_and_recover_keys_internal()
            active_keys = [k for k in self.keys.values() if k.status == KeyStatus.ACTIVE]
            if not active_keys:
                logger.warning("No available API keys.")
                return None
            
            # Simple rotation: try not to use the last used key if others are available
            if len(active_keys) > 1 and self.last_key_used:
                other_keys = [k for k in active_keys if k.key != self.last_key_used]
                if other_keys:
                    selected_key = random.choice(other_keys)
                    self.last_key_used = selected_key.key
                    return selected_key

            selected_key = random.choice(active_keys)
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
        return recovered_count

    async def mark_key_failed(self, key: str, error: str):
        async with self.lock:
            key_info = self.keys.get(key)
            if not key_info: return
            error_type, cooling_time = self._classify_error(error)
            if error_type == 'PERMANENT':
                key_info.status = KeyStatus.FAILED
                logger.error(f"API key {key[:8]}... permanently failed. Reason: {error_type}. Error: {error}")
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
        status_patterns = [r'status code (\d{3})', r'http (\d{3})', r'error (\d{3})']
        for pattern in status_patterns:
            match = re.search(pattern, error_lower, re.IGNORECASE)
            if match:
                status_code = int(match.group(1))
                break

        permanent_patterns = ['invalid api key', 'api key not found', 'api key disabled', 'permission denied']
        if any(p in error_lower for p in permanent_patterns) or status_code in [401, 403]:
            return 'PERMANENT', -1

        rate_limit_patterns = ['quota', 'rate limit', 'rate_limit', 'too many requests', 'resource exhausted']
        if any(p in error_lower for p in rate_limit_patterns) or status_code == 429:
            return 'RATE_LIMIT', self.config.GEMINI_COOLING_PERIOD

        if status_code >= 500:
            return 'SERVER_ERROR', 60

        return 'DEFAULT', self.config.GEMINI_COOLING_PERIOD

    async def mark_key_success(self, key: str):
        async with self.lock:
            key_info = self.keys.get(key)
            if not key_info: return
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
            }

    async def _check_and_recover_keys(self) -> int:
        async with self.lock:
            return await self._check_and_recover_keys_internal()

class LiteLLMAdapter:
    def __init__(self, config: AppConfig, key_manager: GeminiKeyManager, api_config: AnthropicAPIConfig):
        self.config = config
        self.key_manager = key_manager
        self.api_config = api_config
        self.anthropic_to_gemini = api_config.anthropic_to_gemini
        self.gemini_to_anthropic = api_config.gemini_to_anthropic
        
        if config.GEMINI_PROXY_URL:
            os.environ['HTTPS_PROXY'] = config.GEMINI_PROXY_URL
            os.environ['HTTP_PROXY'] = config.GEMINI_PROXY_URL
            logger.info(f"Using proxy: {config.GEMINI_PROXY_URL}")

        litellm.set_verbose = False
        litellm.drop_params = True
        litellm.num_retries = 0

    async def anthropic_messages_completion(self, request: MessagesRequest) -> Union[MessagesResponse, AsyncGenerator[str, None]]:
        try:
            gemini_request_dict = self.anthropic_to_gemini.convert_request(request)
            
            chat_request = ChatRequest(
                model=gemini_request_dict.pop("model"),
                messages=gemini_request_dict.pop("messages"),
                stream=request.stream,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                tools=gemini_request_dict.pop("tools", None),
                tool_choice=gemini_request_dict.pop("tool_choice", None)
            )

            if request.stream:
                gemini_stream = await self.chat_completion_with_litellm(chat_request)
                return self.gemini_to_anthropic.convert_stream_response(gemini_stream, request)
            else:
                gemini_response_model = await self.chat_completion_with_litellm(chat_request)
                gemini_response_dict = gemini_response_model.model_dump()
                return self.gemini_to_anthropic.convert_response(gemini_response_dict, request)

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Critical error in anthropic_messages_completion: {repr(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    async def chat_completion_with_litellm(self, chat_request: ChatRequest) -> Any:
        last_error = None
        
        max_attempts = min(self.config.MAX_RETRIES_PER_REQUEST, len(self.key_manager.keys))
        
        for attempt in range(max_attempts):
            key_info = await self.key_manager.get_available_key()
            if not key_info:
                logger.warning("No available API keys to attempt/retry request.")
                break 

            logger.info(f"Attempt {attempt + 1}/{max_attempts} using key {key_info.key[:8]}...")
            
            litellm_kwargs = chat_request.model_dump(exclude_none=True)
            litellm_kwargs["api_key"] = key_info.key
            litellm_kwargs["model"] = f"gemini/{chat_request.model}"
            litellm_kwargs["timeout"] = self.config.GEMINI_REQUEST_TIMEOUT

            try:
                start_time = time.time()
                response = await litellm.acompletion(**litellm_kwargs)
                response_time = time.time() - start_time
                
                await self.key_manager.mark_key_success(key_info.key)
                logger.info(f"Key {key_info.key[:8]}... succeeded in {response_time:.2f}s.")
                return response
            
            except Exception as e:
                last_error = e
                await self.key_manager.mark_key_failed(key_info.key, str(e))
                logger.warning(f"Attempt {attempt + 1} failed with key {key_info.key[:8]}. Error: {repr(e)}")
                await asyncio.sleep(0.5)

        error_detail = f"All API keys failed after {max_attempts} attempts. Last error: {repr(last_error)}"
        logger.error(error_detail)
        raise HTTPException(status_code=502, detail=error_detail)


key_manager: Optional[GeminiKeyManager] = None
adapter: Optional[LiteLLMAdapter] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global key_manager, adapter, security_config, api_config
    
    os.makedirs("logs", exist_ok=True)
    logger.add("logs/gemini_adapter_{time}.log", rotation="1 day", retention="7 days", level="INFO", enqueue=True, catch=True)
    try:
        app_config = get_config()
        security_config = SecurityConfig(app_config)
        
        working_dir = os.getenv("CLAUDE_CODE_WORKING_DIR", ".")
        api_config = AnthropicAPIConfig(working_directory=working_dir)
        
        key_manager = GeminiKeyManager(app_config)
        adapter = LiteLLMAdapter(app_config, key_manager, api_config)
        
        async def health_check_task():
            while True:
                await asyncio.sleep(app_config.GEMINI_HEALTH_CHECK_INTERVAL)
                await key_manager._check_and_recover_keys()

        health_task = asyncio.create_task(health_check_task())
        logger.info("Gemini Claude Adapter (with retry logic) started successfully.")
        yield
    except Exception as e:
        logger.critical(f"Failed to start application: {e}", exc_info=True)
        raise
    finally:
        if 'health_task' in locals() and not health_task.done():
            health_task.cancel()
        logger.info("Gemini Claude Adapter shutting down.")


app = FastAPI(
    title="Gemini Claude Adapter (Fixed with Retry)",
    description="Adapter with Anthropic API compatibility and automatic key rotation on failure.",
    version="2.4.0-complete",
    lifespan=lifespan
)

@app.post("/v1/messages")
async def create_message(request: MessagesRequest, raw_request: Request, client_key: str = Depends(verify_api_key)):
    if not adapter:
        raise HTTPException(status_code=503, detail="Service not initialized")
    try:
        # Log the mapping
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
        logger.error(f"Unhandled error in /v1/messages endpoint: {repr(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected internal server error occurred.")

@app.get("/", include_in_schema=False)
async def root():
    return {"name": "Gemini Claude Adapter", "version": "2.4.0-complete", "status": "running"}

@app.get("/health")
async def health_check(client_key: str = Depends(verify_api_key)):
    if not key_manager:
        raise HTTPException(status_code=503, detail="Service not initialized")
    stats = await key_manager.get_stats()
    is_healthy = stats["active_keys"] > 0
    return JSONResponse(status_code=200 if is_healthy else 503, content={"status": "healthy" if is_healthy else "degraded", **stats})

if __name__ == "__main__":
    import uvicorn
    # 确保设置了环境变量 GEMINI_API_KEYS
    # 例如: export GEMINI_API_KEYS=your_key_1,your_key_2
    # 同时可以设置 SECURITY_ADAPTER_API_KEYS 来保护你的适配器
    # 例如: export SECURITY_ADAPTER_API_KEYS="a_secret_key_for_claude_code"
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info", reload=True)