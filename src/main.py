import asyncio
import time
import random
import os
from typing import List, Dict, Optional, Any, Union, Set, Tuple, AsyncGenerator
from dataclasses import dataclass
from enum import Enum
import json
import httpx
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, field_validator
from loguru import logger
import litellm
from contextlib import asynccontextmanager
from datetime import datetime
from dotenv import load_dotenv
from starlette.status import HTTP_401_UNAUTHORIZED, HTTP_403_FORBIDDEN
from collections import defaultdict, deque

# Import using relative paths, which is the correct way for a package
from .config import get_config, AppConfig, GeminiConfig
from .error_handling import error_monitor, monitor_errors
from .performance import response_cache, http_client, performance_monitor, monitor_performance, get_performance_stats
from .anthropic_api import (
    MessagesRequest, MessagesResponse,
    AnthropicToGeminiConverter, GeminiToAnthropicConverter,
    StreamingResponseGenerator,
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

class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    model: str = "gemini-2.5-pro"
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, ge=1)
    stream: bool = False
    
    @field_validator('messages')
    def validate_messages(cls, v):
        if not v:
            raise ValueError("Messages cannot be empty")
        for msg in v:
            if 'role' not in msg or 'content' not in msg:
                raise ValueError("Each message must have 'role' and 'content' fields")
        return v

class SecurityManager:
    def __init__(self, app_config: AppConfig):
        self.config = app_config.security
        self.security_enabled = bool(self.config.adapter_api_keys)
        
        if self.security_enabled:
            logger.info(f"Security enabled with {len(self.config.adapter_api_keys)} client keys.")
            if self.config.admin_api_keys:
                logger.info(f"Admin access enabled with {len(self.config.admin_api_keys)} admin keys.")
            else:
                logger.info("No admin keys configured; client keys will have admin access.")
        else:
            logger.warning("Security disabled: ADAPTER_API_KEYS is not configured.")

security_manager: Optional[SecurityManager] = None

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)

async def verify_api_key(
    api_key_header_value: Optional[str] = Depends(api_key_header),
    bearer_token: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme)
) -> str:
    if not security_manager or not security_manager.security_enabled:
        return "insecure_mode"
    
    key_to_check = api_key_header_value or (bearer_token.credentials if bearer_token else None)
    
    adapter_keys = set(security_manager.config.adapter_api_keys)
    admin_keys = set(security_manager.config.admin_api_keys)
    all_valid_keys = adapter_keys.union(admin_keys)

    if not security_manager.config.admin_api_keys:
        all_valid_keys = adapter_keys

    if key_to_check in all_valid_keys:
        return key_to_check
    
    if not key_to_check:
        raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="API key is missing.")
    
    raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Invalid API key.")

async def verify_admin_key(
    api_key: str = Depends(verify_api_key)
) -> str:
    if not security_manager:
         raise HTTPException(status_code=503, detail="Security manager not initialized")

    if not security_manager.config.admin_api_keys:
        return api_key
    
    if api_key in security_manager.config.admin_api_keys:
        return api_key
    
    raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Admin privileges required.")

class GeminiKeyManager:
    def __init__(self, config: GeminiConfig):
        self.config = config
        self.keys: Dict[str, APIKeyInfo] = {key.strip(): APIKeyInfo(key=key.strip()) for key in config.api_keys if key.strip()}
        self.lock = asyncio.Lock()
        if not self.keys:
            raise ValueError("GEMINI_API_KEYS environment variable is not set or empty.")
        logger.info(f"Initialized {len(self.keys)} Gemini API keys.")

    async def get_available_key(self) -> Optional[APIKeyInfo]:
        async with self.lock:
            current_time = time.time()
            active_keys = []
            for key_info in self.keys.values():
                if key_info.status == KeyStatus.COOLING and key_info.cooling_until and current_time > key_info.cooling_until:
                    key_info.status = KeyStatus.ACTIVE
                    key_info.failure_count = 0
                    logger.info(f"API key {key_info.key[:8]}... has cooled down and is now active.")
                if key_info.status == KeyStatus.ACTIVE:
                    active_keys.append(key_info)
            
            if not active_keys:
                logger.warning("No available API keys.")
                return None
            
            return random.choice(active_keys)

    async def mark_key_failed(self, key: str, error: str):
        async with self.lock:
            if key in self.keys:
                key_info = self.keys[key]
                key_info.failure_count += 1
                key_info.last_failure_time = time.time()
                if key_info.failure_count >= self.config.max_failures:
                    key_info.status = KeyStatus.COOLING
                    key_info.cooling_until = time.time() + self.config.cooling_period
                    logger.warning(f"API key {key[:8]}... failed {key_info.failure_count} times. Cooling for {self.config.cooling_period}s. Error: {error}")

    async def mark_key_success(self, key: str):
        async with self.lock:
            if key in self.keys:
                key_info = self.keys[key]
                if key_info.failure_count > 0:
                    key_info.failure_count = 0
                key_info.last_success_time = time.time()

    async def get_stats(self) -> Dict[str, Any]:
        async with self.lock:
            return {
                "total_keys": len(self.keys),
                "active_keys": sum(1 for k in self.keys.values() if k.status == KeyStatus.ACTIVE),
                "cooling_keys": sum(1 for k in self.keys.values() if k.status == KeyStatus.COOLING),
                "failed_keys": sum(1 for k in self.keys.values() if k.status == KeyStatus.FAILED),
            }
            
    async def reset_key(self, key_prefix: str) -> Dict[str, str]:
        async with self.lock:
            for key, key_info in self.keys.items():
                if key.startswith(key_prefix):
                    old_status = key_info.status.value
                    key_info.status = KeyStatus.ACTIVE
                    key_info.failure_count = 0
                    key_info.cooling_until = None
                    logger.info(f"Admin reset key {key[:8]}... from {old_status} to active.")
                    return {"message": f"Key starting with {key_prefix} has been reset to active."}
            raise HTTPException(status_code=404, detail=f"No key found with prefix {key_prefix}.")


class LiteLLMAdapter:
    def __init__(self, config: GeminiConfig, key_manager: GeminiKeyManager):
        self.config = config
        self.key_manager = key_manager
        self.anthropic_to_gemini = AnthropicToGeminiConverter()
        self.gemini_to_anthropic = GeminiToAnthropicConverter()
        if config.proxy_url:
            os.environ['HTTPS_PROXY'] = config.proxy_url
            os.environ['HTTP_PROXY'] = config.proxy_url
            logger.info(f"Using proxy: {config.proxy_url}")
        litellm.set_verbose = False
        litellm.drop_params = True

    @monitor_errors
    async def chat_completion(self, request: ChatRequest) -> Union[Dict, Any]:
        last_error = None
        # Use a copy of keys to try to avoid issues with concurrent modifications
        keys_to_try = list(self.key_manager.keys.keys())
        random.shuffle(keys_to_try)

        for key in keys_to_try:
            key_info = await self.key_manager.get_available_key()
            if not key_info:
                continue # Skip if no key is available right now
            
            try:
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
                
                response = await litellm.acompletion(**kwargs)
                await self.key_manager.mark_key_success(key_info.key)
                return response
            except Exception as e:
                last_error = str(e)
                logger.warning(f"API call failed for key {key_info.key[:8]}... Error: {last_error}")
                await self.key_manager.mark_key_failed(key_info.key, last_error)
        
        raise HTTPException(status_code=502, detail=f"All available keys failed. Last error: {last_error}")

    async def anthropic_messages_completion(self, request: MessagesRequest) -> Union[MessagesResponse, AsyncGenerator[str, None]]:
        gemini_request_dict = self.anthropic_to_gemini.convert_request(request)
        chat_request = ChatRequest(**gemini_request_dict)

        if request.stream:
            gemini_stream = await self.chat_completion(chat_request)
            streaming_generator = StreamingResponseGenerator(request)
            return streaming_generator.generate_sse_events(gemini_stream)
        else:
            gemini_response = await self.chat_completion(chat_request)
            return self.gemini_to_anthropic.convert_response(gemini_response, request)

# Global instances
key_manager: Optional[GeminiKeyManager] = None
adapter: Optional[LiteLLMAdapter] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global key_manager, adapter, security_manager
    try:
        app_config = get_config()
        security_manager = SecurityManager(app_config)
        key_manager = GeminiKeyManager(app_config.gemini)
        adapter = LiteLLMAdapter(app_config.gemini, key_manager)
        logger.info("Gemini Claude Adapter started successfully.")
        yield
    except Exception as e:
        logger.critical(f"Fatal error during application startup: {e}", exc_info=True)
    finally:
        logger.info("Gemini Claude Adapter shutting down.")

app = FastAPI(
    title="Gemini Claude Adapter",
    description="An adapter to use Gemini API with clients compatible with Anthropic's Claude API.",
    version="2.1.0",
    lifespan=lifespan
)

@app.get("/", include_in_schema=False)
async def root():
    return {
        "name": "Gemini Claude Adapter",
        "version": "2.1.0",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health", summary="Health Check")
async def health_check():
    if not key_manager:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    stats = await key_manager.get_stats()
    if stats["active_keys"] > 0:
        return JSONResponse(content={"status": "healthy", **stats})
    else:
        return JSONResponse(status_code=503, content={"status": "degraded", **stats})

@app.post("/v1/messages")
async def create_message(request: MessagesRequest, api_key: str = Depends(verify_api_key)):
    if not adapter:
        raise HTTPException(status_code=503, detail="Service not initialized")

    response = await adapter.anthropic_messages_completion(request)
    if request.stream:
        return StreamingResponse(response, media_type="text/event-stream")
    return response

@app.get("/v1/models", summary="List Models")
async def get_models(api_key: str = Depends(verify_api_key)):
    return {
        "object": "list",
        "data": [
            {"id": "claude-3-5-sonnet", "object": "model", "created": int(time.time()), "owned_by": "anthropic"},
            {"id": "claude-3-5-haiku", "object": "model", "created": int(time.time()), "owned_by": "anthropic"},
            {"id": "claude-3-opus", "object": "model", "created": int(time.time()), "owned_by": "anthropic"},
            {"id": "claude-3-sonnet", "object": "model", "created": int(time.time()), "owned_by": "anthropic"},
            {"id": "claude-3-haiku", "object": "model", "created": int(time.time()), "owned_by": "anthropic"},
        ]
    }

@app.get("/stats", summary="Get Key Manager Stats")
async def get_key_stats(api_key: str = Depends(verify_admin_key)):
    if not key_manager:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return await key_manager.get_stats()

@app.post("/admin/reset-key/{key_prefix}", summary="Reset a Failed Key")
async def reset_failed_key(key_prefix: str, api_key: str = Depends(verify_admin_key)):
    if not key_manager:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return await key_manager.reset_key(key_prefix)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception for request {request.url.path}: {exc}", exc_info=True)
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"type": "api_error", "message": exc.detail}},
        )
    return JSONResponse(
        status_code=500,
        content={"error": {"type": "internal_server_error", "message": "An unexpected error occurred."}},
    )
