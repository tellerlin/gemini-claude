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

# --- MODIFIED: Imports adjusted for flat structure ---
# 假设这些文件与 main.py 在同一目录
# from config import get_config, AppConfig
# from error_handling import error_monitor, monitor_errors, ErrorClassifier
# import performance
from anthropic_api import (
    MessagesRequest, MessagesResponse, TokenCountRequest, TokenCountResponse,
    AnthropicToGeminiConverter, GeminiToAnthropicConverter,
    ToolConverter, log_request_beautifully,
    AnthropicAPIConfig
)

# --- MOCK for standalone execution ---
class AppConfig:
    SECURITY_ADAPTER_API_KEYS = os.getenv("SECURITY_ADAPTER_API_KEYS", "test-key").split(",")
    SECURITY_ADMIN_API_KEYS = []
    SECURITY_ENABLE_IP_BLOCKING = False
    SECURITY_MAX_FAILED_ATTEMPTS = 5
    SECURITY_BLOCK_DURATION = 300
    SECURITY_ENABLE_RATE_LIMITING = False
    SECURITY_RATE_LIMIT_REQUESTS = 100
    SECURITY_RATE_LIMIT_WINDOW = 60
    GEMINI_API_KEYS = os.getenv("GEMINI_API_KEYS", "your-gemini-key-1,your-gemini-key-2").split(",")
    GEMINI_COOLING_PERIOD = 300
    GEMINI_PROXY_URL = os.getenv("GEMINI_PROXY_URL", None)
    GEMINI_REQUEST_TIMEOUT = 60
    CACHE_ENABLED = False
    CACHE_MAX_SIZE = 100
    CACHE_TTL = 3600
    CACHE_KEY_PREFIX = "gemini_adapter"
    GEMINI_HEALTH_CHECK_INTERVAL = 60

def get_config():
    return AppConfig()

def monitor_errors(func):
    return func
# ---------------------------------


load_dotenv()

def safe_format_for_log(message: str, *args, **kwargs) -> str:
    try:
        if args or kwargs:
            return message.format(*args, **kwargs)
        else:
            return message
    except (ValueError, KeyError) as e:
        safe_args = [repr(arg) for arg in args]
        safe_kwargs = {k: repr(v) for k, v in kwargs.items()}
        return f"{message} [FORMAT_ERROR: args={safe_args}, kwargs={safe_kwargs}]"

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
    model: str = "gemini-1.5-pro-latest" # Updated default model
    temperature: Optional[float] = Field(default=0.1, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.95, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    stop_sequences: Optional[List[str]] = Field(default=None, max_length=5)
    stream: bool = Field(default=False)
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[str] = None


    @field_validator('messages')
    @classmethod
    def validate_messages(cls, v):
        if not v:
            raise ValueError("Messages cannot be empty")
        for i, msg in enumerate(v):
            if not isinstance(msg, dict) or 'role' not in msg:
                raise ValueError(f"Message {i} must have 'role' field")
            
            # Content can be a string, a list (for multimodal), or absent (for tool calls)
            content_present = 'content' in msg and msg['content'] is not None
            tool_calls_present = 'tool_calls' in msg and msg['tool_calls'] is not None

            if not content_present and not tool_calls_present:
                 logger.warning(f"Message {i} has neither 'content' nor 'tool_calls'. This is unusual. Message: {msg}")

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
    if not security_config or not security_config.security_enabled:
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
    if not security_config or not security_config.admin_keys:
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
            await self._check_and_recover_keys_internal()
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
        return recovered_count

    async def mark_key_failed(self, key: str, error: str):
        async with self.lock:
            key_info = self.keys.get(key)
            if not key_info:
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
        
        permanent_patterns = ['invalid api key', 'api key not found', 'api key disabled', 'permission denied']
        if any(p in error_lower for p in permanent_patterns) or status_code in [401, 403]:
            return 'PERMANENT', -1
        
        rate_limit_patterns = ['quota', 'rate limit', 'rate_limit', 'too many requests', 'resource exhausted']
        if any(p in error_lower for p in rate_limit_patterns) or status_code == 429:
            return 'EXTENDED_COOLING', 1800 # 30 minutes
            
        if status_code >= 500:
            return 'SERVER_ERROR', 300 # 5 minutes

        return 'DEFAULT', self.config.GEMINI_COOLING_PERIOD

    def _select_best_key(self, active_keys: List[APIKeyInfo]) -> APIKeyInfo:
        return random.choice(active_keys)

    async def record_key_performance(self, key: str, response_time: float, success: bool):
        async with self.lock:
            perf_data = self.key_performance[key]
            perf_data["response_times"].append(response_time)
            if not success:
                perf_data["errors"] += 1

    async def mark_key_success(self, key: str):
        async with self.lock:
            key_info = self.keys.get(key)
            if not key_info: return
            key_info.failure_count = 0
            key_info.last_success_time = time.time()
            key_info.successful_requests += 1
            key_info.total_requests += 1

    async def get_stats(self) -> Dict[str, Any]:
        async with self.lock:
            #... (rest of the class is fine)
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

    def _safe_validate_and_convert_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        FIXED: Validates and converts messages to a LiteLLM/OpenAI-compatible format
        without destroying tool call structures.
        """
        converted_messages = []
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict) or 'role' not in msg:
                logger.warning(f"Skipping invalid message at index {i}: {msg}")
                continue

            # Standardize role: Gemini's 'model' -> 'assistant'
            role = msg['role']
            if role == 'model':
                role = 'assistant'
            
            # If the message is a tool result from the user, it should be a 'tool' role message
            if role == 'user' and 'content' in msg and isinstance(msg['content'], list):
                 if any(item.get('type') == 'tool_result' for item in msg.get('content', [])):
                     # This will be handled by the AnthropicToGeminiConverter, but we check here too
                     pass # Let the converter handle this complex case

            # If message is from assistant and contains tool_calls, it's a tool request
            if role == 'assistant' and 'tool_calls' in msg and msg['tool_calls']:
                converted_messages.append(msg) # Pass tool calls through directly
                continue
            
            # For regular text messages, ensure 'content' is a string
            content = msg.get('content')
            final_content = ""
            if isinstance(content, str):
                final_content = content
            elif isinstance(content, list):
                # Handle multimodal or mixed content by extracting text
                text_parts = [part['text'] for part in content if isinstance(part, dict) and part.get('type') == 'text']
                final_content = "\n\n".join(text_parts)
            
            # Only add message if it has a valid role and some content, or is a tool message
            if final_content or 'tool_calls' in msg or 'tool_call_id' in msg:
                new_msg = {'role': role, 'content': final_content}
                if 'tool_calls' in msg: new_msg['tool_calls'] = msg['tool_calls']
                if 'tool_call_id' in msg: new_msg['tool_call_id'] = msg['tool_call_id']
                converted_messages.append(new_msg)
            else:
                 logger.warning(f"Message {i} resulted in empty content and was skipped: {msg}")

        return converted_messages

    async def anthropic_messages_completion(self, request: MessagesRequest) -> Union[MessagesResponse, AsyncGenerator[str, None]]:
        try:
            logger.info(f"Original Anthropic request - Model: {request.model}, Messages: {len(request.messages)}")
            
            gemini_request_dict = self.anthropic_to_gemini.convert_request(request)
            
            logger.info(f"Converted Gemini request - Model: {gemini_request_dict.get('model')}, Messages: {len(gemini_request_dict.get('messages', []))}")
            if not gemini_request_dict.get("messages"):
                raise HTTPException(status_code=400, detail="Messages list empty after conversion")

            # The converted messages are already in OpenAI format, so we create the ChatRequest
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
                # The GeminiToAnthropic converter now handles streaming responses
                return self.gemini_to_anthropic.convert_stream_response(gemini_stream, request)
            else:
                gemini_response_model = await self.chat_completion_with_litellm(chat_request)
                gemini_response_dict = gemini_response_model.model_dump()
                return self.gemini_to_anthropic.convert_response(gemini_response_dict, request)

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in anthropic_messages_completion: {repr(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    async def chat_completion_with_litellm(self, chat_request: ChatRequest) -> Any:
        key_info = await self.key_manager.get_available_key()
        if not key_info:
            raise HTTPException(status_code=503, detail="No available API keys")
        
        # Prepare kwargs for LiteLLM from the Pydantic model
        litellm_kwargs = chat_request.model_dump(exclude_none=True)
        litellm_kwargs["api_key"] = key_info.key
        
        # LiteLLM expects the provider prefix in the model name
        litellm_kwargs["model"] = f"gemini/{chat_request.model}"

        logger.info(f"LiteLLM call - Model: {litellm_kwargs['model']}, Key: {key_info.key[:8]}...")
        logger.debug(f"LiteLLM kwargs: {json.dumps({k:v for k,v in litellm_kwargs.items() if k != 'api_key' and k != 'messages'}, indent=2)}")

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
            logger.error(f"LiteLLM call failed with key {key_info.key[:8]}: {e}", exc_info=True)
            raise HTTPException(status_code=502, detail=f"Model provider error: {e}")

key_manager: Optional[GeminiKeyManager] = None
adapter: Optional[LiteLLMAdapter] = None
rate_limiter = None # Simplified

@asynccontextmanager
async def lifespan(app: FastAPI):
    global key_manager, adapter, security_config, api_config, rate_limiter
    
    os.makedirs("logs", exist_ok=True)
    logger.add("logs/gemini_adapter_{time}.log", rotation="1 day", retention="7 days", level="INFO", enqueue=True, catch=True)
    try:
        app_config = get_config()
        security_config = SecurityConfig(app_config)
        
        working_dir = os.getenv("CLAUDE_CODE_WORKING_DIR", ".")
        # We pass the simulator to the config, so it's created once
        api_config = AnthropicAPIConfig(working_directory=working_dir)
        
        logger.info(f"Claude Code support enabled. Working directory: {api_config.working_directory}")
        
        key_manager = GeminiKeyManager(app_config)
        adapter = LiteLLMAdapter(app_config, key_manager, api_config)
        
        health_task = asyncio.create_task(key_manager._check_and_recover_keys())
        logger.info("Gemini Claude Adapter started successfully.")
        yield
    except Exception as e:
        logger.critical(f"Failed to start application: {e}", exc_info=True)
        raise
    finally:
        if 'health_task' in locals() and not health_task.done():
            health_task.cancel()
        logger.info("Gemini Claude Adapter shutting down.")

app = FastAPI(
    title="Gemini Claude Adapter (Fixed)",
    description="Adapter with Anthropic API compatibility and corrected Claude Code support.",
    version="2.2.0-fixed",
    lifespan=lifespan
)

@app.post("/v1/messages")
async def create_message(request: MessagesRequest, raw_request: Request):
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
        logger.error(f"Error in /v1/messages: {repr(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

# --- Simplified other endpoints for brevity ---
@app.get("/", include_in_schema=False)
async def root():
    return {"name": "Gemini Claude Adapter", "version": "2.2.0-fixed", "status": "running"}

@app.get("/health")
async def health_check():
    if not key_manager:
        raise HTTPException(status_code=503, detail="Service not initialized")
    stats = await key_manager.get_stats()
    is_healthy = stats["active_keys"] > 0
    return JSONResponse(status_code=200 if is_healthy else 503, content={"status": "healthy" if is_healthy else "degraded", **stats})

if __name__ == "__main__":
    import uvicorn
    # Make sure to set environment variables for GEMINI_API_KEYS
    # e.g., export GEMINI_API_KEYS="your_key_here"
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info", reload=True)