# src/main.py
import asyncio
import time
import random
import os
import json
from typing import Dict, Optional, Any, Set, Tuple, AsyncGenerator, Union
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from loguru import logger
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

# 本地模块导入
# To (absolute imports from the src package)
from src.anthropic_api import (
    MessagesRequest, MessagesResponse, APIConfig, log_request_beautifully
)
from src.config import load_configuration, get_config
from src.performance import initialize_performance_modules, get_performance_stats, monitor_performance

# 加载环境变量
load_dotenv()

# --- 密钥状态与信息 ---
class KeyStatus(Enum):
    ACTIVE = "active"
    COOLING = "cooling"
    FAILED = "failed"

@dataclass
class APIKeyInfo:
    key: str
    status: KeyStatus = KeyStatus.ACTIVE
    failure_count: int = 0
    cooling_until: Optional[float] = None

# --- 依赖注入与安全 ---
config = get_config()
api_config: Optional[APIConfig] = None
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)
valid_api_keys: Set[str] = set(config.SECURITY_ADAPTER_API_KEYS)
admin_api_keys: Set[str] = set(config.SECURITY_ADMIN_API_KEYS)

async def verify_api_key(api_key: Optional[str] = Depends(api_key_header), bearer: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme)):
    if not valid_api_keys:
        return "insecure_mode"
    key = api_key or (bearer.credentials if bearer else None)
    if key in valid_api_keys:
        return key
    raise HTTPException(status_code=403, detail="Invalid API Key")

async def verify_admin_key(api_key: Optional[str] = Depends(api_key_header), bearer: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme)):
    if not admin_api_keys:
        raise HTTPException(status_code=403, detail="Admin API keys not configured")
    key = api_key or (bearer.credentials if bearer else None)
    if key in admin_api_keys:
        return key
    raise HTTPException(status_code=403, detail="Invalid Admin API Key")

# --- Gemini 密钥管理器 ---
class GeminiKeyManager:
    def __init__(self):
        self.keys: Dict[str, APIKeyInfo] = {
            key: APIKeyInfo(key=key) for key in config.GEMINI_API_KEYS if key
        }
        self.lock = asyncio.Lock()
        self.last_used_key_index = -1
        if not self.keys:
            raise ValueError("No valid GEMINI_API_KEYS provided.")
        logger.info(f"Initialized {len(self.keys)} Gemini API keys.")

    async def get_available_key(self) -> Optional[APIKeyInfo]:
        async with self.lock:
            self._recover_keys()
            active_keys = [k for k in self.keys.values() if k.status == KeyStatus.ACTIVE]
            if not active_keys:
                return None
            # 简单轮询选择下一个密钥
            self.last_used_key_index = (self.last_used_key_index + 1) % len(active_keys)
            return active_keys[self.last_used_key_index]

    def _recover_keys(self):
        now = time.time()
        for key_info in self.keys.values():
            if key_info.status == KeyStatus.COOLING and now > (key_info.cooling_until or 0):
                key_info.status = KeyStatus.ACTIVE
                key_info.failure_count = 0
                logger.info(f"Key {key_info.key[:8]}... recovered to ACTIVE.")

    async def mark_key_failed(self, key: str, error: Exception):
        async with self.lock:
            if key not in self.keys:
                return
            key_info = self.keys[key]
            is_permanent = isinstance(error, (google_exceptions.PermissionDenied, google_exceptions.Unauthenticated))
            key_info.status = KeyStatus.FAILED if is_permanent else KeyStatus.COOLING
            key_info.cooling_until = time.time() + config.GEMINI_COOLING_PERIOD
            status_msg = "permanently FAILED" if is_permanent else f"COOLING for {config.GEMINI_COOLING_PERIOD}s"
            logger.warning(f"Key {key_info.key[:8]}... marked as {status_msg}. Reason: {type(error).__name__}")
    
    async def get_stats(self) -> Dict[str, int]:
        async with self.lock:
            return {
                "total": len(self.keys),
                "active": sum(1 for k in self.keys.values() if k.status == KeyStatus.ACTIVE),
                "cooling": sum(1 for k in self.keys.values() if k.status == KeyStatus.COOLING),
                "failed": sum(1 for k in self.keys.values() if k.status == KeyStatus.FAILED),
            }

key_manager: Optional[GeminiKeyManager] = None

# --- 原生 Gemini 适配器 ---
class NativeGeminiAdapter:
    def __init__(self, key_mgr: GeminiKeyManager, api_cfg: APIConfig):
        self.key_manager = key_mgr
        self.api_config = api_cfg

    async def process_completion(self, request: MessagesRequest) -> Union[MessagesResponse, AsyncGenerator[str, None]]:
        last_error = None
        max_attempts = min(config.GEMINI_MAX_RETRIES + 1, len(self.key_manager.keys))
        
        for attempt in range(max_attempts):
            key_info = await self.key_manager.get_available_key()
            if not key_info:
                logger.warning("No available API keys. Waiting for recovery.")
                await asyncio.sleep(5)
                continue

            logger.info(f"Attempt {attempt + 1}/{max_attempts} using key {key_info.key[:8]}...")
            try:
                # --- MODIFIED SECTION START ---
                # 处理 system prompt，将其转换为 Gemini API 需要的字符串格式
                system_prompt_str = ""
                if isinstance(request.system, str):
                    system_prompt_str = request.system
                elif isinstance(request.system, list):
                    # 从字典列表中提取文本并连接
                    system_prompt_str = "\n".join(
                        item.get("text", "") for item in request.system if item.get("type") == "text"
                    )
                
                genai.configure(api_key=key_info.key)
                model = genai.GenerativeModel(
                    model_name=self.api_config.anthropic_to_gemini.convert_model(request.model),
                    system_instruction=system_prompt_str  # 使用处理后的字符串
                )
                # --- MODIFIED SECTION END ---
                
                generation_config = genai.types.GenerationConfig(
                    max_output_tokens=request.max_tokens,
                    temperature=request.temperature
                )
                messages = self.api_config.anthropic_to_gemini.convert_messages(request.messages)
                
                if request.stream:
                    stream = await model.generate_content_async(
                        messages, stream=True, generation_config=generation_config,
                        request_options={'timeout': config.GEMINI_REQUEST_TIMEOUT}
                    )
                    return self.api_config.gemini_to_anthropic.convert_stream_response(stream, request)
                else:
                    response = await model.generate_content_async(
                        messages, generation_config=generation_config,
                        request_options={'timeout': config.GEMINI_REQUEST_TIMEOUT}
                    )
                    return self.api_config.gemini_to_anthropic.convert_response(response, request)
            
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt+1} failed with key {key_info.key[:8]}. Error: {repr(e)}")
                await self.key_manager.mark_key_failed(key_info.key, e)
        
        detail = f"All {max_attempts} attempts failed. Last error: {repr(last_error)}"
        logger.error(detail)
        raise HTTPException(status_code=502, detail=detail)

adapter: Optional[NativeGeminiAdapter] = None

# --- FastAPI 应用生命周期 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global key_manager, adapter, api_config
    os.makedirs("logs", exist_ok=True)
    logger.add("logs/adapter_{time}.log", rotation="1 day", retention="7 days", level="INFO")
    
    try:
        # 初始化性能模块
        initialize_performance_modules(
            cache_enabled=config.CACHE_ENABLED,
            cache_max_size=config.CACHE_MAX_SIZE,
            cache_ttl=config.CACHE_TTL,
            cache_key_prefix=config.CACHE_KEY_PREFIX
        )
        
        api_config = APIConfig()
        key_manager = GeminiKeyManager()
        adapter = NativeGeminiAdapter(key_manager, api_config)
        logger.info("Native Gemini Adapter started successfully.")
        yield
    except Exception as e:
        logger.critical(f"Application failed to start: {e}", exc_info=True)
        raise
    finally:
        logger.info("Native Gemini Adapter shutting down.")

# --- FastAPI 应用实例和路由 ---
app = FastAPI(
    title="Native Gemini-Anthropic Adapter",
    description="Directly bridges Anthropic API requests to Google's Gemini Pro using the native Python SDK.",
    version="3.0.0",
    lifespan=lifespan
)

@app.post("/v1/messages", response_model=None)
async def create_message(request: MessagesRequest, raw_request: Request, client_key: str = Depends(verify_api_key)):
    if not adapter:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    async with monitor_performance("create_message"):
        log_request_beautifully(
            "POST", str(raw_request.url.path), request.model,
            adapter.api_config.anthropic_to_gemini.convert_model(request.model),
            len(request.messages), len(request.tools or [])
        )
        
        response = await adapter.process_completion(request)
        if request.stream:
            return StreamingResponse(response, media_type="text/event-stream")
        return response

@app.get("/v1/models")
async def list_models(client_key: str = Depends(verify_api_key)):
    """List available models"""
    return {
        "object": "list",
        "data": [
            {"id": "claude-3-5-sonnet", "object": "model", "created": 1677649963, "owned_by": "anthropic"},
            {"id": "claude-3-5-haiku", "object": "model", "created": 1677649963, "owned_by": "anthropic"},
            {"id": "claude-3-opus", "object": "model", "created": 1677649963, "owned_by": "anthropic"}
        ]
    }

@app.get("/health")
async def health_check():
    if not key_manager:
        raise HTTPException(status_code=503, detail="Key Manager not initialized")
    stats = await key_manager.get_stats()
    is_healthy = stats["active"] > 0
    status_code = 200 if is_healthy else 503
    return JSONResponse(content={"status": "healthy" if is_healthy else "degraded", **stats}, status_code=status_code)

@app.get("/stats")
async def get_stats(client_key: str = Depends(verify_api_key)):
    """Get key usage statistics"""
    if not key_manager:
        raise HTTPException(status_code=503, detail="Key Manager not initialized")
    
    key_stats = await key_manager.get_stats()
    perf_stats = get_performance_stats()
    
    return {
        "key_stats": key_stats,
        **perf_stats
    }

@app.get("/metrics")
async def get_metrics(client_key: str = Depends(verify_api_key)):
    """Get detailed performance metrics"""
    return get_performance_stats()

@app.post("/admin/reset-key/{key_prefix}")
async def reset_key(key_prefix: str, admin_key: str = Depends(verify_admin_key)):
    """Reset a failed Gemini key"""
    if not key_manager:
        raise HTTPException(status_code=503, detail="Key Manager not initialized")
    
    async with key_manager.lock:
        for key_info in key_manager.keys.values():
            if key_info.key.startswith(key_prefix):
                key_info.status = KeyStatus.ACTIVE
                key_info.failure_count = 0
                key_info.cooling_until = None
                logger.info(f"Admin reset key {key_info.key[:8]}... to ACTIVE")
                return {"message": f"Key {key_prefix}... reset successfully"}
    
    raise HTTPException(status_code=404, detail="Key not found")

@app.get("/")
async def root():
    """Service information"""
    return {
        "service": "Gemini Claude Adapter",
        "version": "3.0.0",
        "status": "running",
        "endpoints": {
            "messages": "/v1/messages",
            "models": "/v1/models",
            "health": "/health",
            "stats": "/stats"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, workers=1)