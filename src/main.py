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
from anthropic_api import (
    MessagesRequest, MessagesResponse, APIConfig, log_request_beautifully
)

# 加载环境变量
load_dotenv()

# --- 密钥状态与信息 ---
class KeyStatus(Enum):
    ACTIVE = "active"; COOLING = "cooling"; FAILED = "failed"
@dataclass
class APIKeyInfo:
    key: str; status: KeyStatus = KeyStatus.ACTIVE; failure_count: int = 0
    cooling_until: Optional[float] = None

# --- 应用配置 (简化，实际可从 config.py 导入) ---
class AppConfig:
    SECURITY_ADAPTER_API_KEYS = os.getenv("SECURITY_ADAPTER_API_KEYS", "").split(",")
    GEMINI_API_KEYS = os.getenv("GEMINI_API_KEYS", "").split(",")
    GEMINI_COOLING_PERIOD = int(os.getenv("GEMINI_COOLING_PERIOD", "300"))
    GEMINI_REQUEST_TIMEOUT = int(os.getenv("GEMINI_REQUEST_TIMEOUT", "120"))
    GEMINI_HEALTH_CHECK_INTERVAL = int(os.getenv("GEMINI_HEALTH_CHECK_INTERVAL", "60"))
    MAX_RETRIES_PER_REQUEST = int(os.getenv("MAX_RETRIES_PER_REQUEST", "3"))

# --- 依赖注入与安全 ---
app_config = AppConfig()
api_config: Optional[APIConfig] = None
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)
valid_api_keys: Set[str] = {k for k in app_config.SECURITY_ADAPTER_API_KEYS if k}

async def verify_api_key(api_key: Optional[str] = Depends(api_key_header), bearer: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme)):
    if not valid_api_keys: return "insecure_mode"
    key = api_key or (bearer.credentials if bearer else None)
    if key in valid_api_keys: return key
    raise HTTPException(status_code=403, detail="Invalid API Key")

# --- Gemini 密钥管理器 ---
class GeminiKeyManager:
    def __init__(self, config: AppConfig):
        self.config = config
        self.keys: Dict[str, APIKeyInfo] = {key: APIKeyInfo(key=key) for key in config.GEMINI_API_KEYS if key}
        self.lock = asyncio.Lock()
        self.last_used_key_index = -1
        if not self.keys: raise ValueError("No valid GEMINI_API_KEYS provided.")
        logger.info(f"Initialized {len(self.keys)} Gemini API keys.")

    async def get_available_key(self) -> Optional[APIKeyInfo]:
        async with self.lock:
            self._recover_keys()
            active_keys = [k for k in self.keys.values() if k.status == KeyStatus.ACTIVE]
            if not active_keys: return None
            # 简单轮询选择下一个密钥
            self.last_used_key_index = (self.last_used_key_index + 1) % len(active_keys)
            return active_keys[self.last_used_key_index]

    def _recover_keys(self):
        now = time.time()
        for key_info in self.keys.values():
            if key_info.status == KeyStatus.COOLING and now > (key_info.cooling_until or 0):
                key_info.status = KeyStatus.ACTIVE; key_info.failure_count = 0
                logger.info(f"Key {key_info.key[:8]}... recovered to ACTIVE.")

    async def mark_key_failed(self, key: str, error: Exception):
        async with self.lock:
            if key not in self.keys: return
            key_info = self.keys[key]
            is_permanent = isinstance(error, (google_exceptions.PermissionDenied, google_exceptions.Unauthenticated))
            key_info.status = KeyStatus.FAILED if is_permanent else KeyStatus.COOLING
            key_info.cooling_until = time.time() + self.config.GEMINI_COOLING_PERIOD
            status_msg = "permanently FAILED" if is_permanent else f"COOLING for {self.config.GEMINI_COOLING_PERIOD}s"
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
    def __init__(self, config: AppConfig, key_mgr: GeminiKeyManager, api_cfg: APIConfig):
        self.config, self.key_manager, self.api_config = config, key_mgr, api_cfg

    async def process_completion(self, request: MessagesRequest) -> Union[MessagesResponse, AsyncGenerator[str, None]]:
        last_error = None
        max_attempts = min(self.config.MAX_RETRIES_PER_REQUEST, len(self.key_manager.keys))
        
        for attempt in range(max_attempts):
            key_info = await self.key_manager.get_available_key()
            if not key_info:
                logger.warning("No available API keys. Waiting for recovery.")
                await asyncio.sleep(5)
                continue

            logger.info(f"Attempt {attempt + 1}/{max_attempts} using key {key_info.key[:8]}...")
            try:
                model = genai.GenerativeModel(
                    model_name=self.api_config.anthropic_to_gemini.convert_model(request.model),
                    system_instruction=request.system
                )
                generation_config = genai.types.GenerationConfig(
                    max_output_tokens=request.max_tokens,
                    temperature=request.temperature
                )
                messages = self.api_config.anthropic_to_gemini.convert_messages(request.messages)
                
                if request.stream:
                    stream = await model.generate_content_async(
                        messages, stream=True, generation_config=generation_config,
                        request_options={'timeout': self.config.GEMINI_REQUEST_TIMEOUT}
                    )
                    return self.api_config.gemini_to_anthropic.convert_stream_response(stream, request)
                else:
                    response = await model.generate_content_async(
                        messages, generation_config=generation_config,
                        request_options={'timeout': self.config.GEMINI_REQUEST_TIMEOUT}
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
        api_config = APIConfig()
        key_manager = GeminiKeyManager(app_config)
        adapter = NativeGeminiAdapter(app_config, key_manager, api_config)
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
    if not adapter: raise HTTPException(status_code=503, detail="Service not initialized")
    
    log_request_beautifully(
        "POST", str(raw_request.url.path), request.model,
        adapter.api_config.anthropic_to_gemini.convert_model(request.model),
        len(request.messages), len(request.tools or [])
    )
    
    response = await adapter.process_completion(request)
    if request.stream:
        return StreamingResponse(response, media_type="text/event-stream")
    return response

@app.get("/health")
async def health_check(client_key: str = Depends(verify_api_key)):
    if not key_manager: raise HTTPException(status_code=503, detail="Key Manager not initialized")
    stats = await key_manager.get_stats()
    is_healthy = stats["active"] > 0
    status_code = 200 if is_healthy else 503
    return JSONResponse(content={"status": "healthy" if is_healthy else "degraded", **stats}, status_code=status_code)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, workers=1)
