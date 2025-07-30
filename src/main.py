import asyncio
import time
import random
from typing import List, Dict, Optional, Any, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import httpx
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, Depends
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from loguru import logger
import litellm
from contextlib import asynccontextmanager
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from starlette.status import HTTP_401_UNAUTHORIZED, HTTP_403_FORBIDDEN

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
        self.current_key_index = 0
        self.lock = asyncio.Lock()

        for key in config.api_keys:
            if key and key.strip():
                self.keys[key] = APIKeyInfo(key=key)

        if not self.keys:
            raise ValueError("No valid API keys provided to key manager")

        logger.info(f"Initialized {len(self.keys)} API keys.")

    async def get_available_key(self) -> Optional[APIKeyInfo]:
        async with self.lock:
            current_time = time.time()
            active_keys = []

            # Check for keys that can be reactivated
            for key_info in self.keys.values():
                if (key_info.status == KeyStatus.COOLING and 
                    key_info.cooling_until and 
                    current_time > key_info.cooling_until):
                    key_info.status = KeyStatus.ACTIVE
                    key_info.failure_count = 0
                    key_info.cooling_until = None
                    logger.info(f"API key {key_info.key[:8]}... has cooled down and is now active.")
                
                if key_info.status == KeyStatus.ACTIVE:
                    active_keys.append(key_info)
            
            if not active_keys:
                logger.warning("No available API keys.")
                return None

            # Simple round-robin with bounds checking
            if self.current_key_index >= len(active_keys):
                self.current_key_index = 0
            
            key_info = active_keys[self.current_key_index]
            self.current_key_index = (self.current_key_index + 1) % len(active_keys)
            return key_info

    async def mark_key_failed(self, key: str, error: str):
        async with self.lock:
            key_info = self.keys.get(key)
            if not key_info:
                logger.warning(f"Attempted to mark unknown key as failed: {key[:8]}...")
                return

            # Immediately put key in cooling on first failure
            key_info.status = KeyStatus.COOLING
            key_info.failure_count += 1
            key_info.last_failure_time = time.time()
            key_info.cooling_until = time.time() + self.config.cooling_period
            logger.warning(f"API key {key[:8]}... failed and is now cooling for {self.config.cooling_period} seconds. Error: {error}")

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
            
            logger.info(f"Reset API key {matched_key[:8]}... from {old_status.value} to active status")
            return {"message": f"Successfully reset key {matched_key[:8]}... from {old_status.value} to active"}

class LiteLLMAdapter:
    def __init__(self, config: GeminiConfig, key_manager: GeminiKeyManager):
        self.config = config
        self.key_manager = key_manager
        
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
        # Try each available key once, no retries for the same key
        for attempt in range(len(self.key_manager.keys) + 1):
            key_info = await self.key_manager.get_available_key()
            if not key_info:
                raise HTTPException(
                    status_code=503, 
                    detail="All API keys are currently unavailable. Please try again later."
                )
            
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
                }
                
                if request.max_tokens:
                    kwargs["max_tokens"] = request.max_tokens
                
                logger.info(f"Attempting request with key {key_info.key[:8]}...")
                
                response = await litellm.acompletion(**kwargs)
                
                await self.key_manager.mark_key_success(key_info.key)
                logger.info(f"Request successful with key {key_info.key[:8]}...")
                return response

            except Exception as e:
                last_error = str(e)
                await self.key_manager.mark_key_failed(key_info.key, last_error)
                logger.warning(f"Key {key_info.key[:8]}... failed, trying next key...")
        
        # All keys exhausted
        raise HTTPException(
            status_code=502, 
            detail=f"Failed to process request with all available keys. Last error: {last_error}"
        )

# Global state
key_manager: Optional[GeminiKeyManager] = None
adapter: Optional[LiteLLMAdapter] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global key_manager, adapter
    
    try:
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
        logger.info("Gemini Claude Adapter shutting down.")

app = FastAPI(
    title="Gemini Claude Code Adapter",
    description="An adapter for Claude Code to use Gemini API with key rotation and fault tolerance.",
    version="1.3.0",
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
            "models": "/v1/models",
            "health": "/health",
            "stats": "/stats",
            "admin": "/admin/reset-key/{key_prefix}"
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
            "service_version": "1.3.0",
            "security_enabled": security_config.security_enabled,
            **stats
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

# Protected endpoints (require authentication)
@app.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
async def chat_completions(request: ChatRequest, client_key: str = Depends(verify_api_key)):
    if not adapter:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        logger.info(f"Chat completion request from client: {client_key[:8] if client_key != 'insecure_mode' else 'insecure_mode'}...")
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
                "id": "gemini-2.5-pro",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "google"
            },
            {
                "id": "gemini-1.5-pro-latest",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "google"
            },
            {
                "id": "gemini-1.5-flash-latest",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "google"
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

@app.get("/admin/security-status", dependencies=[Depends(verify_admin_key)])
async def get_security_status(admin_key: str = Depends(verify_admin_key)):
    """Get security configuration status - admin only"""
    return {
        "security_enabled": security_config.security_enabled,
        "client_keys_count": len(security_config.valid_api_keys),
        "admin_keys_count": len(security_config.admin_keys),
        "has_admin_keys": bool(security_config.admin_keys),
        "authentication_methods": ["X-API-Key header", "Authorization Bearer token"],
        "admin_endpoints": ["/admin/reset-key/{key_prefix}", "/admin/security-status"]
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("main:app", host=host, port=port, log_level="info")