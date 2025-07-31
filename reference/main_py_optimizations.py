# 在 main.py 的 lifespan 函数开始处添加异步健康检查任务
async def health_check_task():
    """定期检查API keys的健康状态"""
    while True:
        if key_manager:
            try:
                # 每60秒检查一次冷却的keys是否可以恢复
                stats = await key_manager.get_stats()
                cooling_keys = stats.get("cooling_keys", 0)
                if cooling_keys > 0:
                    logger.info(f"Health check: {cooling_keys} keys are still cooling")
            except Exception as e:
                logger.error(f"Health check error: {e}")
        await asyncio.sleep(60)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global key_manager, adapter
    
    try:
        # 原有的初始化代码...
        
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

# 在 GeminiKeyManager 类中优化 get_available_key 方法
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
            current_time = time.time()
            active_keys = []

            # 检查冷却中的keys是否可以重新激活
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

    async def mark_key_failed(self, key: str, error: str):
        async with self.lock:
            key_info = self.keys.get(key)
            if not key_info:
                logger.warning(f"Attempted to mark unknown key as failed: {key[:8]}...")
                return

            # 根据错误类型决定冷却时间
            cooling_time = self.config.cooling_period
            
            # 特定错误类型使用更长的冷却时间
            error_lower = error.lower()
            if any(err in error_lower for err in ['quota', 'rate limit', 'billing', 'payment']):
                cooling_time = self.config.cooling_period * 2  # 配额相关错误冷却时间翻倍
                logger.warning(f"Quota/billing error detected for key {key[:8]}..., extending cooling period to {cooling_time}s")
            elif 'invalid' in error_lower or 'unauthorized' in error_lower:
                key_info.status = KeyStatus.FAILED  # 永久失效
                logger.error(f"API key {key[:8]}... permanently failed due to authentication error: {error}")
                return

            key_info.status = KeyStatus.COOLING
            key_info.failure_count += 1
            key_info.last_failure_time = time.time()
            key_info.cooling_until = time.time() + cooling_time
            logger.warning(f"API key {key[:8]}... failed and is now cooling for {cooling_time} seconds. Error: {error}")

# 在 LiteLLMAdapter 类中优化错误处理
class LiteLLMAdapter:
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

# 添加请求限流中间件
import time
from collections import defaultdict, deque

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

# 修改主要endpoint以包含限流
@app.post("/v1/messages", dependencies=[Depends(check_rate_limit)])
async def create_message(
    request: MessagesRequest,
    raw_request: Request,
    client_key: str = Depends(check_rate_limit)
):
    # 原有的处理逻辑...

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