"""
Performance optimization utilities for Gemini Claude Adapter
"""

import time
import json
import hashlib
import asyncio
from typing import Dict, Any, Optional
from cachetools import TTLCache
import logging
from contextlib import asynccontextmanager
from collections import deque, defaultdict

from .config import AppConfig

logger = logging.getLogger(__name__)

class ResponseCache:
    """Intelligent response caching system with two-tier strategy"""

    def __init__(self, config: AppConfig):
        self.enabled = config.CACHE_ENABLED
        self.max_size = config.CACHE_MAX_SIZE
        self.ttl = config.CACHE_TTL
        self.key_prefix = config.CACHE_KEY_PREFIX
        
        self.cache = TTLCache(maxsize=self.max_size, ttl=self.ttl)
        self.frequent_cache = TTLCache(maxsize=self.max_size // 4, ttl=self.ttl * 2)
        self.hit_count = 0
        self.miss_count = 0
        self.lock = asyncio.Lock()
        self._bloom_filter = set()
    
    def _generate_cache_key(self, request_dict: Dict[str, Any]) -> str:
        """Generate a consistent cache key from request data"""
        cache_data = {
            "model": request_dict.get("model", ""),
            "messages": request_dict.get("messages", []),
            "temperature": request_dict.get("temperature", 1.0),
            "max_tokens": request_dict.get("max_tokens", None),
            "tools": request_dict.get("tools", None)
        }
        
        json_str = json.dumps(cache_data, sort_keys=True)
        hash_key = hashlib.md5(json_str.encode()).hexdigest()
        
        return f"{self.key_prefix}:{hash_key}"
    
    def _should_cache(self, request_dict: Dict[str, Any]) -> bool:
        """Intelligent caching strategy: determine if request should be cached"""
        if request_dict.get("stream", False):
            return False
        if request_dict.get("tools"):
            return False
        if request_dict.get("temperature", 0.7) > 1.5:
            return False
        return True
    
    async def get(self, request_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached response for request with two-tier strategy"""
        if not self.enabled or not self._should_cache(request_dict):
            return None
            
        async with self.lock:
            cache_key = self._generate_cache_key(request_dict)
            
            cached_data = self.frequent_cache.get(cache_key)
            if cached_data:
                self.hit_count += 1
                logger.debug(f"Frequent cache hit for key: {cache_key[:16]}...")
                return cached_data
                
            cached_data = self.cache.get(cache_key)
            if cached_data:
                self.hit_count += 1
                self.frequent_cache[cache_key] = cached_data
                logger.debug(f"Cache hit for key: {cache_key[:16]}...")
                return cached_data
                
            self.miss_count += 1
            logger.debug(f"Cache miss for key: {cache_key[:16]}...")
            return None
    
    async def set(self, request_dict: Dict[str, Any], response_data: Dict[str, Any]) -> None:
        """Cache response for request with intelligent strategy"""
        if not self.enabled or not self._should_cache(request_dict):
            return
        
        async with self.lock:
            cache_key = self._generate_cache_key(request_dict)
            self.cache[cache_key] = response_data
            logger.debug(f"Cached response for key: {cache_key[:16]}...")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get enhanced cache statistics"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "enabled": self.enabled,
            "total_requests": total_requests,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": round(hit_rate, 2),
            "cache_size": len(self.cache),
            "frequent_cache_size": len(self.frequent_cache),
            "max_size": self.max_size,
            "ttl": self.ttl,
            "frequent_cache_ttl": self.ttl * 2,
            "bloom_filter_size": len(self._bloom_filter)
        }
    
    def clear(self) -> None:
        """Clear all cached responses"""
        self.cache.clear()
        self.frequent_cache.clear()
        self.hit_count = 0
        self.miss_count = 0
        logger.info("Cache cleared")

class PerformanceMonitor:
    """Performance monitoring for the adapter"""
    
    def __init__(self):
        self.request_times = deque(maxlen=1000)
        self.request_counts = defaultdict(int)
        self.error_counts = defaultdict(int)
        self.lock = asyncio.Lock()
    
    async def record_request(self, endpoint: str, duration: float, success: bool = True):
        """Record a request for performance monitoring"""
        async with self.lock:
            self.request_times.append(duration)
            self.request_counts[endpoint] += 1
            
            if not success:
                self.error_counts[endpoint] += 1
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.request_times:
            return {
                "total_requests": 0,
                "avg_response_time": 0,
                "p95_response_time": 0,
                "p99_response_time": 0,
                "endpoint_stats": {}
            }
        
        sorted_times = sorted(self.request_times)
        total_requests = len(sorted_times)
        
        p95_index = int(total_requests * 0.95)
        p99_index = int(total_requests * 0.99)
        
        p95_time = sorted_times[p95_index] if p95_index < total_requests else sorted_times[-1]
        p99_time = sorted_times[p99_index] if p99_index < total_requests else sorted_times[-1]
        
        endpoint_stats = {}
        for endpoint in self.request_counts:
            endpoint_stats[endpoint] = {
                "request_count": self.request_counts[endpoint],
                "error_count": self.error_counts[endpoint],
                "error_rate": (self.error_counts[endpoint] / self.request_counts[endpoint] * 100) if self.request_counts[endpoint] > 0 else 0
            }
        
        return {
            "total_requests": total_requests,
            "avg_response_time": round(sum(sorted_times) / total_requests, 3),
            "p95_response_time": round(p95_time, 3),
            "p99_response_time": round(p99_time, 3),
            "endpoint_stats": endpoint_stats
        }

class BatchProcessor:
    """Batch request processor"""
    def __init__(self, max_batch_size: int = 10, batch_timeout: float = 0.1):
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout
        self.pending_requests = []
        self.batch_event = asyncio.Event()
        self.lock = asyncio.Lock()
        
    async def add_request(self, request_data: Dict[str, Any]) -> Any:
        """Add a request to the batch processing queue"""
        future = asyncio.Future()
        
        async with self.lock:
            self.pending_requests.append((request_data, future))
            
            if len(self.pending_requests) >= self.max_batch_size:
                self.batch_event.set()
        
        return await future
    
    async def process_batch(self):
        """Process batch requests"""
        while True:
            try:
                await asyncio.wait_for(self.batch_event.wait(), timeout=self.batch_timeout)
                
                async with self.lock:
                    if not self.pending_requests:
                        self.batch_event.clear()
                        continue
                        
                    current_batch = self.pending_requests[:self.max_batch_size]
                    self.pending_requests = self.pending_requests[self.max_batch_size:]
                    
                    if not self.pending_requests:
                        self.batch_event.clear()
                
                tasks = []
                for request_data, future in current_batch:
                    task = asyncio.create_task(self._process_single_request(request_data, future))
                    tasks.append(task)
                
                await asyncio.gather(*tasks, return_exceptions=True)
                
            except asyncio.TimeoutError:
                async with self.lock:
                    if self.pending_requests:
                        current_batch = self.pending_requests[:]
                        self.pending_requests = []
                        
                        tasks = []
                        for request_data, future in current_batch:
                            task = asyncio.create_task(self._process_single_request(request_data, future))
                            tasks.append(task)
                        
                        await asyncio.gather(*tasks, return_exceptions=True)
                
                self.batch_event.clear()
                
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                self.batch_event.clear()
    
    async def _process_single_request(self, request_data: Dict[str, Any], future: asyncio.Future):
        """Process a single request"""
        try:
            result = await self._handle_request(request_data)
            future.set_result(result)
        except Exception as e:
            future.set_exception(e)
    
    async def _handle_request(self, request_data: Dict[str, Any]) -> Any:
        """Actual request processing logic"""
        pass

# Initialize these variables as None at the module level.
response_cache: Optional[ResponseCache] = None
performance_monitor: PerformanceMonitor = PerformanceMonitor()
batch_processor: BatchProcessor = BatchProcessor()

def initialize_performance_modules(config: AppConfig):
    """
    Initializes performance-related modules after config is loaded.
    """
    global response_cache
    response_cache = ResponseCache(config)
    logger.info("Performance modules initialized.")

@asynccontextmanager
async def monitor_performance(endpoint: str):
    """Context manager for monitoring performance"""
    start_time = time.time()
    success = True
    
    try:
        yield
    except Exception as e:
        success = False
        raise e
    finally:
        duration = time.time() - start_time
        await performance_monitor.record_request(endpoint, duration, success)

def get_performance_stats() -> Dict[str, Any]:
    """Get comprehensive performance statistics"""
    return {
        "cache_stats": response_cache.get_stats(),
        "performance_stats": performance_monitor.get_performance_stats()
    }
