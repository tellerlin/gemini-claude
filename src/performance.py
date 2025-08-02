"""
Performance optimization utilities for Gemini Claude Adapter
"""

import time
import json
import hashlib
import asyncio
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import httpx
from cachetools import TTLCache
import logging
from contextlib import asynccontextmanager
from collections import deque, defaultdict

# +++ Import the centralized config classes and loader +++
from .config import get_config, CacheConfig, PerformanceConfig

logger = logging.getLogger(__name__)

class ResponseCache:
    """Intelligent response caching system with two-tier strategy"""
    
    # +++ Use the imported CacheConfig for type hinting +++
    def __init__(self, config: CacheConfig):
        self.config = config
        # Two-tier caching strategy
        self.cache = TTLCache(maxsize=config.max_size, ttl=config.ttl)
        self.frequent_cache = TTLCache(maxsize=config.max_size // 4, ttl=config.ttl * 2)  # High-frequency cache
        self.hit_count = 0
        self.miss_count = 0
        self.lock = asyncio.Lock()
        self._bloom_filter = set()  # Simple bloom filter for cache avoidance
    
    def _generate_cache_key(self, request_dict: Dict[str, Any]) -> str:
        """Generate a consistent cache key from request data"""
        # Create a deterministic string representation
        cache_data = {
            "model": request_dict.get("model", ""),
            "messages": request_dict.get("messages", []),
            "temperature": request_dict.get("temperature", 1.0),
            "max_tokens": request_dict.get("max_tokens", None),
            "tools": request_dict.get("tools", None)
        }
        
        # Convert to JSON string and hash
        json_str = json.dumps(cache_data, sort_keys=True)
        hash_key = hashlib.md5(json_str.encode()).hexdigest()
        
        return f"{self.config.key_prefix}:{hash_key}"
    
    def _should_cache(self, request_dict: Dict[str, Any]) -> bool:
        """Intelligent caching strategy: determine if request should be cached"""
        # Don't cache streaming requests
        if request_dict.get("stream", False):
            return False
            
        # Don't cache requests with tool calls
        if request_dict.get("tools"):
            return False
            
        # Don't cache high-temperature requests (too creative)
        if request_dict.get("temperature", 0.7) > 1.5:
            return False
            
        return True
    
    async def get(self, request_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached response for request with two-tier strategy"""
        if not self.config.enabled or not self._should_cache(request_dict):
            return None
            
        async with self.lock:
            cache_key = self._generate_cache_key(request_dict)
            
            # Check frequent cache first
            cached_data = self.frequent_cache.get(cache_key)
            if cached_data:
                self.hit_count += 1
                logger.debug(f"Frequent cache hit for key: {cache_key[:16]}...")
                return cached_data
                
            # Check regular cache
            cached_data = self.cache.get(cache_key)
            if cached_data:
                self.hit_count += 1
                # Move to frequent cache on hit
                self.frequent_cache[cache_key] = cached_data
                logger.debug(f"Cache hit for key: {cache_key[:16]}...")
                return cached_data
                
            self.miss_count += 1
            logger.debug(f"Cache miss for key: {cache_key[:16]}...")
            return None
    
    async def set(self, request_dict: Dict[str, Any], response_data: Dict[str, Any]) -> None:
        """Cache response for request with intelligent strategy"""
        if not self.config.enabled or not self._should_cache(request_dict):
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
            "enabled": self.config.enabled,
            "total_requests": total_requests,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": round(hit_rate, 2),
            "cache_size": len(self.cache),
            "frequent_cache_size": len(self.frequent_cache),
            "max_size": self.config.max_size,
            "ttl": self.config.ttl,
            "frequent_cache_ttl": self.config.ttl * 2,
            "bloom_filter_size": len(self._bloom_filter)
        }
    
    def clear(self) -> None:
        """Clear all cached responses"""
        self.cache.clear()
        # CORRECTED: Clear the frequent_cache as well
        self.frequent_cache.clear()
        self.hit_count = 0
        self.miss_count = 0
        logger.info("Cache cleared")

class OptimizedHTTPClient:
    """Optimized HTTP client with enhanced connection pooling and performance monitoring"""
    
    # +++ Use the imported PerformanceConfig for type hinting +++
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.client: Optional[httpx.AsyncClient] = None
        self.request_count = 0
        self.error_count = 0
        self.total_request_time = 0.0
        self.lock = asyncio.Lock()
        self._session_pool = {}  # Session pool for advanced connection management
        
    async def initialize(self):
        """Initialize the HTTP client with enhanced connection pooling"""
        async with self.lock:
            if self.client is None:
                # Enhanced connection limits
                limits = httpx.Limits(
                    max_keepalive_connections=self.config.max_keepalive_connections,
                    max_connections=self.config.max_connections,
                    keepalive_expiry=self.config.keepalive_expiry
                )
                
                # Optimized timeout settings
                timeout = httpx.Timeout(
                    connect=self.config.connect_timeout,
                    read=self.config.read_timeout,
                    write=self.config.write_timeout,
                    pool=self.config.pool_timeout
                )
                
                self.client = httpx.AsyncClient(
                    limits=limits,
                    timeout=timeout,
                    # +++ Use the correct field name from PerformanceConfig +++
                    http2=self.config.http2_enabled,
                    trust_env=self.config.trust_env,
                    verify=self.config.verify_ssl
                )
                logger.info("Optimized HTTP client initialized with enhanced connection pooling")
    
    async def close(self):
        """Close the HTTP client"""
        async with self.lock:
            if self.client:
                await self.client.aclose()
                self.client = None
                logger.info("HTTP client closed")
    
    async def request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Make an HTTP request with performance monitoring"""
        # CORRECTED: Ensure client is initialized before use for robustness
        if not self.client:
            await self.initialize()
        
        start_time = time.time()
        self.request_count += 1
        
        try:
            # +++ This ensures the client is not None before making a request +++
            response = await self.client.request(method, url, **kwargs)
            request_time = time.time() - start_time
            self.total_request_time += request_time
            
            logger.debug(f"HTTP {method} {url} - {response.status_code} - {request_time:.3f}s")
            return response
            
        except Exception as e:
            self.error_count += 1
            request_time = time.time() - start_time
            logger.error(f"HTTP {method} {url} - Error: {e} - {request_time:.3f}s")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get HTTP client statistics"""
        avg_request_time = (self.total_request_time / self.request_count) if self.request_count > 0 else 0
        error_rate = (self.error_count / self.request_count * 100) if self.request_count > 0 else 0
        
        return {
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": round(error_rate, 2),
            "avg_request_time": round(avg_request_time, 3),
            "total_request_time": round(self.total_request_time, 3),
            "client_initialized": self.client is not None,
            "config": {
                "max_keepalive_connections": self.config.max_keepalive_connections,
                "max_connections": self.config.max_connections,
                # +++ Use the correct field name from PerformanceConfig +++
                "http2_enabled": self.config.http2_enabled
            }
        }

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
        
        # Calculate percentiles
        p95_index = int(total_requests * 0.95)
        p99_index = int(total_requests * 0.99)
        
        p95_time = sorted_times[p95_index] if p95_index < total_requests else sorted_times[-1]
        p99_time = sorted_times[p99_index] if p99_index < total_requests else sorted_times[-1]
        
        # Endpoint statistics
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
                # Wait for batch processing conditions
                await asyncio.wait_for(self.batch_event.wait(), timeout=self.batch_timeout)
                
                async with self.lock:
                    if not self.pending_requests:
                        self.batch_event.clear()
                        continue
                        
                    # Get the current batch
                    current_batch = self.pending_requests[:self.max_batch_size]
                    self.pending_requests = self.pending_requests[self.max_batch_size:]
                    
                    if not self.pending_requests:
                        self.batch_event.clear()
                
                # Process requests in the batch concurrently
                tasks = []
                for request_data, future in current_batch:
                    task = asyncio.create_task(self._process_single_request(request_data, future))
                    tasks.append(task)
                
                await asyncio.gather(*tasks, return_exceptions=True)
                
            except asyncio.TimeoutError:
                # Timeout also triggers batch processing
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
            # Call the actual processing logic here
            result = await self._handle_request(request_data)
            future.set_result(result)
        except Exception as e:
            future.set_exception(e)
    
    async def _handle_request(self, request_data: Dict[str, Any]) -> Any:
        """Actual request processing logic"""
        # Implement the specific processing logic here
        pass

# +++ Initialize global instances using the centralized configuration loader +++
app_config = get_config()
response_cache = ResponseCache(app_config.cache)
http_client = OptimizedHTTPClient(app_config.performance)
performance_monitor = PerformanceMonitor()
batch_processor = BatchProcessor()


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

async def get_optimized_http_client() -> httpx.AsyncClient:
    """Get the optimized HTTP client"""
    if not http_client.client:
        await http_client.initialize()
    return http_client.client

def get_performance_stats() -> Dict[str, Any]:
    """Get comprehensive performance statistics"""
    return {
        "cache_stats": response_cache.get_stats(),
        "http_client_stats": http_client.get_stats(),
        "performance_stats": performance_monitor.get_performance_stats()
    }
