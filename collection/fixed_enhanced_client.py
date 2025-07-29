# enhanced_client.py - Enhanced local client for Gemini Claude Adapter
import os
import asyncio
import httpx
import json
from typing import List, Dict, Optional, AsyncGenerator
import time
from dataclasses import dataclass
from datetime import datetime
import argparse
import sys

@dataclass
class ClientConfig:
    """Client configuration"""
    vps_url: str
    timeout: int = 120
    retries: int = 3
    preferred_model: str = "gemini-1.5-pro"
    default_temperature: float = 0.7
    max_concurrent: int = 10

class RemoteGeminiClient:
    """Enhanced client for connecting to Gemini Claude Adapter on VPS"""
    
    def __init__(self, config: ClientConfig):
        self.config = config
        self.server_url = config.vps_url.rstrip('/')
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(config.timeout),
            limits=httpx.Limits(
                max_connections=config.max_concurrent,
                max_keepalive_connections=5
            )
        )
    
    async def chat_completion(self, 
                            messages: List[Dict[str, str]], 
                            model: Optional[str] = None,
                            temperature: Optional[float] = None,
                            max_tokens: Optional[int] = None,
                            stream: bool = False) -> Dict:
        """Send chat completion request with enhanced error handling"""
        
        url = f"{self.server_url}/v1/chat/completions"
        payload = {
            "messages": messages,
            "model": model or self.config.preferred_model,
            "temperature": temperature or self.config.default_temperature,
            "stream": stream
        }
        
        if max_tokens:
            payload["max_tokens"] = max_tokens
        
        # Enhanced retry mechanism with exponential backoff
        last_error = None
        for attempt in range(self.config.retries + 1):
            try:
                if stream:
                    async for chunk in self._stream_request(url, payload):
                        yield chunk
                    return
                else:
                    response = await self.client.post(url, json=payload)
                    response.raise_for_status()
                    return response.json()
                    
            except httpx.TimeoutException as e:
                last_error = f"Request timeout: {e}"
                if attempt < self.config.retries:
                    wait_time = min(2 ** attempt, 30)  # Max 30 seconds
                    print(f"请求超时，{wait_time}秒后重试... (尝试 {attempt + 1}/{self.config.retries + 1})")
                    await asyncio.sleep(wait_time)
                else:
                    raise Exception(last_error)
                    
            except httpx.ConnectError as e:
                last_error = f"连接错误: {e}"
                # Connection errors usually don't benefit from retries unless it's intermittent
                if attempt < self.config.retries:
                    wait_time = min(2 ** attempt, 15)
                    print(f"连接错误，{wait_time}秒后重试... (尝试 {attempt + 1}/{self.config.retries + 1})")
                    await asyncio.sleep(wait_time)
                else:
                    raise Exception(last_error)
                    
            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code
                if status_code == 503:
                    last_error = "服务暂时不可用 (所有API密钥可能都在冷却中)"
                elif status_code == 502:
                    last_error = "网关错误 (后端服务不可用)"
                elif status_code == 429:
                    last_error = "请求频率过高，请稍后再试"
                elif status_code == 400:
                    last_error = f"请求格式错误: {e.response.text}"
                    break  # Don't retry on client errors
                else:
                    last_error = f"HTTP {status_code}: {e.response.text}"
                
                if attempt < self.config.retries and status_code in [502, 503, 429]:
                    wait_time = min(2 ** attempt, 30)
                    print(f"服务错误，{wait_time}秒后重试... (尝试 {attempt + 1}/{self.config.retries + 1})")
                    await asyncio.sleep(wait_time)
                else:
                    raise Exception(last_error)
                    
            except Exception as e:
                last_error = str(e)
                if attempt < self.config.retries:
                    wait_time = min(2 ** attempt, 10)
                    print(f"请求失败，{wait_time}秒后重试... (尝试 {attempt + 1}/{self.config.retries + 1})")
                    await asyncio.sleep(wait_time)
                else:
                    raise Exception(last_error)
    
    async def _stream_request(self, url: str, payload: Dict) -> AsyncGenerator[Dict, None]:
        """Handle streaming requests with better error handling"""
        try:
            async with self.client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if not line:
                        continue
                        
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            yield chunk
                        except json.JSONDecodeError as e:
                            # Skip malformed JSON chunks
                            continue
                            
        except httpx.HTTPStatusError as e:
            yield {"error": {"message": f"HTTP {e.response.status_code}: {e.response.text}", "type": "http_error"}}
        except Exception as e:
            yield {"error": {"message": str(e), "type": "stream_error"}}
    
    async def health_check(self) -> Dict:
        """Comprehensive health check"""
        try:
            start_time = time.time()
            response = await self.client.get(f"{self.server_url}/health")
            response_time = time.time() - start_time
            response.raise_for_status()
            
            health_data = response.json()
            health_data["response_time"] = round(response_time, 3)
            return health_data
        except httpx.HTTPStatusError as e:
            return {"error": f"HTTP {e.response.status_code}: {e.response.text}", "status": "error"}
        except Exception as e:
            return {"error": str(e), "status": "error"}
    
    async def get_stats(self) -> Dict:
        """Get detailed statistics"""
        try:
            response = await self.client.get(f"{self.server_url}/stats")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            return {"error": f"HTTP {e.response.status_code}: {e.response.text}"}
        except Exception as e:
            return {"error": str(e)}
    
    async def get_models(self) -> Dict:
        """Get available models"""
        try:
            response = await self.client.get(f"{self.server_url}/v1/models")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    async def reset_key(self, key_prefix: str) -> Dict:
        """Reset API key status"""
        try:
            response = await self.client.post(f"{self.server_url}/admin/reset-key/{key_prefix}")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            return {"error": f"HTTP {e.response.status_code}: {e.response.text}"}
        except Exception as e:
            return {"error": str(e)}
    
    async def close(self):
        """Close the client"""
        await self.client.aclose()

class ConfigManager:
    """Enhanced configuration manager"""
    
    CONFIG_FILE = "vps_config.json"
    
    @classmethod
    def load_config(cls) -> ClientConfig:
        """Load configuration from file"""
        if os.path.exists(cls.CONFIG_FILE):
            try:
                with open(cls.CONFIG_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return ClientConfig(**data)
            