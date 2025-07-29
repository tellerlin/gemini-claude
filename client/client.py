#!/usr/bin/env python3
# client.py - Enhanced client for Gemini Claude Adapter

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
    preferred_model: str = "gemini-2.5-pro"
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
                            stream: bool = False):
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
                    print(f"Request timeout, retrying in {wait_time} seconds... (Attempt {attempt + 1}/{self.config.retries + 1})")
                    await asyncio.sleep(wait_time)
                else:
                    raise Exception(last_error)
                    
            except httpx.ConnectError as e:
                last_error = f"连接错误: {e}"
                # Connection errors usually don't benefit from retries unless it's intermittent
                if attempt < self.config.retries:
                    wait_time = min(2 ** attempt, 15)
                    print(f"Connection error, retrying in {wait_time} seconds... (Attempt {attempt + 1}/{self.config.retries + 1})")
                    await asyncio.sleep(wait_time)
                else:
                    raise Exception(last_error)
                    
            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code
                if status_code == 503:
                    last_error = "Service temporarily unavailable (all API keys may be cooling down)"
                elif status_code == 502:
                    last_error = "Gateway error (backend service unavailable)"
                elif status_code == 429:
                    last_error = "Request rate too high, please try again later"
                elif status_code == 400:
                    last_error = f"Request format error: {e.response.text}"
                    break  # Don't retry on client errors
                else:
                    last_error = f"HTTP {status_code}: {e.response.text}"
                
                if attempt < self.config.retries:
                    wait_time = min(2 ** attempt, 10)
                    print(f"Server error, retrying in {wait_time} seconds... (Attempt {attempt + 1}/{self.config.retries + 1})")
                    await asyncio.sleep(wait_time)
                else:
                    raise Exception(last_error)
                    
            except Exception as e:
                last_error = f"未知错误: {e}"
                if attempt < self.config.retries:
                    wait_time = min(2 ** attempt, 5)
                    print(f"Unknown error, retrying in {wait_time} seconds... (Attempt {attempt + 1}/{self.config.retries + 1})")
                    await asyncio.sleep(wait_time)
                else:
                    raise Exception(last_error)
        
        raise Exception(f"Request failed: {last_error}")
    
    async def _stream_request(self, url: str, payload: Dict) -> AsyncGenerator[Dict, None]:
        """Handle streaming requests"""
        async with self.client.stream('POST', url, json=payload) as response:
            response.raise_for_status()
            
            async for line in response.aiter_lines():
                line = line.strip()
                if line.startswith('data: '):
                    data_str = line[6:]  # Remove 'data: ' prefix
                    if data_str == '[DONE]':
                        break
                    
                    try:
                        data = json.loads(data_str)
                        yield data
                    except json.JSONDecodeError:
                        continue
    
    async def get_server_health(self) -> Dict:
        """Check server health and statistics"""
        try:
            response = await self.client.get(f"{self.server_url}/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    async def get_server_stats(self) -> Dict:
        """Get detailed server statistics"""
        try:
            response = await self.client.get(f"{self.server_url}/stats")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()

class InteractiveChat:
    """Interactive chat interface"""
    
    def __init__(self, client: RemoteGeminiClient):
        self.client = client
        self.conversation_history = []
    
    async def start_interactive_chat(self):
        """Start interactive chat session"""
        print("🤖 Gemini Claude Adapter - Interactive Chat")
        print("Type 'quit' or 'exit' to exit chat")
        print("Type 'clear' to clear conversation history")
        print("Type 'health' to check server status")
        print("Type 'stats' to view server statistics")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\n你: ").strip()
                
                if user_input.lower() in ['quit', 'exit']:
                    print("Goodbye!")
                    break
                
                if user_input.lower() == 'clear':
                    self.conversation_history = []
                    print("Conversation history cleared")
                    continue
                
                if user_input.lower() == 'health':
                    await self._show_health()
                    continue
                
                if user_input.lower() == 'stats':
                    await self._show_stats()
                    continue
                
                if not user_input:
                    continue
                
                # Add user message to history
                self.conversation_history.append({"role": "user", "content": user_input})
                
                # Get assistant response
                print("\n助手: ", end="", flush=True)
                
                full_response = ""
                async for chunk in self.client.chat_completion(
                    messages=self.conversation_history,
                    stream=True
                ):
                    if 'choices' in chunk and len(chunk['choices']) > 0:
                        delta = chunk['choices'][0].get('delta', {})
                        content = delta.get('content', '')
                        if content:
                            print(content, end="", flush=True)
                            full_response += content
                
                print()  # New line after response
                
                # Add assistant response to history
                if full_response:
                    self.conversation_history.append({"role": "assistant", "content": full_response})
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
    
    async def _show_health(self):
        """Show server health"""
        health = await self.client.get_server_health()
        if 'error' in health:
            print(f"❌ Server health check failed: {health['error']}")
        else:
            status = health.get('status', 'unknown')
            active_keys = health.get('active_keys', 0)
            total_keys = health.get('total_keys', 0)
            print(f"✅ Server status: {status}")
            print(f"🔑 Active keys: {active_keys}/{total_keys}")
    
    async def _show_stats(self):
        """Show server statistics"""
        stats = await self.client.get_server_stats()
        if 'error' in stats:
            print(f"❌ Failed to get statistics: {stats['error']}")
        else:
            print("📊 Server Statistics:")
            print(f"  Total keys: {stats.get('total_keys', 0)}")
            print(f"  Active keys: {stats.get('active_keys', 0)}")
            print(f"  Cooling keys: {stats.get('cooling_keys', 0)}")
            print(f"  Failed keys: {stats.get('failed_keys', 0)}")

async def main():
    parser = argparse.ArgumentParser(description="Gemini Claude Adapter Client")
    parser.add_argument("url", help="Server URL (e.g., http://your-vps-ip)")
    parser.add_argument("-m", "--model", default="gemini-2.5-pro", help="Model to use")
    parser.add_argument("-t", "--temperature", type=float, default=0.7, help="Temperature")
    parser.add_argument("--timeout", type=int, default=120, help="Request timeout")
    parser.add_argument("--retries", type=int, default=3, help="Number of retries")
    parser.add_argument("--max-concurrent", type=int, default=10, help="Max concurrent connections")
    
    args = parser.parse_args()
    
    # Create client configuration
    config = ClientConfig(
        vps_url=args.url,
        timeout=args.timeout,
        retries=args.retries,
        preferred_model=args.model,
        default_temperature=args.temperature,
        max_concurrent=args.max_concurrent
    )
    
    # Create client
    client = RemoteGeminiClient(config)
    
    try:
        # Test connection
        print("Connecting to server...")
        health = await client.get_server_health()
        if 'error' in health:
            print(f"❌ Connection failed: {health['error']}")
            return 1
        
        print("✅ Connection successful!")
        
        # Start interactive chat
        chat = InteractiveChat(client)
        await chat.start_interactive_chat()
        
    except KeyboardInterrupt:
        print("\nProgram terminated")
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    finally:
        await client.close()
    
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))