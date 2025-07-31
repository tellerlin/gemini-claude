#!/usr/bin/env python3
"""
Simple test script for the enhanced Gemini Claude Adapter
Tests both Anthropic Messages API and legacy OpenAI-compatible endpoints
"""

import asyncio
import json
import httpx
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
BASE_URL = "http://localhost:8000"
CLIENT_KEY = os.getenv("ADAPTER_API_KEYS", "test-key").split(",")[0].strip()

# Test scenarios
TESTS = [
    {
        "name": "Anthropic Messages API - Simple Chat",
        "endpoint": "/v1/messages",
        "method": "POST",
        "headers": {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {CLIENT_KEY}",
            "Anthropic-Version": "2023-06-01"
        },
        "body": {
            "model": "claude-3-5-sonnet",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "Hello! Can you tell me about Paris in 2-3 sentences?"}
            ]
        }
    },
    {
        "name": "Anthropic Messages API - Token Count",
        "endpoint": "/v1/messages/count_tokens",
        "method": "POST",
        "headers": {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {CLIENT_KEY}",
            "Anthropic-Version": "2023-06-01"
        },
        "body": {
            "model": "claude-3-5-sonnet",
            "messages": [
                {"role": "user", "content": "Hello! Can you tell me about Paris in 2-3 sentences?"}
            ]
        }
    },
    {
        "name": "Legacy OpenAI API - Chat Completions",
        "endpoint": "/v1/chat/completions",
        "method": "POST",
        "headers": {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {CLIENT_KEY}"
        },
        "body": {
            "model": "gemini-2.5-pro",
            "messages": [
                {"role": "user", "content": "Hello! Can you tell me about Paris in 2-3 sentences?"}
            ],
            "max_tokens": 100
        }
    },
    {
        "name": "Models List",
        "endpoint": "/v1/models",
        "method": "GET",
        "headers": {
            "Authorization": f"Bearer {CLIENT_KEY}"
        }
    },
    {
        "name": "Health Check",
        "endpoint": "/health",
        "method": "GET"
    }
]

async def run_test(test):
    """Run a single test"""
    print(f"\n🧪 Running: {test['name']}")
    print("-" * 50)
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            if test['method'] == 'POST':
                response = await client.post(
                    BASE_URL + test['endpoint'],
                    headers=test.get('headers', {}),
                    json=test['body']
                )
            else:
                response = await client.get(
                    BASE_URL + test['endpoint'],
                    headers=test.get('headers', {})
                )
            
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                print("✅ Test passed!")
                
                # Show a preview of the response
                try:
                    data = response.json()
                    if 'object' in data and data['object'] == 'list':
                        # Models list
                        print(f"Found {len(data.get('data', []))} models")
                    elif 'input_tokens' in data:
                        # Token count
                        print(f"Token count: {data['input_tokens']}")
                    elif 'content' in data:
                        # Message response
                        content = data['content'][0].get('text', '')[:100] + '...'
                        print(f"Response preview: {content}")
                    else:
                        print(f"Response keys: {list(data.keys())}")
                except:
                    print("Response is not JSON")
            else:
                print("❌ Test failed!")
                print(f"Response: {response.text}")
                
    except Exception as e:
        print(f"❌ Test failed with error: {e}")

async def main():
    """Run all tests"""
    print("🚀 Gemini Claude Adapter v2.0.0 Test Suite")
    print("=" * 60)
    
    # First check if the server is running
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(BASE_URL)
            if response.status_code == 200:
                print("✅ Server is running")
            else:
                print("❌ Server is not responding correctly")
                return
    except:
        print("❌ Could not connect to server. Make sure it's running on http://localhost:8000")
        return
    
    # Run all tests
    for test in TESTS:
        await run_test(test)
    
    print("\n" + "=" * 60)
    print("🎉 Test suite completed!")
    print("\n📖 Next steps:")
    print("1. Configure your client to use: http://localhost:8000/v1")
    print("2. Set your API key to one of your ADAPTER_API_KEYS")
    print("3. Try with Claude Code or other Anthropic-compatible clients")

if __name__ == "__main__":
    asyncio.run(main())