# comprehensive_test.py - 完整的测试脚本
import asyncio
import httpx
import json
import time
import sys
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import argparse

@dataclass
class TestResult:
    name: str
    success: bool
    duration: float
    error: Optional[str] = None
    details: Optional[Dict] = None

class GeminiAdapterTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.client = httpx.AsyncClient(timeout=120.0)
        self.results: List[TestResult] = []
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    def log_result(self, result: TestResult):
        """记录测试结果"""
        self.results.append(result)
        status = "✅ PASS" if result.success else "❌ FAIL"
        print(f"{status} {result.name} ({result.duration:.2f}s)")
        if result.error:
            print(f"   Error: {result.error}")
        if result.details:
            print(f"   Details: {json.dumps(result.details, indent=2)}")
    
    async def test_health_check(self) -> TestResult:
        """测试健康检查端点"""
        start_time = time.time()
        try:
            response = await self.client.get(f"{self.base_url}/health")
            response.raise_for_status()
            data = response.json()
            
            # 验证响应格式
            required_fields = ["status", "timestamp", "total_keys", "active_keys"]
            missing_fields = [field for field in required_fields if field not in data]
            
            if missing_fields:
                return TestResult(
                    "Health Check",
                    False,
                    time.time() - start_time,
                    f"Missing fields: {missing_fields}",
                    data
                )
            
            return TestResult(
                "Health Check",
                True,
                time.time() - start_time,
                details=data
            )
        except Exception as e:
            return TestResult(
                "Health Check",
                False,
                time.time() - start_time,
                str(e)
            )
    
    async def test_models_endpoint(self) -> TestResult:
        """测试模型列表端点"""
        start_time = time.time()
        try:
            response = await self.client.get(f"{self.base_url}/v1/models")
            response.raise_for_status()
            data = response.json()
            
            # 验证响应格式
            if "data" not in data or not isinstance(data["data"], list):
                return TestResult(
                    "Models Endpoint",
                    False,
                    time.time() - start_time,
                    "Invalid response format",
                    data
                )
            
            # 检查是否有Gemini模型
            gemini_models = [model for model in data["data"] if "gemini" in model.get("id", "").lower()]
            if not gemini_models:
                return TestResult(
                    "Models Endpoint",
                    False,
                    time.time() - start_time,
                    "No Gemini models found",
                    data
                )
            
            return TestResult(
                "Models Endpoint",
                True,
                time.time() - start_time,
                details={"model_count": len(data["data"]), "gemini_models": len(gemini_models)}
            )
        except Exception as e:
            return TestResult(
                "Models Endpoint",
                False,
                time.time() - start_time,
                str(e)
            )
    
    async def test_basic_chat(self) -> TestResult:
        """测试基本聊天功能"""
        start_time = time.time()
        try:
            payload = {
                "messages": [{"role": "user", "content": "Hello, please respond with 'Test successful'"}],
                "model": "gemini-1.5-pro",
                "temperature": 0.1,
                "stream": False
            }
            
            response = await self.client.post(f"{self.base_url}/v1/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
            
            # 验证响应格式
            if "choices" not in data or not data["choices"]:
                return TestResult(
                    "Basic Chat",
                    False,
                    time.time() - start_time,
                    "No choices in response",
                    data
                )
            
            content = data["choices"][0].get("message", {}).get("content", "")
            if not content:
                return TestResult(
                    "Basic Chat",
                    False,
                    time.time() - start_time,
                    "Empty response content",
                    data
                )
            
            return TestResult(
                "Basic Chat",
                True,
                time.time() - start_time,
                details={"response_length": len(content), "content_preview": content[:100]}
            )
        except Exception as e:
            return TestResult(
                "Basic Chat",
                False,
                time.time() - start_time,
                str(e)
            )
    
    async def test_streaming_chat(self) -> TestResult:
        """测试流式聊天功能"""
        start_time = time.time()
        try:
            payload = {
                "messages": [{"role": "user", "content": "Count from 1 to 5, each number on a new line"}],
                "model": "gemini-1.5-pro",
                "temperature": 0.1,
                "stream": True
            }
            
            chunks_received = 0
            content_received = ""
            
            async with self.client.stream("POST", f"{self.base_url}/v1/chat/completions", json=payload) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        
                        try:
                            chunk_data = json.loads(data_str)
                            chunks_received += 1
                            
                            if "choices" in chunk_data and chunk_data["choices"]:
                                delta = chunk_data["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                content_received += content
                        except json.JSONDecodeError:
                            continue
            
            if chunks_received == 0:
                return TestResult(
                    "Streaming Chat",
                    False,
                    time.time() - start_time,
                    "No chunks received"
                )
            
            return TestResult(
                "Streaming Chat",
                True,
                time.time() - start_time,
                details={
                    "chunks_received": chunks_received,
                    "content_length": len(content_received),
                    "content_preview": content_received[:100]
                }
            )
        except Exception as e:
            return TestResult(
                "Streaming Chat",
                False,
                time.time() - start_time,
                str(e)
            )
    
    async def test_concurrent_requests(self, concurrent_count: int = 5) -> TestResult:
        """测试并发请求"""
        start_time = time.time()
        try:
            async def single_request(request_id: int):
                payload = {
                    "messages": [{"role": "user", "content": f"This is concurrent test request #{request_id}. Please respond briefly."}],
                    "model": "gemini-1.5-pro",
                    "temperature": 0.1
                }
                
                try:
                    response = await self.client.post(f"{self.base_url}/v1/chat/completions", json=payload)
                    response.raise_for_status()
                    return True, response.json()
                except Exception as e:
                    return False, str(e)
            
            # 执行并发请求
            tasks = [single_request(i) for i in range(1, concurrent_count + 1)]
            results = await asyncio.gather(*tasks)
            
            successful = sum(1 for success, _ in results if success)
            failed = len(results) - successful
            
            return TestResult(
                f"Concurrent Requests ({concurrent_count})",
                failed == 0,
                time.time() - start_time,
                f"{failed} requests failed" if failed > 0 else None,
                {
                    "total_requests": len(results),
                    "successful": successful,
                    "failed": failed,
                    "success_rate": f"{successful/len(results)*100:.1f}%"
                }
            )
        except Exception as e:
            return TestResult(
                f"Concurrent Requests ({concurrent_count})",
                False,
                time.time() - start_time,
                str(e)
            )
    
    async def test_different_models(self) -> TestResult:
        """测试不同的模型"""
        start_time = time.time()
        models_to_test = ["gemini-1.5-pro", "gemini-1.5-flash"]
        results = {}
        
        try:
            for model in models_to_test:
                payload = {
                    "messages": [{"role": "user", "content": f"Hello from {model}"}],
                    "model": model,
                    "temperature": 0.1
                }
                
                try:
                    response = await self.client.post(f"{self.base_url}/v1/chat/completions", json=payload)
                    response.raise_for_status()
                    data = response.json()
                    results[model] = "success"
                except Exception as e:
                    results[model] = f"failed: {str(e)}"
            
            successful_models = sum(1 for result in results.values() if result == "success")
            
            return TestResult(
                "Different Models",
                successful_models > 0,
                time.time() - start_time,
                None if successful_models == len(models_to_test) else f"Some models failed",
                results
            )
        except Exception as e:
            return TestResult(
                "Different Models",
                False,
                time.time() - start_time,
                str(e)
            )
    
    async def test_error_handling(self) -> TestResult:
        """测试错误处理"""
        start_time = time.time()
        try:
            # 测试无效的请求格式
            invalid_payload = {
                "messages": "invalid format",  # 应该是列表
                "model": "gemini-1.5-pro"
            }
            
            response = await self.client.post(f"{self.base_url}/v1/chat/completions", json=invalid_payload)
            
            # 应该返回400错误
            if response.status_code == 400:
                return TestResult(
                    "Error Handling",
                    True,
                    time.time() - start_time,
                    details={"status_code": response.status_code, "handled_correctly": True}
                )
            else:
                return TestResult(
                    "Error Handling",
                    False,
                    time.time() - start_time,
                    f"Expected 400, got {response.status_code}",
                    {"status_code": response.status_code}
                )
        except Exception as e:
            return TestResult(
                "Error Handling",
                False,
                time.time() - start_time,
                str(e)
            )
    
    async def test_stats_endpoint(self) -> TestResult:
        """测试统计端点"""
        start_time = time.time()
        try:
            response = await self.client.get(f"{self.base_url}/stats")
            response.raise_for_status()
            data = response.json()
            
            required_fields = ["total_keys", "active_keys", "cooling_keys", "failed_keys", "keys_detail"]
            missing_fields = [field for field in required_fields if field not in data]
            
            if missing_fields:
                return TestResult(
                    "Stats Endpoint",
                    False,
                    time.time() - start_time,
                    f"Missing fields: {missing_fields}",
                    data
                )
            
            return TestResult(
                "Stats Endpoint",
                True,
                time.time() - start_time,
                details={
                    "total_keys": data["total_keys"],
                    "active_keys": data["active_keys"],
                    "keys_with_details": len(data["keys_detail"])
                }
            )
        except Exception as e:
            return TestResult(
                "Stats Endpoint",
                False,
                time.time() - start_time,
                str(e)
            )
    
    async def test_performance_benchmark(self, duration_seconds: int = 30) -> TestResult:
        """性能基准测试"""
        start_time = time.time()
        try:
            requests_completed = 0
            total_response_time = 0
            errors = 0
            
            async def benchmark_request():
                nonlocal requests_completed, total_response_time, errors
                
                request_start = time.time()
                try:
                    payload = {
                        "messages": [{"role": "user", "content": "Hello, please respond briefly."}],
                        "model": "gemini-1.5-pro",
                        "temperature": 0.1
                    }
                    
                    response = await self.client.post(f"{self.base_url}/v1/chat/completions", json=payload)
                    response.raise_for_status()
                    
                    requests_completed += 1
                    total_response_time += time.time() - request_start
                except Exception:
                    errors += 1
            
            # 运行基准测试
            end_time = start_time + duration_seconds
            tasks = []
            
            while time.time() < end_time:
                # 限制并发数以避免过载
                if len(tasks) < 10:
                    task = asyncio.create_task(benchmark_request())
                    tasks.append(task)
                
                # 清理完成的任务
                tasks = [task for task in tasks if not task.done()]
                await asyncio.sleep(0.1)
            
            # 等待所有任务完成
            await asyncio.gather(*tasks, return_exceptions=True)
            
            avg_response_time = total_response_time / requests_completed if requests_completed > 0 else 0
            requests_per_second = requests_completed / duration_seconds
            
            return TestResult(
                f"Performance Benchmark ({duration_seconds}s)",
                errors < requests_completed * 0.1,  # 允许10%的错误率
                time.time() - start_time,
                f"{errors} errors occurred" if errors > 0 else None,
                {
                    "duration": duration_seconds,
                    "requests_completed": requests_completed,
                    "errors": errors,
                    "avg_response_time": f"{avg_response_time:.2f}s",
                    "requests_per_second": f"{requests_per_second:.2f}",
                    "error_rate": f"{errors/max(requests_completed + errors, 1)*100:.1f}%"
                }
            )
        except Exception as e:
            return TestResult(
                f"Performance Benchmark ({duration_seconds}s)",
                False,
                time.time() - start_time,
                str(e)
            )
    
    async def run_all_tests(self, include_performance: bool = False, performance_duration: int = 30, concurrent_count: int = 5):
        """运行所有测试"""
        print(f"🚀 开始测试 Gemini Claude Adapter ({self.base_url})")
        print("=" * 60)
        
        # 基础功能测试
        basic_tests = [
            self.test_health_check(),
            self.test_models_endpoint(),
            self.test_stats_endpoint(),
            self.test_basic_chat(),
            self.test_streaming_chat(),
            self.test_different_models(),
            self.test_error_handling(),
        ]
        
        for test_coro in basic_tests:
            result = await test_coro
            self.log_result(result)
        
        # 并发测试
        result = await self.test_concurrent_requests(concurrent_count)
        self.log_result(result)
        
        # 性能测试（可选）
        if include_performance:
            result = await self.test_performance_benchmark(performance_duration)
            self.log_result(result)
        
        # 输出总结
        self.print_summary()
    
    def print_summary(self):
        """打印测试总结"""
        total_tests = len(self.results)
        passed_tests = sum(1 for result in self.results if result.success)
        failed_tests = total_tests - passed_tests
        
        print("\n" + "=" * 60)
        print("📊 测试总结")
        print("=" * 60)
        print(f"总测试数: {total_tests}")
        print(f"通过: {passed_tests} ✅")
        print(f"失败: {failed_tests} ❌")
        print(f"成功率: {passed_tests/total_tests*100:.1f}%")
        
        if failed_tests > 0:
            print("\n❌ 失败的测试:")
            for result in self.results:
                if not result.success:
                    print(f"  - {result.name}: {result.error}")
        
        total_duration = sum(result.duration for result in self.results)
        print(f"\n⏱️ 总耗时: {total_duration:.2f}秒")
        
        if passed_tests == total_tests:
            print("\n🎉 所有测试通过！系统运行正常。")
        else:
            print(f"\n⚠️ 有{failed_tests}个测试失败，请检查系统状态。")

# 命令行工具
async def main():
    parser = argparse.ArgumentParser(description="Gemini Claude Adapter 综合测试工具")
    parser.add_argument("--url", default="http://localhost:8000", help="服务器URL")
    parser.add_argument("--performance", action="store_true", help="包含性能测试")
    parser.add_argument("--perf-duration", type=int, default=30, help="性能测试持续时间(秒)")
    parser.add_argument("--concurrent", type=int, default=5, help="并发测试请求数")
    parser.add_argument("--quick", action="store_true", help="快速测试模式(跳过耗时测试)")
    
    args = parser.parse_args()
    
    # 快速模式配置
    if args.quick:
        args.performance = False
        args.concurrent = 3
        args.perf_duration = 10
    
    async with GeminiAdapterTester(args.url) as tester:
        await tester.run_all_tests(
            include_performance=args.performance,
            performance_duration=args.perf_duration,
            concurrent_count=args.concurrent
        )

# 独立的快速健康检查
async def quick_health_check(url: str = "http://localhost:8000"):
    """快速健康检查"""
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            # 健康检查
            response = await client.get(f"{url}/health")
            response.raise_for_status()
            health = response.json()
            
            print(f"🏥 服务健康状态:")
            print(f"  状态: {health.get('status', 'unknown')}")
            print(f"  总密钥数: {health.get('total_keys', 0)}")
            print(f"  活跃密钥: {health.get('active_keys', 0)}")
            print(f"  冷却密钥: {health.get('cooling_keys', 0)}")
            print(f"  失效密钥: {health.get('failed_keys', 0)}")
            
            # 简单请求测试
            start_time = time.time()
            chat_response = await client.post(f"{url}/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "gemini-1.5-pro"
            })
            chat_response.raise_for_status()
            response_time = time.time() - start_time
            
            print(f"📡 响应测试:")
            print(f"  响应时间: {response_time:.2f}秒")
            print(f"  状态码: {chat_response.status_code}")
            
            if health.get('active_keys', 0) > 0:
                print("✅ 系统运行正常")
            else:
                print("⚠️ 没有可用的API密钥")
                
        except Exception as e:
            print(f"❌ 健康检查失败: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "health":
        # 快速健康检查
        url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000"
        asyncio.run(quick_health_check(url))
    else:
        # 完整测试
        asyncio.run(main())