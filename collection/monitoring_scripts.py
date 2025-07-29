# monitor.py - 监控脚本
import asyncio
import httpx
import json
import time
from datetime import datetime
import argparse
from typing import Dict, Any

class AdapterMonitor:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=10.0)
    
    async def get_health(self) -> Dict[str, Any]:
        """获取健康状态"""
        try:
            response = await self.client.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e), "status": "error"}
    
    async def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        try:
            response = await self.client.get(f"{self.base_url}/stats")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    async def reset_key(self, key_prefix: str) -> Dict[str, Any]:
        """重置API key状态"""
        try:
            response = await self.client.post(f"{self.base_url}/admin/reset-key/{key_prefix}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def format_health_info(self, health: Dict[str, Any]) -> str:
        """格式化健康信息"""
        if "error" in health:
            return f"❌ 服务不可用: {health['error']}"
        
        status_emoji = "✅" if health.get("status") == "healthy" else "⚠️"
        return f"""
{status_emoji} 服务状态: {health.get('status', 'unknown')}
📊 API Keys状态:
  - 总计: {health.get('total_keys', 0)}
  - 活跃: {health.get('active_keys', 0)}
  - 冷却中: {health.get('cooling_keys', 0)}
  - 失效: {health.get('failed_keys', 0)}
🕐 检查时间: {health.get('timestamp', 'unknown')}
"""
    
    def format_stats(self, stats: Dict[str, Any]) -> str:
        """格式化统计信息"""
        if "error" in stats:
            return f"❌ 无法获取统计信息: {stats['error']}"
        
        result = f"""
📈 详细统计信息:
总API Keys: {stats.get('total_keys', 0)}
活跃Keys: {stats.get('active_keys', 0)}
冷却Keys: {stats.get('cooling_keys', 0)}
失效Keys: {stats.get('failed_keys', 0)}

📋 Keys详情:
"""
        
        for key_detail in stats.get('keys_detail', []):
            status_emoji = {
                'active': '✅',
                'cooling': '❄️',
                'failed': '❌'
            }.get(key_detail.get('status'), '❓')
            
            result += f"""
  {status_emoji} Key: {key_detail.get('key', 'unknown')}
    状态: {key_detail.get('status', 'unknown')}
    失败次数: {key_detail.get('failure_count', 0)}
    总请求: {key_detail.get('total_requests', 0)}
    成功请求: {key_detail.get('successful_requests', 0)}
    成功率: {key_detail.get('success_rate', 0):.1f}%
"""
        
        return result
    
    async def continuous_monitor(self, interval: int = 30):
        """持续监控"""
        print(f"开始监控服务 ({self.base_url})，检查间隔: {interval}秒")
        print("按 Ctrl+C 停止监控\n")
        
        try:
            while True:
                print(f"\n{'='*50}")
                print(f"监控时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print('='*50)
                
                # 获取健康状态
                health = await self.get_health()
                print(self.format_health_info(health))
                
                # 如果服务健康，显示详细统计
                if health.get("status") == "healthy":
                    stats = await self.get_stats()
                    print(self.format_stats(stats))
                
                await asyncio.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n监控已停止")
        except Exception as e:
            print(f"监控错误: {e}")
        finally:
            await self.client.aclose()

# admin.py - 管理脚本
class AdapterAdmin:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=10.0)
    
    async def show_status(self):
        """显示当前状态"""
        monitor = AdapterMonitor(self.base_url)
        
        print("获取服务状态...")
        health = await monitor.get_health()
        print(monitor.format_health_info(health))
        
        if health.get("status") in ["healthy", "degraded"]:
            stats = await monitor.get_stats()
            print(monitor.format_stats(stats))
        
        await self.client.aclose()
    
    async def reset_key_interactive(self):
        """交互式重置API key"""
        monitor = AdapterMonitor(self.base_url)
        
        # 先显示当前状态
        stats = await monitor.get_stats()
        if "error" in stats:
            print(f"无法获取key信息: {stats['error']}")
            return
        
        print("\n当前API Keys状态:")
        keys_detail = stats.get('keys_detail', [])
        
        for i, key_detail in enumerate(keys_detail):
            status = key_detail.get('status', 'unknown')
            key_name = key_detail.get('key', 'unknown')
            failure_count = key_detail.get('failure_count', 0)
            
            status_emoji = {
                'active': '✅',
                'cooling': '❄️', 
                'failed': '❌'
            }.get(status, '❓')
            
            print(f"{i+1}. {status_emoji} {key_name} (状态: {status}, 失败: {failure_count})")
        
        # 让用户选择要重置的key
        try:
            choice = input(f"\n选择要重置的key (1-{len(keys_detail)}) 或输入key前缀: ").strip()
            
            if choice.isdigit():
                index = int(choice) - 1
                if 0 <= index < len(keys_detail):
                    key_prefix = keys_detail[index]['key'].split('...')[0]
                else:
                    print("无效的选择")
                    return
            else:
                key_prefix = choice
            
            print(f"重置key {key_prefix}...")
            result = await monitor.reset_key(key_prefix)
            
            if "error" in result:
                print(f"重置失败: {result['error']}")
            else:
                print(f"✅ {result.get('message', '重置成功')}")
                
        except KeyboardInterrupt:
            print("\n操作已取消")
        except Exception as e:
            print(f"操作失败: {e}")
        finally:
            await self.client.aclose()
    
    async def test_connection(self):
        """测试连接"""
        print(f"测试连接到 {self.base_url}...")
        
        try:
            response = await self.client.get(f"{self.base_url}/health")
            response.raise_for_status()
            
            health = response.json()
            if health.get("status") == "healthy":
                print("✅ 连接成功，服务健康")
            else:
                print(f"⚠️ 连接成功，但服务状态: {health.get('status')}")
                
        except httpx.ConnectError:
            print("❌ 连接失败：无法连接到服务")
        except httpx.TimeoutException:
            print("❌ 连接超时")
        except Exception as e:
            print(f"❌ 连接错误: {e}")
        finally:
            await self.client.aclose()

# 命令行工具
async def main():
    parser = argparse.ArgumentParser(description="Gemini Claude适配器监控和管理工具")
    parser.add_argument("--url", default="http://localhost:8000", help="服务URL")
    
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # 监控命令
    monitor_parser = subparsers.add_parser("monitor", help="持续监控服务")
    monitor_parser.add_argument("--interval", type=int, default=30, help="检查间隔(秒)")
    
    # 状态命令
    subparsers.add_parser("status", help="显示当前状态")
    
    # 重置命令
    subparsers.add_parser("reset", help="重置API key状态")
    
    # 测试命令
    subparsers.add_parser("test", help="测试连接")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == "monitor":
        monitor = AdapterMonitor(args.url)
        await monitor.continuous_monitor(args.interval)
    
    elif args.command == "status":
        admin = AdapterAdmin(args.url)
        await admin.show_status()
    
    elif args.command == "reset":
        admin = AdapterAdmin(args.url)
        await admin.reset_key_interactive()
    
    elif args.command == "test":
        admin = AdapterAdmin(args.url)
        await admin.test_connection()

if __name__ == "__main__":
    asyncio.run(main())

# 使用示例脚本
"""
使用方法:

1. 测试连接:
python monitor.py test

2. 查看状态:
python monitor.py status

3. 持续监控(30秒间隔):
python monitor.py monitor

4. 持续监控(自定义间隔):
python monitor.py monitor --interval 60

5. 重置API key:
python monitor.py reset

6. 指定服务URL:
python monitor.py --url http://192.168.1.100:8000 status
"""