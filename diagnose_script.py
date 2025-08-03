#!/usr/bin/env python3
"""
诊断脚本 - 检查 main.py 导入问题
"""

import sys
import os
import traceback
from pathlib import Path

def check_python_version():
    print(f"Python 版本: {sys.version}")
    print(f"Python 路径: {sys.executable}")

def check_working_directory():
    print(f"当前工作目录: {os.getcwd()}")
    print(f"Python 路径: {sys.path}")

def check_files_exist():
    required_files = [
        'main.py',
        'config.py', 
        'error_handling.py',
        'performance.py',
        'anthropic_api.py',
        'requirements.txt'
    ]
    
    print("\n检查必需文件:")
    for file in required_files:
        exists = os.path.exists(file)
        print(f"  {file}: {'✓' if exists else '✗'}")
        if not exists:
            print(f"    错误: 缺少文件 {file}")

def check_imports():
    print("\n检查模块导入:")
    
    # 检查标准库
    standard_libs = ['asyncio', 'time', 'json', 'os', 'hashlib']
    for lib in standard_libs:
        try:
            __import__(lib)
            print(f"  {lib}: ✓")
        except ImportError as e:
            print(f"  {lib}: ✗ - {e}")
    
    # 检查第三方库
    third_party_libs = [
        'fastapi', 'uvicorn', 'pydantic', 'loguru', 
        'litellm', 'dotenv', 'cachetools'
    ]
    
    for lib in third_party_libs:
        try:
            __import__(lib)
            print(f"  {lib}: ✓")
        except ImportError as e:
            print(f"  {lib}: ✗ - {e}")

def check_local_imports():
    print("\n检查本地模块导入:")
    
    local_modules = ['config', 'error_handling', 'performance', 'anthropic_api']
    
    for module in local_modules:
        try:
            __import__(module)
            print(f"  {module}: ✓")
        except ImportError as e:
            print(f"  {module}: ✗ - {e}")
            traceback.print_exc()

def check_main_module():
    print("\n检查 main.py:")
    
    try:
        # 尝试导入 main 模块
        import main
        print("  导入 main 模块: ✓")
        
        # 检查是否有 app 属性
        if hasattr(main, 'app'):
            print("  找到 app 属性: ✓")
            print(f"  app 类型: {type(main.app)}")
        else:
            print("  找到 app 属性: ✗")
            print("  可用属性:", [attr for attr in dir(main) if not attr.startswith('_')])
            
    except ImportError as e:
        print(f"  导入 main 模块: ✗ - {e}")
        traceback.print_exc()
    except Exception as e:
        print(f"  检查 main 模块时出错: {e}")
        traceback.print_exc()

def check_environment_variables():
    print("\n检查环境变量:")
    
    required_env_vars = [
        'GEMINI_API_KEYS',
        'SECURITY_ADAPTER_API_KEYS'
    ]
    
    optional_env_vars = [
        'CLAUDE_CODE_WORKING_DIR',
        'GEMINI_PROXY_URL',
        'SERVICE_WORKERS'
    ]
    
    print("  必需的环境变量:")
    for var in required_env_vars:
        value = os.getenv(var)
        if value:
            # 只显示前几个字符，保护敏感信息
            display_value = value[:10] + "..." if len(value) > 10 else value
            print(f"    {var}: ✓ ({display_value})")
        else:
            print(f"    {var}: ✗ (未设置)")
    
    print("  可选的环境变量:")
    for var in optional_env_vars:
        value = os.getenv(var)
        if value:
            print(f"    {var}: ✓ ({value})")
        else:
            print(f"    {var}: - (未设置)")

def main():
    print("=== Gemini Claude Adapter 诊断工具 ===\n")
    
    check_python_version()
    check_working_directory()
    check_files_exist()
    check_imports()
    check_local_imports()
    check_environment_variables()
    check_main_module()
    
    print("\n=== 诊断完毕 ===")
    
    # 给出建议
    print("\n建议:")
    print("1. 如果缺少文件，请确保所有源代码文件都已正确复制到容器中")
    print("2. 如果缺少第三方库，请检查 requirements.txt 和 pip 安装")
    print("3. 如果本地模块导入失败，请检查文件语法和依赖关系")
    print("4. 如果环境变量缺失，请检查 .env 文件或 docker-compose.yml")

if __name__ == "__main__":
    main()
