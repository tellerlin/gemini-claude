#!/bin/bash

# 设置默认值
: "${SERVICE_HOST:=0.0.0.0}"
: "${SERVICE_PORT:=8000}"

# 智能计算 Gunicorn 工作进程数
# 优先使用环境变量 SERVICE_WORKERS
# 如果未设置，则根据 CPU 核心数计算，公式为 (2 * CPU核心数) + 1
# 如果无法获取 CPU 核心数，则默认为 3
if [ -z "$SERVICE_WORKERS" ]; then
    CPU_CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)
    WORKERS=$((2 * CPU_CORES + 1))
else
    WORKERS=$SERVICE_WORKERS
fi

echo "🚀 Starting Gunicorn server..."
echo "➤ Host: $SERVICE_HOST"
echo "➤ Port: $SERVICE_PORT"
echo "➤ Workers: $WORKERS"

# 使用 gunicorn 启动应用
# -w: 工作进程数
# -k: 使用 uvicorn.workers.UvicornWorker 作为工作进程类，以支持 FastAPI 的异步特性
# -b: 绑定地址和端口
exec gunicorn "main:app" \
    --workers $WORKERS \
    --worker-class "uvicorn.workers.UvicornWorker" \
    --bind "$SERVICE_HOST:$SERVICE_PORT"
