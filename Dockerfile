# --- 1. Builder Stage ---
# 使用一个包含构建工具的镜像来安装依赖
FROM python:3.11-slim as builder

# 设置环境变量，避免交互式提示
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 安装依赖到一个特定目录，以便后续复制
# --prefix 指定安装路径，而不是使用 --user
RUN pip install --no-cache-dir --prefix="/install" -r requirements.txt

# --- 2. Production Stage ---
# 使用一个干净、轻量的基础镜像
FROM python:3.11-slim

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
# 将安装的包路径添加到 PYTHONPATH
ENV PYTHONPATH=/install/lib/python3.11/site-packages

# 创建一个非 root 用户来运行应用，增加安全性
RUN useradd --create-home --shell /bin/bash appuser

# 从 builder 阶段复制已安装的依赖
COPY --from=builder /install /install

# 设置工作目录
WORKDIR /app

# 复制应用源代码，并设置所有者为 appuser
# 注意：假设你的代码在 src 目录下
COPY --chown=appuser:appuser ./src .

# 创建日志目录并设置权限
RUN mkdir -p logs && chown appuser:appuser logs

# 切换到非 root 用户
USER appuser

# 暴露应用运行的端口
EXPOSE 8000

# 健康检查，确保服务正常运行
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; exit(0) if urllib.request.urlopen('http://localhost:8000/health', timeout=10).getcode() == 200 else exit(1)"

# 使用 Gunicorn 启动应用，这是生产环境推荐的方式
# 从环境变量读取工作进程数，默认为1
CMD ["gunicorn", "main:app", "-w", "${SERVICE_WORKERS:-1}", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000"]
