# Dockerfile

# --- 基础镜像 ---
# 使用官方的 Python 3.11 slim 镜像，它体积小且稳定。
FROM python:3.11-slim

# --- 设置工作目录 ---
# 在容器内创建一个 /app 目录，并将其设置为工作目录。
WORKDIR /app

# --- 安装依赖 ---
# 为了利用 Docker 的缓存机制，先只复制依赖文件。
# 这样只有在 requirements.txt 变化时，才会重新安装依赖，加快后续构建速度。
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- 复制项目代码 ---
# 将项目的所有文件复制到工作目录 /app。
COPY . .

# --- 暴露端口 ---
# 声明容器将会在 8000 端口上监听。
EXPOSE 8000

# --- 启动命令 ---
# 设置容器启动时要执行的命令。
# 这里我们使用 gunicorn，这是一个更适合生产环境的 WSGI 服务器，它会管理 uvicorn 工作进程。
# -w 4 表示启动4个工作进程，您可以根据服务器CPU核心数调整。
# --bind 0.0.0.0:8000 表示监听所有网络接口的8000端口，这样容器外部才能访问。
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "src.main:app", "--bind", "0.0.0.0:8000"]
