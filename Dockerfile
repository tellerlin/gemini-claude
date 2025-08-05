# --- 1. Builder Stage ---
FROM python:3.11-slim as builder

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix="/install" -r requirements.txt

# --- 2. Production Stage ---
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/install/lib/python3.11/site-packages
# Add /install/bin to the PATH
ENV PATH=/install/bin:$PATH

RUN useradd --create-home --shell /bin/bash appuser

COPY --from=builder /install /install

WORKDIR /app

# 复制启动脚本并赋予执行权限
COPY --chown=appuser:appuser start.sh .
RUN chmod +x ./start.sh

COPY --chown=appuser:appuser ./src .

RUN mkdir -p logs && chown appuser:appuser logs

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; exit(0) if urllib.request.urlopen('http://localhost:8000/health', timeout=10).getcode() == 200 else exit(1)"

# 使用启动脚本来启动应用
CMD ["./start.sh"]
