# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖（ChromaDB 需要）
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖声明
COPY pyproject.toml .

# 安装 Python 包（editable mode）
RUN pip install --no-cache-dir -e .

# 复制整个项目（包括 models/ 和 data/）
COPY . .

# 设置环境变量
ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=file:///app/mlruns
ENV OLLAMA_HOST=http://host.docker.internal:11434

# 暴露端口（MLflow 默认 8080，也可改 5001）
EXPOSE 8000

# fastapi
CMD ["uvicorn", "src.ecu_agent.api:app", "--host", "0.0.0.0", "--port", "8000"]