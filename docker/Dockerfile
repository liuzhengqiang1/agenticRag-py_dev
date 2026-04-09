# 使用官方 Python 3.11 轻量级镜像
FROM python:3.11.15-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量（防止 Python 生成 .pyc 文件和缓冲输出）
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# 安装系统依赖（如果需要编译某些 Python 包）
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件（利用 Docker 缓存加速构建）
COPY ../requirements.txt .

# 安装 Python 依赖
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install gunicorn

# 复制项目代码
COPY .. .

# 暴露端口
EXPOSE 8100

# 使用 Gunicorn 启动（生产级）
CMD ["gunicorn", "app.main:app", \
     "-w", "4", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8100", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
