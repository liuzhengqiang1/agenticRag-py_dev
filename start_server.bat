@echo off
echo ========================================
echo   RAG 流式对话系统启动脚本
echo ========================================
echo.

echo [1/3] 检查依赖...
pip show fastapi >nul 2>&1
if errorlevel 1 (
    echo 正在安装依赖...
    pip install -r requirements.txt
) else (
    echo 依赖已安装
)

echo.
echo [2/3] 启动服务...
echo 服务地址: http://localhost:8000
echo API 文档: http://localhost:8000/docs
echo.

start http://localhost:8000/docs
start test_stream.html

echo [3/3] 运行中...
echo 按 Ctrl+C 停止服务
echo.

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
