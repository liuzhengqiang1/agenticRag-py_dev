# -*- coding: utf-8 -*-
"""FastAPI 应用实例"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import chat

app = FastAPI(
    title="RAG 问答系统 MVP",
    description="基于 FastAPI 和 LangChain 的企业级问答系统",
    version="0.2.0"
)

# 配置 CORS（跨域资源共享）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境建议指定具体域名
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有 HTTP 方法（包括 OPTIONS）
    allow_headers=["*"],  # 允许所有请求头
)

# 注册路由
app.include_router(chat.router, prefix="/api", tags=["聊天"])


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "ok"}