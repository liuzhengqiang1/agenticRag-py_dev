# -*- coding: utf-8 -*-
"""FastAPI 应用实例"""
from fastapi import FastAPI

from app.api import chat

app = FastAPI(
    title="RAG 问答系统 MVP",
    description="基于 FastAPI 和 LangChain 的企业级问答系统",
    version="0.1.0"
)

# 注册路由
app.include_router(chat.router, prefix="/api", tags=["聊天"])


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "ok"}