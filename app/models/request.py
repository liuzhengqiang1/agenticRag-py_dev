# -*- coding: utf-8 -*-
"""请求模型"""
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """聊天请求模型"""
    query: str = Field(..., description="用户问题")
    session_id: str = Field(default="default", description="会话 ID，用于区分不同用户的对话")