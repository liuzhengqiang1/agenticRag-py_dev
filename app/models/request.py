# -*- coding: utf-8 -*-
"""请求模型"""
from pydantic import BaseModel


class ChatRequest(BaseModel):
    """聊天请求模型"""
    query: str