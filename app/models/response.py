# -*- coding: utf-8 -*-
"""响应模型"""
from pydantic import BaseModel


class ChatResponse(BaseModel):
    """聊天响应模型"""
    reply: str