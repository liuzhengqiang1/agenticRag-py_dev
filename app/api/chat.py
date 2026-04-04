# -*- coding: utf-8 -*-
"""聊天 API 路由"""
from fastapi import APIRouter, HTTPException

from app.models.request import ChatRequest
from app.models.response import ChatResponse
from app.services import rag_service

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """聊天接口"""
    try:
        reply = rag_service.chat(request.query)
        return ChatResponse(reply=reply)
    except Exception as e:
        print(f"❌ RAG 调用失败：{e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"问答服务暂时不可用，请稍后重试。错误信息：{str(e)}"
        )