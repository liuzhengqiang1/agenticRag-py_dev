# -*- coding: utf-8 -*-
"""聊天 API 路由 - 支持流式输出"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.models.request import ChatRequest
from app.models.response import ChatResponse
from app.services import rag_service

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    同步聊天接口（非流式）
    
    保留此接口是为了向后兼容，如果客户端不支持流式，可以使用这个接口
    """
    try:
        reply = rag_service.chat(request.query, request.session_id)
        return ChatResponse(reply=reply)
    except Exception as e:
        print(f"❌ RAG 调用失败：{e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"问答服务暂时不可用，请稍后重试。错误信息：{str(e)}"
        )


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    ⭐ 流式聊天接口（企业级核心功能）
    
    工作原理（给 Java 开发者的解释）：
    1. 客户端发起 HTTP 请求，保持连接不断开
    2. 服务端通过 Server-Sent Events (SSE) 协议，逐块推送数据
    3. 类似 Java 的 ResponseBodyEmitter 或 SseEmitter
    
    返回格式：text/event-stream（SSE 标准格式）
    """
    try:
        async def generate():
            """
            异步生成器函数
            
            ⭐ 关键概念：
            - async def：声明异步函数
            - async for：异步遍历流式数据
            - yield：每次返回一个数据块（不会阻塞）
            """
            async for chunk in rag_service.chat_stream(request.query, request.session_id):
                # 将每个文本片段返回给客户端
                # 注意：这里不需要加 "data: " 前缀，StreamingResponse 会自动处理
                yield chunk
        
        # 返回流式响应
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",  # SSE 标准 MIME 类型
            headers={
                "Cache-Control": "no-cache",  # 禁止缓存
                "Connection": "keep-alive",   # 保持连接
                "X-Accel-Buffering": "no"     # 禁用 Nginx 缓冲（如果使用 Nginx）
            }
        )
    
    except Exception as e:
        print(f"❌ 流式 RAG 调用失败：{e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"流式问答服务暂时不可用，请稍后重试。错误信息：{str(e)}"
        )