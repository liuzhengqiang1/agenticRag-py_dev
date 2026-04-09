# -*- coding: utf-8 -*-
"""Agentic RAG 聊天接口"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional

from app.services.agents import run_agentic_rag, run_agentic_rag_stream
from app.services.tools import (
    search_knowledge_base,
    get_current_weather,
    query_database_order,
    get_web_search_tool,
)

router = APIRouter()


# ============================================================
# 请求/响应模型
# ============================================================


class AgenticChatRequest(BaseModel):
    """Agentic RAG 聊天请求"""

    question: str = Field(..., description="用户问题", min_length=1, max_length=1000)
    session_id: Optional[str] = Field(
        default="default", description="会话 ID（用于多轮对话）"
    )
    stream: bool = Field(default=False, description="是否使用流式输出")


class AgenticChatResponse(BaseModel):
    """Agentic RAG 聊天响应"""

    answer: str = Field(..., description="回答内容")
    session_id: str = Field(..., description="会话 ID")


# ============================================================
# API 路由
# ============================================================


@router.post("/agentic-chat", response_model=AgenticChatResponse)
async def agentic_chat(request: AgenticChatRequest):
    """
    Agentic RAG 聊天接口（非流式）
    TODO 没有同步聊天需求，暂未修复bug

    支持的功能：
    - 知识库检索（培训制度、报销流程等）
    - 天气查询
    - 订单查询
    - 互联网搜索（需配置 TAVILY_API_KEY）

    示例请求：
    ```json
    {
        "question": "帮我查一下订单 9982 的状态，顺便看看上海天气",
        "session_id": "user_123",
        "stream": false
    }
    ```
    """
    try:
        # 如果请求流式输出，返回错误提示
        if request.stream:
            raise HTTPException(
                status_code=400,
                detail="请使用 /api/agentic-chat-stream 接口进行流式输出",
            )

        # 调用 Agentic RAG
        answer = run_agentic_rag(
            user_question=request.question, session_id=request.session_id
        )

        return AgenticChatResponse(answer=answer, session_id=request.session_id)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理失败：{str(e)}")


@router.post("/agentic-chat-stream")
async def agentic_chat_stream(request: AgenticChatRequest):
    """
    Agentic RAG 聊天接口（流式输出）

    使用 Server-Sent Events (SSE) 实现流式响应。

    示例请求：
    ```json
    {
        "question": "公司的培训类型有哪些？",
        "session_id": "user_123"
    }
    ```

    响应格式：
    ```
    data: 公司
    data: 的
    data: 培训
    data: 类型
    data: 有...
    ```
    """
    try:

        async def event_generator():
            """SSE 事件生成器"""
            try:
                async for chunk in run_agentic_rag_stream(
                    user_question=request.question, session_id=request.session_id
                ):
                    # SSE 格式：data: <content>\n\n
                    yield f"data: {chunk}\n\n"

                # 发送结束标记
                yield "data: [DONE]\n\n"

            except Exception as e:
                # 发送错误信息
                yield f"data: [ERROR] {str(e)}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # 禁用 Nginx 缓冲
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理失败：{str(e)}")


@router.get("/agentic-tools")
async def list_agentic_tools():
    """
    列出所有可用的 Agentic RAG 工具

    返回：
    ```json
    {
        "tools": [
            {
                "name": "search_knowledge_base",
                "description": "在企业知识库中检索相关信息"
            },
            {
                "name": "get_current_weather",
                "description": "获取指定城市的当前天气信息"
            },
            ...
        ]
    }
    ```
    """
    # 整合所有工具
    tools = [
        search_knowledge_base,
        get_current_weather,
        query_database_order,
    ]

    # 如果 Tavily 可用，添加到工具列表
    web_search_tool = get_web_search_tool()
    if web_search_tool:
        tools.append(web_search_tool)

    tool_list = [{"name": tool.name, "description": tool.description} for tool in tools]

    return {"tools": tool_list, "count": len(tool_list)}
