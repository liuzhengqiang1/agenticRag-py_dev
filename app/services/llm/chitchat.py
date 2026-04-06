# -*- coding: utf-8 -*-
"""闲聊服务"""
from typing import AsyncIterator
from langchain_openai import ChatOpenAI
from app.services.history.session_manager import get_recent_history, get_session_history
from app.services.utils.formatters import format_history_for_prompt

# 初始化轻量级模型
_chitchat_llm = ChatOpenAI(model="qwen-turbo", temperature=0.7)


async def chitchat_stream(question: str, session_id: str) -> AsyncIterator[str]:
    """
    闲聊流式回复（使用 qwen-turbo，保持 HR 助手人设）

    参数：
        question: 用户问题
        session_id: 会话 ID

    返回：
        异步生成器，逐块返回回复
    """
    # 获取历史对话
    history_pairs = get_recent_history(session_id, max_rounds=3)
    history_text = format_history_for_prompt(history_pairs)

    chitchat_prompt = f"""你是一个专业、友好的企业 HR 助手。

你的职责：
- 回答员工关于公司培训制度的问题
- 在闲聊时保持专业、礼貌、热情的态度
- 适当引导用户询问培训相关问题

历史对话：
{history_text}

用户：{question}

请简短、自然地回复（1-2句话即可）："""

    try:
        full_response = ""
        # 使用 astream 进行流式输出
        async for chunk in _chitchat_llm.astream(chitchat_prompt):
            if chunk.content:
                full_response += chunk.content
                yield chunk.content

        # 保存到历史记录
        history = get_session_history(session_id)
        history.add_user_message(question)
        history.add_ai_message(full_response)

    except Exception as e:
        print(f"⚠️ 闲聊回复失败：{e}")
        yield "抱歉，我遇到了一些问题，请稍后再试。"
