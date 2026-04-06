# -*- coding: utf-8 -*-
"""查询重写服务"""
from typing import List
from langchain_openai import ChatOpenAI
from app.services.history.session_manager import get_recent_history
from app.services.utils.formatters import format_history_for_prompt

# 初始化轻量级模型
_rewrite_llm = ChatOpenAI(model="qwen-turbo", temperature=0)


def need_query_rewrite(question: str, history_pairs: List[tuple]) -> bool:
    """
    判断是否需要查询重写（智能触发）

    触发条件：
    1. 问题中包含指代词（它、这个、那个、他、她等）
    2. 问题过短且有历史对话（可能是追问）

    参数：
        question: 用户问题
        history_pairs: 历史对话

    返回：
        True 需要重写 / False 不需要重写
    """
    # 规则1：包含指代词
    pronouns = [
        "它",
        "这个",
        "那个",
        "这",
        "那",
        "他",
        "她",
        "这些",
        "那些",
        "哪个",
        "哪些",
    ]
    if any(p in question for p in pronouns):
        print(f"✏️ 触发查询重写：检测到指代词 | 问题：{question}")
        return True

    # 规则2：问题过短（<8字）且有历史对话（可能是追问）
    if len(question) < 8 and history_pairs:
        print(f"✏️ 触发查询重写：问题过短且有历史对话 | 问题：{question}")
        return True

    print(f"⏭️ 跳过查询重写：问题已完整 | 问题：{question}")
    return False


async def rewrite_query(question: str, session_id: str) -> str:
    """
    查询重写：将多轮对话改写为独立问题

    参数：
        question: 用户问题
        session_id: 会话 ID

    返回：
        改写后的独立问题
    """
    # 获取最近 2-3 轮历史
    history_pairs = get_recent_history(session_id, max_rounds=3)

    # 判断是否需要重写
    if not need_query_rewrite(question, history_pairs):
        return question

    history_text = format_history_for_prompt(history_pairs)

    rewrite_prompt = f"""你是一个查询重写助手。请将用户的问题改写为一个完整、独立、适合检索的问题。

改写规则：
1. 补全缺失的主语、宾语（从历史对话中推断）
2. 将指代词（它、这个、那个、他、她）替换为具体内容
3. 保持原问题的核心意图不变
4. 只返回改写后的问题，不要解释，不要加引号

历史对话：
{history_text}

当前问题：{question}

改写后的问题："""

    try:
        response = await _rewrite_llm.ainvoke(rewrite_prompt)
        rewritten = response.content.strip().strip('"').strip("'")
        print(f"📝 查询重写：\n  原问题：{question}\n  改写后：{rewritten}")
        return rewritten
    except Exception as e:
        print(f"⚠️ 查询重写失败，使用原问题：{e}")
        return question
