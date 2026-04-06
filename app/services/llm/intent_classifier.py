# -*- coding: utf-8 -*-
"""意图分类服务"""
import json
from langchain_openai import ChatOpenAI
from app.services.history.session_manager import get_recent_history
from app.services.utils.formatters import format_history_for_prompt

# 初始化轻量级模型
_intent_llm = ChatOpenAI(model="qwen-turbo", temperature=0)


async def classify_intent(question: str, session_id: str) -> str:
    """
    意图分类：判断用户问题是闲聊还是需要检索

    参数：
        question: 用户问题
        session_id: 会话 ID

    返回：
        "chitchat" 或 "retrieval"
    """
    # 获取最近 1-2 轮历史
    history_pairs = get_recent_history(session_id, max_rounds=2)
    history_text = format_history_for_prompt(history_pairs)

    intent_prompt = f"""你是一个意图分类助手。请判断用户的问题是否需要查询知识库。

知识库内容：企业 HR 培训制度相关信息（培训类型、申请流程、费用报销、考核标准等）

请判断以下问题的意图类型，只返回 JSON 格式：
{{"intent": "chitchat"}} 或 {{"intent": "retrieval"}}

判断规则：
- chitchat（闲聊）：问候、感谢、情感表达、与培训制度无关的话题
  例如："你好"、"谢谢"、"今天天气怎么样"、"你是谁"、"再见"
  
- retrieval（需要检索）：询问培训制度、政策、流程、费用等具体信息
  例如："培训有哪些类型"、"如何申请培训"、"费用怎么报销"、"考核标准是什么"

历史对话：
{history_text}

当前问题：{question}

请只返回 JSON，不要有任何其他内容："""

    try:
        response = await _intent_llm.ainvoke(intent_prompt)
        result = json.loads(response.content.strip())
        intent = result.get("intent", "retrieval")
        print(f"🔍 意图分类结果：{intent} | 问题：{question}")
        return intent
    except Exception as e:
        print(f"⚠️ 意图分类失败，默认走检索路径：{e}")
        return "retrieval"
