# -*- coding: utf-8 -*-
"""RAG 服务模块 - 支持多轮对话记忆、流式输出与 Elasticsearch 混合检索"""
from dotenv import load_dotenv

load_dotenv()  # 加载 .env 文件中的环境变量

from typing import AsyncIterator
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory

# 导入拆分后的模块
from app.services.history.session_manager import get_session_history
from app.services.retrievers.ensemble_retriever import create_hybrid_retriever
from app.services.utils.formatters import format_docs
from app.services.llm.intent_classifier import classify_intent
from app.services.llm.query_rewriter import rewrite_query
from app.services.llm.chitchat import chitchat_stream


# ============================================================
# 🚀 RAG 服务（单例模式）
# ============================================================
class RAGService:
    """RAG 服务单例类 - 编排层"""

    _instance = None
    _conversational_chain = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def initialize(self):
        """初始化带有会话记忆的 RAG 组件（只会执行一次）"""
        if self._conversational_chain is not None:
            return self._conversational_chain

        print(
            "正在初始化 RAG 组件（支持意图理解 + 查询重写 + Elasticsearch 混合检索）..."
        )

        # 1. LLM 模型
        llm = ChatOpenAI(model="qwen-max", temperature=0.7)

        # 2. 创建混合检索器（向量 + BM25 + 重排序）
        final_retriever = create_hybrid_retriever()

        # 3. Prompt 模板（支持历史对话）
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是一个专业的企业 HR 助手，负责回答员工关于公司培训制度的问题。\n"
                    "请根据以下检索到的知识库内容来回答用户的问题。\n"
                    '如果知识库中没有相关信息，请明确告知用户"抱歉，我在知识库中未找到相关信息"。\n\n'
                    "知识库内容：\n{context}",
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )

        # 4. 基础 RAG Chain
        base_chain = (
            {
                "context": lambda x: format_docs(final_retriever.invoke(x["input"])),
                "input": lambda x: x["input"],
                "chat_history": lambda x: x.get("chat_history", []),
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        # 5. 包装为带历史记录的 Chain
        self._conversational_chain = RunnableWithMessageHistory(
            base_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

        print("✓ RAG 组件初始化完成（已启用意图理解 + 查询重写 + 两阶段检索）！")
        return self._conversational_chain

    @property
    def chain(self):
        """获取 conversational chain 实例"""
        if self._conversational_chain is None:
            self.initialize()
        return self._conversational_chain


# 全局 RAG Service 实例（启动时初始化）
_rag_service = RAGService()
_rag_service.initialize()


# ============================================================
# 💬 对外接口
# ============================================================
def chat(question: str, session_id: str = "default") -> str:
    """
    同步问答接口（非流式）

    参数：
        question: 用户问题
        session_id: 会话 ID，用于区分不同用户
    """
    return _rag_service.chain.invoke(
        {"input": question}, config={"configurable": {"session_id": session_id}}
    )


async def chat_stream(question: str, session_id: str) -> AsyncIterator[str]:
    """
    异步流式问答接口（优化版：意图理解 + 查询重写 + RAG 检索）

    流程：
    1. 意图理解（qwen-turbo + 最近1-2轮历史）
    2. 如果是闲聊 → qwen-turbo 直接回复
    3. 如果需要检索 → 智能判断是否需要查询重写 → RAG 检索（qwen-max）

    参数：
        question: 用户问题
        session_id: 会话 ID

    返回：
        异步生成器，逐块返回大模型的回复
    """
    print(f"\n{'='*80}")
    print(f"📨 收到用户问题：{question} | 会话ID：{session_id}")
    print(f"{'='*80}")

    # 步骤1：意图理解
    intent = await classify_intent(question, session_id)

    if intent == "chitchat":
        # 闲聊路径：直接用 qwen-turbo 回复
        print("🎯 路由决策：闲聊模式（不触发检索）")
        async for chunk in chitchat_stream(question, session_id):
            yield chunk

    else:
        # 检索路径：查询重写 → RAG 检索
        print("🎯 路由决策：检索模式（触发 RAG）")

        # 步骤2：查询重写（智能触发）
        rewritten_question = await rewrite_query(question, session_id)

        # 步骤3：RAG 检索 + 生成
        async for chunk in _rag_service.chain.astream(
            {"input": rewritten_question},
            config={"configurable": {"session_id": session_id}},
        ):
            if chunk:
                yield chunk

    print(f"{'='*80}")
    print(f"✅ 问题处理完成")
    print(f"{'='*80}\n")
