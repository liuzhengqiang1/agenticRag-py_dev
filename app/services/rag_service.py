# -*- coding: utf-8 -*-
"""RAG 服务模块 - 支持多轮对话记忆与流式输出"""
import os
from dotenv import load_dotenv
load_dotenv()  # 加载 .env 文件中的环境变量

from typing import AsyncIterator, List
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.embeddings import DashScopeEmbeddings

# ============================================================
# 📦 会话存储（内存版）
# ============================================================
# 在真实企业级生产环境中，这里应该替换为 RedisChatMessageHistory
# 以支持分布式部署和持久化存储
store = {}


def get_session_history(session_id: str) -> ChatMessageHistory:
    """
    根据 session_id 获取或创建聊天历史记录
    
    类比 Java：这就像一个 ConcurrentHashMap<String, ChatMessageHistory>
    如果 key 不存在，就创建一个新的 ChatMessageHistory 对象
    """
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# ============================================================
# 🔧 工具函数
# ============================================================
def format_docs(docs):
    """将检索到的文档列表格式化为字符串"""
    if not docs:
        return "未在知识库中找到相关信息"
    # 过滤掉空文档
    valid_docs = [doc.page_content for doc in docs if doc.page_content and doc.page_content.strip()]
    if not valid_docs:
        return "未在知识库中找到相关信息"
    return "\n\n".join(valid_docs)

# ============================================================
# 🚀 初始化 RAG 组件（单例模式）
# ============================================================
class RAGService:
    """RAG 服务单例类"""
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
            
        print("正在初始化 RAG 组件（支持多轮对话）...")

        # 1. Embedding 模型
        embeddings = DashScopeEmbeddings(model="text-embedding-v4")

        # 2. LLM 模型
        llm = ChatOpenAI(model="qwen-max", temperature=0.7)

        # 3. 向量数据库
        vector_store = Chroma(
            persist_directory="vector_store",
            embedding_function=embeddings
        )
        # ⭐ 修复：使用 search_type="similarity" 确保异步流式调用时正确工作
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 2}
        )

        # 4. Prompt 模板（⭐ 关键改动：加入 chat_history）
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "你是一个专业的企业 HR 助手，负责回答员工关于公司培训制度的问题。\n"
             "请根据以下检索到的知识库内容来回答用户的问题。\n"
             "如果知识库中没有相关信息，请明确告知用户\"抱歉，我在知识库中未找到相关信息\"。\n\n"
             "知识库内容：\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),  # ⭐ 历史消息占位符
            ("human", "{input}")  # ⭐ 注意：这里改为 input，与后面的配置对应
        ])

        # 5. 基础 RAG Chain
        # ⭐ 修复：使用 lambda 函数从输入中提取 input 字段 避免RunnablePassthrough直接传递整个字典
        base_chain = (
            {
                "context": lambda x: format_docs(retriever.invoke(x["input"])),
                "input": lambda x: x["input"],
                "chat_history": lambda x: x.get("chat_history", [])
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        # 6. ⭐ 包装为带历史记录的 Chain
        self._conversational_chain = RunnableWithMessageHistory(
            base_chain,
            get_session_history,
            input_messages_key="input",      # 用户输入的 key
            history_messages_key="chat_history"  # 历史记录的 key
        )

        print("✓ RAG 组件初始化完成（已启用会话记忆）！")
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
# 💬 同步问答接口（保留兼容性）
# ============================================================
def chat(question: str, session_id: str = "default") -> str:
    """
    同步问答接口（非流式）
    
    参数：
        question: 用户问题
        session_id: 会话 ID，用于区分不同用户
    """
    return _rag_service.chain.invoke(
        {"input": question},
        config={"configurable": {"session_id": session_id}}
    )


# ============================================================
# 🌊 异步流式问答接口（核心功能）
# ============================================================
async def chat_stream(question: str, session_id: str) -> AsyncIterator[str]:
    """
    异步流式问答接口
    
    ⭐ 关键概念解释（给 Java 开发者）：
    1. async def：声明这是一个异步函数（类似 Java 的 CompletableFuture）
    2. AsyncIterator[str]：返回一个异步迭代器（类似 Java 的 Stream<String>）
    3. yield：生成器关键字，每次返回一个片段（类似 Java 的 Iterator.next()）
    4. async for：异步遍历（类似 Java 的 stream().forEach()）
    
    参数：
        question: 用户问题
        session_id: 会话 ID
        
    返回：
        异步生成器，逐块返回大模型的回复
    """
    # ⭐ 使用 astream() 方法进行异步流式调用
    async for chunk in _rag_service.chain.astream(
        {"input": question},
        config={"configurable": {"session_id": session_id}}
    ):
        # chunk 是大模型返回的文本片段
        if chunk:
            yield chunk