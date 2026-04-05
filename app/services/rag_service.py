# -*- coding: utf-8 -*-
"""RAG 服务模块 - 支持多轮对话记忆、流式输出与混合检索"""
from dotenv import load_dotenv
load_dotenv()  # 加载 .env 文件中的环境变量

from typing import AsyncIterator
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.retrievers import BM25Retriever  # ⭐ 关键字检索
from langchain_classic.retrievers import EnsembleRetriever  # ⭐ 混合检索聚合器
from langchain_text_splitters import RecursiveCharacterTextSplitter  # ⭐ 文本切片

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
    
    # ⭐ 打印检索到的上下文，便于观察混合检索效果
    print("\n" + "="*60)
    print("🔍 检索到的上下文（Context）：")
    print("="*60)
    for i, doc in enumerate(docs, 1):
        print(f"\n📄 文档片段 {i}:")
        print(f"内容: {doc.page_content[:200]}...")  # 只打印前200字符
        # 如果有元数据，也打印出来（可以看到来源）
        if hasattr(doc, 'metadata') and doc.metadata:
            print(f"元数据: {doc.metadata}")
    print("="*60 + "\n")
    
    return "\n\n".join(valid_docs)


def load_and_split_documents(file_path: str):
    """
    加载并切片文档（用于 BM25 检索器初始化）
    
    ⭐ 架构说明（给 Java 开发者）：
    在真正的企业级生产环境中，关键字检索通常由 ElasticSearch 承担：
    - ES 会在索引时就完成分词和倒排索引构建
    - 查询时直接调用 ES 的 REST API
    
    但为了当前项目的轻量化 MVP 演示，我们采用内存方案：
    - 启动时读取 data/training_doc.txt
    - 使用 RecursiveCharacterTextSplitter 切片
    - 在内存中初始化 BM25Retriever（基于 rank_bm25 库）
    
    类比 Java：这就像在 Spring Boot 启动时，用 @PostConstruct 
    加载配置文件到内存缓存中
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 使用与向量库相同的切片策略，保证一致性
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        )
        
        # 切片并转换为 Document 对象
        from langchain_core.documents import Document
        chunks = text_splitter.split_text(content)
        documents = [Document(page_content=chunk) for chunk in chunks if chunk.strip()]
        
        print(f"✓ 已加载并切片文档：{file_path}，共 {len(documents)} 个片段")
        return documents
    
    except Exception as e:
        print(f"⚠️ 加载文档失败：{e}")
        return []

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
            
        print("正在初始化 RAG 组件（支持多轮对话 + 混合检索）...")

        # 1. Embedding 模型
        embeddings = DashScopeEmbeddings(model="text-embedding-v4")

        # 2. LLM 模型
        llm = ChatOpenAI(model="qwen-max", temperature=0.7)

        # 3. 向量检索器（Chroma）
        vector_store = Chroma(
            persist_directory="vector_store",
            embedding_function=embeddings
        )
        vector_retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 2}
        )
        print("✓ 向量检索器（Chroma）初始化完成")

        # 4. ⭐ 关键字检索器（BM25）
        # 加载并切片训练文档
        documents = load_and_split_documents("data/training_doc.txt")
        if not documents:
            print("⚠️ 警告：未能加载文档，BM25 检索器将不可用")
            # 降级为仅使用向量检索
            ensemble_retriever = vector_retriever
        else:
            bm25_retriever = BM25Retriever.from_documents(
                documents,
                k=2  # 返回 Top-2 结果
            )
            print("✓ 关键字检索器（BM25）初始化完成")

            # 5. ⭐ 混合检索器（EnsembleRetriever）
            # 【架构说明】类比 Java 的策略模式聚合器：
            # - retrievers: List<Retriever> 多个检索策略
            # - weights: 各策略的权重（使用 RRF 算法融合排序）
            # - 0.5 + 0.5 = 向量检索和关键字检索各占 50%
            ensemble_retriever = EnsembleRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                weights=[0.5, 0.5]
            )
            print("✓ 混合检索器（Ensemble）初始化完成，权重：向量 50% + BM25 50%")

        # 6. Prompt 模板（支持历史对话）
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "你是一个专业的企业 HR 助手，负责回答员工关于公司培训制度的问题。\n"
             "请根据以下检索到的知识库内容来回答用户的问题。\n"
             "如果知识库中没有相关信息，请明确告知用户\"抱歉，我在知识库中未找到相关信息\"。\n\n"
             "知识库内容：\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        # 7. 基础 RAG Chain（⭐ 使用混合检索器）
        base_chain = (
            {
                "context": lambda x: format_docs(ensemble_retriever.invoke(x["input"])),
                "input": lambda x: x["input"],
                "chat_history": lambda x: x.get("chat_history", [])
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        # 8. 包装为带历史记录的 Chain
        self._conversational_chain = RunnableWithMessageHistory(
            base_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history"
        )

        print("✓ RAG 组件初始化完成（已启用会话记忆 + 混合检索）！")
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