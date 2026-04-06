# -*- coding: utf-8 -*-
"""RAG 服务模块 - 支持多轮对话记忆、流式输出与 Elasticsearch 混合检索"""
from dotenv import load_dotenv

from app.services.topk import TopKRetriever
from app.core.redis_config import RedisConfig
from app.core.es_config import ESConfig

load_dotenv()  # 加载 .env 文件中的环境变量

from typing import AsyncIterator, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_classic.retrievers import EnsembleRetriever  # ⭐ 混合检索聚合器
from langchain_classic.retrievers import ContextualCompressionRetriever  # ⭐ 压缩检索器（用于重排）
from langchain_community.document_compressors import FlashrankRerank  # ⭐ Flashrank 重排模型
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from elasticsearch import Elasticsearch

# ============================================================
# 📦 会话存储（Redis 版）
# ============================================================
# 初始化 Redis 配置（从环境变量或 .env 文件读取）
redis_config = RedisConfig()


def get_session_history(session_id: str) -> RedisChatMessageHistory:
    """
    根据 session_id 获取或创建 Redis 聊天历史记录
    
    架构说明：
    - 使用 RedisChatMessageHistory 替代内存版的 ChatMessageHistory
    - 支持分布式部署：多个服务实例共享同一个 Redis
    - 支持持久化：服务重启后会话历史不丢失
    - 支持 TTL：可以设置会话过期时间（通过 Redis 的 EXPIRE 命令）
    
    类比 Java：
    - 内存版 = ConcurrentHashMap（单机，重启丢失）
    - Redis 版 = Redis + Spring Session（分布式，持久化）
    
    参数：
        session_id: 会话 ID，用于区分不同用户的对话历史
        一般来说：前端自己生成session_id(UUID) + 后端获取的user_id 组合在一起 = 上面所说的session_id
        
    返回：
        RedisChatMessageHistory 实例
    """
    return RedisChatMessageHistory(
        session_id=session_id,
        url=redis_config.get_url(),
        key_prefix="chat_history:",  # Redis key 前缀，实际存储为 chat_history:{session_id}
        ttl=3600  # 会话过期时间（秒），1小时后自动清理，可根据需求调整 主要看用户量，使用量，多就调小
    )


# ============================================================
# 🔧 工具函数
# ============================================================
def format_docs(docs):
    """将检索到的文档列表格式化为字符串"""
    if not docs:
        return "未在知识库中找到相关信息"
    # 过滤掉空文档    非0非空即为True
    valid_docs = [doc.page_content for doc in docs if doc.page_content and doc.page_content.strip()]
    if not valid_docs:
        return "未在知识库中找到相关信息"
    
    # ⭐ 打印检索到的上下文，便于观察重排效果
    print("\n" + "="*80)
    print("🎯 经过 Reranker 重排后，最终发给大模型的上下文（Top-3）：")
    print("="*80)
    for i, doc in enumerate(docs, 1):
        print(f"\n📄 文档片段 {i} (重排后排名):")
        print(f"内容: {doc.page_content[:300]}...")  # 打印前300字符
        # 如果有元数据，也打印出来（可以看到来源和重排分数）
        if hasattr(doc, 'metadata') and doc.metadata:
            print(f"元数据: {doc.metadata}")
    print("="*80)
    print("💡 提示：这些是从召回的 6 个候选中，经过 Flashrank 精排后的最优结果")
    print("="*80 + "\n")
    
    return "\n\n".join(valid_docs)


# ============================================================
# 🔍 Elasticsearch 检索器封装
# ============================================================
class ESVectorRetriever(BaseRetriever):
    """
    Elasticsearch 向量检索器（kNN）
    
    架构说明：
    - 使用 ES 的 kNN 搜索功能进行向量检索
    - 支持余弦相似度计算
    - 返回 LangChain 标准的 Document 对象
    """
    es_client: Elasticsearch
    index_name: str
    embeddings: DashScopeEmbeddings
    k: int = 10
    
    class Config:
        arbitrary_types_allowed = True
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """执行向量检索"""
        # 1. 将查询文本向量化
        query_vector = self.embeddings.embed_query(query)
        
        # 2. 构建 kNN 查询
        knn_query = {
            "knn": {
                "field": "vector",
                "query_vector": query_vector,
                "k": self.k,
                "num_candidates": self.k * 2  # 候选数量（建议是 k 的 2-10 倍）
            },
            "_source": ["text", "metadata"]
        }
        
        # 3. 执行查询
        response = self.es_client.search(index=self.index_name, body=knn_query)
        
        # 4. 转换为 Document 对象
        documents = []
        for hit in response['hits']['hits']:
            doc = Document(
                page_content=hit['_source']['text'],
                metadata={
                    **hit['_source'].get('metadata', {}),
                    'score': hit['_score'],
                    'retriever': 'es_vector'
                }
            )
            documents.append(doc)
        
        return documents


class ESBM25Retriever(BaseRetriever):
    """
    Elasticsearch BM25 检索器（全文检索）
    
    架构说明：
    - 使用 ES 的 match 查询进行 BM25 全文检索
    - 支持中文分词（需要安装 IK 分词器插件）
    - 返回 LangChain 标准的 Document 对象
    """
    es_client: Elasticsearch
    index_name: str
    k: int = 10
    
    class Config:
        arbitrary_types_allowed = True
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """执行 BM25 检索"""
        # 1. 构建 BM25 查询
        bm25_query = {
            "query": {
                "match": {
                    "text": {
                        "query": query,
                        "analyzer": "ik_smart"  # 使用 IK 分词器
                    }
                }
            },
            "size": self.k,
            "_source": ["text", "metadata"]
        }
        
        # 2. 执行查询
        response = self.es_client.search(index=self.index_name, body=bm25_query)
        
        # 3. 转换为 Document 对象
        documents = []
        for hit in response['hits']['hits']:
            doc = Document(
                page_content=hit['_source']['text'],
                metadata={
                    **hit['_source'].get('metadata', {}),
                    'score': hit['_score'],
                    'retriever': 'es_bm25'
                }
            )
            documents.append(doc)
        
        return documents

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
            
        print("正在初始化 RAG 组件（支持多轮对话 + Elasticsearch 混合检索）...")

        # 1. Embedding 模型
        embeddings = DashScopeEmbeddings(model="text-embedding-v4")

        # 2. LLM 模型
        llm = ChatOpenAI(model="qwen-max", temperature=0.7)

        # 3. 初始化 Elasticsearch 客户端
        es_config = ESConfig()
        es_client = Elasticsearch(**es_config.get_connection_params())
        
        # 测试连接
        if not es_client.ping():
            raise ConnectionError(
                f"无法连接到 Elasticsearch ({es_config.get_url()})，"
                "请检查 Docker 容器是否启动，以及 .env 配置是否正确"
            )
        print(f"✓ 成功连接到 Elasticsearch：{es_config.get_url()}")

        # 4. ⭐ 向量检索器（ES kNN）
        vector_retriever = ESVectorRetriever(
            es_client=es_client,
            index_name=es_config.index_name,
            embeddings=embeddings,
            k=10  # 召回 10 个候选
        )
        print("✓ ES 向量检索器（kNN）初始化完成，召回数：10")

        # 5. ⭐ 关键字检索器（ES BM25）
        bm25_retriever = ESBM25Retriever(
            es_client=es_client,
            index_name=es_config.index_name,
            k=10  # 召回 10 个候选
        )
        print("✓ ES 关键字检索器（BM25）初始化完成，召回数：10")

        # 6. ⭐ 混合检索器（EnsembleRetriever）
        # 【架构说明】类比 Java 的策略模式聚合器：
        # - retrievers: List<Retriever> 多个检索策略
        # - weights: 各策略的权重（使用 RRF 算法融合排序）
        # - 0.5 + 0.5 = 向量检索和关键字检索各占 50%
        base_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.5, 0.5]
        )
        print("✓ 混合检索器（Ensemble）初始化完成，权重：向量 50% + BM25 50%")

        # 7. ⭐⭐⭐ 重排序器（Reranker）—— 两阶段检索的核心！
        # 【架构说明】类比 Java 的装饰器模式：
        # - base_retriever：第一阶段召回（粗筛，快速但不精准）
        # - compressor：第二阶段精排（细筛，慢但精准）
        # - ContextualCompressionRetriever：将两者组合
        #
        # 为什么不直接用 Reranker 扫全库？
        # - 假设知识库有 10 万个文档片段
        # - Reranker 是 Cross-Encoder 模型，对每个片段都要做深度语义计算
        # - 如果直接扫全库，耗时可能达到分钟级
        # - 而先用向量检索 + BM25 快速筛到 20 个候选，先用RRF加权融合排序后筛出6个，再用 Reranker 精排这 6 个，只需要秒级
        #
        # 类比 Java：
        # - 第一阶段 = MySQL 的 WHERE + LIMIT（快速筛选）
        # - 第二阶段 = Java 的 Comparator.comparing()（精细排序）
        
        # 先导入并初始化 Flashrank 的 Ranker
        from flashrank import Ranker
        ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")  # 轻量级模型，无需 GPU
        
        compressor = FlashrankRerank(
            client=ranker,  # 传入 Ranker 实例
            top_n=3  # 最终只取最精准的前 3 个
        )

        # ⭐ 预过滤器（TopKRetriever） k的值代表RRF加权融合排序后的文档，你要保留几个文档交给重排序
        pre_filtered_retriever = TopKRetriever(retriever=base_retriever, k=6)

        final_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=pre_filtered_retriever
        )
        print("✓ 重排序器（Flashrank Reranker）初始化完成，精排数：3")
        print("✓ 两阶段检索架构已启用：召回（10+10=20）→ 过滤(Top-6) → 重排（Top-3）")

        # 8. Prompt 模板（支持历史对话）
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "你是一个专业的企业 HR 助手，负责回答员工关于公司培训制度的问题。\n"
             "请根据以下检索到的知识库内容来回答用户的问题。\n"
             "如果知识库中没有相关信息，请明确告知用户\"抱歉，我在知识库中未找到相关信息\"。\n\n"
             "知识库内容：\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        # 9. 基础 RAG Chain（⭐ 使用重排后的检索器）
        base_chain = (
            {
                "context": lambda x: format_docs(final_retriever.invoke(x["input"])),
                "input": lambda x: x["input"],
                "chat_history": lambda x: x.get("chat_history", [])
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        # 10. 包装为带历史记录的 Chain
        self._conversational_chain = RunnableWithMessageHistory(
            base_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history"
        )

        print("✓ RAG 组件初始化完成（已启用会话记忆 + Elasticsearch 两阶段检索）！")
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