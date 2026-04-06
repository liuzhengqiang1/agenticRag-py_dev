# -*- coding: utf-8 -*-
"""RAG 服务模块 - 支持多轮对话记忆、流式输出与 Elasticsearch 混合检索"""
from dotenv import load_dotenv

from app.services.topk import TopKRetriever
from app.core.redis_config import RedisConfig
from app.core.es_config import ESConfig

load_dotenv()  # 加载 .env 文件中的环境变量

from typing import AsyncIterator, List
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_classic.retrievers import EnsembleRetriever  # ⭐ 混合检索聚合器
from langchain_classic.retrievers import (
    ContextualCompressionRetriever,
)  # ⭐ 压缩检索器（用于重排）
from langchain_community.document_compressors import (
    FlashrankRerank,
)  # ⭐ Flashrank 重排模型
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
        ttl=3600,  # 会话过期时间（秒），1小时后自动清理，可根据需求调整 主要看用户量，使用量，多就调小
    )


# ============================================================
# 🔧 工具函数
# ============================================================
def format_docs(docs):
    """将检索到的文档列表格式化为字符串"""
    if not docs:
        return "未在知识库中找到相关信息"
    # 过滤掉空文档    非0非空即为True
    valid_docs = [
        doc.page_content
        for doc in docs
        if doc.page_content and doc.page_content.strip()
    ]
    if not valid_docs:
        return "未在知识库中找到相关信息"

    # ⭐ 打印检索到的上下文，便于观察重排效果
    print("\n" + "=" * 80)
    print("🎯 经过 Reranker 重排后，最终发给大模型的上下文（Top-3）：")
    print("=" * 80)
    for i, doc in enumerate(docs, 1):
        print(f"\n📄 文档片段 {i} (重排后排名):")
        print(f"内容: {doc.page_content[:300]}...")  # 打印前300字符
        # 如果有元数据，也打印出来（可以看到来源和重排分数）
        if hasattr(doc, "metadata") and doc.metadata:
            print(f"元数据: {doc.metadata}")
    print("=" * 80)
    print("💡 提示：这些是从召回的 6 个候选中，经过 Flashrank 精排后的最优结果")
    print("=" * 80 + "\n")

    return "\n\n".join(valid_docs)


def get_recent_history(session_id: str, max_rounds: int = 3) -> List[tuple]:
    """
    获取最近 N 轮对话历史（滑动窗口）

    参数：
        session_id: 会话 ID
        max_rounds: 最多保留几轮对话（1轮 = 1个用户消息 + 1个助手回复）

    返回：
        [(user_msg, assistant_msg), ...] 格式的对话历史
    """
    history = get_session_history(session_id)
    messages = history.messages

    # 提取最近的对话轮次（user + assistant 成对）
    recent_pairs = []
    i = len(messages) - 1

    while i >= 1 and len(recent_pairs) < max_rounds:
        # 从后往前找成对的 user + assistant
        if messages[i].type == "ai" and messages[i - 1].type == "human":
            recent_pairs.insert(0, (messages[i - 1].content, messages[i].content))
            i -= 2
        else:
            i -= 1

    return recent_pairs


def format_history_for_prompt(history_pairs: List[tuple]) -> str:
    """将对话历史格式化为 Prompt 字符串"""
    if not history_pairs:
        return "无历史对话"

    formatted = []
    for i, (user_msg, assistant_msg) in enumerate(history_pairs, 1):
        formatted.append(f"第{i}轮 - 用户: {user_msg}")
        formatted.append(f"第{i}轮 - 助手: {assistant_msg}")

    return "\n".join(formatted)


# ============================================================
# 🧠 意图理解与查询重写（智能路由层）
# ============================================================
# 初始化轻量级模型（用于意图分类、查询重写和闲聊）
_intent_llm = ChatOpenAI(model="qwen-turbo", temperature=0)
_rewrite_llm = ChatOpenAI(model="qwen-turbo", temperature=0)
_chitchat_llm = ChatOpenAI(model="qwen-turbo", temperature=0.7)


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
        intent = result.get("intent", "retrieval")  # 默认走检索（保守策略）
        print(f"🔍 意图分类结果：{intent} | 问题：{question}")
        return intent
    except Exception as e:
        print(f"⚠️ 意图分类失败，默认走检索路径：{e}")
        return "retrieval"  # 失败时默认走检索


def need_query_rewrite(question: str, history_pairs: List[tuple]) -> bool:
    """
    判断是否需要查询重写（智能触发）

    触发条件：
    1. 问题中包含指代词（它、这个、那个、他、她等）
    2. 问题过短且有历史对话（可能是追问）
    3. 问题缺少主语（通过简单规则判断）

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

    # # 规则3：缺少常见主语（简单规则）
    # common_subjects = [
    #     "培训",
    #     "课程",
    #     "费用",
    #     "申请",
    #     "考核",
    #     "证书",
    #     "公司",
    #     "员工",
    #     "我",
    # ]
    # if not any(s in question for s in common_subjects) and history_pairs:
    #     print(f"✏️ 触发查询重写：可能缺少主语 | 问题：{question}")
    #     return True

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
        return question  # 不需要重写，直接返回原问题

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
                "num_candidates": self.k * 2,  # 候选数量（建议是 k 的 2-10 倍）
            },
            "_source": ["text", "metadata"],
        }

        # 3. 执行查询
        response = self.es_client.search(index=self.index_name, body=knn_query)

        # 4. 转换为 Document 对象
        documents = []
        for hit in response["hits"]["hits"]:
            doc = Document(
                page_content=hit["_source"]["text"],
                metadata={
                    **hit["_source"].get("metadata", {}),
                    "score": hit["_score"],
                    "retriever": "es_vector",
                },
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
                    "text": {"query": query, "analyzer": "ik_smart"}  # 使用 IK 分词器
                }
            },
            "size": self.k,
            "_source": ["text", "metadata"],
        }

        # 2. 执行查询
        response = self.es_client.search(index=self.index_name, body=bm25_query)

        # 3. 转换为 Document 对象
        documents = []
        for hit in response["hits"]["hits"]:
            doc = Document(
                page_content=hit["_source"]["text"],
                metadata={
                    **hit["_source"].get("metadata", {}),
                    "score": hit["_score"],
                    "retriever": "es_bm25",
                },
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

        print(
            "正在初始化 RAG 组件（支持意图理解 + 查询重写 + Elasticsearch 混合检索）..."
        )

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
            k=10,  # 召回 10 个候选
        )
        print("✓ ES 向量检索器（kNN）初始化完成，召回数：10")

        # 5. ⭐ 关键字检索器（ES BM25）
        bm25_retriever = ESBM25Retriever(
            es_client=es_client, index_name=es_config.index_name, k=10  # 召回 10 个候选
        )
        print("✓ ES 关键字检索器（BM25）初始化完成，召回数：10")

        # 6. ⭐ 混合检索器（EnsembleRetriever）
        # 【架构说明】类比 Java 的策略模式聚合器：
        # - retrievers: List<Retriever> 多个检索策略
        # - weights: 各策略的权重（使用 RRF 算法融合排序）
        # - 0.5 + 0.5 = 向量检索和关键字检索各占 50%
        base_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever], weights=[0.5, 0.5]
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
            client=ranker, top_n=3  # 传入 Ranker 实例  # 最终只取最精准的前 3 个
        )

        # ⭐ 预过滤器（TopKRetriever） k的值代表RRF加权融合排序后的文档，你要保留几个文档交给重排序
        pre_filtered_retriever = TopKRetriever(retriever=base_retriever, k=6)

        final_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=pre_filtered_retriever
        )
        print("✓ 重排序器（Flashrank Reranker）初始化完成，精排数：3")
        print("✓ 两阶段检索架构已启用：召回（10+10=20）→ 过滤(Top-6) → 重排（Top-3）")

        # 8. Prompt 模板（支持历史对话）
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

        # 9. 基础 RAG Chain（⭐ 使用重排后的检索器）
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

        # 10. 包装为带历史记录的 Chain
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
        {"input": question}, config={"configurable": {"session_id": session_id}}
    )


# ============================================================
# 🌊 异步流式问答接口（核心功能 - 带智能路由）
# ============================================================
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

    # ⭐ 步骤1：意图理解
    intent = await classify_intent(question, session_id)

    if intent == "chitchat":
        # 🗨️ 闲聊路径：直接用 qwen-turbo 回复
        print("🎯 路由决策：闲聊模式（不触发检索）")
        async for chunk in chitchat_stream(question, session_id):
            yield chunk

    else:
        # 🔍 检索路径：查询重写 → RAG 检索
        print("🎯 路由决策：检索模式（触发 RAG）")

        # ⭐ 步骤2：查询重写（智能触发）
        rewritten_question = await rewrite_query(question, session_id)

        # ⭐ 步骤3：RAG 检索 + 生成
        async for chunk in _rag_service.chain.astream(
            {"input": rewritten_question},
            config={"configurable": {"session_id": session_id}},
        ):
            if chunk:
                yield chunk

    print(f"{'='*80}")
    print(f"✅ 问题处理完成")
    print(f"{'='*80}\n")
