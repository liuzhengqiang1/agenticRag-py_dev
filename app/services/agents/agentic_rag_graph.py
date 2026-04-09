"""
Agentic RAG 状态机：LangGraph 编排所有工具

【核心理念】
把 RAG 检索器包装成一个 Tool，和业务 API（天气、订单）、互联网搜索一起
扔给 LangGraph，打造一个"无所不能"的企业级智能体。

【架构设计】
1. 查询重写节点：优化用户查询（补全上下文、替换指代词）
2. Agent 节点：决策调用哪些工具
3. Tool 节点：执行具体任务
4. 记忆管理：优先使用 Redis 持久化存储，失败时降级到内存（带 LRU 限制）

【单例模式】
项目启动时调用 init_agentic_rag() 初始化，之后直接使用 get_agentic_rag_graph() 获取实例
"""

import asyncio
from typing import Literal

from cachetools import LRUCache
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
import redis.asyncio as redis_async

from app.core.redis_config import RedisConfig
from app.services.tools import (
    search_knowledge_base,
    get_current_weather,
    query_database_order,
    get_web_search_tool,
)
from app.services.llm.query_rewriter import (
    need_query_rewrite_from_messages,
    rewrite_query_from_messages_async,
)


class AgenticRAGState(MessagesState):
    """
    扩展的状态定义，支持查询重写

    属性：
        messages: 对话消息列表（继承自 MessagesState）
        rewritten_query: 重写后的查询（用于优化检索）
    """

    rewritten_query: str = ""


class AgenticRAGGraph:
    """
    Agentic RAG 状态图单例类

    使用方式：
        # 项目启动时初始化
        await AgenticRAGGraph.initialize()

        # 之后获取实例使用
        graph = AgenticRAGGraph.get_instance()
    """

    _instance = None
    _initialized = False
    _using_redis = False

    def __init__(self):
        """私有构造函数，防止外部实例化"""
        if AgenticRAGGraph._instance is not None:
            raise RuntimeError("请使用 AgenticRAGGraph.get_instance() 获取实例")
        AgenticRAGGraph._instance = self

    @classmethod
    def get_instance(cls):
        """获取单例实例"""
        if cls._instance is None:
            raise RuntimeError("请先调用 AgenticRAGGraph.initialize() 初始化")
        return cls._instance

    @classmethod
    async def initialize(cls):
        """
        初始化 Agentic RAG 系统（项目启动时调用一次）

        返回:
            编译后的 LangGraph 状态图
        """
        if cls._initialized:
            return cls._instance

        print("=" * 80)
        print("🚀 Agentic RAG 系统初始化中...")
        print("=" * 80)

        # 创建实例
        instance = cls()

        # 1. 整合全量工具箱
        tools = [
            search_knowledge_base,  # RAG 知识库检索
            get_current_weather,  # 天气查询
            query_database_order,  # 订单查询
        ]

        # 如果 Tavily 可用，添加到工具列表
        web_search_tool = get_web_search_tool()
        if web_search_tool:
            tools.append(web_search_tool)

        print(f"✓ 已注册 {len(tools)} 个工具：{[t.name for t in tools]}")

        # 2. 初始化大模型并绑定工具
        instance._llm = ChatOpenAI(model="qwen-turbo", temperature=0)
        instance._llm_with_tools = instance._llm.bind_tools(tools)
        instance._tools = tools
        print("✓ 大模型已绑定所有工具")

        # 3. 获取 checkpointer
        instance._checkpointer = cls._create_checkpointer()

        # 4. 如果使用 Redis，初始化表结构
        if cls._using_redis:
            try:
                await instance._checkpointer.asetup()
                print("✓ Redis 表结构初始化完成")
            except Exception as e:
                print(f"⚠️  Redis 表结构初始化失败（可能已存在）：{e}")

        # 5. 构建状态图
        instance._graph = cls._build_graph(instance)

        cls._initialized = True

        print("✓ LangGraph 状态机构建完成")
        print("  - 查询重写节点：优化用户查询")
        print("  - Agent 节点：决策工具调用")
        print("  - Tool 节点：执行具体任务")
        print(
            f"  - 记忆管理：{'Redis 异步持久化存储' if cls._using_redis else 'MemorySaver (LRU 限制)'}"
        )
        print("=" * 80)
        print()

        return instance

    @classmethod
    def _create_checkpointer(cls):
        """
        创建检查点存储器（优先 Redis，失败时降级到内存）

        返回:
            检查点存储器实例（异步版本）
        """
        try:
            # 尝试使用 Redis（异步客户端）
            redis_config = RedisConfig(decode_responses=False)  # LangGraph 需要字节模式
            redis_client = redis_async.Redis(**redis_config.get_connection_kwargs())

            # 导入 AsyncRedisSaver
            from langgraph.checkpoint.redis.aio import AsyncRedisSaver

            cls._using_redis = True
            print("✓ Redis 连接成功，使用异步持久化存储")
            return AsyncRedisSaver(redis_client=redis_client)

        except Exception as e:
            # Redis 连接失败，降级到内存存储
            print(f"⚠️  Redis 连接失败：{e}")
            print("⚠️  降级到内存存储（LRU 限制：最多 100 个会话）")

            cls._using_redis = False

            # 使用 LRU 缓存包装 MemorySaver，防止 OOM
            class LRUMemorySaver(MemorySaver):
                """带 LRU 限制的内存存储器"""

                def __init__(self, maxsize=100):
                    super().__init__()
                    self.storage = LRUCache(maxsize=maxsize)

            return LRUMemorySaver(maxsize=100)

    @classmethod
    def _build_graph(cls, instance):
        """构建状态图"""

        # 定义查询重写节点
        async def query_rewrite_node(state: AgenticRAGState):
            """
            查询重写节点：在调用工具前优化查询

            功能：
            - 补全缺失的主语、宾语
            - 替换指代词为具体内容
            - 将追问改写为独立问题
            """
            messages = state["messages"]
            last_message = messages[-1]

            # 只处理用户消息
            if not isinstance(last_message, HumanMessage):
                return {"rewritten_query": ""}

            user_question = last_message.content

            # 判断是否需要重写
            if need_query_rewrite_from_messages(user_question, messages[:-1]):
                rewritten = await rewrite_query_from_messages_async(
                    user_question, messages[:-1]
                )
                return {"rewritten_query": rewritten}

            return {"rewritten_query": user_question}

        # 定义 Agent 节点
        def agent_node(state: AgenticRAGState):
            """
            Agent 决策节点：分析用户问题，决定调用哪些工具

            如果有重写后的查询，使用重写后的查询进行决策
            """
            messages = state["messages"]
            rewritten_query = state.get("rewritten_query", "")

            # 如果有重写查询，替换最后一条用户消息
            if rewritten_query and isinstance(messages[-1], HumanMessage):
                modified_messages = messages[:-1] + [
                    HumanMessage(content=rewritten_query)
                ]
                response = instance._llm_with_tools.invoke(modified_messages)
            else:
                response = instance._llm_with_tools.invoke(messages)

            return {"messages": [response]}

        # 定义路由逻辑
        def should_continue(state: AgenticRAGState) -> Literal["tools", END]:
            """路由函数：判断是否需要继续调用工具"""
            messages = state["messages"]
            last_message = messages[-1]

            # 如果 LLM 决定调用工具，则路由到 tools 节点
            if last_message.tool_calls:
                return "tools"

            # 否则结束流程
            return END

        # 构建状态图
        workflow = StateGraph(AgenticRAGState)

        workflow.add_node("query_rewrite", query_rewrite_node)
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", ToolNode(instance._tools))

        workflow.add_edge(START, "query_rewrite")
        workflow.add_edge("query_rewrite", "agent")
        workflow.add_conditional_edges("agent", should_continue)
        workflow.add_edge("tools", "agent")

        return workflow.compile(checkpointer=instance._checkpointer)

    @property
    def graph(self):
        """获取编译后的状态图"""
        return self._graph

    @property
    def using_redis(self):
        """是否使用 Redis"""
        return self._using_redis


# 便捷函数
def get_agentic_rag_graph():
    """获取 Agentic RAG 状态图实例"""
    return AgenticRAGGraph.get_instance()


async def init_agentic_rag():
    """初始化 Agentic RAG 系统（项目启动时调用）"""
    return await AgenticRAGGraph.initialize()


def run_agentic_rag(user_question: str, session_id: str = "default") -> str:
    """
    运行 Agentic RAG（同步版本）

    参数:
        user_question: 用户问题
        session_id: 会话 ID（用于多轮对话记忆）

    返回:
        最终回答
    """
    print(f"👤 用户问题：{user_question}")
    print(f"🔑 会话 ID：{session_id}")
    print("=" * 80)

    # 获取状态图
    graph = get_agentic_rag_graph().graph

    # 配置会话
    config = {"configurable": {"thread_id": session_id}}

    # 异步调用
    async def _run():
        return await graph.ainvoke(
            {"messages": [HumanMessage(content=user_question)]}, config=config
        )

    # 在事件循环中运行
    try:
        loop = asyncio.get_running_loop()
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, _run())
            result = future.result()
    except RuntimeError:
        result = asyncio.run(_run())

    # 提取最终回答
    final_answer = result["messages"][-1].content

    print("=" * 80)
    print(f"💬 最终回答：{final_answer}")
    print("=" * 80)
    print()

    return final_answer


async def run_agentic_rag_stream(user_question: str, session_id: str = "default"):
    """
    运行 Agentic RAG（流式版本）

    参数:
        user_question: 用户问题
        session_id: 会话 ID

    返回:
        异步生成器，逐块返回回答
    """
    print(f"👤 用户问题：{user_question}")
    print(f"🔑 会话 ID：{session_id}")
    print("=" * 80)

    # 获取状态图
    graph = get_agentic_rag_graph().graph

    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": 50,
    }

    # 使用 astream_events v2 版本
    async for event in graph.astream_events(
        input={"messages": [HumanMessage(content=user_question)]},
        config=config,
        version="v2",
    ):
        kind = event["event"]

        # 只处理 LLM 的流式输出
        if kind == "on_chat_model_stream":
            chunk = event.get("data", {}).get("chunk")
            if chunk and hasattr(chunk, "content"):
                content = chunk.content
                if content:
                    yield content

    print("=" * 80)
    print("✅ 流式输出完成")
    print("=" * 80)
    print()
