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

【并发控制】
- 分布式锁：防止同一 session_id 并发请求导致 State 脏写
- 熔断机制：Tool 连续失败 3 次后强制跳出循环

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
from app.services.llm.intent_classifier import classify_intent_async
from app.services.agents.query_cache import QueryCache
from app.services.agents.session_guard import SessionGuard


class AgenticRAGState(MessagesState):
    """
    扩展的状态定义，支持查询重写、意图识别和缓存

    属性：
        messages: 对话消息列表（继承自 MessagesState）
        rewritten_query: 重写后的查询（用于优化检索）
        intent_status: 意图状态（accept/reject/unclear）
        intent_reason: 意图判断原因
        should_use_cache: 是否应该使用缓存
        rewrite_count: 查询重写次数（防止无限循环）
        cache_hit: 是否命中缓存
        cached_answer: 缓存的回答
        session_id: 会话 ID（用于熔断判断）
        circuit_break: 是否触发熔断
    """

    rewritten_query: str = ""
    intent_status: str = ""
    intent_reason: str = ""
    should_use_cache: bool = True
    rewrite_count: int = 0
    cache_hit: bool = False
    cached_answer: str = ""
    session_id: str = "default"
    circuit_break: bool = False


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
        print("Agentic RAG 系统初始化中...")
        print("=" * 80)

        # 创建实例
        instance = cls()

        # 1. 初始化 Query 缓存
        await QueryCache.initialize()

        # 2. 初始化会话守护器（分布式锁 + 熔断）
        await SessionGuard.initialize()

        # 3. 整合全量工具箱
        tools = [
            search_knowledge_base,  # RAG 知识库检索
            get_current_weather,  # 天气查询
            query_database_order,  # 订单查询
        ]

        # 如果 Tavily 可用，添加到工具列表
        web_search_tool = get_web_search_tool()
        if web_search_tool:
            tools.append(web_search_tool)

        print(f"  已注册 {len(tools)} 个工具：{[t.name for t in tools]}")

        # 4. 初始化大模型并绑定工具
        instance._llm = ChatOpenAI(model="qwen-turbo", temperature=0)
        instance._llm_with_tools = instance._llm.bind_tools(tools)
        instance._tools = tools
        print("  大模型已绑定所有工具")

        # 5. 获取 checkpointer
        instance._checkpointer = cls._create_checkpointer()

        # 6. 如果使用 Redis，初始化表结构
        if cls._using_redis:
            try:
                await instance._checkpointer.asetup()
                print("  Redis 表结构初始化完成")
            except Exception as e:
                print(f"  Redis 表结构初始化失败（可能已存在）：{e}")

        # 7. 构建状态图
        instance._graph = cls._build_graph(instance)

        cls._initialized = True

        print("  LangGraph 状态机构建完成")
        print("  - 查询重写节点：优化用户查询")
        print("  - 意图识别节点：业务护栏 + 缓存策略 + 意图验证")
        print("  - 缓存查询节点：语义相似度匹配")
        print("  - Agent 节点：决策工具调用")
        print("  - Tool 节点：执行具体任务（带异常捕获）")
        print(
            f"  - 记忆管理：{'Redis 异步持久化存储' if cls._using_redis else 'MemorySaver (LRU 限制)'}"
        )
        print("  - 并发控制：分布式锁 + 熔断机制")
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

            # 获取当前重写次数
            rewrite_count = state.get("rewrite_count", 0)

            # 判断是否需要重写
            if need_query_rewrite_from_messages(user_question, messages[:-1]):
                rewritten = await rewrite_query_from_messages_async(
                    user_question, messages[:-1]
                )
                return {
                    "rewritten_query": rewritten,
                    "rewrite_count": rewrite_count + 1,
                }

            return {
                "rewritten_query": user_question,
                "rewrite_count": rewrite_count + 1,
            }

        # 定义意图识别节点
        async def intent_classification_node(state: AgenticRAGState):
            """
            意图识别节点：判断查询意图和业务范围

            功能：
            1. 业务护栏：拒绝超出能力范围的请求
            2. 缓存策略：判断是否应该使用缓存
            3. 意图验证：检测意图是否明确
            """
            rewritten_query = state.get("rewritten_query", "")
            if not rewritten_query:
                messages = state["messages"]
                if messages and isinstance(messages[-1], HumanMessage):
                    rewritten_query = messages[-1].content

            if not rewritten_query:
                return {
                    "intent_status": "unclear",
                    "intent_reason": "查询为空",
                    "should_use_cache": False,
                }

            # 调用意图分类
            status, reason, use_cache = await classify_intent_async(rewritten_query)

            print(f"  [意图识别] 状态: {status}, 原因: {reason}, 使用缓存: {use_cache}")

            return {
                "intent_status": status,
                "intent_reason": reason,
                "should_use_cache": use_cache,
            }

        # 定义缓存查询节点
        async def cache_lookup_node(state: AgenticRAGState):
            """
            缓存查询节点：查找相似问题的缓存回答

            如果命中缓存，直接返回缓存结果，跳过 LLM 调用
            注意：只有 should_use_cache=True 时才会执行查询
            """
            # 检查是否应该使用缓存
            if not state.get("should_use_cache", True):
                print("  [缓存策略] 跳过缓存查询")
                return {"cache_hit": False, "cached_answer": ""}

            rewritten_query = state.get("rewritten_query", "")
            if not rewritten_query:
                # 使用原始问题
                messages = state["messages"]
                if messages and isinstance(messages[-1], HumanMessage):
                    rewritten_query = messages[-1].content

            if not rewritten_query:
                return {"cache_hit": False, "cached_answer": ""}

            # 查找相似问题
            result = await QueryCache.get_similar(rewritten_query)

            if result:
                cached_answer, similarity = result
                print(f"  [缓存命中] 相似度: {similarity:.2%}")
                return {"cache_hit": True, "cached_answer": cached_answer}

            return {"cache_hit": False, "cached_answer": ""}

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

        # 定义 Tool 节点（带熔断检测）
        async def tool_node_with_circuit_breaker(state: AgenticRAGState):
            """
            Tool 执行节点：执行工具调用，检测异常并触发熔断
            """
            messages = state["messages"]
            last_message = messages[-1]
            session_id = state.get("session_id", "default")

            # 执行工具
            tool_node = ToolNode(instance._tools)
            result = await tool_node.ainvoke(state)

            # 检查工具执行结果是否有错误
            result_messages = result.get("messages", [])
            for msg in result_messages:
                content = getattr(msg, "content", "")
                # 检测工具执行失败的标志
                if isinstance(content, str) and "Tool execution failed:" in content:
                    # 增加错误计数
                    error_count = await SessionGuard.increment_error(session_id)
                    print(f"  [Tool Error] 错误计数: {error_count}/3")

                    # 检查是否需要熔断
                    if error_count >= SessionGuard.MAX_RETRIES:
                        print(f"  [Circuit Breaker] 触发熔断，session_id: {session_id}")
                        return {
                            "messages": result_messages,
                            "circuit_break": True,
                        }

            return {"messages": result_messages, "circuit_break": False}

        # 定义缓存写入节点
        async def cache_write_node(state: AgenticRAGState):
            """
            缓存写入节点：将问题和回答写入缓存
            """
            # 只有未命中缓存时才写入
            if state.get("cache_hit"):
                return {}

            rewritten_query = state.get("rewritten_query", "")
            if not rewritten_query:
                messages = state["messages"]
                if messages and isinstance(messages[-1], HumanMessage):
                    rewritten_query = messages[-1].content

            if not rewritten_query:
                return {}

            # 获取最终回答
            messages = state["messages"]
            final_answer = messages[-1].content if messages else ""

            if final_answer:
                await QueryCache.set(rewritten_query, final_answer)
                print("  [缓存写入] 问题已缓存")

            return {}

        # 定义路由逻辑
        def route_after_intent(
            state: AgenticRAGState,
        ) -> Literal["query_rewrite", "cache_lookup", END]:
            """意图识别后的路由"""
            intent_status = state.get("intent_status", "accept")
            rewrite_count = state.get("rewrite_count", 0)

            # 如果意图被拒绝，直接结束
            if intent_status == "reject":
                print(f"  [业务护栏] 拒绝查询: {state.get('intent_reason', '')}")
                return END

            # 如果意图不明确且重写次数未超限，重新改写
            if intent_status == "unclear" and rewrite_count < 3:
                print(f"  [意图验证] 意图不明确，重新改写（第 {rewrite_count} 次）")
                return "query_rewrite"

            # 如果重写次数超限，也结束流程
            if intent_status == "unclear" and rewrite_count >= 3:
                print("  [意图验证] 重写次数超限，终止流程")
                return END

            # 意图明确，继续处理
            return "cache_lookup"

        def should_continue(
            state: AgenticRAGState,
        ) -> Literal["tools", "cache_write"]:
            """路由函数：判断是否需要继续调用工具"""
            messages = state["messages"]
            last_message = messages[-1]

            # 如果 LLM 决定调用工具，则路由到 tools 节点
            if last_message.tool_calls:
                return "tools"

            # 否则写入缓存后结束
            return "cache_write"

        def route_after_tools(state: AgenticRAGState) -> Literal["agent", END]:
            """Tool 执行后的路由：检查是否触发熔断"""
            # 如果触发熔断，直接结束
            if state.get("circuit_break"):
                return END
            # 否则继续 Agent 决策
            return "agent"

        def route_after_cache_lookup(state: AgenticRAGState) -> Literal["agent", END]:
            """缓存查询后的路由：命中则直接结束"""
            if state.get("cache_hit"):
                return END
            return "agent"

        # 构建状态图
        workflow = StateGraph(AgenticRAGState)

        workflow.add_node("query_rewrite", query_rewrite_node)
        workflow.add_node("intent_classification", intent_classification_node)
        workflow.add_node("cache_lookup", cache_lookup_node)
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", tool_node_with_circuit_breaker)
        workflow.add_node("cache_write", cache_write_node)

        workflow.add_edge(START, "query_rewrite")
        workflow.add_edge("query_rewrite", "intent_classification")
        workflow.add_conditional_edges("intent_classification", route_after_intent)
        workflow.add_conditional_edges("cache_lookup", route_after_cache_lookup)
        workflow.add_conditional_edges("agent", should_continue)
        workflow.add_conditional_edges("tools", route_after_tools)
        workflow.add_edge("cache_write", END)

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
    print(f"用户问题：{user_question}")
    print(f"会话 ID：{session_id}")
    print("=" * 80)

    # 获取状态图
    graph = get_agentic_rag_graph().graph

    # 配置会话
    config = {"configurable": {"thread_id": session_id}}

    # 异步调用（包含分布式锁）
    async def _run():
        # 获取分布式锁
        acquired = await SessionGuard.acquire(session_id)
        if not acquired:
            return {
                "circuit_break": True,
                "messages": [],
                "lock_error": True,
            }

        try:
            return await graph.ainvoke(
                {
                    "messages": [HumanMessage(content=user_question)],
                    "session_id": session_id,
                },
                config=config,
            )
        finally:
            # 释放锁
            await SessionGuard.release(session_id)

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
    if result.get("lock_error"):
        final_answer = "您的请求正在处理中，请稍后再试。"
    elif result.get("circuit_break"):
        final_answer = "系统当前较忙或遇到未知错误，请稍后再试。"
    elif result.get("cache_hit"):
        final_answer = result.get("cached_answer", "")
    elif result.get("intent_status") == "reject":
        # 业务护栏：拒绝超出范围的请求
        reason = result.get("intent_reason", "该请求不在支持范围内")
        final_answer = f"抱歉，{reason}。\n\n我目前支持以下功能：\n- 知识库检索：回答基于已有文档的问题\n- 天气查询：查询指定城市的天气信息\n- 订单查询：查询用户的订单信息\n- 互联网搜索：搜索最新的互联网信息\n\n请问有什么我可以帮助您的吗？"
    elif result.get("intent_status") == "unclear":
        # 意图不明确
        reason = result.get("intent_reason", "您的问题不够明确")
        final_answer = f"抱歉，{reason}。能否提供更多信息或换个方式描述您的问题？"
    else:
        final_answer = result["messages"][-1].content

    print("=" * 80)
    print(f"最终回答：{final_answer}")
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
    print(f"用户问题：{user_question}")
    print(f"会话 ID：{session_id}")
    print("=" * 80)

    # 获取分布式锁
    acquired = await SessionGuard.acquire(session_id)
    if not acquired:
        error_msg = "您的请求正在处理中，请稍后再试。"
        for char in error_msg:
            yield char
            await asyncio.sleep(0.01)
        return

    try:
        # 获取状态图
        graph = get_agentic_rag_graph().graph

        config = {
            "configurable": {"thread_id": session_id},
            "recursion_limit": 50,
        }

        cache_hit = False
        cached_answer = ""
        intent_status = ""
        intent_reason = ""
        circuit_break = False

        # 使用 astream_events v2 版本
        async for event in graph.astream_events(
            input={
                "messages": [HumanMessage(content=user_question)],
                "session_id": session_id,
            },
            config=config,
            version="v2",
        ):
            kind = event["event"]

            # 检测意图识别结果
            if kind == "on_chain_end" and event.get("name") == "intent_classification":
                data = event.get("data", {}).get("output", {})
                intent_status = data.get("intent_status", "")
                intent_reason = data.get("intent_reason", "")

            # 检测缓存命中事件
            if kind == "on_chain_end" and event.get("name") == "cache_lookup":
                data = event.get("data", {}).get("output", {})
                if data.get("cache_hit"):
                    cache_hit = True
                    cached_answer = data.get("cached_answer", "")

            # 检测熔断事件
            if kind == "on_chain_end" and event.get("name") == "tools":
                data = event.get("data", {}).get("output", {})
                if data.get("circuit_break"):
                    circuit_break = True

            # 只处理 LLM 的流式输出
            if kind == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content"):
                    content = chunk.content
                    if content:
                        yield content

        # 如果触发熔断，返回兜底信息
        if circuit_break:
            error_msg = "系统当前较忙或遇到未知错误，请稍后再试。"
            for char in error_msg:
                yield char
                await asyncio.sleep(0.01)
            return

        # 如果意图被拒绝，返回提示信息
        if intent_status == "reject":
            reject_message = f"抱歉，{intent_reason}。\n\n我目前支持以下功能：\n- 知识库检索：回答基于已有文档的问题\n- 天气查询：查询指定城市的天气信息\n- 订单查询：查询用户的订单信息\n- 互联网搜索：搜索最新的互联网信息\n\n请问有什么我可以帮助您的吗？"
            for char in reject_message:
                yield char
                await asyncio.sleep(0.01)
            return

        # 如果意图不明确，返回提示信息
        if intent_status == "unclear":
            unclear_message = (
                f"抱歉，{intent_reason}。能否提供更多信息或换个方式描述您的问题？"
            )
            for char in unclear_message:
                yield char
                await asyncio.sleep(0.01)
            return

        # 如果命中缓存，逐字符返回缓存内容（模拟流式输出）
        if cache_hit and cached_answer:
            print("  [缓存命中] 流式返回缓存内容")
            for char in cached_answer:
                yield char
                await asyncio.sleep(0.01)  # 模拟打字效果

        print("=" * 80)
        print("流式输出完成")
        print("=" * 80)
        print()

    finally:
        # 释放锁
        await SessionGuard.release(session_id)
