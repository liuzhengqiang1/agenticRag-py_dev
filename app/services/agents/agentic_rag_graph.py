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
"""

from typing import Literal, TypedDict
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from cachetools import LRUCache
import redis

from app.core.redis_config import RedisConfig
from app.services.tools import (
    search_knowledge_base,
    get_current_weather,
    query_database_order,
    get_web_search_tool,
)
from app.services.llm.query_rewriter import (
    need_query_rewrite_from_messages,
    rewrite_query_from_messages,
)

# 全局变量：缓存编译后的状态图
_compiled_graph = None

# 全局变量：记录是否使用 Redis
_using_redis = False


class AgenticRAGState(MessagesState):
    """
    扩展的状态定义，支持查询重写

    属性：
        messages: 对话消息列表（继承自 MessagesState）
        rewritten_query: 重写后的查询（用于优化检索）
    """

    rewritten_query: str = ""


def get_checkpointer():
    """
    获取检查点存储器（优先 Redis，失败时降级到内存）

    降级策略：
    1. 尝试连接 Redis，成功则使用 RedisSaver
    2. 连接失败则降级到 MemorySaver + LRU 缓存（防止 OOM）

    返回:
        检查点存储器实例
    """
    global _using_redis

    try:
        # 尝试使用 Redis
        redis_config = RedisConfig(decode_responses=False)  # LangGraph 需要字节模式
        redis_client = redis.Redis(**redis_config.get_connection_kwargs())

        # 健康检查
        redis_client.ping()

        # 导入 RedisSaver（延迟导入，避免没有安装时报错）
        try:
            from langgraph.checkpoint.redis import RedisSaver

            _using_redis = True
            print("✓ Redis 连接成功，使用持久化存储")
            saver = RedisSaver(redis_client=redis_client)
            saver.setup()
            return saver
        except ImportError:
            print("⚠️  langgraph.checkpoint.redis 未安装，降级到内存存储")
            raise

    except Exception as e:
        # Redis 连接失败，降级到内存存储
        print(f"⚠️  Redis 连接失败：{e}")
        print("⚠️  降级到内存存储（LRU 限制：最多 100 个会话）")

        _using_redis = False

        # 使用 LRU 缓存包装 MemorySaver，防止 OOM
        class LRUMemorySaver(MemorySaver):
            """带 LRU 限制的内存存储器"""

            def __init__(self, maxsize=100):
                super().__init__()
                # 用 LRU 缓存替换原有的存储字典
                self.storage = LRUCache(maxsize=maxsize)

        return LRUMemorySaver(maxsize=100)


def create_agentic_rag_graph():
    """
    创建 Agentic RAG 状态图

    返回:
        编译后的 LangGraph 状态图
    """
    global _compiled_graph

    # 如果已经编译过，直接返回
    if _compiled_graph is not None:
        return _compiled_graph

    print("=" * 80)
    print("🚀 Agentic RAG 系统初始化中...")
    print("=" * 80)

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
    llm = ChatOpenAI(model="qwen-turbo", temperature=0)
    llm_with_tools = llm.bind_tools(tools)
    print("✓ 大模型已绑定所有工具")

    # 3. 定义查询重写节点
    def query_rewrite_node(state: AgenticRAGState):
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
            rewritten = rewrite_query_from_messages(user_question, messages[:-1])
            return {"rewritten_query": rewritten}

        return {"rewritten_query": user_question}

    # 4. 定义 Agent 节点（决策中心）
    def agent_node(state: AgenticRAGState):
        """
        Agent 决策节点：分析用户问题，决定调用哪些工具

        如果有重写后的查询，使用重写后的查询进行决策
        """
        messages = state["messages"]
        rewritten_query = state.get("rewritten_query", "")

        # 如果有重写查询，替换最后一条用户消息
        if rewritten_query and isinstance(messages[-1], HumanMessage):
            modified_messages = messages[:-1] + [HumanMessage(content=rewritten_query)]
            response = llm_with_tools.invoke(modified_messages)
        else:
            response = llm_with_tools.invoke(messages)

        return {"messages": [response]}

    # 5. 定义路由逻辑（决定下一步去哪）
    def should_continue(state: AgenticRAGState) -> Literal["tools", END]:
        """
        路由函数：判断是否需要继续调用工具
        """
        messages = state["messages"]
        last_message = messages[-1]

        # 如果 LLM 决定调用工具，则路由到 tools 节点
        if last_message.tool_calls:
            return "tools"

        # 否则结束流程
        return END

    # 6. 构建状态图
    workflow = StateGraph(AgenticRAGState)

    # 添加节点
    workflow.add_node("query_rewrite", query_rewrite_node)  # 查询重写节点
    workflow.add_node("agent", agent_node)  # Agent 决策节点
    workflow.add_node("tools", ToolNode(tools))  # 工具执行节点

    # 添加边（定义流程）
    workflow.add_edge(START, "query_rewrite")  # 开始 -> 查询重写
    workflow.add_edge("query_rewrite", "agent")  # 查询重写 -> Agent
    workflow.add_conditional_edges("agent", should_continue)  # Agent -> 工具 or 结束
    workflow.add_edge("tools", "agent")  # 工具 -> Agent（闭环反馈）

    # 7. 编译状态图（添加检查点存储）
    checkpointer = get_checkpointer()
    _compiled_graph = workflow.compile(checkpointer=checkpointer)

    print("✓ LangGraph 状态机构建完成")
    print("  - 查询重写节点：优化用户查询")
    print("  - Agent 节点：决策工具调用")
    print("  - Tool 节点：执行具体任务")
    print(
        f"  - 记忆管理：{'Redis 持久化存储' if _using_redis else 'MemorySaver (LRU 限制)'}"
    )
    print("=" * 80)
    print()

    return _compiled_graph


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
    graph = create_agentic_rag_graph()

    # 配置会话
    config = {"configurable": {"thread_id": session_id}}

    # 调用状态图
    result = graph.invoke(
        {"messages": [HumanMessage(content=user_question)]}, config=config
    )

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
    graph = create_agentic_rag_graph()

    config = {"configurable": {"thread_id": session_id}}

    async for event in graph.astream_events(
        {"messages": [HumanMessage(content=user_question)]}, config=config, version="v1"
    ):
        kind = event["event"]

        # 只处理 LLM 的流式输出
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                yield content

    print("=" * 80)
    print("✅ 流式输出完成")
    print("=" * 80)
    print()
