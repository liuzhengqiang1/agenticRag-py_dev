"""
互联网搜索工具：Tavily 搜索
"""

from typing import Optional
from langchain_tavily import TavilySearch
from app.services.tools.tool_wrapper import safe_tool


def get_web_search_tool() -> Optional[TavilySearch]:
    """
    获取 Tavily 搜索工具实例

    返回:
        TavilySearch 实例，如果初始化失败则返回 None
    """
    try:
        web_search_tool = TavilySearch(
            max_results=3,
            topic="general",
            search_depth="basic",
            include_answer=True,
            include_raw_content=False,
        )
        print("[Web Search] Tavily 搜索工具初始化成功")
        return web_search_tool
    except Exception as e:
        print(f"[Web Search] Tavily 搜索工具初始化失败: {e}")
        print("  如需使用互联网搜索，请在 .env 中配置 TAVILY_API_KEY")
        return None


def create_safe_web_search_tool():
    """
    创建带异常捕获的 Tavily 搜索工具

    返回:
        包装后的安全工具，或 None
    """
    tavily_tool = get_web_search_tool()
    if tavily_tool is None:
        return None

    # TavilySearch 是 LangChain Tool 子类，需要包装其 invoke 方法
    original_invoke = tavily_tool.invoke

    @safe_tool
    def safe_web_search(query: str) -> str:
        """在互联网上搜索最新信息"""
        result = original_invoke(query)
        # Tavily 返回的是 dict，需要转换为字符串
        if isinstance(result, dict):
            return result.get("answer", str(result))
        return str(result)

    # 复制工具属性
    safe_web_search.name = tavily_tool.name
    safe_web_search.description = tavily_tool.description

    return safe_web_search
