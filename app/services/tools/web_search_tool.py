"""
互联网搜索工具：Tavily 搜索
"""

from typing import Optional
from langchain_tavily import TavilySearch


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
        print("✓ Tavily 搜索工具初始化成功")
        return web_search_tool
    except Exception as e:
        print(f"⚠️  Tavily 搜索工具初始化失败：{e}")
        print("   如需使用互联网搜索，请在 .env 中配置 TAVILY_API_KEY")
        return None
