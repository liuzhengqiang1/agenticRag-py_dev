"""
知识库检索工具：将 RAG 检索器封装为 Tool
"""

from langchain_core.tools import tool
from app.services.retrievers.ensemble_retriever import create_hybrid_retriever
from app.services.utils.formatters import format_docs
from app.services.tools.tool_wrapper import safe_tool


# ============================================================
# 检索器单例（模块加载时初始化一次）
# ============================================================
_retriever = None


def _get_retriever():
    """获取检索器单例"""
    global _retriever
    if _retriever is None:
        print("[RAG Tool] 初始化检索器...")
        _retriever = create_hybrid_retriever()
        print("[RAG Tool] 检索器初始化完成")
    return _retriever


@tool
@safe_tool
def search_knowledge_base(query: str) -> str:
    """
    在企业知识库中检索相关信息（培训制度、报销流程等内部文档）。

    适用场景：
    - 查询公司内部政策、制度、流程
    - 培训相关问题
    - 报销、福利等 HR 问题

    参数:
        query: 要检索的问题或关键词（已经过查询重写优化）

    返回:
        知识库中检索到的相关内容
    """
    print(f"[RAG Tool] 检索查询：{query}")

    # 使用单例检索器
    retriever = _get_retriever()
    docs = retriever.invoke(query)

    if not docs:
        return "知识库中未找到相关信息。"

    # 格式化结果
    result = format_docs(docs)
    print(f"[RAG Tool] 检索到 {len(docs)} 条相关文档")
    return result
