"""
知识库检索工具：将 RAG 检索器封装为 Tool
"""

from langchain_core.tools import tool
from app.services.retrievers.ensemble_retriever import create_hybrid_retriever
from app.services.utils.formatters import format_docs


@tool
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
    print(f"🔍 [RAG Tool] 检索查询：{query}")

    try:
        # 混合检索
        retriever = create_hybrid_retriever()
        docs = retriever.invoke(query)

        if not docs:
            return "知识库中未找到相关信息。"

        # 格式化结果
        result = format_docs(docs)
        print(f"✓ 检索到 {len(docs)} 条相关文档")
        return result

    except Exception as e:
        error_msg = f"知识库检索失败：{str(e)}"
        print(f"❌ {error_msg}")
        return error_msg
