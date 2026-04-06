# -*- coding: utf-8 -*-
"""格式化工具函数"""
from typing import List


def format_docs(docs):
    """将检索到的文档列表格式化为字符串"""
    if not docs:
        return "未在知识库中找到相关信息"

    # 过滤掉空文档
    valid_docs = [
        doc.page_content
        for doc in docs
        if doc.page_content and doc.page_content.strip()
    ]
    if not valid_docs:
        return "未在知识库中找到相关信息"

    # 打印检索到的上下文，便于观察重排效果
    print("\n" + "=" * 80)
    print("🎯 经过 Reranker 重排后，最终发给大模型的上下文（Top-3）：")
    print("=" * 80)
    for i, doc in enumerate(docs, 1):
        print(f"\n📄 文档片段 {i} (重排后排名):")
        print(f"内容: {doc.page_content[:300]}...")
        if hasattr(doc, "metadata") and doc.metadata:
            print(f"元数据: {doc.metadata}")
    print("=" * 80)
    print("💡 提示：这些是从召回的 6 个候选中，经过 Flashrank 精排后的最优结果")
    print("=" * 80 + "\n")

    return "\n\n".join(valid_docs)


def format_history_for_prompt(history_pairs: List[tuple]) -> str:
    """将对话历史格式化为 Prompt 字符串"""
    if not history_pairs:
        return "无历史对话"

    formatted = []
    for i, (user_msg, assistant_msg) in enumerate(history_pairs, 1):
        formatted.append(f"第{i}轮 - 用户: {user_msg}")
        formatted.append(f"第{i}轮 - 助手: {assistant_msg}")

    return "\n".join(formatted)
