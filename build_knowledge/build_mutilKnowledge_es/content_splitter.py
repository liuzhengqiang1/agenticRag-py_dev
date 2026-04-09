# -*- coding: utf-8 -*-
"""
内容切分模块
"""

from typing import List, Dict
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


def split_markdown_content(content: str, doc_metadata: Dict) -> List[Dict]:
    """
    使用混合策略切分 markdown 内容
    1. 优先按标题切分
    2. 对过长章节再用递归分割器切分
    """
    chunks = []

    headers_to_split_on = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )

    try:
        md_chunks = markdown_splitter.split_text(content)
    except Exception as e:
        print(f"    ⚠️  标题切分失败，使用递归切分: {e}")
        md_chunks = []

    if not md_chunks:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        )
        text_chunks = text_splitter.split_text(content)

        for i, chunk_text in enumerate(text_chunks):
            chunks.append(
                {
                    "content": chunk_text,
                    "metadata": {
                        **doc_metadata,
                        "chunk_id": i + 1,
                        "h1": "",
                        "h2": "",
                        "h3": "",
                        "sub_chunk_id": 0,
                    },
                }
            )
        return chunks

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )

    chunk_id = 0
    for md_chunk in md_chunks:
        chunk_content = md_chunk.page_content
        chunk_meta = md_chunk.metadata

        if len(chunk_content) > 800:
            sub_chunks = text_splitter.split_text(chunk_content)
            for sub_id, sub_chunk in enumerate(sub_chunks):
                chunk_id += 1
                chunks.append(
                    {
                        "content": sub_chunk,
                        "metadata": {
                            **doc_metadata,
                            "chunk_id": chunk_id,
                            "h1": chunk_meta.get("h1", ""),
                            "h2": chunk_meta.get("h2", ""),
                            "h3": chunk_meta.get("h3", ""),
                            "sub_chunk_id": sub_id,
                        },
                    }
                )
        else:
            chunk_id += 1
            chunks.append(
                {
                    "content": chunk_content,
                    "metadata": {
                        **doc_metadata,
                        "chunk_id": chunk_id,
                        "h1": chunk_meta.get("h1", ""),
                        "h2": chunk_meta.get("h2", ""),
                        "h3": chunk_meta.get("h3", ""),
                        "sub_chunk_id": 0,
                    },
                }
            )

    return chunks
