# -*- coding: utf-8 -*-
"""
Elasticsearch 索引操作模块
"""

import re
from typing import List, Dict
from elasticsearch import Elasticsearch


def create_es_index(es_client: Elasticsearch, index_name: str, vector_dim: int = 1024):
    """创建 ES 索引（支持向量检索 + BM25 + 元数据过滤 + Multi-Vector）"""
    if es_client.indices.exists(index=index_name):
        print(f"  ✓ 索引 '{index_name}' 已存在，使用现有索引")
        return

    index_mapping = {
        "mappings": {
            "properties": {
                "text": {
                    "type": "text",
                    "analyzer": "ik_smart",
                    "search_analyzer": "ik_smart",
                },
                "vector": {
                    "type": "dense_vector",
                    "dims": vector_dim,
                    "index": True,
                    "similarity": "cosine",
                },
                "doc_id": {"type": "keyword"},
                "title": {
                    "type": "text",
                    "analyzer": "ik_smart",
                    "fields": {"keyword": {"type": "keyword"}},
                },
                "author": {"type": "keyword"},
                "department": {"type": "keyword"},
                "access_level": {"type": "keyword"},
                "year": {"type": "keyword"},
                "created_at": {"type": "date", "format": "yyyy-MM-dd"},
                "file_path": {"type": "keyword"},
                "file_hash": {"type": "keyword"},
                "chunk_id": {"type": "integer"},
                "h1": {"type": "text", "analyzer": "ik_smart"},
                "h2": {"type": "text", "analyzer": "ik_smart"},
                "h3": {"type": "text", "analyzer": "ik_smart"},
                "sub_chunk_id": {"type": "integer"},
                "element_type": {"type": "keyword"},
                "has_image": {"type": "boolean"},
                "has_table": {"type": "boolean"},
                "image_count": {"type": "integer"},
                "table_count": {"type": "integer"},
                "element_summary": {"type": "text", "analyzer": "ik_smart"},
                "element_keywords": {"type": "keyword"},
                "element_columns": {"type": "keyword"},
                "original_table": {"type": "text", "index": False},
                "original_image_url": {"type": "keyword", "index": False},
                "parent_chunk_id": {"type": "keyword"},
                "is_summary_chunk": {"type": "boolean"},
                "element_id": {"type": "keyword"},
            }
        },
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
    }

    es_client.indices.create(index=index_name, body=index_mapping)
    print(f"  ✓ 索引 '{index_name}' 创建成功（支持 Multi-Vector）")


def index_chunks_to_es(
    es_client: Elasticsearch,
    index_name: str,
    chunks: List[Dict],
    vectors: List[List[float]],
    doc_metadata: Dict,
) -> int:
    """批量索引文本块到 ES"""
    indexed_count = 0

    for chunk, vector in zip(chunks, vectors):
        es_doc_id = f"{doc_metadata['doc_id']}_chunk_{chunk['metadata']['chunk_id']}"

        chunk_text = chunk["content"]
        has_image = "<figure" in chunk_text and "![" in chunk_text
        has_table = "<figure" in chunk_text and "|" in chunk_text

        summary_match = re.search(r"<summary>(.*?)</summary>", chunk_text, re.DOTALL)
        keywords_match = re.search(r"<keywords>(.*?)</keywords>", chunk_text, re.DOTALL)
        columns_match = re.search(r"<columns>(.*?)</columns>", chunk_text, re.DOTALL)

        element_summary = summary_match.group(1).strip() if summary_match else ""
        element_keywords = (
            [k.strip() for k in keywords_match.group(1).split(",")]
            if keywords_match
            else []
        )
        element_columns = (
            [c.strip() for c in columns_match.group(1).split(",")]
            if columns_match
            else []
        )

        if has_image and has_table:
            element_type = "mixed"
        elif has_image:
            element_type = "image"
        elif has_table:
            table_size = len(chunk_text)
            element_type = "small_table" if table_size < 800 else "large_table"
        else:
            element_type = "text"

        doc_body = {
            "text": chunk["content"],
            "vector": vector,
            **chunk["metadata"],
            "element_type": element_type,
            "has_image": has_image,
            "has_table": has_table,
            "image_count": chunk_text.count("!["),
            "table_count": chunk_text.count("<figure") if has_table else 0,
            "element_summary": element_summary,
            "element_keywords": element_keywords,
            "element_columns": element_columns,
            "is_summary_chunk": bool(element_summary),
        }

        es_client.index(index=index_name, id=es_doc_id, body=doc_body)
        indexed_count += 1

    return indexed_count
