# -*- coding: utf-8 -*-
"""Elasticsearch 检索器封装"""
from typing import List
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.embeddings import DashScopeEmbeddings
from elasticsearch import Elasticsearch


class ESVectorRetriever(BaseRetriever):
    """
    Elasticsearch 向量检索器（kNN）

    架构说明：
    - 使用 ES 的 kNN 搜索功能进行向量检索
    - 支持余弦相似度计算
    - 返回 LangChain 标准的 Document 对象
    """

    es_client: Elasticsearch
    index_name: str
    embeddings: DashScopeEmbeddings
    k: int = 10

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """执行向量检索"""
        # 1. 将查询文本向量化
        query_vector = self.embeddings.embed_query(query)

        # 2. 构建 kNN 查询
        knn_query = {
            "knn": {
                "field": "vector",
                "query_vector": query_vector,
                "k": self.k,
                "num_candidates": self.k * 2,
            },
            "_source": ["text", "metadata"],
        }

        # 3. 执行查询
        response = self.es_client.search(index=self.index_name, body=knn_query)

        # 4. 转换为 Document 对象
        documents = []
        for hit in response["hits"]["hits"]:
            doc = Document(
                page_content=hit["_source"]["text"],
                metadata={
                    **hit["_source"].get("metadata", {}),
                    "score": hit["_score"],
                    "retriever": "es_vector",
                },
            )
            documents.append(doc)

        return documents


class ESBM25Retriever(BaseRetriever):
    """
    Elasticsearch BM25 检索器（全文检索）

    架构说明：
    - 使用 ES 的 match 查询进行 BM25 全文检索
    - 支持中文分词（需要安装 IK 分词器插件）
    - 返回 LangChain 标准的 Document 对象
    """

    es_client: Elasticsearch
    index_name: str
    k: int = 10

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """执行 BM25 检索"""
        # 1. 构建 BM25 查询
        bm25_query = {
            "query": {"match": {"text": {"query": query, "analyzer": "ik_smart"}}},
            "size": self.k,
            "_source": ["text", "metadata"],
        }

        # 2. 执行查询
        response = self.es_client.search(index=self.index_name, body=bm25_query)

        # 3. 转换为 Document 对象
        documents = []
        for hit in response["hits"]["hits"]:
            doc = Document(
                page_content=hit["_source"]["text"],
                metadata={
                    **hit["_source"].get("metadata", {}),
                    "score": hit["_score"],
                    "retriever": "es_bm25",
                },
            )
            documents.append(doc)

        return documents
