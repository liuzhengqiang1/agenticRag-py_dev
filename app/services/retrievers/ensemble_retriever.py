# -*- coding: utf-8 -*-
"""混合检索器配置"""
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
from flashrank import Ranker
from elasticsearch import Elasticsearch

from app.core.es_config import ESConfig
from app.services.retrievers.es_retriever import ESVectorRetriever, ESBM25Retriever
from app.services.topk import TopKRetriever


def create_hybrid_retriever():
    """
    创建混合检索器（向量 + BM25 + 重排序）

    返回：
        ContextualCompressionRetriever 实例
    """
    # 1. Embedding 模型
    embeddings = DashScopeEmbeddings(model="text-embedding-v4")

    # 2. 初始化 Elasticsearch 客户端
    es_config = ESConfig()
    es_client = Elasticsearch(**es_config.get_connection_params())

    # 测试连接
    if not es_client.ping():
        raise ConnectionError(
            f"无法连接到 Elasticsearch ({es_config.get_url()})，"
            "请检查 Docker 容器是否启动，以及 .env 配置是否正确"
        )
    print(f"✓ 成功连接到 Elasticsearch：{es_config.get_url()}")

    # 3. 向量检索器（ES kNN）
    vector_retriever = ESVectorRetriever(
        es_client=es_client,
        index_name=es_config.index_name,
        embeddings=embeddings,
        k=10,
    )
    print("✓ ES 向量检索器（kNN）初始化完成，召回数：10")

    # 4. 关键字检索器（ES BM25）
    bm25_retriever = ESBM25Retriever(
        es_client=es_client,
        index_name=es_config.index_name,
        k=10,
    )
    print("✓ ES 关键字检索器（BM25）初始化完成，召回数：10")

    # 5. 混合检索器（EnsembleRetriever）
    base_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever], weights=[0.5, 0.5]
    )
    print("✓ 混合检索器（Ensemble）初始化完成，权重：向量 50% + BM25 50%")

    # 6. 重排序器（Reranker）
    ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
    compressor = FlashrankRerank(client=ranker, top_n=3)

    # 7. 预过滤器（TopKRetriever）
    pre_filtered_retriever = TopKRetriever(retriever=base_retriever, k=6)

    # 8. 最终检索器
    final_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=pre_filtered_retriever
    )
    print("✓ 重排序器（Flashrank Reranker）初始化完成，精排数：3")
    print("✓ 两阶段检索架构已启用：召回（10+10=20）→ 过滤(Top-6) → 重排（Top-3）")

    return final_retriever
