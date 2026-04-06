# -*- coding: utf-8 -*-
"""
Elasticsearch 知识库构建脚本
用途：将原始文档切分、向量化并索引到 Elasticsearch 中
支持：向量检索（kNN）+ BM25 关键字检索
"""

import os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from app.core.es_config import ESConfig

load_dotenv()


def create_es_index(es_client: Elasticsearch, index_name: str, vector_dim: int = 1024):
    """
    创建 ES 索引（支持向量检索 + BM25）
    
    架构说明：
    - text 字段：使用 text 类型 + ik_smart 分词器，支持 BM25 全文检索
    - vector 字段：使用 dense_vector 类型，支持 kNN 向量检索
    - metadata 字段：存储文档元数据（来源、chunk_id 等）
    
    参数：
        es_client: ES 客户端实例
        index_name: 索引名称
        vector_dim: 向量维度（DashScope text-embedding-v4 默认是 1024 维，维度有很多选择）
    """
    # 检查索引是否已存在
    if es_client.indices.exists(index=index_name):
        print(f"⚠️  索引 '{index_name}' 已存在，正在删除...")
        es_client.indices.delete(index=index_name)
        print(f"✓ 旧索引已删除")
    
    # 创建索引映射（Mapping）
    index_mapping = {
        "mappings": {
            "properties": {
                "text": {
                    "type": "text",
                    "analyzer": "ik_smart",  # 使用 IK 分词器（需要安装 IK 插件）
                    "search_analyzer": "ik_smart"
                },
                "vector": {
                    "type": "dense_vector",
                    "dims": vector_dim,
                    "index": True,
                    "similarity": "cosine"  # 余弦相似度
                },
                "metadata": {
                    "type": "object",
                    "enabled": True
                }
            }
        },
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        }
    }
    
    es_client.indices.create(index=index_name, body=index_mapping)
    print(f"✓ 索引 '{index_name}' 创建成功")


def build_es_knowledge_base():
    """
    构建 Elasticsearch 知识库的主流程
    """
    print("=" * 60)
    print("开始构建 Elasticsearch 知识库...")
    print("=" * 60)
    
    # ==================== 步骤 1：连接 ES ====================
    print("\n[1/5] 正在连接 Elasticsearch...")
    es_config = ESConfig()
    es_client = Elasticsearch(**es_config.get_connection_params())
    
    # 测试连接
    if not es_client.ping():
        raise ConnectionError("无法连接到 Elasticsearch，请检查配置和服务状态")
    
    print(f"✓ 成功连接到 ES：{es_config.get_url()}")
    
    # ==================== 步骤 2：加载文档 ====================
    print("\n[2/5] 正在加载文档...")
    doc_path = "data/training_doc.txt"
    
    if not os.path.exists(doc_path):
        raise FileNotFoundError(f"文档文件不存在：{doc_path}")
    
    with open(doc_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(f"✓ 成功加载文档：{doc_path}")
    
    # ==================== 步骤 3：文档切分 ====================
    print("\n[3/5] 正在切分文档...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
    )
    
    chunks = text_splitter.split_text(content)
    print(f"✓ 文档切分完成，共生成 {len(chunks)} 个文本块")
    
    # 打印前 2 个切片示例
    print("\n切片示例（前 2 个）：")
    for i, chunk in enumerate(chunks[:2]):
        print(f"\n--- Chunk {i+1} ---")
        print(chunk[:100] + "..." if len(chunk) > 100 else chunk)
    
    # ==================== 步骤 4：初始化 Embedding 模型 ====================
    print("\n[4/5] 正在初始化 Embedding 模型...")
    embeddings = DashScopeEmbeddings(model="text-embedding-v4")
    print("✓ Embedding 模型初始化成功（text-embedding-v4，1024 维）")
    
    # ==================== 步骤 5：创建索引并批量导入 ====================
    print("\n[5/5] 正在创建索引并导入数据...")
    
    # 创建索引
    create_es_index(es_client, es_config.index_name, vector_dim=1024)
    
    # 批量向量化并导入
    print(f"正在向量化 {len(chunks)} 个文本块...")
    
    # 批量处理（每次 10 个，避免 API 限流）
    batch_size = 10
    total_imported = 0
    
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        
        # 批量向量化
        batch_vectors = embeddings.embed_documents(batch_chunks)
        
        # 批量导入 ES
        for j, (chunk_text, vector) in enumerate(zip(batch_chunks, batch_vectors)):
            doc_id = f"doc_{i + j + 1}"
            doc_body = {
                "text": chunk_text,
                "vector": vector,
                "metadata": {
                    "source": doc_path,
                    "chunk_id": i + j + 1
                }
            }
            
            es_client.index(index=es_config.index_name, id=doc_id, body=doc_body)
            total_imported += 1
        
        print(f"  进度：{total_imported}/{len(chunks)}")
    
    # 刷新索引（确保数据可搜索）
    es_client.indices.refresh(index=es_config.index_name)
    
    print(f"\n✓ 数据导入完成！共导入 {total_imported} 个文档")
    
    print("\n" + "=" * 60)
    print("Elasticsearch 知识库构建成功！")
    print("=" * 60)
    print(f"\n索引信息：")
    print(f"  - 索引名称：{es_config.index_name}")
    print(f"  - 文档数量：{total_imported}")
    print(f"  - ES 地址：{es_config.get_url()}")
    print("\n⚠️  重要提示：")
    print("   如果 data/training_doc.txt 内容有更新，")
    print("   请重新运行此脚本以更新索引！")
    print("   运行命令：python build_knowledge_es.py")
    print("=" * 60)


if __name__ == "__main__":
    try:
        build_es_knowledge_base()
    except Exception as e:
        print(f"\n❌ 构建失败：{e}")
        import traceback
        traceback.print_exc()
