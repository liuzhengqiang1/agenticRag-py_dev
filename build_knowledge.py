# -*- coding: utf-8 -*-
"""
离线知识库构建脚本
类比 Java：这是一个独立的数据预处理工具类，类似 ETL 脚本
用途：将原始文档切分、向量化并持久化到向量数据库中
"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings

# 加载环境变量（获取 API Key 和 Base URL）
load_dotenv()


def build_vector_store():
    """
    构建向量知识库的主流程
    
    流程说明（类比 Java 的 ETL Pipeline）：
    1. Extract（提取）：从文件加载原始文档
    2. Transform（转换）：将文档切分成小块（chunk）
    3. Load（加载）：向量化并持久化到 Chroma 数据库
    """
    
    print("=" * 60)
    print("开始构建向量知识库...")
    print("=" * 60)
    
    # ==================== 步骤 1：加载文档 ====================
    print("\n[1/4] 正在加载文档...")
    doc_path = "data/training_doc.txt"
    
    if not os.path.exists(doc_path):
        raise FileNotFoundError(f"文档文件不存在：{doc_path}")
    
    # TextLoader：文本文件加载器（类似 Java 的 FileReader）
    loader = TextLoader(doc_path, encoding="utf-8")
    documents = loader.load()
    print(f"✓ 成功加载文档，共 {len(documents)} 个文件")
    
    
    # ==================== 步骤 2：文档切分 ====================
    print("\n[2/4] 正在切分文档...")
    
    # RecursiveCharacterTextSplitter：递归字符文本切分器
    # 类比 Java：类似 StringTokenizer，但更智能，会按段落、句子、词语递归切分
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,        # 每个切片的最大字符数（测试用小值，生产环境建议 500-1000）
        chunk_overlap=20,      # 切片之间的重叠字符数（保证上下文连贯性）
        length_function=len,   # 计算长度的函数
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]  # 优先按段落、句子切分
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"✓ 文档切分完成，共生成 {len(chunks)} 个文本块")
    
    # 打印前 2 个切片示例（方便调试）
    print("\n切片示例（前 2 个）：")
    for i, chunk in enumerate(chunks[:2]):
        print(f"\n--- Chunk {i+1} ---")
        print(chunk.page_content[:100] + "..." if len(chunk.page_content) > 100 else chunk.page_content)


    # ==================== 步骤 3：初始化 Embedding 模型 ====================
    print("\n[3/4] 正在初始化 Embedding 模型...")
    
    # DashScopeEmbeddings：阿里云 格式的向量化模型
    # 类比 Java：类似一个 @Service 组件，负责将文本转换为向量
    # 注意：会自动读取环境变量 DASHSCOPE_API_KEY
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v4"
    )
    print("✓ Embedding 模型初始化成功（使用阿里云百炼 text-embedding-v4）")
    
    
    # ==================== 步骤 4：向量化并持久化 ====================
    print("\n[4/4] 正在向量化并保存到 Chroma 数据库...")
    
    # Chroma：轻量级向量数据库（类比 Java 的 H2 或 SQLite）
    # 注意：每次重新构建时会清空旧数据
    vector_store = Chroma(persist_directory="vector_store", embedding_function=embeddings)

    # 清空已有数据（防止重复添加）
    vector_store.delete_collection()

    # persist_directory：持久化目录（类似数据库的 data 目录）
    vector_store = Chroma.from_documents(
        documents=chunks,              # 要向量化的文档切片
        embedding=embeddings,          # Embedding 模型实例
        persist_directory="vector_store"  # 持久化路径
    )
    
    print(f"✓ 向量数据库构建完成！已保存到 vector_store/ 目录")
    print(f"✓ 共存储 {len(chunks)} 个向量")
    
    print("\n" + "=" * 60)
    print("知识库构建成功！")
    print("=" * 60)
    print("\n⚠️  重要提示：")
    print("   如果 data/training_doc.txt 内容有更新，")
    print("   请重新运行此脚本以更新向量库！")
    print("   运行命令：python build_knowledge.py")
    print("=" * 60)


if __name__ == "__main__":
    """
    脚本入口（类比 Java 的 main 方法）
    """
    try:
        build_vector_store()
    except Exception as e:
        print(f"\n❌ 构建失败：{e}")
        import traceback
        traceback.print_exc()
