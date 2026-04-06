# -*- coding: utf-8 -*-
"""
Elasticsearch 知识库构建脚本
用途：将 Markdown 文档切分、向量化并索引到 Elasticsearch 中
支持：向量检索（kNN）+ BM25 关键字检索 + 元数据过滤
"""

import os
import json
import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_community.embeddings import DashScopeEmbeddings
from app.core.es_config import ESConfig

load_dotenv()

# 已导入文件记录路径
IMPORTED_FILES_RECORD = "data/.imported_files.json"


def calculate_file_hash(file_path: str) -> str:
    """计算文件 MD5 哈希值，用于检测文件是否变更"""
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def load_imported_files() -> Dict[str, str]:
    """加载已导入文件记录（文件路径 -> 文件哈希）"""
    if os.path.exists(IMPORTED_FILES_RECORD):
        with open(IMPORTED_FILES_RECORD, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_imported_files(imported_files: Dict[str, str]):
    """保存已导入文件记录"""
    os.makedirs(os.path.dirname(IMPORTED_FILES_RECORD), exist_ok=True)
    with open(IMPORTED_FILES_RECORD, "w", encoding="utf-8") as f:
        json.dump(imported_files, f, ensure_ascii=False, indent=2)


def parse_filename(filename: str) -> Tuple[str, str, Optional[str]]:
    """
    解析文件名提取元数据
    格式：{标题}_{作者/部门}_{YYYYMMDD}.md

    示例：
        销售培训_张三_20240115.md -> ("销售培训", "张三", "2024-01-15")
        软件培训_技术部_20240201.md -> ("软件培训", "技术部", "2024-02-01")
    """
    # 移除扩展名
    name_without_ext = os.path.splitext(filename)[0]

    # 按下划线分割
    parts = name_without_ext.split("_")

    if len(parts) < 3:
        # 格式不符合规范，返回默认值
        return (name_without_ext, "未知", None)

    title = parts[0]
    author = parts[1]
    date_str = parts[2]

    # 解析日期 YYYYMMDD -> YYYY-MM-DD
    try:
        if len(date_str) == 8 and date_str.isdigit():
            year = date_str[:4]
            month = date_str[4:6]
            day = date_str[6:8]
            created_at = f"{year}-{month}-{day}"
        else:
            created_at = None
    except:
        created_at = None

    return (title, author, created_at)


def extract_metadata_from_path(file_path: str) -> Dict:
    """
    从文件路径提取元数据

    路径示例：
        data/documents/employee/销售部/2024/销售培训_张三_20240115.md

    提取信息：
        - access_level: employee (从路径)
        - department: 销售部 (从路径)
        - year: 2024 (从路径)
        - title, author, created_at: 从文件名解析
    """
    # 标准化路径分隔符
    normalized_path = file_path.replace("\\", "/")
    parts = normalized_path.split("/")

    # 提取权限级别
    access_level = "employee"  # 默认
    if "public" in parts:
        access_level = "public"
    elif "employee" in parts:
        access_level = "employee"
    elif "manager" in parts:
        access_level = "manager"
    elif "executive" in parts:
        access_level = "executive"

    # 提取部门（假设在 access_level 后面）
    department = "未知部门"
    try:
        access_idx = next(
            i
            for i, p in enumerate(parts)
            if p in ["public", "employee", "manager", "executive"]
        )
        if access_idx + 1 < len(parts) and not parts[access_idx + 1].isdigit():
            department = parts[access_idx + 1]
    except:
        pass

    # 提取年份
    year = None
    for part in parts:
        if part.isdigit() and len(part) == 4:
            year = part
            break

    # 解析文件名
    filename = os.path.basename(file_path)
    title, author, created_at = parse_filename(filename)

    # 生成 doc_id
    doc_id = f"{access_level}_{year or 'unknown'}_{title}"

    # 使用文件创建时间作为备用
    if not created_at:
        try:
            file_ctime = os.path.getctime(file_path)
            created_at = datetime.fromtimestamp(file_ctime).strftime("%Y-%m-%d")
        except:
            created_at = datetime.now().strftime("%Y-%m-%d")

    return {
        "doc_id": doc_id,
        "title": title,
        "author": author,
        "department": department,
        "access_level": access_level,
        "year": year,
        "created_at": created_at,
        "file_path": file_path,
        "file_hash": calculate_file_hash(file_path),
    }


def scan_markdown_files(root_dir: str = "data") -> List[str]:
    """递归扫描目录下的所有 markdown 文件"""
    markdown_files = []

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                markdown_files.append(file_path)

    return markdown_files


def split_markdown_content(content: str, doc_metadata: Dict) -> List[Dict]:
    """
    使用混合策略切分 markdown 内容
    1. 优先按标题切分
    2. 对过长章节再用递归分割器切分
    """
    chunks = []

    # 第一步：按 markdown 标题切分
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

    # 如果没有标题或切分失败，直接用递归切分
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

    # 第二步：对过长的章节再次切分
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
            # 章节过长，再次切分
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


def create_es_index(es_client: Elasticsearch, index_name: str, vector_dim: int = 1024):
    """
    创建 ES 索引（支持向量检索 + BM25 + 元数据过滤）
    """
    # 检查索引是否已存在
    if es_client.indices.exists(index=index_name):
        print(f"  ✓ 索引 '{index_name}' 已存在，使用现有索引")
        return

    # 创建索引映射
    index_mapping = {
        "mappings": {
            "properties": {
                # 原有字段
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
                # 文档级元数据
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
                # 切片级元数据
                "chunk_id": {"type": "integer"},
                "h1": {"type": "text", "analyzer": "ik_smart"},
                "h2": {"type": "text", "analyzer": "ik_smart"},
                "h3": {"type": "text", "analyzer": "ik_smart"},
                "sub_chunk_id": {"type": "integer"},
            }
        },
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
    }

    es_client.indices.create(index=index_name, body=index_mapping)
    print(f"  ✓ 索引 '{index_name}' 创建成功")


def build_es_knowledge_base():
    """
    构建 Elasticsearch 知识库的主流程（支持增量更新）
    """
    print("=" * 70)
    print("开始构建 Elasticsearch 知识库（Markdown + 元数据 + 增量更新）")
    print("=" * 70)

    # ==================== 步骤 1：连接 ES ====================
    print("\n[1/6] 正在连接 Elasticsearch...")
    es_config = ESConfig()
    es_client = Elasticsearch(**es_config.get_connection_params())

    if not es_client.ping():
        raise ConnectionError("无法连接到 Elasticsearch，请检查配置和服务状态")

    print(f"  ✓ 成功连接到 ES：{es_config.get_url()}")

    # ==================== 步骤 2：创建索引 ====================
    print("\n[2/6] 正在检查/创建索引...")
    create_es_index(es_client, es_config.index_name, vector_dim=1024)

    # ==================== 步骤 3：扫描文档 ====================
    print("\n[3/6] 正在扫描 Markdown 文档...")
    markdown_files = scan_markdown_files("data")

    if not markdown_files:
        print("  ⚠️  未找到任何 Markdown 文件，请检查 data 目录")
        return

    print(f"  ✓ 找到 {len(markdown_files)} 个 Markdown 文件")

    # ==================== 步骤 4：增量过滤 ====================
    print("\n[4/6] 正在检查增量更新...")
    imported_files = load_imported_files()

    files_to_process = []
    for file_path in markdown_files:
        file_hash = calculate_file_hash(file_path)

        # 检查文件是否已导入且未变更
        if file_path in imported_files and imported_files[file_path] == file_hash:
            continue  # 跳过已导入且未变更的文件

        files_to_process.append(file_path)

    if not files_to_process:
        print("  ✓ 所有文件已是最新，无需更新")
        return

    print(f"  ✓ 需要处理 {len(files_to_process)} 个文件（新增或已变更）")

    # ==================== 步骤 5：初始化 Embedding 模型 ====================
    print("\n[5/6] 正在初始化 Embedding 模型...")
    embeddings = DashScopeEmbeddings(model="text-embedding-v4")
    print("  ✓ Embedding 模型初始化成功（text-embedding-v4，1024 维）")

    # ==================== 步骤 6：处理文档并导入 ====================
    print(f"\n[6/6] 正在处理并导入文档...")

    total_chunks = 0
    failed_files = []

    for idx, file_path in enumerate(files_to_process, 1):
        print(f"\n  [{idx}/{len(files_to_process)}] 处理文件: {file_path}")

        try:
            # 读取文件内容
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            if not content.strip():
                print(f"    ⚠️  文件为空，跳过")
                failed_files.append((file_path, "文件为空"))
                continue

            # 提取元数据
            doc_metadata = extract_metadata_from_path(file_path)
            print(f"    - 标题: {doc_metadata['title']}")
            print(f"    - 作者: {doc_metadata['author']}")
            print(f"    - 部门: {doc_metadata['department']}")
            print(f"    - 权限: {doc_metadata['access_level']}")
            print(f"    - 日期: {doc_metadata['created_at']}")

            # 切分文档
            chunks = split_markdown_content(content, doc_metadata)
            print(f"    - 切分: {len(chunks)} 个文本块")

            # 批量向量化并导入
            batch_size = 10
            file_chunk_count = 0

            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i : i + batch_size]

                # 提取文本内容
                batch_texts = [chunk["content"] for chunk in batch_chunks]

                # 批量向量化
                batch_vectors = embeddings.embed_documents(batch_texts)

                # 批量导入 ES
                for chunk, vector in zip(batch_chunks, batch_vectors):
                    es_doc_id = f"{doc_metadata['doc_id']}_chunk_{chunk['metadata']['chunk_id']}"

                    doc_body = {
                        "text": chunk["content"],
                        "vector": vector,
                        **chunk["metadata"],
                    }

                    es_client.index(
                        index=es_config.index_name, id=es_doc_id, body=doc_body
                    )
                    file_chunk_count += 1

            print(f"    ✓ 成功导入 {file_chunk_count} 个文本块")
            total_chunks += file_chunk_count

            # 记录已导入文件
            imported_files[file_path] = doc_metadata["file_hash"]

        except Exception as e:
            print(f"    ❌ 处理失败: {e}")
            failed_files.append((file_path, str(e)))
            continue

    # 刷新索引
    es_client.indices.refresh(index=es_config.index_name)

    # 保存已导入文件记录
    save_imported_files(imported_files)

    # ==================== 总结 ====================
    print("\n" + "=" * 70)
    print("知识库构建完成！")
    print("=" * 70)
    print(f"\n统计信息：")
    print(f"  - 处理文件数: {len(files_to_process)}")
    print(f"  - 成功文件数: {len(files_to_process) - len(failed_files)}")
    print(f"  - 失败文件数: {len(failed_files)}")
    print(f"  - 导入文本块: {total_chunks}")
    print(f"  - 索引名称: {es_config.index_name}")
    print(f"  - ES 地址: {es_config.get_url()}")

    if failed_files:
        print(f"\n⚠️  失败文件列表：")
        for file_path, error in failed_files:
            print(f"  - {file_path}")
            print(f"    原因: {error}")

    print("\n" + "=" * 70)
    print("提示：")
    print("  - 新增文件后，重新运行此脚本即可增量导入")
    print("  - 修改文件后，重新运行此脚本会自动更新")
    print("  - 运行命令：python build_knowledge_es.py")
    print("=" * 70)


if __name__ == "__main__":
    try:
        build_es_knowledge_base()
    except Exception as e:
        print(f"\n❌ 构建失败：{e}")
        import traceback

        traceback.print_exc()
