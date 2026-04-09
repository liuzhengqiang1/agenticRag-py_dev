# -*- coding: utf-8 -*-
"""
Elasticsearch 知识库构建脚本 v2.0（重构版）
用途：将 Markdown 文档切分、向量化并索引到 Elasticsearch 中
"""

import asyncio
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from langchain_community.embeddings import DashScopeEmbeddings

# 导入模块化组件
from config import ProcessConfig, IMPORTED_FILES_RECORD, ELEMENT_CACHE_FILE
from file_utils import (
    calculate_file_hash,
    load_json_file,
    save_json_file,
    scan_markdown_files,
)
from metadata_extractor import extract_metadata_from_path
from markdown_parser import scan_markdown_elements, inject_summaries_to_markdown
from content_splitter import split_markdown_content
from llm_analyzer import batch_analyze_elements
from es_indexer import create_es_index, index_chunks_to_es
from app.core.es_config import ESConfig

load_dotenv()


class KnowledgeBaseBuilder:
    """知识库构建器"""

    def __init__(self):
        self.es_config = ESConfig()
        self.es_client = None
        self.embeddings = None
        self.imported_files = {}
        self.element_cache = {}
        self.stats = {
            "total_files": 0,
            "success_files": 0,
            "failed_files": 0,
            "total_chunks": 0,
            "total_images": 0,
            "total_tables": 0,
        }
        self.failed_files = []

    def connect_es(self):
        """连接 Elasticsearch"""
        print("\n[1/8] 正在连接 Elasticsearch...")
        self.es_client = Elasticsearch(**self.es_config.get_connection_params())

        if not self.es_client.ping():
            raise ConnectionError("无法连接到 Elasticsearch，请检查配置和服务状态")

        print(f"  ✓ 成功连接到 ES：{self.es_config.get_url()}")

    def setup_index(self):
        """创建索引"""
        print("\n[2/8] 正在检查/创建索引...")
        create_es_index(self.es_client, self.es_config.index_name, vector_dim=1024)

    def scan_files(self):
        """扫描文档"""
        print("\n[3/8] 正在扫描 Markdown 文档...")
        markdown_files = scan_markdown_files("../../data")

        if not markdown_files:
            print("  ⚠️  未找到任何 Markdown 文件，请检查 data 目录")
            return []

        print(f"  ✓ 找到 {len(markdown_files)} 个 Markdown 文件")
        return markdown_files

    def filter_incremental(self, markdown_files):
        """增量过滤"""
        print("\n[4/8] 正在检查增量更新...")
        self.imported_files = load_json_file(IMPORTED_FILES_RECORD)

        files_to_process = []
        for file_path in markdown_files:
            file_hash = calculate_file_hash(file_path)

            if (
                file_path in self.imported_files
                and self.imported_files[file_path] == file_hash
            ):
                continue

            files_to_process.append(file_path)

        if not files_to_process:
            print("  ✓ 所有文件已是最新，无需更新")
            return []

        print(f"  ✓ 需要处理 {len(files_to_process)} 个文件（新增或已变更）")
        return files_to_process

    def load_cache(self):
        """加载缓存"""
        print("\n[5/8] 正在加载元素分析缓存...")
        self.element_cache = load_json_file(ELEMENT_CACHE_FILE)
        print(f"  ✓ 缓存中已有 {len(self.element_cache)} 条分析记录")

    def init_embeddings(self):
        """初始化 Embedding 模型"""
        print("\n[6/8] 正在初始化 Embedding 模型...")
        self.embeddings = DashScopeEmbeddings(model="text-embedding-v4")
        print("  ✓ Embedding 模型初始化成功（text-embedding-v4，1024 维）")

    def process_file(self, file_path: str, idx: int, total: int):
        """处理单个文件"""
        print(f"\n  [{idx}/{total}] 处理文件: {file_path}")

        try:
            # 读取文件
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            if not content.strip():
                print(f"    ⚠️  文件为空，跳过")
                self.failed_files.append((file_path, "文件为空"))
                return

            # 提取元数据
            doc_metadata = extract_metadata_from_path(file_path)
            print(f"    - 标题: {doc_metadata['title']}")
            print(f"    - 作者: {doc_metadata['author']}")
            print(f"    - 部门: {doc_metadata['department']}")

            # 扫描图片和表格
            print(f"    - 正在扫描图片和表格...")
            elements = scan_markdown_elements(content, file_path)

            images = elements["images"]
            tables = elements["tables"]

            print(f"      发现 {len(images)} 个图片，{len(tables)} 个表格")

            # 统计表格分类
            small_tables = [t for t in tables if t["classification"] == "small"]
            large_tables = [t for t in tables if t["classification"] == "large"]
            giant_tables = [t for t in tables if t["classification"] == "giant"]

            print(
                f"      表格分类: 小表 {len(small_tables)}, 大表 {len(large_tables)}, 巨型表 {len(giant_tables)}"
            )

            # 异步分析元素
            images_to_analyze = [img for img in images if img["action"] == "process"]
            tables_to_analyze = [
                t for t in tables if t["action"] in ["summarize", "split_and_summarize"]
            ]

            analyzed_images = []
            analyzed_tables = []

            if images_to_analyze or tables_to_analyze:
                print(
                    f"    - 需要分析: {len(images_to_analyze)} 个图片, {len(tables_to_analyze)} 个表格"
                )

                if images_to_analyze:
                    analyzed_images = asyncio.run(
                        batch_analyze_elements(
                            images_to_analyze, "图片", self.element_cache
                        )
                    )

                if tables_to_analyze:
                    analyzed_tables = asyncio.run(
                        batch_analyze_elements(
                            tables_to_analyze, "表格", self.element_cache
                        )
                    )

            # 注入摘要到原文
            if analyzed_images or analyzed_tables:
                print(f"    - 正在注入分析结果...")
                content = inject_summaries_to_markdown(
                    content, analyzed_images, analyzed_tables
                )

            # 切分文档
            chunks = split_markdown_content(content, doc_metadata)
            print(f"    - 切分: {len(chunks)} 个文本块")

            # 批量向量化并导入
            file_chunk_count = self._vectorize_and_index(chunks, doc_metadata)

            print(f"    ✓ 成功导入 {file_chunk_count} 个文本块")

            # 更新统计
            self.stats["total_chunks"] += file_chunk_count
            self.stats["total_images"] += len(images)
            self.stats["total_tables"] += len(tables)
            self.stats["success_files"] += 1

            # 记录已导入文件
            self.imported_files[file_path] = doc_metadata["file_hash"]

        except Exception as e:
            print(f"    ❌ 处理失败: {e}")
            import traceback

            traceback.print_exc()
            self.failed_files.append((file_path, str(e)))
            self.stats["failed_files"] += 1

    def _vectorize_and_index(self, chunks, doc_metadata):
        """向量化并索引文本块"""
        batch_size = 10
        max_text_length = 24000
        file_chunk_count = 0

        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i : i + batch_size]

            # 提取并截断文本
            batch_texts = []
            for chunk in batch_chunks:
                text = chunk["content"]
                if len(text) > max_text_length:
                    print(
                        f"      ⚠️  文本过长 ({len(text)} 字符)，截断至 {max_text_length} 字符"
                    )
                    text = text[:max_text_length]
                batch_texts.append(text)

            # 批量向量化
            try:
                batch_vectors = self.embeddings.embed_documents(batch_texts)
            except Exception as e:
                print(f"      ⚠️  批量向量化失败: {e}")
                batch_vectors = []
                for text in batch_texts:
                    try:
                        vector = self.embeddings.embed_query(text)
                        batch_vectors.append(vector)
                    except Exception as e2:
                        print(f"      ❌ 单个向量化也失败: {e2}")
                        batch_vectors.append([0.0] * 1024)

            if len(batch_vectors) != len(batch_chunks):
                print(f"      ⚠️  向量数量不匹配，跳过此批次")
                continue

            # 批量索引
            indexed = index_chunks_to_es(
                self.es_client,
                self.es_config.index_name,
                batch_chunks,
                batch_vectors,
                doc_metadata,
            )
            file_chunk_count += indexed

        return file_chunk_count

    def process_files(self, files_to_process):
        """处理所有文件"""
        print(f"\n[7/8] 正在处理文档（Multi-Vector 模式）...")

        self.stats["total_files"] = len(files_to_process)

        for idx, file_path in enumerate(files_to_process, 1):
            self.process_file(file_path, idx, len(files_to_process))

        # 刷新索引
        self.es_client.indices.refresh(index=self.es_config.index_name)

        # 保存记录
        save_json_file(IMPORTED_FILES_RECORD, self.imported_files)
        save_json_file(ELEMENT_CACHE_FILE, self.element_cache)

    def print_summary(self):
        """打印总结"""
        print("\n" + "=" * 70)
        print("知识库构建完成！")
        print("=" * 70)
        print(f"\n统计信息：")
        print(f"  - 处理文件数: {self.stats['total_files']}")
        print(f"  - 成功文件数: {self.stats['success_files']}")
        print(f"  - 失败文件数: {self.stats['failed_files']}")
        print(f"  - 导入文本块: {self.stats['total_chunks']}")
        print(f"  - 处理图片数: {self.stats['total_images']}")
        print(f"  - 处理表格数: {self.stats['total_tables']}")
        print(f"  - 缓存记录数: {len(self.element_cache)}")
        print(f"  - 索引名称: {self.es_config.index_name}")

        if self.failed_files:
            print(f"\n⚠️  失败文件列表：")
            for file_path, error in self.failed_files:
                print(f"  - {file_path}")
                print(f"    原因: {error}")

        print("\n" + "=" * 70)
        print("提示：")
        print("  - 图片和表格已智能处理（Multi-Vector 模式）")
        print("  - 小表直接入库，大表生成摘要，巨型表表头广播")
        print("  - 分析结果已缓存，重复运行不会重复调用 API")
        print("  - 运行命令：python build_knowledge/build_knowledge_es.py")
        print("=" * 70)

    def build(self):
        """执行构建流程"""
        print("=" * 70)
        print("开始构建 Elasticsearch 知识库 v2.0")
        print("支持：Multi-Vector 检索 + 图表智能处理 + 大小表分流")
        print("=" * 70)

        self.connect_es()
        self.setup_index()

        markdown_files = self.scan_files()
        if not markdown_files:
            return

        files_to_process = self.filter_incremental(markdown_files)
        if not files_to_process:
            return

        self.load_cache()
        self.init_embeddings()
        self.process_files(files_to_process)
        self.print_summary()


def build_es_knowledge_base():
    """构建知识库的主入口"""
    builder = KnowledgeBaseBuilder()
    builder.build()


if __name__ == "__main__":
    try:
        build_es_knowledge_base()
    except Exception as e:
        print(f"\n❌ 构建失败：{e}")
        import traceback

        traceback.print_exc()
