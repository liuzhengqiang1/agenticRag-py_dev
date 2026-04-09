# -*- coding: utf-8 -*-
"""
Elasticsearch 知识库构建脚本 v2.1（重构优化版）
用途：将 Markdown 文档切分、向量化并索引到 Elasticsearch 中

主要改进：
- 修复路径硬编码问题，使用动态路径
- 完善错误处理和日志记录
- 优化资源管理，避免连接泄漏
- 改进向量化批处理逻辑
- 增加进度显示和性能监控
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from langchain_community.embeddings import DashScopeEmbeddings

# 导入模块化组件
from config import (
    ProcessConfig,
    IMPORTED_FILES_RECORD,
    ELEMENT_CACHE_FILE,
    LOG_FILE,
    DATA_DIR,
)
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


# 配置日志
def setup_logging():
    """配置日志系统"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


logger = setup_logging()


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
            "start_time": 0,
            "end_time": 0,
            "api_calls": 0,
        }
        self.failed_files = []

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出，确保资源清理"""
        self.cleanup()
        return False

    def cleanup(self):
        """清理资源"""
        if self.es_client:
            try:
                self.es_client.close()
                logger.info("ES 连接已关闭")
            except Exception as e:
                logger.warning(f"关闭 ES 连接时出错: {e}")

    def connect_es(self):
        """连接 Elasticsearch"""
        logger.info("[1/8] 正在连接 Elasticsearch...")

        try:
            self.es_client = Elasticsearch(**self.es_config.get_connection_params())

            if not self.es_client.ping():
                raise ConnectionError("无法连接到 Elasticsearch，请检查配置和服务状态")

            logger.info(f"成功连接到 ES：{self.es_config.get_url()}")
        except Exception as e:
            logger.error(f"连接 ES 失败: {e}")
            raise

    def setup_index(self):
        """创建索引"""
        logger.info("[2/8] 正在检查/创建索引...")
        try:
            create_es_index(self.es_client, self.es_config.index_name, vector_dim=1024)
        except Exception as e:
            logger.error(f"创建索引失败: {e}")
            raise

    def scan_files(self) -> List[str]:
        """扫描文档"""
        logger.info("[3/8] 正在扫描 Markdown 文档...")

        # 使用绝对路径
        data_path = str(DATA_DIR)
        markdown_files = scan_markdown_files(data_path)

        if not markdown_files:
            logger.warning(f"未找到任何 Markdown 文件，请检查目录: {data_path}")
            return []

        logger.info(f"找到 {len(markdown_files)} 个 Markdown 文件")
        return markdown_files

    def filter_incremental(self, markdown_files: List[str]) -> List[str]:
        """增量过滤"""
        logger.info("[4/8] 正在检查增量更新...")
        self.imported_files = load_json_file(IMPORTED_FILES_RECORD)

        files_to_process = []
        for file_path in markdown_files:
            try:
                file_hash = calculate_file_hash(file_path)

                if (
                    file_path in self.imported_files
                    and self.imported_files[file_path] == file_hash
                ):
                    continue

                files_to_process.append(file_path)
            except Exception as e:
                logger.warning(f"计算文件哈希失败 {file_path}: {e}")
                files_to_process.append(file_path)

        if not files_to_process:
            logger.info("所有文件已是最新，无需更新")
            return []

        logger.info(f"需要处理 {len(files_to_process)} 个文件（新增或已变更）")
        return files_to_process

    def load_cache(self):
        """加载缓存"""
        logger.info("[5/8] 正在加载元素分析缓存...")
        try:
            self.element_cache = load_json_file(ELEMENT_CACHE_FILE)
            logger.info(f"缓存中已有 {len(self.element_cache)} 条分析记录")
        except Exception as e:
            logger.warning(f"加载缓存失败: {e}，将使用空缓存")
            self.element_cache = {}

    def init_embeddings(self):
        """初始化 Embedding 模型"""
        logger.info("[6/8] 正在初始化 Embedding 模型...")
        try:
            self.embeddings = DashScopeEmbeddings(model="text-embedding-v4")
            logger.info("Embedding 模型初始化成功（text-embedding-v4，1024 维）")
        except Exception as e:
            logger.error(f"初始化 Embedding 模型失败: {e}")
            raise

    def process_file(self, file_path: str, idx: int, total: int):
        """处理单个文件"""
        logger.info(f"[{idx}/{total}] 处理文件: {file_path}")
        file_start_time = time.time()

        try:
            # 读取文件
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except UnicodeDecodeError:
                # 尝试其他编码
                with open(file_path, "r", encoding="gbk") as f:
                    content = f.read()
                logger.warning(f"文件使用 GBK 编码: {file_path}")

            if not content.strip():
                logger.warning(f"文件为空，跳过: {file_path}")
                self.failed_files.append((file_path, "文件为空"))
                self.stats["failed_files"] += 1
                return

            # 提取元数据
            doc_metadata = extract_metadata_from_path(file_path)
            logger.info(
                f"  标题: {doc_metadata['title']}, 作者: {doc_metadata['author']}, 部门: {doc_metadata['department']}"
            )

            # 扫描图片和表格
            logger.info(f"  正在扫描图片和表格...")
            elements = scan_markdown_elements(content, file_path)

            images = elements["images"]
            tables = elements["tables"]

            logger.info(f"  发现 {len(images)} 个图片，{len(tables)} 个表格")

            # 统计表格分类
            small_tables = [t for t in tables if t["classification"] == "small"]
            large_tables = [t for t in tables if t["classification"] == "large"]
            giant_tables = [t for t in tables if t["classification"] == "giant"]

            logger.info(
                f"  表格分类: 小表 {len(small_tables)}, 大表 {len(large_tables)}, 巨型表 {len(giant_tables)}"
            )

            # 异步分析元素
            images_to_analyze = [img for img in images if img["action"] == "process"]
            tables_to_analyze = [
                t for t in tables if t["action"] in ["summarize", "split_and_summarize"]
            ]

            analyzed_images = []
            analyzed_tables = []

            if images_to_analyze or tables_to_analyze:
                logger.info(
                    f"  需要分析: {len(images_to_analyze)} 个图片, {len(tables_to_analyze)} 个表格"
                )

                if images_to_analyze:
                    analyzed_images = asyncio.run(
                        batch_analyze_elements(
                            images_to_analyze, "图片", self.element_cache
                        )
                    )
                    self.stats["api_calls"] += len(images_to_analyze)

                if tables_to_analyze:
                    analyzed_tables = asyncio.run(
                        batch_analyze_elements(
                            tables_to_analyze, "表格", self.element_cache
                        )
                    )
                    self.stats["api_calls"] += len(tables_to_analyze)

            # 注入摘要到原文
            if analyzed_images or analyzed_tables:
                logger.info(f"  正在注入分析结果...")
                content = inject_summaries_to_markdown(
                    content, analyzed_images, analyzed_tables
                )

            # 切分文档
            chunks = split_markdown_content(content, doc_metadata)
            logger.info(f"  切分: {len(chunks)} 个文本块")

            # 批量向量化并导入
            file_chunk_count = self._vectorize_and_index(chunks, doc_metadata)

            file_elapsed = time.time() - file_start_time
            logger.info(
                f"  成功导入 {file_chunk_count} 个文本块 (耗时 {file_elapsed:.2f}s)"
            )

            # 更新统计
            self.stats["total_chunks"] += file_chunk_count
            self.stats["total_images"] += len(images)
            self.stats["total_tables"] += len(tables)
            self.stats["success_files"] += 1

            # 记录已导入文件
            self.imported_files[file_path] = doc_metadata["file_hash"]

        except Exception as e:
            logger.error(f"  处理失败: {e}", exc_info=True)
            self.failed_files.append((file_path, str(e)))
            self.stats["failed_files"] += 1

    def _vectorize_and_index(self, chunks: List[Dict], doc_metadata: Dict) -> int:
        """向量化并索引文本块"""
        batch_size = ProcessConfig.EMBEDDING_BATCH_SIZE
        max_text_length = ProcessConfig.EMBEDDING_MAX_TEXT_LENGTH
        file_chunk_count = 0

        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i : i + batch_size]

            # 提取并截断文本
            batch_texts = []
            for chunk in batch_chunks:
                text = chunk["content"]
                if len(text) > max_text_length:
                    logger.warning(
                        f"  文本过长 ({len(text)} 字符)，截断至 {max_text_length} 字符"
                    )
                    # 智能截断：保留完整句子
                    text = self._smart_truncate(text, max_text_length)
                batch_texts.append(text)

            # 批量向量化
            try:
                batch_vectors = self.embeddings.embed_documents(batch_texts)
            except Exception as e:
                logger.error(f"  批量向量化失败: {e}")
                # 降级为单个向量化
                batch_vectors = []
                for idx, text in enumerate(batch_texts):
                    try:
                        vector = self.embeddings.embed_query(text)
                        batch_vectors.append(vector)
                    except Exception as e2:
                        logger.error(f"  单个向量化也失败 (chunk {i+idx}): {e2}")
                        # 跳过这个chunk，不使用零向量
                        continue

            if len(batch_vectors) != len(batch_chunks):
                logger.warning(f"  向量数量不匹配，跳过此批次")
                continue

            # 批量索引
            try:
                indexed = index_chunks_to_es(
                    self.es_client,
                    self.es_config.index_name,
                    batch_chunks,
                    batch_vectors,
                    doc_metadata,
                )
                file_chunk_count += indexed
            except Exception as e:
                logger.error(f"  索引失败: {e}")

        return file_chunk_count

    def _smart_truncate(self, text: str, max_length: int) -> str:
        """智能截断文本，保留完整句子"""
        if len(text) <= max_length:
            return text

        # 截断到最大长度
        truncated = text[:max_length]

        # 尝试在句子边界截断
        for sep in ["。", "！", "？", "\n\n", "\n", ".", "!", "?"]:
            last_sep = truncated.rfind(sep)
            if last_sep > max_length * 0.8:  # 至少保留80%
                return truncated[: last_sep + 1]

        # 如果找不到合适的分隔符，直接截断
        return truncated

    def process_files(self, files_to_process: List[str]):
        """处理所有文件"""
        logger.info(f"[7/8] 正在处理文档（Multi-Vector 模式）...")

        self.stats["total_files"] = len(files_to_process)

        for idx, file_path in enumerate(files_to_process, 1):
            self.process_file(file_path, idx, len(files_to_process))

            # 定期保存进度
            if idx % 5 == 0:
                save_json_file(IMPORTED_FILES_RECORD, self.imported_files)
                save_json_file(ELEMENT_CACHE_FILE, self.element_cache)
                logger.info(f"  进度已保存 ({idx}/{len(files_to_process)})")

        # 刷新索引
        try:
            self.es_client.indices.refresh(index=self.es_config.index_name)
            logger.info("ES 索引已刷新")
        except Exception as e:
            logger.warning(f"刷新索引失败: {e}")

        # 最终保存记录
        save_json_file(IMPORTED_FILES_RECORD, self.imported_files)
        save_json_file(ELEMENT_CACHE_FILE, self.element_cache)
        logger.info("所有记录已保存")

    def print_summary(self):
        """打印总结"""
        elapsed_time = self.stats["end_time"] - self.stats["start_time"]

        logger.info("=" * 70)
        logger.info("知识库构建完成")
        logger.info("=" * 70)
        logger.info(f"统计信息：")
        logger.info(f"  处理文件数: {self.stats['total_files']}")
        logger.info(f"  成功文件数: {self.stats['success_files']}")
        logger.info(f"  失败文件数: {self.stats['failed_files']}")
        logger.info(f"  导入文本块: {self.stats['total_chunks']}")
        logger.info(f"  处理图片数: {self.stats['total_images']}")
        logger.info(f"  处理表格数: {self.stats['total_tables']}")
        logger.info(f"  API 调用数: {self.stats['api_calls']}")
        logger.info(f"  缓存记录数: {len(self.element_cache)}")
        logger.info(f"  总耗时: {elapsed_time:.2f}s")
        logger.info(f"  索引名称: {self.es_config.index_name}")

        if self.failed_files:
            logger.warning(f"失败文件列表：")
            for file_path, error in self.failed_files:
                logger.warning(f"  {file_path}")
                logger.warning(f"    原因: {error}")

        logger.info("=" * 70)
        logger.info("提示：")
        logger.info("  图片和表格已智能处理（Multi-Vector 模式）")
        logger.info("  小表直接入库，大表生成摘要，巨型表表头广播")
        logger.info("  分析结果已缓存，重复运行不会重复调用 API")
        logger.info(f"  日志文件: {LOG_FILE}")
        logger.info("=" * 70)

    def build(self):
        """执行构建流程"""
        logger.info("=" * 70)
        logger.info("开始构建 Elasticsearch 知识库 v2.1")
        logger.info("支持：Multi-Vector 检索 + 图表智能处理 + 大小表分流")
        logger.info("=" * 70)

        self.stats["start_time"] = time.time()

        try:
            # 验证环境变量
            ProcessConfig.validate_env()

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

        except Exception as e:
            logger.error(f"构建过程出错: {e}", exc_info=True)
            raise
        finally:
            self.stats["end_time"] = time.time()
            self.print_summary()


def build_es_knowledge_base():
    """构建知识库的主入口"""
    with KnowledgeBaseBuilder() as builder:
        builder.build()


if __name__ == "__main__":
    try:
        build_es_knowledge_base()
    except KeyboardInterrupt:
        logger.warning("用户中断执行")
    except Exception as e:
        logger.error(f"构建失败：{e}", exc_info=True)
        exit(1)
