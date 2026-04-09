# -*- coding: utf-8 -*-
"""
配置管理模块
"""

import os
from pathlib import Path


class ProcessConfig:
    """处理配置参数"""

    # 表格分类阈值
    SMALL_TABLE_ROWS = 10
    SMALL_TABLE_COLS = 5
    SMALL_TABLE_CHARS = 800

    LARGE_TABLE_ROWS = 30
    LARGE_TABLE_COLS = 10
    LARGE_TABLE_CHARS = 3000

    # 图片过滤阈值
    MIN_IMAGE_WIDTH = 50
    MIN_IMAGE_HEIGHT = 50
    MAX_ASPECT_RATIO = 20
    MIN_IMAGE_SIZE = 1024  # 1KB

    # 表头广播
    GIANT_TABLE_CHUNK_ROWS = 10

    # 并发控制
    MAX_CONCURRENT_REQUESTS = 10
    REQUEST_TIMEOUT = 30
    MAX_RETRIES = 3

    # 上下文窗口
    CONTEXT_BEFORE_CHARS = 1000
    CONTEXT_AFTER_CHARS = 500

    # Embedding API 限制（text-embedding-v4）
    EMBEDDING_BATCH_SIZE = 10
    EMBEDDING_MAX_TOKENS = 8192
    EMBEDDING_MAX_CHARS = 8192
    EMBEDDING_MAX_TEXT_LENGTH = 24000

    # 文本切分参数
    CHUNK_SIZE = 1200
    CHUNK_OVERLAP = 360  # 30% overlap

    # 模式开关
    ENABLE_IMAGE_ANALYSIS = True
    ENABLE_TABLE_ANALYSIS = True
    ENABLE_CACHE = True
    FAST_MODE = False

    @classmethod
    def validate_env(cls):
        """验证必需的环境变量"""
        required_vars = ["DASHSCOPE_API_KEY"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]

        if missing_vars:
            raise EnvironmentError(
                f"缺少必需的环境变量: {', '.join(missing_vars)}\n"
                f"请在 .env 文件中配置这些变量"
            )


# 获取项目根目录（向上两级）
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# 确保数据目录存在
DATA_DIR.mkdir(exist_ok=True)

# 文件记录路径（使用绝对路径）
IMPORTED_FILES_RECORD = str(DATA_DIR / ".imported_files.json")
ELEMENT_CACHE_FILE = str(DATA_DIR / ".element_analysis_cache.json")
LOG_FILE = str(DATA_DIR / ".build_knowledge.log")
