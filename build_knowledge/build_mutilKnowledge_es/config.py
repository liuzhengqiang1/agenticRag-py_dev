# -*- coding: utf-8 -*-
"""
配置管理模块
"""


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

    # 模式开关
    ENABLE_IMAGE_ANALYSIS = True
    ENABLE_TABLE_ANALYSIS = True
    ENABLE_CACHE = True
    FAST_MODE = False


# 文件记录路径
IMPORTED_FILES_RECORD = "data/.imported_files.json"
ELEMENT_CACHE_FILE = "data/.element_analysis_cache.json"
