# -*- coding: utf-8 -*-
"""
知识库构建脚本配置示例
复制此文件并根据需要修改参数
"""


class ProcessConfig:
    """处理配置参数"""

    # ==================== 表格分类阈值 ====================
    # 小表：直接入库，不调用 LLM（节省成本）
    SMALL_TABLE_ROWS = 10  # 最大行数
    SMALL_TABLE_COLS = 5  # 最大列数
    SMALL_TABLE_CHARS = 800  # 最大字符数

    # 大表：生成摘要后入库（提升检索效果）
    LARGE_TABLE_ROWS = 30  # 最大行数
    LARGE_TABLE_COLS = 10  # 最大列数
    LARGE_TABLE_CHARS = 3000  # 最大字符数

    # 巨型表：超过大表阈值，需要表头广播切分

    # ==================== 图片过滤阈值 ====================
    # 过滤无效图片（Logo、图标、装饰线等）
    MIN_IMAGE_WIDTH = 50  # 最小宽度（像素）
    MIN_IMAGE_HEIGHT = 50  # 最小高度（像素）
    MAX_ASPECT_RATIO = 20  # 最大长宽比（过滤装饰线）
    MIN_IMAGE_SIZE = 1024  # 最小文件大小（字节，1KB）

    # ==================== 表头广播 ====================
    # 巨型表格切分策略
    GIANT_TABLE_CHUNK_ROWS = 10  # 每个切片包含的数据行数

    # ==================== 并发控制 ====================
    # 异步批处理参数
    MAX_CONCURRENT_REQUESTS = 10  # 最大并发请求数
    REQUEST_TIMEOUT = 30  # 单个请求超时时间（秒）
    MAX_RETRIES = 3  # 失败重试次数

    # ==================== 上下文窗口 ====================
    # LLM 分析时提取的上下文长度
    CONTEXT_BEFORE_CHARS = 1000  # 元素前的上下文字符数
    CONTEXT_AFTER_CHARS = 500  # 元素后的上下文字符数

    # ==================== 模式开关 ====================
    # 功能开关
    ENABLE_IMAGE_ANALYSIS = True  # 是否分析图片
    ENABLE_TABLE_ANALYSIS = True  # 是否分析表格
    ENABLE_CACHE = True  # 是否启用缓存
    FAST_MODE = False  # 快速模式：跳过所有 LLM 分析


# ==================== 使用场景配置示例 ====================


class FastConfig(ProcessConfig):
    """快速模式：跳过所有分析，快速导入"""

    FAST_MODE = True
    ENABLE_IMAGE_ANALYSIS = False
    ENABLE_TABLE_ANALYSIS = False


class HighQualityConfig(ProcessConfig):
    """高质量模式：更严格的过滤，更详细的分析"""

    # 更严格的表格分类
    SMALL_TABLE_ROWS = 5
    SMALL_TABLE_COLS = 3
    SMALL_TABLE_CHARS = 500

    # 更严格的图片过滤
    MIN_IMAGE_WIDTH = 100
    MIN_IMAGE_HEIGHT = 100
    MIN_IMAGE_SIZE = 5120  # 5KB

    # 更多的上下文
    CONTEXT_BEFORE_CHARS = 1500
    CONTEXT_AFTER_CHARS = 800


class LowCostConfig(ProcessConfig):
    """低成本模式：最大化节省 API 调用"""

    # 更宽松的小表定义（更多表格直接入库）
    SMALL_TABLE_ROWS = 15
    SMALL_TABLE_COLS = 8
    SMALL_TABLE_CHARS = 1200

    # 禁用图片分析
    ENABLE_IMAGE_ANALYSIS = False

    # 降低并发（避免限流）
    MAX_CONCURRENT_REQUESTS = 5


class HighSpeedConfig(ProcessConfig):
    """高速模式：最大化处理速度"""

    # 提高并发数
    MAX_CONCURRENT_REQUESTS = 20

    # 减少上下文长度（加快处理）
    CONTEXT_BEFORE_CHARS = 500
    CONTEXT_AFTER_CHARS = 300

    # 减少重试次数
    MAX_RETRIES = 1


# ==================== 如何使用 ====================
"""
在 build_knowledge_es.py 中：

# 方式 1：直接修改 ProcessConfig 类
class ProcessConfig:
    SMALL_TABLE_ROWS = 15  # 修改参数
    ...

# 方式 2：使用预设配置
from config_example import LowCostConfig as ProcessConfig

# 方式 3：运行时动态修改
config = ProcessConfig()
config.FAST_MODE = True
config.MAX_CONCURRENT_REQUESTS = 5
"""


# ==================== 参数调优建议 ====================
"""
1. 成本优先：
   - 使用 LowCostConfig
   - 或提高 SMALL_TABLE_* 阈值
   - 禁用 ENABLE_IMAGE_ANALYSIS

2. 质量优先：
   - 使用 HighQualityConfig
   - 降低 SMALL_TABLE_* 阈值
   - 增加 CONTEXT_* 长度

3. 速度优先：
   - 使用 HighSpeedConfig
   - 提高 MAX_CONCURRENT_REQUESTS
   - 减少 CONTEXT_* 长度

4. 平衡模式：
   - 使用默认 ProcessConfig
   - 根据实际情况微调

5. API 限流问题：
   - 降低 MAX_CONCURRENT_REQUESTS
   - 增加 REQUEST_TIMEOUT

6. 内存不足：
   - 降低 MAX_CONCURRENT_REQUESTS
   - 分批处理文档
"""
