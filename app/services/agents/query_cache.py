# -*- coding: utf-8 -*-
"""
Query 缓存模块：基于 Redis Stack 向量检索的问题缓存

【核心功能】
- 使用 Redis Stack 的向量检索功能 (FT.SEARCH + KNN)
- 缓存最终回答，避免重复调用 LLM
- 支持 Redis Stack 向量存储，降级到内存存储

【使用方式】
    # 初始化
    await QueryCache.initialize()

    # 查询缓存
    result = await QueryCache.get_similar("用户问题")

    # 写入缓存
    await QueryCache.set("用户问题", "最终回答")
"""

import json
import time
import hashlib
from typing import Optional, Tuple, List
from cachetools import LRUCache

from langchain_community.embeddings import DashScopeEmbeddings

from app.core.redis_config import RedisConfig


class QueryCache:
    """
    Query 缓存管理器（单例模式）

    使用 Redis Stack 向量检索匹配相似问题，缓存最终回答。
    """

    _instance = None
    _initialized = False
    _using_redis = False

    # 缓存配置
    TTL_SECONDS = 12 * 60 * 60  # 12小时
    SIMILARITY_THRESHOLD = 0.90  # 相似度阈值
    MAX_MEMORY_CACHE_SIZE = 500  # 内存缓存最大条数
    EMBEDDING_DIM = 1024  # text-embedding-v4 向量维度

    # Redis 键名
    INDEX_NAME = "query_cache_idx"
    KEY_PREFIX = "query_cache:"

    def __init__(self):
        """私有构造函数"""
        if QueryCache._instance is not None:
            raise RuntimeError("请使用 QueryCache.get_instance() 获取实例")
        QueryCache._instance = self

        self._embeddings = None
        self._redis_client = None
        self._memory_cache = None
        self._cache_index = []

    @classmethod
    def get_instance(cls):
        """获取单例实例"""
        if cls._instance is None:
            raise RuntimeError("请先调用 QueryCache.initialize() 初始化")
        return cls._instance

    @classmethod
    async def initialize(cls):
        """
        初始化 Query 缓存系统

        返回:
            QueryCache 实例
        """
        if cls._initialized:
            return cls._instance

        print("=" * 60)
        print("Query 缓存系统初始化中...")

        instance = cls()

        # 1. 初始化 Embedding 模型
        instance._embeddings = DashScopeEmbeddings(model="text-embedding-v4")
        print("  - Embedding 模型已加载 (text-embedding-v4, 1024维)")

        # 2. 初始化存储
        await instance._init_storage()

        cls._initialized = True

        storage_type = "Redis Stack (向量检索)" if cls._using_redis else "内存 (LRU)"
        print(f"  - 存储模式: {storage_type}")
        print(f"  - TTL: {cls.TTL_SECONDS // 3600}小时")
        print(f"  - 相似度阈值: {cls.SIMILARITY_THRESHOLD}")
        print("=" * 60)

        return instance

    async def _init_storage(self):
        """初始化存储（优先 Redis Stack，失败时降级到内存）"""
        try:
            import redis.asyncio as redis_async

            redis_config = RedisConfig(decode_responses=True)
            self._redis_client = redis_async.Redis(
                **redis_config.get_connection_kwargs()
            )

            # 测试连接
            await self._redis_client.ping()

            # 创建向量索引
            await self._create_vector_index()

            QueryCache._using_redis = True
            print("  - Redis Stack 连接成功，向量索引已就绪")

        except Exception as e:
            print(f"  - Redis Stack 初始化失败: {e}")
            print("  - 降级到内存存储")

            QueryCache._using_redis = False
            self._memory_cache = LRUCache(maxsize=self.MAX_MEMORY_CACHE_SIZE)
            self._cache_index = []

    async def _create_vector_index(self):
        """
        创建 Redis Stack 向量索引

        使用 HNSW 算法进行近似最近邻搜索
        """
        try:
            # 检查索引是否已存在
            indices = await self._redis_client.execute_command("FT._LIST")
            if self.INDEX_NAME.encode() in indices or self.INDEX_NAME in indices:
                print("  - 向量索引已存在，跳过创建")
                # 输出索引信息
                try:
                    info = await self._redis_client.execute_command(
                        "FT.INFO", self.INDEX_NAME
                    )
                    print(f"  - 索引信息: {info[:20]}...")  # 只显示前20项
                except Exception as e:
                    print(f"  - 获取索引信息失败: {e}")
                return

            # 创建索引
            # 使用 HNSW 算法，COSINE 相似度
            await self._redis_client.execute_command(
                "FT.CREATE",
                self.INDEX_NAME,
                "ON",
                "HASH",
                "PREFIX",
                "1",
                self.KEY_PREFIX,
                "SCHEMA",
                "query",
                "TEXT",
                "answer",
                "TEXT",
                "timestamp",
                "NUMERIC",
                "embedding",
                "VECTOR",
                "HNSW",
                "6",
                "TYPE",
                "FLOAT32",
                "DIM",
                str(self.EMBEDDING_DIM),
                "DISTANCE_METRIC",
                "COSINE",
            )
            print("  - 向量索引创建成功 (HNSW, COSINE)")

        except Exception as e:
            # 索引可能已存在，忽略错误
            if "Index already exists" not in str(e):
                print(f"  - 创建向量索引失败: {e}")

    @classmethod
    async def get_similar(cls, query: str) -> Optional[Tuple[str, float]]:
        """
        查找相似问题的缓存回答

        参数:
            query: 用户问题

        返回:
            (cached_answer, similarity) 如果找到相似问题
            None 如果未找到
        """
        instance = cls.get_instance()

        # 1. 生成查询向量
        query_embedding = await instance._embeddings.aembed_query(query)

        # 2. 查找最相似的问题
        if cls._using_redis:
            result = await instance._search_redis_vector(query_embedding)
        else:
            result = instance._search_memory(query_embedding)

        # 3. 输出日志
        if result:
            answer, similarity = result
            print(f"✓ 缓存命中 | 相似度: {similarity:.4f} | 问题: {query[:50]}...")
        else:
            print(f"✗ 缓存未命中 | 问题: {query[:50]}...")

        return result

    @classmethod
    async def set(cls, query: str, answer: str):
        """
        写入缓存

        参数:
            query: 用户问题
            answer: 最终回答
        """
        instance = cls.get_instance()

        # 1. 生成问题向量
        query_embedding = await instance._embeddings.aembed_query(query)

        # 2. 写入存储
        if cls._using_redis:
            await instance._set_redis_vector(query, answer, query_embedding)
        else:
            instance._set_memory(query, answer, query_embedding)

        # 3. 输出日志
        storage_type = "Redis" if cls._using_redis else "内存"
        print(f"→ 缓存已写入 ({storage_type}) | 问题: {query[:50]}...")

    async def _search_redis_vector(
        self, query_embedding: List[float]
    ) -> Optional[Tuple[str, float]]:
        """
        使用 Redis Stack 向量检索搜索相似问题

        参数:
            query_embedding: 查询向量

        返回:
            (answer, similarity) 或 None
        """
        try:
            import numpy as np

            # 将向量转换为 bytes (FLOAT32)
            query_vector = np.array(query_embedding, dtype=np.float32).tobytes()

            # 使用 KNN 搜索最相似的 1 个结果
            # FT.SEARCH 使用 COSINE 距离，需要转换为相似度
            results = await self._redis_client.execute_command(
                "FT.SEARCH",
                self.INDEX_NAME,
                "*=>[KNN 1 @embedding $query_vector AS vector_score]",
                "PARAMS",
                "2",
                "query_vector",
                query_vector,
                "RETURN",
                "3",
                "answer",
                "query",
                "vector_score",
                "DIALECT",
                "2",
            )

            print(f"  [调试] Redis 搜索结果: {results}")

            # 解析结果
            # 格式: [total_results, key1, [field1, value1, field2, value2, ...]]
            if not results or results[0] == 0:
                print(f"  [调试] 未找到任何结果")
                return None

            # 提取字段值
            fields = results[2]  # [field1, value1, field2, value2, ...]
            field_dict = {}
            for i in range(0, len(fields), 2):
                field_dict[fields[i]] = fields[i + 1]

            answer = field_dict.get("answer")
            vector_score = field_dict.get("vector_score")

            print(f"  [调试] 原始距离分数: {vector_score}")

            if not answer or vector_score is None:
                print(f"  [调试] answer 或 vector_score 为空")
                return None

            # __vector_score 是 COSINE 距离 (0-2)，转换为相似度 (0-1)
            # COSINE 距离 = 1 - COSINE 相似度
            # 但 Redis Stack 返回的是距离，需要转换
            similarity = 1.0 - float(vector_score)

            print(
                f"  [调试] 计算相似度: {similarity:.4f}, 阈值: {self.SIMILARITY_THRESHOLD}"
            )

            # 检查是否超过阈值
            if similarity >= self.SIMILARITY_THRESHOLD:
                return (answer, similarity)

            print(f"  [调试] 相似度未达到阈值")
            return None

        except Exception as e:
            print(f"Redis 向量搜索失败: {e}")
            import traceback

            traceback.print_exc()
            return None

    def _search_memory(
        self, query_embedding: List[float]
    ) -> Optional[Tuple[str, float]]:
        """在内存中搜索相似问题"""
        if not self._cache_index:
            return None

        import numpy as np

        best_similarity = 0.0
        best_answer = None

        for cached_embedding, answer in self._cache_index:
            similarity = self._cosine_similarity(query_embedding, cached_embedding)

            if similarity > best_similarity:
                best_similarity = similarity
                best_answer = answer

        if best_similarity >= self.SIMILARITY_THRESHOLD and best_answer:
            return (best_answer, best_similarity)

        return None

    async def _set_redis_vector(self, query: str, answer: str, embedding: List[float]):
        """
        写入 Redis Stack 向量缓存

        参数:
            query: 用户问题
            answer: 最终回答
            embedding: 问题向量
        """
        try:
            import numpy as np

            # 生成 key
            key = f"{self.KEY_PREFIX}{hashlib.md5(query.encode()).hexdigest()}"

            # 将向量转换为 bytes (FLOAT32)
            vector_bytes = np.array(embedding, dtype=np.float32).tobytes()

            print(f"  [调试] 写入 key: {key}")
            print(f"  [调试] 向量维度: {len(embedding)}, 字节长度: {len(vector_bytes)}")

            # 写入 Redis Hash
            await self._redis_client.hset(
                key,
                mapping={
                    "query": query,
                    "answer": answer,
                    "timestamp": str(time.time()),
                    "embedding": vector_bytes,
                },
            )

            # 设置 TTL
            await self._redis_client.expire(key, self.TTL_SECONDS)

            # 验证写入
            exists = await self._redis_client.exists(key)
            print(f"  [调试] 写入验证: key 存在={exists}")

        except Exception as e:
            print(f"Redis 向量写入失败: {e}")
            import traceback

            traceback.print_exc()

    def _set_memory(self, query: str, answer: str, embedding: List[float]):
        """写入内存缓存"""
        key = hashlib.md5(query.encode()).hexdigest()

        # 写入 LRU 缓存
        self._memory_cache[key] = {
            "query": query,
            "answer": answer,
            "embedding": embedding,
        }

        # 更新索引
        self._cache_index.append((embedding, answer))

        # 限制索引大小
        if len(self._cache_index) > self.MAX_MEMORY_CACHE_SIZE:
            self._cache_index = self._cache_index[-self.MAX_MEMORY_CACHE_SIZE :]

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        import numpy as np

        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)

        dot_product = np.dot(vec1_np, vec2_np)
        norm1 = np.linalg.norm(vec1_np)
        norm2 = np.linalg.norm(vec2_np)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))


# 便捷函数
async def init_query_cache():
    """初始化 Query 缓存"""
    return await QueryCache.initialize()


def get_query_cache():
    """获取 Query 缓存实例"""
    return QueryCache.get_instance()
