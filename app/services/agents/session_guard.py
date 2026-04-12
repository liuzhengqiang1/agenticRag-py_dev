# -*- coding: utf-8 -*-
"""
会话守护器：分布式锁 + 异常计数

【核心功能】
1. 分布式锁：防止同一 session_id 并发请求导致 State 脏写
2. 异常计数：记录当前请求的工具错误次数，支持熔断

【Redis Key 设计】
- Key: agentic_guard:{session_id}
- Value: JSON {"locked": 1, "error_count": n}
- TTL: 30秒（自动过期防死锁）

【降级策略】
Redis 不可用时，降级到进程内 asyncio.Lock + dict
"""

import json
import asyncio
from typing import Optional, Dict
from functools import wraps

from app.core.redis_config import RedisConfig


class SessionGuard:
    """
    会话守护器（单例模式）

    使用方式：
        # 获取锁
        acquired = await SessionGuard.acquire("session_123")
        if not acquired:
            raise RuntimeError("请勿重复提交")

        try:
            # 执行业务逻辑
            ...
        finally:
            # 释放锁
            await SessionGuard.release("session_123")
    """

    _instance = None
    _initialized = False
    _using_redis = False
    _redis_client = None

    # 进程内降级存储
    _memory_locks: Dict[str, asyncio.Lock] = {}
    _memory_error_counts: Dict[str, int] = {}
    _memory_lock = asyncio.Lock()  # 保护上述 dict 的全局锁

    # Redis 配置
    KEY_PREFIX = "agentic_guard:"
    LOCK_TTL = 30  # 锁超时时间（秒）
    MAX_RETRIES = 3  # 最大重试次数

    def __init__(self):
        """私有构造函数"""
        if SessionGuard._instance is not None:
            raise RuntimeError("请使用 SessionGuard.get_instance() 获取实例")
        SessionGuard._instance = self

    @classmethod
    def get_instance(cls):
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    async def initialize(cls):
        """
        初始化会话守护器（项目启动时调用）
        """
        if cls._initialized:
            return cls._instance

        print("=" * 60)
        print("会话守护器初始化中...")

        instance = cls.get_instance()

        # 尝试连接 Redis
        try:
            import redis.asyncio as redis_async

            redis_config = RedisConfig(decode_responses=True)
            cls._redis_client = redis_async.Redis(
                **redis_config.get_connection_kwargs()
            )

            # 测试连接
            await cls._redis_client.ping()

            cls._using_redis = True
            print("  - Redis 连接成功，使用分布式锁")

        except Exception as e:
            print(f"  - Redis 连接失败: {e}")
            print("  - 降级到进程内锁")
            cls._using_redis = False

        cls._initialized = True
        print(f"  - 锁超时: {cls.LOCK_TTL}秒")
        print(f"  - 最大重试次数: {cls.MAX_RETRIES}")
        print("=" * 60)

        return instance

    @classmethod
    async def acquire(cls, session_id: str) -> bool:
        """
        获取分布式锁

        参数:
            session_id: 会话 ID

        返回:
            True 表示获取成功，False 表示已被锁定
        """
        key = f"{cls.KEY_PREFIX}{session_id}"

        if cls._using_redis:
            try:
                # SET NX EX：原子操作，不存在时设置，并设置过期时间
                result = await cls._redis_client.set(
                    key,
                    json.dumps({"locked": 1, "error_count": 0}),
                    nx=True,
                    ex=cls.LOCK_TTL,
                )
                return result is not None
            except Exception as e:
                print(f"[SessionGuard] Redis 获取锁失败: {e}")
                # Redis 失败，降级到内存
                return await cls._acquire_memory(session_id)
        else:
            return await cls._acquire_memory(session_id)

    @classmethod
    async def _acquire_memory(cls, session_id: str) -> bool:
        """进程内锁获取"""
        async with cls._memory_lock:
            if session_id not in cls._memory_locks:
                cls._memory_locks[session_id] = asyncio.Lock()

            lock = cls._memory_locks[session_id]
            acquired = lock.locked() == False

            if acquired:
                await lock.acquire()
                cls._memory_error_counts[session_id] = 0

            return acquired

    @classmethod
    async def increment_error(cls, session_id: str) -> int:
        """
        增加错误计数

        参数:
            session_id: 会话 ID

        返回:
            当前错误次数
        """
        key = f"{cls.KEY_PREFIX}{session_id}"

        if cls._using_redis:
            try:
                # 获取当前值
                value = await cls._redis_client.get(key)
                if not value:
                    return 0

                data = json.loads(value)
                data["error_count"] = data.get("error_count", 0) + 1

                # 写回（保持 TTL）
                await cls._redis_client.set(key, json.dumps(data), ex=cls.LOCK_TTL)

                return data["error_count"]

            except Exception as e:
                print(f"[SessionGuard] Redis 增加计数失败: {e}")
                return await cls._increment_error_memory(session_id)
        else:
            return await cls._increment_error_memory(session_id)

    @classmethod
    async def _increment_error_memory(cls, session_id: str) -> int:
        """进程内增加计数"""
        async with cls._memory_lock:
            count = cls._memory_error_counts.get(session_id, 0) + 1
            cls._memory_error_counts[session_id] = count
            return count

    @classmethod
    async def get_error_count(cls, session_id: str) -> int:
        """
        获取当前错误计数

        参数:
            session_id: 会话 ID

        返回:
            当前错误次数
        """
        key = f"{cls.KEY_PREFIX}{session_id}"

        if cls._using_redis:
            try:
                value = await cls._redis_client.get(key)
                if not value:
                    return 0
                data = json.loads(value)
                return data.get("error_count", 0)
            except Exception as e:
                print(f"[SessionGuard] Redis 获取计数失败: {e}")
                return cls._memory_error_counts.get(session_id, 0)
        else:
            return cls._memory_error_counts.get(session_id, 0)

    @classmethod
    async def should_break(cls, session_id: str) -> bool:
        """
        判断是否应该熔断

        参数:
            session_id: 会话 ID

        返回:
            True 表示应该熔断
        """
        count = await cls.get_error_count(session_id)
        return count >= cls.MAX_RETRIES

    @classmethod
    async def release(cls, session_id: str):
        """
        释放锁

        参数:
            session_id: 会话 ID
        """
        key = f"{cls.KEY_PREFIX}{session_id}"

        if cls._using_redis:
            try:
                await cls._redis_client.delete(key)
            except Exception as e:
                print(f"[SessionGuard] Redis 释放锁失败: {e}")
                await cls._release_memory(session_id)
        else:
            await cls._release_memory(session_id)

    @classmethod
    async def _release_memory(cls, session_id: str):
        """进程内释放锁"""
        async with cls._memory_lock:
            # 释放锁
            if session_id in cls._memory_locks:
                lock = cls._memory_locks[session_id]
                if lock.locked():
                    lock.release()

            # 清理计数
            cls._memory_error_counts.pop(session_id, None)


# 便捷函数
async def init_session_guard():
    """初始化会话守护器"""
    return await SessionGuard.initialize()


def get_session_guard():
    """获取会话守护器实例"""
    return SessionGuard.get_instance()
