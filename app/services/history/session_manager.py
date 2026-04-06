# -*- coding: utf-8 -*-
"""会话历史管理模块"""
from typing import List
from langchain_community.chat_message_histories import RedisChatMessageHistory
from app.core.redis_config import RedisConfig

# 初始化 Redis 配置
redis_config = RedisConfig()


def get_session_history(session_id: str) -> RedisChatMessageHistory:
    """
    根据 session_id 获取或创建 Redis 聊天历史记录

    架构说明：
    - 使用 RedisChatMessageHistory 替代内存版的 ChatMessageHistory
    - 支持分布式部署：多个服务实例共享同一个 Redis
    - 支持持久化：服务重启后会话历史不丢失
    - 支持 TTL：可以设置会话过期时间（通过 Redis 的 EXPIRE 命令）

    参数：
        session_id: 会话 ID，用于区分不同用户的对话历史
        一般来说：前端自己生成session_id(UUID) + 后端获取的user_id 组合在一起 = 上面所说的session_id

    返回：
        RedisChatMessageHistory 实例
    """
    return RedisChatMessageHistory(
        session_id=session_id,
        url=redis_config.get_url(),
        key_prefix="chat_history:",
        ttl=3600,  # 会话过期时间（秒），1小时后自动清理
    )


def get_recent_history(session_id: str, max_rounds: int = 3) -> List[tuple]:
    """
    获取最近 N 轮对话历史（滑动窗口）

    参数：
        session_id: 会话 ID
        max_rounds: 最多保留几轮对话（1轮 = 1个用户消息 + 1个助手回复）

    返回：
        [(user_msg, assistant_msg), ...] 格式的对话历史
    """
    history = get_session_history(session_id)
    messages = history.messages

    # 提取最近的对话轮次（user + assistant 成对）
    recent_pairs = []
    i = len(messages) - 1

    while i >= 1 and len(recent_pairs) < max_rounds:
        # 从后往前找成对的 user + assistant
        if messages[i].type == "ai" and messages[i - 1].type == "human":
            recent_pairs.insert(0, (messages[i - 1].content, messages[i].content))
            i -= 2
        else:
            i -= 1

    return recent_pairs
