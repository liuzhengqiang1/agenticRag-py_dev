# -*- coding: utf-8 -*-
"""Redis 配置模块"""
import os
from typing import Optional


class RedisConfig:
    """
    Redis 连接配置类
    
    使用方式：
    1. 在 .env 文件中配置环境变量
    2. 或者直接在代码中实例化时传入参数
    """
    
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        db: Optional[int] = None,
        password: Optional[str] = None,
        username: Optional[str] = None,
        decode_responses: bool = True,
        socket_timeout: int = 5,
        socket_connect_timeout: int = 5,
        max_connections: int = 50
    ):
        """
        初始化 Redis 配置
        
        参数：
            host: Redis 服务器地址（默认从环境变量 REDIS_HOST 读取，或使用 localhost）
            port: Redis 端口（默认从环境变量 REDIS_PORT 读取，或使用 6379）
            db: Redis 数据库编号（默认从环境变量 REDIS_DB 读取，或使用 0）
            password: Redis 密码（默认从环境变量 REDIS_PASSWORD 读取）
            username: Redis 用户名（默认从环境变量 REDIS_USERNAME 读取，Redis 6.0+ 支持）
            decode_responses: 是否自动解码响应为字符串
            socket_timeout: Socket 超时时间（秒）
            socket_connect_timeout: Socket 连接超时时间（秒）
            max_connections: 连接池最大连接数
        """
        self.host = host or os.getenv("REDIS_HOST", "localhost")
        self.port = port or int(os.getenv("REDIS_PORT", "6379"))
        self.db = db or int(os.getenv("REDIS_DB", "0"))
        self.password = password or os.getenv("REDIS_PASSWORD")
        self.username = username or os.getenv("REDIS_USERNAME")
        self.decode_responses = decode_responses
        self.socket_timeout = socket_timeout
        self.socket_connect_timeout = socket_connect_timeout
        self.max_connections = max_connections
    
    def get_connection_kwargs(self) -> dict:
        """
        获取 Redis 连接参数字典
        
        返回：
            适用于 redis.Redis() 或 RedisChatMessageHistory 的连接参数
        """
        kwargs = {
            "host": self.host,
            "port": self.port,
            "db": self.db,
            "decode_responses": self.decode_responses,
            "socket_timeout": self.socket_timeout,
            "socket_connect_timeout": self.socket_connect_timeout,
            "max_connections": self.max_connections
        }
        
        # 只有在设置了密码时才添加 password 参数
        if self.password:
            kwargs["password"] = self.password
        
        # 只有在设置了用户名时才添加 username 参数（Redis 6.0+ ACL 功能）
        if self.username:
            kwargs["username"] = self.username
        
        return kwargs
    
    def get_url(self) -> str:
        """
        获取 Redis 连接 URL
        
        返回：
            格式：redis://[username:password@]host:port/db
        """
        auth = ""
        if self.username and self.password:
            auth = f"{self.username}:{self.password}@"
        elif self.password:
            auth = f":{self.password}@"
        
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"
