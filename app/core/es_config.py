# -*- coding: utf-8 -*-
"""Elasticsearch 配置模块"""
import os
from dotenv import load_dotenv

load_dotenv()


class ESConfig:
    """Elasticsearch 配置类（单例模式）"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        # ES 连接配置
        self.host = os.getenv("ES_HOST", "localhost")
        self.port = int(os.getenv("ES_PORT", "9200"))
        self.username = os.getenv("ES_USERNAME", "")
        self.password = os.getenv("ES_PASSWORD", "")
        
        # 索引名称
        self.index_name = os.getenv("ES_INDEX_NAME", "training_docs")
    
    def get_url(self) -> str:
        """获取 ES 连接 URL"""
        return f"http://{self.host}:{self.port}"
    
    def get_connection_params(self) -> dict:
        """获取 ES 连接参数"""
        params = {
            "hosts": [self.get_url()],
        }
        
        # 如果配置了用户名密码，添加认证信息
        if self.username and self.password:
            params["basic_auth"] = (self.username, self.password)
        
        return params
