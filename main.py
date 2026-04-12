# -*- coding: utf-8 -*-
"""应用入口"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        app="app.main:app",
        host="0.0.0.0",
        port=8100, # 应对wsl隐性占用端口
        reload=True
    )
    # 生产环境增加 Uvicorn workers
    # 限流采用令牌桶，但我微服务架构部署有sentinel没必要再写了，限流熔断降级一次性搞定