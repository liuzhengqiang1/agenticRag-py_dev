# -*- coding: utf-8 -*-
"""
工具安全包装器：捕获异常，返回自然语言

【核心功能】
1. 捕获工具执行中的所有异常
2. 将异常转换为自然语言返回给 LLM
3. 记录完整堆栈到日志

【使用方式】
    @tool
    @safe_tool
    def my_tool(query: str) -> str:
        ...
"""

import functools
import traceback
from typing import Callable, Any


def safe_tool(func: Callable) -> Callable:
    """
    工具安全装饰器

    捕获所有异常，返回自然语言格式：
    "Tool execution failed: {tool_name} - {error_detail}"

    参数:
        func: 被装饰的工具函数

    返回:
        包装后的函数
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            tool_name = func.__name__
            error_detail = str(e)

            # 记录完整堆栈
            print(f"[Tool Error] {tool_name} 执行失败")
            print(f"[Tool Error] 错误信息: {error_detail}")
            print(f"[Tool Error] 堆栈追踪:\n{traceback.format_exc()}")

            # 返回自然语言，让 LLM 知道失败了
            return f"Tool execution failed: {tool_name} - {error_detail}"

    return wrapper


def safe_tool_async(func: Callable) -> Callable:
    """
    异步工具安全装饰器

    参数:
        func: 被装饰的异步工具函数

    返回:
        包装后的异步函数
    """

    @functools.wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            tool_name = func.__name__
            error_detail = str(e)

            # 记录完整堆栈
            print(f"[Tool Error] {tool_name} 执行失败")
            print(f"[Tool Error] 错误信息: {error_detail}")
            print(f"[Tool Error] 堆栈追踪:\n{traceback.format_exc()}")

            # 返回自然语言
            return f"Tool execution failed: {tool_name} - {error_detail}"

    return wrapper
