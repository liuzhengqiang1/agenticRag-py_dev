"""
意图识别模块：判断用户查询的意图和是否符合业务范围

功能：
1. 业务护栏：识别并拒绝超出能力范围的请求
2. 缓存策略：判断查询是否适合使用缓存
3. 意图清晰度：判断查询意图是否明确
"""

from typing import Tuple, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


# Agent 能力范围描述
AGENT_CAPABILITIES = """
当前 Agent 支持的能力范围：
1. 知识库检索：回答基于已有文档的问题
2. 天气查询：查询指定城市的天气信息
3. 订单查询：查询用户的订单信息
4. 互联网搜索：搜索最新的互联网信息

不支持的能力：
- 代码编写和调试
- 创意内容生成（故事、诗歌等）
- 报表生成和数据分析
- 图片生成和处理
- 文件操作和系统命令
"""


async def classify_intent_async(
    query: str,
) -> Tuple[Literal["accept", "reject", "unclear"], str, bool]:
    """
    异步版本：分类用户查询意图

    参数:
        query: 用户查询（已重写后的）

    返回:
        (意图状态, 原因说明, 是否应该使用缓存)
        - accept: 接受查询，继续处理
        - reject: 拒绝查询，超出能力范围
        - unclear: 意图不明确，需要重新改写
    """
    llm = ChatOpenAI(model="qwen-turbo", temperature=0)

    system_prompt = f"""你是一个意图识别专家，负责判断用户查询是否符合业务范围。

{AGENT_CAPABILITIES}

你需要完成三个任务：

任务1：判断查询是否在能力范围内
- 如果查询明确属于支持的能力范围，返回 "accept"
- 如果查询明确超出能力范围，返回 "reject"
- 如果查询意图不明确、过于模糊或缺少关键信息，返回 "unclear"

任务2：判断是否应该使用缓存
- 事实性查询（天气、订单、知识库检索）：应该使用缓存
- 生成式查询（需要创意、个性化回答）：不应该使用缓存
- 时效性查询（最新新闻、实时数据）：不应该使用缓存

任务3：提供简短的原因说明

请严格按照以下格式返回（每行一个字段）：
状态: [accept/reject/unclear]
缓存: [yes/no]
原因: [简短说明]

示例1：
用户查询：北京今天天气怎么样？
你的回答：
状态: accept
缓存: yes
原因: 天气查询属于支持范围

示例2：
用户查询：帮我写一个 Python 爬虫
你的回答：
状态: reject
缓存: no
原因: 代码编写不在支持范围内

示例3：
用户查询：那个怎么样？
你的回答：
状态: unclear
缓存: no
原因: 指代不明确，缺少上下文信息
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"用户查询：{query}"),
    ]

    response = await llm.ainvoke(messages)
    result_text = response.content.strip()

    # 解析返回结果
    status = "accept"
    use_cache = True
    reason = ""

    for line in result_text.split("\n"):
        line = line.strip()
        if line.startswith("状态:") or line.startswith("状态："):
            status_value = line.split(":", 1)[1].strip()
            if status_value in ["accept", "reject", "unclear"]:
                status = status_value
        elif line.startswith("缓存:") or line.startswith("缓存："):
            cache_value = line.split(":", 1)[1].strip().lower()
            use_cache = cache_value == "yes"
        elif line.startswith("原因:") or line.startswith("原因："):
            reason = line.split(":", 1)[1].strip()

    return status, reason, use_cache


def classify_intent(
    query: str,
) -> Tuple[Literal["accept", "reject", "unclear"], str, bool]:
    """
    同步版本：分类用户查询意图（内部调用异步版本）

    参数:
        query: 用户查询（已重写后的）

    返回:
        (意图状态, 原因说明, 是否应该使用缓存)
    """
    import asyncio

    try:
        loop = asyncio.get_running_loop()
        # 如果已经在事件循环中，使用 create_task
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, classify_intent_async(query))
            return future.result()
    except RuntimeError:
        # 没有运行中的事件循环，直接运行
        return asyncio.run(classify_intent_async(query))
