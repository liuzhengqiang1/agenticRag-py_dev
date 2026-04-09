# -*- coding: utf-8 -*-
"""
LLM 分析模块：异步分析图片和表格
"""

import re
import json
import asyncio
from typing import List, Dict
from langchain_community.chat_models import ChatTongyi
from config import ProcessConfig


SUMMARY_PROMPT_TEMPLATE = """你是专业的数据分析师。请分析以下内容：

【上文】
{context_before}

【当前元素】
{element_content}

【下文】
{context_after}

请按以下JSON格式输出：
{{
  "summary": "100字以内的核心结论",
  "keywords": ["关键实体", "时间节点", "指标名称"],
  "columns": ["列名1", "列名2"],
  "data_type": "财务报表/产品数据/其他"
}}

要求：
1. 关键词必须包含可检索的实体词
2. 如果有时间信息，必须提取
3. 如果有数值指标，必须说明
4. 如果是表格，columns 字段必须列出所有列名
5. 如果是图片，columns 字段为空数组
"""


async def analyze_element_async(
    element: Dict, element_type: str, llm_client: ChatTongyi, cache: Dict
) -> Dict:
    """异步分析单个元素（图片或表格）"""
    element_hash = element["hash"]

    if ProcessConfig.ENABLE_CACHE and element_hash in cache:
        print(f"      - 使用缓存: {element['id']}")
        return {**element, "analysis": cache[element_hash]}

    prompt = SUMMARY_PROMPT_TEMPLATE.format(
        context_before=element["context_before"][-500:],
        element_content=element.get("content", element.get("url", "")),
        context_after=element["context_after"][:300],
    )

    try:
        response = await asyncio.to_thread(llm_client.invoke, prompt)
        result_text = response.content

        json_match = re.search(r"\{.*\}", result_text, re.DOTALL)
        if json_match:
            analysis = json.loads(json_match.group(0))
        else:
            analysis = {
                "summary": result_text[:200],
                "keywords": [],
                "columns": [],
                "data_type": "未知",
            }

        if ProcessConfig.ENABLE_CACHE:
            cache[element_hash] = analysis

        return {**element, "analysis": analysis}

    except Exception as e:
        print(f"      ⚠️  分析失败 {element['id']}: {e}")
        return {
            **element,
            "analysis": {
                "summary": f"{element_type}内容",
                "keywords": [],
                "columns": [],
                "data_type": "未知",
            },
        }


async def batch_analyze_elements(
    elements: List[Dict], element_type: str, cache: Dict
) -> List[Dict]:
    """批量异步分析元素"""
    if not elements or ProcessConfig.FAST_MODE:
        return elements

    print(f"    - 正在分析 {len(elements)} 个{element_type}...")

    llm_client = ChatTongyi(model="qwen-max")
    semaphore = asyncio.Semaphore(ProcessConfig.MAX_CONCURRENT_REQUESTS)

    async def analyze_with_semaphore(element):
        async with semaphore:
            return await analyze_element_async(element, element_type, llm_client, cache)

    tasks = [analyze_with_semaphore(elem) for elem in elements]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    analyzed_elements = []
    for result in results:
        if isinstance(result, Exception):
            print(f"      ⚠️  分析异常: {result}")
        else:
            analyzed_elements.append(result)

    print(f"    ✓ 完成分析: {len(analyzed_elements)}/{len(elements)}")

    return analyzed_elements
