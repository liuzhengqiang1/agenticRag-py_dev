# -*- coding: utf-8 -*-
"""
Markdown 解析模块：提取图片、表格等元素
"""

import os
import re
from typing import List, Dict, Tuple
from config import ProcessConfig
from file_utils import calculate_content_hash


def extract_context(
    content: str, position: int, before_chars: int = 1000, after_chars: int = 500
) -> Tuple[str, str]:
    """提取指定位置的上下文"""
    start = max(0, position - before_chars)
    end = min(len(content), position + after_chars)
    context_before = content[start:position]
    context_after = content[position:end]
    return context_before, context_after


def extract_images_from_markdown(content: str) -> List[Dict]:
    """提取 Markdown 中的所有图片"""
    images = []
    pattern = r"!\[(.*?)\]\((.*?)\)"

    for match in re.finditer(pattern, content):
        alt_text = match.group(1)
        image_url = match.group(2)
        position = match.start()

        context_before, context_after = extract_context(content, position)

        images.append(
            {
                "id": f"img_{len(images) + 1}",
                "url": image_url,
                "alt": alt_text,
                "position": position,
                "context_before": context_before,
                "context_after": context_after,
                "match_text": match.group(0),
            }
        )

    return images


def parse_markdown_table(table_text: str) -> Tuple[int, int, int]:
    """解析 Markdown 表格，返回 (行数, 列数, 字符数)"""
    lines = [line.strip() for line in table_text.split("\n") if line.strip()]
    data_lines = [line for line in lines if not re.match(r"^\|[\s\-:]+\|$", line)]

    rows = len(data_lines)
    cols = 0

    if data_lines:
        cols = data_lines[0].count("|") - 1

    chars = len(table_text)
    return rows, cols, chars


def classify_table(rows: int, cols: int, chars: int) -> str:
    """分类表格：small / large / giant"""
    config = ProcessConfig()

    if (
        rows <= config.SMALL_TABLE_ROWS
        and cols <= config.SMALL_TABLE_COLS
        and chars <= config.SMALL_TABLE_CHARS
    ):
        return "small"

    if (
        rows <= config.LARGE_TABLE_ROWS
        and cols <= config.LARGE_TABLE_COLS
        and chars <= config.LARGE_TABLE_CHARS
    ):
        return "large"

    return "giant"


def extract_tables_from_markdown(content: str) -> List[Dict]:
    """提取 Markdown 中的所有表格"""
    tables = []
    lines = content.split("\n")
    i = 0

    while i < len(lines):
        if lines[i].strip().startswith("|"):
            table_start = i
            table_lines = []

            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1

            position = len("\n".join(lines[:table_start]))
            table_text = "\n".join(table_lines)

            rows, cols, chars = parse_markdown_table(table_text)
            classification = classify_table(rows, cols, chars)

            context_before, context_after = extract_context(content, position)

            tables.append(
                {
                    "id": f"table_{len(tables) + 1}",
                    "content": table_text,
                    "rows": rows,
                    "cols": cols,
                    "chars": chars,
                    "position": position,
                    "context_before": context_before,
                    "context_after": context_after,
                    "classification": classification,
                }
            )
        else:
            i += 1

    return tables


def filter_image(image_info: Dict, file_dir: str) -> bool:
    """过滤无效图片（装饰线、图标等）"""
    config = ProcessConfig
    image_url = image_info["url"]

    if image_url.startswith("http://") or image_url.startswith("https://"):
        return True

    try:
        if not os.path.isabs(image_url):
            image_path = os.path.join(file_dir, image_url)
        else:
            image_path = image_url

        if not os.path.exists(image_path):
            return False

        file_size = os.path.getsize(image_path)
        if file_size < config.MIN_IMAGE_SIZE:
            return False

        try:
            from PIL import Image

            with Image.open(image_path) as img:
                width, height = img.size

                if width < config.MIN_IMAGE_WIDTH or height < config.MIN_IMAGE_HEIGHT:
                    return False

                aspect_ratio = max(width, height) / min(width, height)
                if aspect_ratio > config.MAX_ASPECT_RATIO:
                    return False
        except ImportError:
            # PIL 不可用时，只检查文件大小
            pass
        except Exception:
            # 图片打开失败，保守处理：保留
            pass

        return True

    except Exception:
        # 发生异常时保守处理：保留图片
        return True


def scan_markdown_elements(content: str, file_path: str) -> Dict:
    """扫描 Markdown 中的所有图片和表格"""
    file_dir = os.path.dirname(file_path)

    # 提取并过滤图片
    images = extract_images_from_markdown(content)
    filtered_images = []
    image_hashes = set()

    for img in images:
        img_hash = calculate_content_hash(img["url"])
        if img_hash in image_hashes:
            continue

        if not filter_image(img, file_dir):
            continue

        img["hash"] = img_hash
        img["action"] = "process" if ProcessConfig.ENABLE_IMAGE_ANALYSIS else "skip"
        filtered_images.append(img)
        image_hashes.add(img_hash)

    # 提取表格
    tables = extract_tables_from_markdown(content)

    for table in tables:
        table["hash"] = calculate_content_hash(table["content"])

        if table["classification"] == "small":
            table["action"] = "bypass"
        elif table["classification"] == "large":
            table["action"] = (
                "summarize" if ProcessConfig.ENABLE_TABLE_ANALYSIS else "bypass"
            )
        else:  # giant
            table["action"] = (
                "split_and_summarize"
                if ProcessConfig.ENABLE_TABLE_ANALYSIS
                else "bypass"
            )

    return {"images": filtered_images, "tables": tables}


def inject_summaries_to_markdown(
    content: str, analyzed_images: List[Dict], analyzed_tables: List[Dict]
) -> str:
    """将分析结果注入到原 Markdown 中"""
    all_elements = []

    for img in analyzed_images:
        if "analysis" in img:
            all_elements.append(
                {
                    "position": img["position"],
                    "original": img["match_text"],
                    "replacement": f"""<figure id="{img['id']}">
<summary>{img['analysis']['summary']}</summary>
<keywords>{', '.join(img['analysis']['keywords'])}</keywords>
{img['match_text']}
</figure>""",
                }
            )

    for table in analyzed_tables:
        if "analysis" in table:
            all_elements.append(
                {
                    "position": table["position"],
                    "original": table["content"],
                    "replacement": f"""<figure id="{table['id']}">
<summary>{table['analysis']['summary']}</summary>
<keywords>{', '.join(table['analysis']['keywords'])}</keywords>
<columns>{', '.join(table['analysis'].get('columns', []))}</columns>
{table['content']}
</figure>""",
                }
            )

    all_elements.sort(key=lambda x: x["position"], reverse=True)

    modified_content = content
    for elem in all_elements:
        pos = elem["position"]
        original_len = len(elem["original"])
        modified_content = (
            modified_content[:pos]
            + elem["replacement"]
            + modified_content[pos + original_len :]
        )

    return modified_content
