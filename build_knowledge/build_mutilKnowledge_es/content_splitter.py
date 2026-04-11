# -*- coding: utf-8 -*-
"""
内容切分模块 v2.0 - 语义感知的结构化切分策略
核心改进：保护表格、代码块等结构化元素不被切断
"""

import re
from typing import List, Dict, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import ProcessConfig


class ElementProtector:
    """元素保护器：识别并保护结构化元素"""

    def __init__(self):
        self.protected_elements = {}
        self.counter = 0
        # 占位符正则模式，用于检测和保护
        self.placeholder_pattern = re.compile(r"__PROTECTED_\w+_\d+__")

    def _generate_placeholder(self, element_type: str) -> str:
        """生成唯一占位符（确保独立成行）"""
        self.counter += 1
        return f"\n__PROTECTED_{element_type.upper()}_{self.counter}__\n"

    def contains_placeholder(self, text: str) -> bool:
        """检查文本是否包含占位符"""
        return bool(self.placeholder_pattern.search(text))

    def extract_placeholders(self, text: str) -> List[str]:
        """提取文本中的所有占位符"""
        return self.placeholder_pattern.findall(text)

    def protect(self, content: str) -> str:
        """保护表格和代码块，返回带占位符的内容"""
        # 先保护代码块（可能包含表格语法）
        content = self._protect_code_blocks(content)
        # 再保护表格
        content = self._protect_tables(content)
        return content

    def _protect_code_blocks(self, content: str) -> str:
        """保护代码块"""
        # 匹配 ```...``` 代码块
        pattern = r"```[\s\S]*?```"

        def replace(match):
            placeholder = self._generate_placeholder("CODE")
            # 去除占位符中的换行符，存储时使用纯占位符
            clean_placeholder = placeholder.strip()
            self.protected_elements[clean_placeholder] = match.group(0)
            return placeholder  # 返回带换行的占位符

        return re.sub(pattern, replace, content)

    def _protect_tables(self, content: str) -> str:
        """保护 Markdown 表格"""
        # 匹配表格：以 | 开头的连续行
        pattern = r"(?:^|\n)(\|[^\n]+\n\|[-:\s|]+\n(?:\|[^\n]+\n?)+)"

        def replace(match):
            placeholder = self._generate_placeholder("TABLE")
            # 去除占位符中的换行符，存储时使用纯占位符
            clean_placeholder = placeholder.strip()
            self.protected_elements[clean_placeholder] = match.group(1).strip()
            return placeholder  # 返回带换行的占位符

        return re.sub(pattern, replace, content)

    def restore(self, content: str) -> str:
        """恢复被保护的元素"""
        for placeholder, original in self.protected_elements.items():
            content = content.replace(placeholder, original)
        return content


class SemanticTextSplitter:
    """语义感知的文本切分器"""

    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # 语义边界分隔符（优先级从高到低）
        self.separators = [
            "\n\n\n",  # 多个空行
            "\n\n",  # 段落
            "\n",  # 行
            "。",  # 句号
            "！",  # 感叹号
            "？",  # 问号
            "；",  # 分号
            "，",  # 逗号
            " ",  # 空格
            "",  # 字符
        ]

    def split(self, text: str, protector: ElementProtector = None) -> List[str]:
        """切分文本，保护关键元素"""
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        current_chunk = ""
        lines = text.split("\n")
        i = 0

        while i < len(lines):
            line = lines[i]

            # 检查是否是受保护元素的占位符
            if protector and line.strip() in protector.protected_elements:
                element_content = protector.protected_elements[line.strip()]

                # 如果当前块 + 元素不超过限制，直接添加
                if len(current_chunk) + len(element_content) + 1 <= self.chunk_size:
                    current_chunk += (
                        "\n" + element_content if current_chunk else element_content
                    )
                else:
                    # 保存当前块
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    current_chunk = element_content
            else:
                # 检查行中是否包含占位符（防止占位符被切分）
                if protector and protector.contains_placeholder(line):
                    # 包含占位符的行不切分，整行处理
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    current_chunk = line
                else:
                    # 普通行处理
                    test_chunk = current_chunk + "\n" + line if current_chunk else line

                    if len(test_chunk) <= self.chunk_size:
                        current_chunk = test_chunk
                    else:
                        # 当前块已满，需要切分
                        if current_chunk.strip():
                            chunks.append(current_chunk.strip())

                        # 如果单行超过 chunk_size，需要进一步切分
                        if len(line) > self.chunk_size:
                            sub_chunks = self._split_long_line(line, protector)
                            chunks.extend(sub_chunks[:-1])
                            current_chunk = sub_chunks[-1] if sub_chunks else ""
                        else:
                            current_chunk = line

            i += 1

        # 处理最后一个块
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # 添加重叠（需要保护占位符）
        if self.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._add_overlap(chunks, protector)

        return chunks

    def _split_long_line(
        self, line: str, protector: ElementProtector = None
    ) -> List[str]:
        """切分超长行（保护占位符）"""
        if len(line) <= self.chunk_size:
            return [line]

        # 如果包含占位符，不切分
        if protector and protector.contains_placeholder(line):
            return [line]

        chunks = []
        for sep in self.separators:
            if sep in line:
                parts = line.split(sep)
                current = ""
                for part in parts:
                    test = current + part + sep if current else part + sep
                    if len(test) <= self.chunk_size:
                        current = test
                    else:
                        if current:
                            chunks.append(current)
                        current = part + sep
                if current:
                    chunks.append(current)
                break
        else:
            # 无法按语义切分，强制按字符切分（但仍需保护占位符）
            if protector and protector.contains_placeholder(line):
                # 包含占位符，不切分
                return [line]

            for j in range(0, len(line), self.chunk_size):
                chunks.append(line[j : j + self.chunk_size])

        return chunks if chunks else [line]

    def _add_overlap(
        self, chunks: List[str], protector: ElementProtector = None
    ) -> List[str]:
        """添加块间重叠（保护占位符完整性）"""
        overlapped = []
        for i, chunk in enumerate(chunks):
            if i > 0:
                prev_chunk = chunks[i - 1]

                # 如果启用了保护器，检查截断位置是否会破坏占位符
                if protector:
                    # 获取重叠文本
                    overlap_start = max(0, len(prev_chunk) - self.chunk_overlap)
                    overlap_text = prev_chunk[overlap_start:]

                    # 检查重叠文本是否包含不完整的占位符
                    placeholders = protector.extract_placeholders(prev_chunk)
                    if placeholders:
                        # 找到最后一个完整占位符的位置
                        last_placeholder = placeholders[-1]
                        last_pos = prev_chunk.rfind(last_placeholder)

                        # 如果重叠区域会截断占位符，调整起始位置
                        if last_pos >= overlap_start and last_pos + len(
                            last_placeholder
                        ) > len(prev_chunk):
                            # 从占位符开始位置取重叠
                            overlap_text = prev_chunk[last_pos:]
                        elif (
                            overlap_start > last_pos
                            and overlap_start < last_pos + len(last_placeholder)
                        ):
                            # 重叠起点在占位符中间，向前调整到占位符开始
                            overlap_text = prev_chunk[last_pos:]
                else:
                    overlap_text = (
                        prev_chunk[-self.chunk_overlap :]
                        if len(prev_chunk) > self.chunk_overlap
                        else prev_chunk
                    )

                chunk = overlap_text + "\n" + chunk
            overlapped.append(chunk)
        return overlapped


class TableSplitter:
    """表格切分器：按行切分大表格，保留表头"""

    @staticmethod
    def split_table(table_text: str, max_size: int) -> List[str]:
        """切分大表格，保留表头"""
        lines = table_text.strip().split("\n")

        if len(table_text) <= max_size or len(lines) <= 2:
            return [table_text]

        # 提取表头（第一行 + 分隔行）
        header_lines = lines[:2]
        header = "\n".join(header_lines)

        # 数据行
        data_lines = lines[2:]

        # 按行切分，确保每个块不超过 max_size
        chunks = []
        current_rows = []
        current_size = len(header)

        for row in data_lines:
            row_size = len(row) + 1  # +1 for newline
            if current_size + row_size > max_size and current_rows:
                # 保存当前块
                chunk = header + "\n" + "\n".join(current_rows)
                chunks.append(chunk)
                current_rows = []
                current_size = len(header)

            current_rows.append(row)
            current_size += row_size

        # 处理剩余行
        if current_rows:
            chunk = header + "\n" + "\n".join(current_rows)
            chunks.append(chunk)

        return chunks if chunks else [table_text]


def extract_headers(content: str) -> List[Dict]:
    """提取标题结构"""
    headers = []
    pattern = r"^(#{1,6})\s+(.+)$"

    for match in re.finditer(pattern, content, re.MULTILINE):
        level = len(match.group(1))
        title = match.group(2).strip()
        position = match.start()
        headers.append(
            {
                "level": level,
                "title": title,
                "position": position,
                "line": match.group(0),
            }
        )

    return headers


def split_by_headers(content: str, headers: List[Dict]) -> List[Dict]:
    """按标题切分内容"""
    if not headers:
        return [{"content": content, "h1": "", "h2": "", "h3": ""}]

    sections = []
    content_len = len(content)

    for i, header in enumerate(headers):
        start = header["position"]
        end = headers[i + 1]["position"] if i + 1 < len(headers) else content_len

        section_content = content[start:end].strip()

        # 确定当前标题层级
        h1, h2, h3 = "", "", ""
        for h in reversed(headers[: i + 1]):
            if h["level"] == 1 and not h1:
                h1 = h["title"]
            elif h["level"] == 2 and not h2:
                h2 = h["title"]
            elif h["level"] == 3 and not h3:
                h3 = h["title"]

        sections.append({"content": section_content, "h1": h1, "h2": h2, "h3": h3})

    # 处理标题前的内容
    if headers[0]["position"] > 0:
        pre_content = content[: headers[0]["position"]].strip()
        if pre_content:
            sections.insert(0, {"content": pre_content, "h1": "", "h2": "", "h3": ""})

    return sections


def split_markdown_content(content: str, doc_metadata: Dict) -> List[Dict]:
    """
    使用语义感知策略切分 markdown 内容

    流程：
    1. 保护表格和代码块
    2. 按标题层级切分
    3. 对超长章节进行语义切分
    4. 恢复被保护的元素
    5. 处理超大表格
    """
    chunks = []
    chunk_size = ProcessConfig.CHUNK_SIZE
    chunk_overlap = ProcessConfig.CHUNK_OVERLAP

    # 1. 保护结构化元素
    protector = ElementProtector()
    protected_content = protector.protect(content)

    # 2. 按标题切分
    headers = extract_headers(protected_content)
    sections = split_by_headers(protected_content, headers)

    # 3. 对每个章节进行切分
    splitter = SemanticTextSplitter(chunk_size, chunk_overlap)
    table_splitter = TableSplitter()
    chunk_id = 0

    for section in sections:
        section_content = section["content"]

        if not section_content.strip():
            continue

        # 恢复当前章节中的保护元素
        restored_content = protector.restore(section_content)

        # 检查是否需要切分
        if len(restored_content) <= chunk_size:
            chunk_id += 1
            chunks.append(
                {
                    "content": restored_content,
                    "metadata": {
                        **doc_metadata,
                        "chunk_id": chunk_id,
                        "h1": section["h1"],
                        "h2": section["h2"],
                        "h3": section["h3"],
                        "sub_chunk_id": 0,
                    },
                }
            )
        else:
            # 需要切分
            sub_chunks = splitter.split(restored_content, protector)

            for sub_id, sub_chunk in enumerate(sub_chunks):
                # 检查是否包含大表格需要进一步切分
                if _contains_large_table(sub_chunk, chunk_size):
                    table_chunks = _split_tables_in_chunk(
                        sub_chunk, table_splitter, chunk_size
                    )
                    for tc in table_chunks:
                        chunk_id += 1
                        chunks.append(
                            {
                                "content": tc,
                                "metadata": {
                                    **doc_metadata,
                                    "chunk_id": chunk_id,
                                    "h1": section["h1"],
                                    "h2": section["h2"],
                                    "h3": section["h3"],
                                    "sub_chunk_id": sub_id,
                                },
                            }
                        )
                else:
                    chunk_id += 1
                    chunks.append(
                        {
                            "content": sub_chunk,
                            "metadata": {
                                **doc_metadata,
                                "chunk_id": chunk_id,
                                "h1": section["h1"],
                                "h2": section["h2"],
                                "h3": section["h3"],
                                "sub_chunk_id": sub_id,
                            },
                        }
                    )

    return chunks


def _contains_large_table(content: str, max_size: int) -> bool:
    """检查内容是否包含超大表格"""
    pattern = r"(?:^|\n)(\|[^\n]+\n\|[-:\s|]+\n(?:\|[^\n]+\n?)+)"
    for match in re.finditer(pattern, content):
        table = match.group(1)
        if len(table) > max_size:
            return True
    return False


def _split_tables_in_chunk(
    content: str, table_splitter: TableSplitter, max_size: int
) -> List[str]:
    """切分内容中的超大表格"""
    result = []
    pattern = r"(?:^|\n)(\|[^\n]+\n\|[-:\s|]+\n(?:\|[^\n]+\n?)+)"

    last_end = 0
    for match in re.finditer(pattern, content):
        # 添加表格前的内容
        if match.start() > last_end:
            result.append(content[last_end : match.start()].strip())

        table = match.group(1)
        if len(table) > max_size:
            # 切分大表格
            table_chunks = table_splitter.split_table(table, max_size)
            result.extend(table_chunks)
        else:
            result.append(table)

        last_end = match.end()

    # 添加最后的内容
    if last_end < len(content):
        remaining = content[last_end:].strip()
        if remaining:
            result.append(remaining)

    return [r for r in result if r.strip()]
