# -*- coding: utf-8 -*-
"""
文件操作工具模块
"""

import os
import json
import hashlib
from typing import Dict, List


def calculate_file_hash(file_path: str) -> str:
    """计算文件 MD5 哈希值"""
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def calculate_content_hash(content: str) -> str:
    """计算内容 MD5 哈希值"""
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def load_json_file(file_path: str) -> Dict:
    """加载 JSON 文件"""
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    return {}
                return json.loads(content)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_json_file(file_path: str, data: Dict):
    """保存 JSON 文件"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def scan_markdown_files(root_dir: str = "data") -> List[str]:
    """递归扫描目录下的所有 markdown 文件"""
    markdown_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                markdown_files.append(file_path)
    return markdown_files
