# -*- coding: utf-8 -*-
"""
元数据提取模块
"""

import os
from datetime import datetime
from typing import Dict, Tuple, Optional
from file_utils import calculate_file_hash


def parse_filename(filename: str) -> Tuple[str, str, Optional[str]]:
    """
    解析文件名提取元数据
    格式：{标题}_{作者/部门}_{YYYYMMDD}.md
    """
    name_without_ext = os.path.splitext(filename)[0]
    parts = name_without_ext.split("_")

    if len(parts) < 3:
        return (name_without_ext, "未知", None)

    title = parts[0]
    author = parts[1]
    date_str = parts[2]

    try:
        if len(date_str) == 8 and date_str.isdigit():
            year = date_str[:4]
            month = date_str[4:6]
            day = date_str[6:8]
            created_at = f"{year}-{month}-{day}"
        else:
            created_at = None
    except:
        created_at = None

    return (title, author, created_at)


def extract_metadata_from_path(file_path: str) -> Dict:
    """从文件路径提取元数据"""
    normalized_path = file_path.replace("\\", "/")
    parts = normalized_path.split("/")

    # 提取权限级别
    access_level = "employee"
    if "public" in parts:
        access_level = "public"
    elif "employee" in parts:
        access_level = "employee"
    elif "manager" in parts:
        access_level = "manager"
    elif "executive" in parts:
        access_level = "executive"

    # 提取部门
    department = "未知部门"
    try:
        access_idx = next(
            i
            for i, p in enumerate(parts)
            if p in ["public", "employee", "manager", "executive"]
        )
        if access_idx + 1 < len(parts) and not parts[access_idx + 1].isdigit():
            department = parts[access_idx + 1]
    except:
        pass

    # 提取年份
    year = None
    for part in parts:
        if part.isdigit() and len(part) == 4:
            year = part
            break

    # 解析文件名
    filename = os.path.basename(file_path)
    title, author, created_at = parse_filename(filename)

    # 使用文件创建时间作为备用
    if not created_at:
        try:
            file_ctime = os.path.getctime(file_path)
            created_at = datetime.fromtimestamp(file_ctime).strftime("%Y-%m-%d")
        except:
            created_at = datetime.now().strftime("%Y-%m-%d")

    doc_id = f"{access_level}_{year or 'unknown'}_{title}"

    return {
        "doc_id": doc_id,
        "title": title,
        "author": author,
        "department": department,
        "access_level": access_level,
        "year": year,
        "created_at": created_at,
        "file_path": file_path,
        "file_hash": calculate_file_hash(file_path),
    }
