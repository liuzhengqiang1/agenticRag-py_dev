# -*- coding: utf-8 -*-
"""
快速启动脚本
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from build_knowledge.build_mutilKnowledge_es.build_knowledge_es import (
    build_es_knowledge_base,
    logger,
)


if __name__ == "__main__":
    try:
        build_es_knowledge_base()
    except KeyboardInterrupt:
        logger.warning("用户中断执行")
    except Exception as e:
        logger.error(f"构建失败：{e}", exc_info=True)
        sys.exit(1)
