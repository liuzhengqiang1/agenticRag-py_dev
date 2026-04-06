# -*- coding: utf-8 -*-
"""
Elasticsearch 连接测试脚本
用途：快速验证 ES 是否正常运行
"""

from elasticsearch import Elasticsearch
from app.core.es_config import ESConfig


def test_es_connection():
    """测试 ES 连接"""
    print("=" * 60)
    print("Elasticsearch 连接测试")
    print("=" * 60)
    
    try:
        # 1. 加载配置
        print("\n[1/3] 正在加载配置...")
        es_config = ESConfig()
        print(f"  ES 地址：{es_config.get_url()}")
        print(f"  索引名称：{es_config.index_name}")
        
        # 2. 创建客户端
        print("\n[2/3] 正在连接 Elasticsearch...")
        es_client = Elasticsearch(**es_config.get_connection_params())

        # 3. 获取集群信息
        info = es_client.info()
        print("✓ 连接成功！")
        print(f"\nES 集群信息：")
        print(f"  - 集群名称：{info['cluster_name']}")
        print(f"  - 节点名称：{info['name']}")
        print(f"  - ES 版本：{info['version']['number']}")
        
        # 4. 检查索引是否存在
        print(f"\n检查索引 '{es_config.index_name}'...")
        if es_client.indices.exists(index=es_config.index_name):
            # 获取文档数量
            count_response = es_client.count(index=es_config.index_name)
            doc_count = count_response['count']
            print(f"✓ 索引已存在，包含 {doc_count} 个文档")
            
            # 获取索引映射
            mapping = es_client.indices.get_mapping(index=es_config.index_name)
            fields = mapping[es_config.index_name]['mappings']['properties'].keys()
            print(f"  字段：{', '.join(fields)}")
        else:
            print(f"⚠️  索引不存在，请先运行：python build_knowledge_es.py")
        
        print("\n" + "=" * 60)
        print("测试完成！ES 运行正常 ✓")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败：{e}")
        return False

if __name__ == "__main__":
    test_es_connection()
