# Elasticsearch 知识库构建脚本 v2.0

## 新增功能

### 1. Multi-Vector 检索架构

- 支持图片和表格的智能处理
- 摘要向量化用于检索，原始内容用于展示
- 实现检索召回率和答案质量的双重提升

### 2. 大小表分流策略

- **小表**（行数≤10，列数≤5，字符数≤800）：直接入库，不调用LLM
- **大表**（行数≤30，列数≤10，字符数≤3000）：生成摘要后入库
- **巨型表**（超过大表阈值）：表头广播切分后处理

### 3. 图片智能过滤

- 尺寸过滤：过滤小于 50x50 的图标
- 比例过滤：过滤长宽比超过 20:1 的装饰线
- 文件大小过滤：过滤小于 1KB 的小图标
- Hash 去重：同一图片只处理一次

### 4. 异步批处理优化

- 使用 asyncio 并发调用 LLM API
- 可配置并发数（默认 10）
- 自动重试机制（最多 3 次）
- 显著提升处理速度

### 5. 智能缓存机制

- 基于内容 Hash 的缓存
- 避免重复调用 API
- 缓存文件：`data/.element_analysis_cache.json`

## 配置参数

在 `ProcessConfig` 类中可以调整以下参数：

```python
class ProcessConfig:
    # 表格分类阈值
    SMALL_TABLE_ROWS = 10
    SMALL_TABLE_COLS = 5
    SMALL_TABLE_CHARS = 800

    LARGE_TABLE_ROWS = 30
    LARGE_TABLE_COLS = 10
    LARGE_TABLE_CHARS = 3000

    # 图片过滤阈值
    MIN_IMAGE_WIDTH = 50
    MIN_IMAGE_HEIGHT = 50
    MAX_ASPECT_RATIO = 20
    MIN_IMAGE_SIZE = 1024  # 1KB

    # 表头广播
    GIANT_TABLE_CHUNK_ROWS = 10

    # 并发控制
    MAX_CONCURRENT_REQUESTS = 10
    REQUEST_TIMEOUT = 30
    MAX_RETRIES = 3

    # 上下文窗口
    CONTEXT_BEFORE_CHARS = 1000
    CONTEXT_AFTER_CHARS = 500

    # 模式开关
    ENABLE_IMAGE_ANALYSIS = True
    ENABLE_TABLE_ANALYSIS = True
    ENABLE_CACHE = True
    FAST_MODE = False  # 快速模式：跳过所有LLM分析
```

## 使用方法

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

新增依赖：

- `pillow>=10.0.0`：用于图片尺寸检测

### 2. 配置环境变量

确保 `.env` 文件中配置了：

- `DASHSCOPE_API_KEY`：阿里云 DashScope API Key
- `ES_HOST`、`ES_PORT`：Elasticsearch 连接信息

### 3. 运行脚本

```bash
python build_knowledge/build_knowledge_es.py
```

### 4. 快速模式（跳过LLM分析）

如果只想快速导入文档，不进行图表分析：

```python
# 在脚本开头设置
ProcessConfig.FAST_MODE = True
```

## 工作流程

```
1. 快速扫描阶段
   ├─ 提取所有图片和表格
   ├─ 图片过滤（尺寸/比例/Hash去重）
   └─ 表格分类（小表/大表/巨型表）

2. 智能分流阶段
   ├─ 小表 → bypass（保持原样）
   ├─ 大表 → summarize（需要摘要）
   └─ 巨型表 → split_and_summarize（表头广播）

3. 并发处理阶段（异步IO）
   ├─ 有效图片 → 批量调用 LLM
   ├─ 大表 → 批量调用 LLM
   └─ 巨型表 → 先切分再批量调用

4. 回填与注入阶段
   └─ 将摘要注入原文（<figure> 标签）

5. 智能切分阶段
   ├─ 保护 <figure> 标签完整性
   └─ 小表随文本自然切分

6. 向量化与索引阶段
   └─ Multi-Vector 存储到 ES
```

## ES 索引结构

新增字段：

```json
{
  "element_type": "text/small_table/large_table/image/mixed",
  "has_image": true,
  "has_table": true,
  "image_count": 2,
  "table_count": 1,
  "element_summary": "图表核心结论",
  "element_keywords": ["关键词1", "关键词2"],
  "element_columns": ["列名1", "列名2"],
  "original_table": "原始表格内容（不索引）",
  "original_image_url": "图片URL（不索引）",
  "is_summary_chunk": true
}
```

## 成本优化

### API 调用成本节省

- **小表直接入库**：节省约 90% 的表格分析成本
- **图片过滤**：节省约 70% 的图片分析成本
- **Hash 去重**：避免重复图片的重复调用
- **智能缓存**：重复运行不会重复调用 API

### 示例成本对比

假设处理 100 个文档：

- 原方案：500 个表格 + 200 个图片 = 700 次 API 调用
- 优化方案：50 个大表 + 60 个有效图片 = 110 次 API 调用
- **节省约 84% 的 API 成本**

## 检索优化

### 检索锚点设计

LLM 生成的摘要包含：

1. **核心结论**：100 字以内的概括
2. **关键词**：实体、时间、指标名称
3. **列名**：表格的所有列名（用于精确匹配）
4. **数据类型**：财务报表/产品数据/其他

这些信息都会被向量化，大幅提升检索召回率。

### 检索示例

用户提问："2023年Q4营收怎么样？"

1. 向量检索命中图表摘要（包含"2023"、"Q4"、"营收"关键词）
2. 通过 `element_id` 获取原始表格
3. 将完整表格喂给问答模型
4. 生成准确答案

## 注意事项

1. **首次运行较慢**：需要调用 LLM 分析图表，后续运行会使用缓存
2. **图片路径**：确保 Markdown 中的图片路径正确（相对路径或绝对路径）
3. **并发限制**：根据 API 限流调整 `MAX_CONCURRENT_REQUESTS`
4. **内存占用**：处理大量文档时注意内存使用

## 故障排查

### 1. 图片过滤失败

如果 PIL 不可用，图片过滤会降级为保守策略（保留所有图片）。

解决方案：

```bash
pip install pillow
```

### 2. LLM 调用失败

检查：

- `DASHSCOPE_API_KEY` 是否配置正确
- API 余额是否充足
- 网络连接是否正常

### 3. 缓存文件损坏

删除缓存文件重新运行：

```bash
rm data/.element_analysis_cache.json
```

## 未来优化方向

1. **VLM 支持**：集成 qwen-vl-max 进行图片内容理解
2. **表格结构化**：将表格转为 JSON 存储，支持结构化查询
3. **增量更新优化**：仅重新分析变更的图表
4. **分布式处理**：支持多机并行处理大规模文档
