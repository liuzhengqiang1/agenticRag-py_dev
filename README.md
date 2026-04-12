# 企业级 RAG 问答系统

基于 FastAPI + LangChain + Elasticsearch 构建的智能问答系统，支持传统 RAG（用于Ragas快速分析） 和 Agentic RAG 两种模式。
docker包下提供了docker-compose.prod.yml文件，可一键部署于Linux服务器上对外提供RESTful接口。

## 核心特性

- **Agentic RAG**：Agent 自动路由到知识库/天气/订单等多工具，支持并发调用和拒答。
- **Redis加锁**：确保同一会话不被重复调用。污染LangGraph的state。 
                结束自动释放，且30s自动过期
- **查询重写**：自动处理追问和指代词，提升检索精度。
              在用户query<8且有记忆时，或用户query存在指代词时触发查询重写。
- **意图识别**：业务护栏：识别并拒绝代码编写、创意生成、报表生成等超出能力范围的请求。
              缓存策略：判断查询类型（事实性/生成式/时效性），决定是否使用缓存。
              意图清晰度：检测查询是否明确，避免模糊问题进入后续流程。
- **语义缓存**：Redis缓存用户query，向量相似走缓存，不调用LLM。
              针对否定词敏感度不足会导致缓存污染，添加否定词检测，会导致一部分缓存重复。
              该项目核心点是通过限制AI能力边界，功能有限便于实现缓存的内容可控，没有花时间去做各种query的校验。
              TODO 用户觉得回答不满意，前端点踩删除缓存。
- **混合检索**：ES 向量检索（kNN）+ BM25 全文检索 + Reranker 重排。
- **多轮对话**：Redis 会话存储，支持上下文连续对话。
              原生redisSaver默认不会做记忆轮数限制，也不会设置 TTL。
              TODO Redis（短期） + 向量数据库（长期） + 数据库（历史对话+审计） + 记忆压缩（运行时每10轮调用LLM总结，在redis过期从数据库读取时，整体摘要） + Memory Service 抽象层（将前面提到的逻辑抽取出来，形成独立的服务，便于多agent共享。
- **流式输出**：SSE 实时响应
- **异常处理**：redis记录重试次数，避免agent死循环，超过3次返回预定义信息降级。 
              采用装饰器避免每个工具都手写异常捕获
- **限流**： 限流可采用令牌桶，但我微服务架构部署有sentinel没必要再写了，限流熔断降级一次性搞定
- **Multi-Vector 检索**：不同格式的文档统一MinerU转markdown，图片和表格智能处理，大幅提升检索召回率
- **大小表分流**：小表直接入库，大表生成摘要。如果超出切片大小：重要表格全量存，一般表格表头广播存，历史表格只存摘要
- **图片处理**：过滤小文件和小尺寸图片，元数据记录图片地址，TODO 回答时根据元数据中图片地址，前端展示图片
- **异步批处理**：并发调用 LLM，默认10

## 技术栈

- **框架**：FastAPI + LangChain + LangGraph
- **LLM**：阿里云百炼（Qwen-Max/Turbo）
- **检索**：Elasticsearch 8.12.2（kNN + BM25 + IK 分词）
- **重排**：Flashrank（无需 GPU）
- **存储**：Redis（会话管理）
- **Embedding**：DashScope text-embedding-v4（1024 维）

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <your-repo-url>
cd langchain-mvp

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境变量

创建 `.env` 文件：

```env
# 阿里云百炼 API
DASHSCOPE_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# Elasticsearch
ES_HOST=localhost
ES_PORT=9200
ES_INDEX_NAME=training_docs

# Tavily 搜索（可选，用于 Agentic RAG）
TAVILY_API_KEY=your_tavily_key_here
```

### 3. 启动 Elasticsearch

```bash
cd docker
docker-compose up -d

# 安装 IK 中文分词（推荐）
docker exec -it es bash
./bin/elasticsearch-plugin install https://github.com/medcl/elasticsearch-analysis-ik/releases/download/v8.12.2/elasticsearch-analysis-ik-8.12.2.zip
exit
docker restart es
```

### 4. 构建知识库

```bash
# 使用 v2.0 版本（支持图表智能处理）
python build_knowledge/build_knowledge_es.py

# 首次运行会分析图片和表格（较慢）
# 后续运行会使用缓存（快速）
```

**v2.0 新特性**：

- 自动识别和分类表格（小表/大表/巨型表）
- 过滤无效图片（装饰线、图标等）
- 异步批处理，显著提升速度
- 智能缓存，避免重复调用 API

详细说明见：[构建知识库脚本/README_v2.md](build_knowledge/README_v2.md)

### 5. 启动服务

```bash
python main.py
# 访问 http://localhost:8000/docs
```

## 使用方式

### 传统 RAG 模式

```bash
# 网页测试
open static/test_stream.html

# API 调用
POST /api/chat-stream
{
  "question": "公司有哪些培训类型？",
  "session_id": "user_123"
}
```

### Agentic RAG 模式（推荐）

```bash
# 命令行测试
python agentic_rag.py

# 网页测试
open static/test_agentic_stream.html

# API 调用
POST /api/agentic-chat-stream
{
  "question": "查询订单 9982 状态，公司出差打车报销额度是多少？上海今天天气如何？",
  "session_id": "user_123"
}
```

**Agentic RAG 优势**：

- Agent 自动识别意图，路由到不同工具（知识库/天气/订单/搜索）
- 支持并发调用多个工具
- 具备拒答能力，避免幻觉

## 项目结构

```
├── app/
│   ├── api/                    # API 路由层
│   │   ├── chat.py             # 传统 RAG 接口
│   │   └── agentic_chat.py     # Agentic RAG 接口
│   ├── core/                   # 核心配置
│   ├── services/               # 业务逻辑层
│   │   ├── history/            # 会话管理
│   │   ├── llm/                # 意图分类、查询重写、闲聊
│   │   ├── retrievers/         # ES 检索器（向量 + BM25）
│   │   └── rag_service.py      # RAG 核心编排
├── data/
│   └── training_doc.txt        # 知识库源文件
├── docker/
│   └── docker-compose.yml      # ES 容器配置
├── agentic_rag.py              # Agentic RAG 主程序
├── static/
│   ├── test_stream.html        # 传统 RAG 测试页面
│   └── test_agentic_stream.html # Agentic RAG 测试页面
└── 构建知识库脚本/
    └── build_knowledge_es.py   # 知识库构建脚本
```

## 核心架构

### 传统 RAG 流程

```
用户问题
  ↓
意图理解（闲聊/检索）
  ↓
查询重写（处理追问和指代词）
  ↓
混合检索（向量 kNN + BM25）
  ↓
重排序（Flashrank Top-3）
  ↓
LLM 生成答案
```

### Agentic RAG 流程

```
用户问题
  ↓
Agent 分析意图
  ↓
动态路由到工具
  ├─ 知识库检索（RAG Tool）
  ├─ 天气查询（Weather API）
  ├─ 订单查询（Database Tool）
  └─ 互联网搜索（Tavily）
  ↓
LLM 整合结果
```

## 可用工具（Agentic RAG）

| 工具名称                   | 功能描述       | 适用场景                                |
| -------------------------- | -------------- | --------------------------------------- |
| search_knowledge_base      | 检索企业知识库 | 培训制度、报销流程等内部文档            |
| get_current_weather        | 查询天气信息   | 天气查询                                |
| query_database_order       | 查询订单状态   | 订单查询                                |
| tavily_search_results_json | 互联网搜索     | 实时信息、新闻（需配置 TAVILY_API_KEY） |

## 常见问题

**Q: 如何更新知识库？**

```bash
# 修改 data/documents/ 下的 Markdown 文件后运行
python build_knowledge/build_knowledge_es.py

# 支持增量更新，只处理新增或变更的文件
```

**Q: 如何跳过图表分析（快速模式）？**

```python
# 在脚本开头设置
ProcessConfig.FAST_MODE = True
```

**Q: 如何调整表格分类阈值？**

修改 `build_knowledge_es.py` 中的 `ProcessConfig` 类参数。

**Q: 图表分析失败怎么办？**

检查：

- `DASHSCOPE_API_KEY` 是否配置正确
- API 余额是否充足
- 缓存文件是否损坏（可删除 `data/.element_analysis_cache.json` 重试）

**Q: 如何验证 ES 是否正常？**

```bash
python test_es_connection.py
# 或访问 http://localhost:9200
```

**Q: 如何清空会话历史？**

```bash
redis-cli DEL chat_history:user_123
```

**Q: 如何切换大模型？**

修改 `app/services/rag_service.py` 中的模型名称（qwen-max/qwen-plus/qwen-turbo）

## 配置调优

| 参数         | 位置                  | 默认值 | 说明                   |
| ------------ | --------------------- | ------ | ---------------------- |
| 向量召回数   | rag_service.py        | 10     | 向量检索召回的候选数   |
| BM25 召回数  | rag_service.py        | 10     | 关键字检索召回的候选数 |
| 重排 TopN    | rag_service.py        | 3      | 最终发给大模型的文档数 |
| chunk_size   | build_knowledge_es.py | 500    | 文档切片大小           |
| 会话过期时间 | rag_service.py        | 3600   | Redis 会话 TTL（秒）   |

## 许可证

MIT License
