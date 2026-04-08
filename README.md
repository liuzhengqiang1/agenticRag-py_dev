# RAG 问答系统 MVP

## 📋 项目简介

这是一个基于 **FastAPI** 和 **LangChain** 构建的企业级 RAG（检索增强生成）问答系统，专为企业 HR 场景设计，用于回答员工关于公司培训制度的问题。

### 核心特性

- ✅ **智能路由**：意图理解自动判断闲聊/检索，闲聊场景不触发检索（节省成本）
- ✅ **查询重写**：智能检测追问和指代词，自动改写为独立问题（提升检索精度）
- ✅ **多轮对话记忆**：基于 Redis 的分布式会话存储，支持上下文连续对话
- ✅ **混合检索**：结合向量检索（ES kNN）和关键字检索（ES BM25），提升召回率
- ✅ **两阶段检索**：粗筛（召回 20 个候选）→ 精排（Reranker 重排 Top-3），兼顾速度与精度
- ✅ **流式输出**：基于 SSE（Server-Sent Events）的实时流式响应
- ✅ **企业级架构**：使用 Elasticsearch 作为检索引擎，支持分布式部署
- ✅ **轻量级部署**：使用 Flashrank 重排模型，无需 GPU

---

## 🏗️ 系统架构

```
用户问题
    ↓
[意图理解层]：Qwen-Turbo + 最近 1-2 轮历史
    ↓
    ├─→ 闲聊 → Qwen-Turbo 直接回复（不触发检索）
    └─→ 检索
         ↓
       [查询重写层]：智能判断是否需要重写（Qwen-Turbo + 最近 2-3 轮历史）
         ↓
       [混合检索层]
         ├─ 向量检索（ES kNN）：召回 10 个候选
         └─ 关键字检索（ES BM25 + IK 分词）：召回 10 个候选
         ↓
       [RRF 融合排序]：加权融合（50% + 50%）→ 20 个候选
         ↓
       [预过滤器]：TopK 过滤 → 保留 Top-6
         ↓
       [重排序层]：Flashrank Reranker → 精排 Top-3
         ↓
       [大模型生成]：Qwen-Max + 历史对话上下文
         ↓
       流式返回答案
```

---

## 📦 技术栈

| 组件           | 技术选型                    | 说明                          |
| -------------- | --------------------------- | ----------------------------- |
| Web 框架       | FastAPI 0.109.0             | 高性能异步框架                |
| LLM 框架       | LangChain 1.2.14            | 大模型应用开发框架            |
| 大模型（生成） | 阿里云百炼 Qwen-Max         | RAG 最终回答生成              |
| 大模型（路由） | 阿里云百炼 Qwen-Turbo       | 意图理解、查询重写、闲聊回复  |
| Embedding      | DashScope text-embedding-v4 | 阿里云向量化模型（1024 维）   |
| 检索引擎       | Elasticsearch 8.12.2        | 企业级搜索引擎（向量 + BM25） |
| 中文分词       | IK Analyzer                 | ES 中文分词插件               |
| 重排模型       | Flashrank (ms-marco-MiniLM) | 轻量级重排，无需 GPU          |
| 会话存储       | Redis 5.0+                  | 分布式会话历史管理            |
| 流式响应       | SSE (sse-starlette)         | Server-Sent Events            |

---

## 🚀 快速开始

### 1. 环境要求

- Python 3.8+
- Redis 服务（本地或远程）
- Elasticsearch 8.12.2（通过 Docker 启动）
- 阿里云百炼 API Key

### 2. 启动 Elasticsearch

使用 Docker Compose 启动 ES：

```bash
cd docker
docker-compose up -d
```

验证 ES 是否启动成功：

```bash
curl http://localhost:9200
```

**可选但推荐**：安装 IK 中文分词器（提升中文检索效果）

```bash
# 进入容器
docker exec -it es bash

# 安装 IK 插件
./bin/elasticsearch-plugin install https://github.com/medcl/elasticsearch-analysis-ik/releases/download/v8.12.2/elasticsearch-analysis-ik-8.12.2.zip

# 退出并重启
exit
docker restart es
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置环境变量

在项目根目录创建 `.env` 文件：

```env
# 阿里云百炼 API 配置
DASHSCOPE_API_KEY=your_dashscope_api_key_here
OPENAI_API_KEY=your_dashscope_api_key_here
OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1

# Redis 配置
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_DB=0

# Elasticsearch 配置
ES_HOST=localhost
ES_PORT=9200
ES_USERNAME=
ES_PASSWORD=
ES_INDEX_NAME=training_docs
```

### 5. 测试 ES 连接

```bash
python test_es_connection.py
```

### 6. 构建知识库

首次运行前，需要将训练文档索引到 Elasticsearch：

```bash
python build_knowledge_es.py
```

**说明**：

- 该脚本会读取 `data/training_doc.txt` 文件
- 将文档切分为 500 字符的小块（chunk_size=500，chunk_overlap=50）
- 使用 DashScope Embedding 模型向量化（1024 维）
- 索引到 Elasticsearch（支持向量检索 + BM25）

⚠️ **重要**：每次更新 `data/training_doc.txt` 后，需要重新运行此脚本！

### 7. 启动服务

#### 方式一：使用 Python 直接启动

```bash
python main.py
```

#### 方式二：使用批处理脚本（Windows）

```bash
start_server.bat
```

服务启动后，访问：

- API 文档：http://localhost:8000/docs
- 健康检查：http://localhost:8000/health

---

## 🗂️ 项目结构

```
.
├── app/                          # 应用主目录
│   ├── api/                      # API 路由层
│   │   ├── chat.py               # 传统 RAG 聊天接口
│   │   └── agentic_chat.py       # ⭐ Agentic RAG 聊天接口
│   ├── core/                     # 核心配置
│   │   ├── config.py             # 通用配置
│   │   ├── redis_config.py       # Redis 配置管理
│   │   └── es_config.py          # Elasticsearch 配置管理
│   ├── models/                   # 数据模型
│   ├── services/                 # 业务逻辑层
│   │   ├── history/              # 会话历史管理
│   │   │   ├── __init__.py
│   │   │   └── session_manager.py  # Redis 会话管理
│   │   ├── llm/                  # LLM 相关服务
│   │   │   ├── __init__.py
│   │   │   ├── intent_classifier.py  # 意图分类
│   │   │   ├── query_rewriter.py     # 查询重写
│   │   │   └── chitchat.py           # 闲聊服务
│   │   ├── retrievers/           # 检索器模块
│   │   │   ├── __init__.py
│   │   │   ├── es_retriever.py       # ES 检索器（向量 + BM25）
│   │   │   └── ensemble_retriever.py # 混合检索器配置
│   │   ├── utils/                # 工具函数
│   │   │   ├── __init__.py
│   │   │   └── formatters.py         # 格式化工具
│   │   ├── rag_service.py        # RAG 核心服务（⭐ 编排层）
│   │   └── topk.py               # TopK 预过滤器
│   └── main.py                   # FastAPI 应用实例
├── data/                         # 数据目录
│   └── training_doc.txt          # 训练文档（知识库源文件）
├── docker/                       # Docker 配置
│   └── docker-compose.yml        # ES 容器配置
├── agentic_rag.py                # ⭐ Agentic RAG 主程序（命令行版本）
├── agent_demo.py                 # Agent 入门演示（工具调用与闭环反馈）
├── langgraph_demo.py             # LangGraph 状态机演示
├── build_knowledge_es.py         # ES 知识库构建脚本
├── build_knowledge_chroma.py     # Chroma 知识库构建脚本（已废弃）
├── test_es_connection.py         # ES 连接测试脚本
├── test_stream.html              # 传统 RAG 流式接口测试页面
├── test_agentic_stream.html      # ⭐ Agentic RAG 流式接口测试页面
├── test_agentic_rag.bat          # ⭐ Agentic RAG 快速测试脚本（Windows）
├── main.py                       # 应用入口
├── requirements.txt              # 依赖清单
├── start_server.bat              # Windows 启动脚本
├── README.md                     # 项目说明文档
├── AGENTIC_RAG_ARCHITECTURE.md   # ⭐ Agentic RAG 架构深度解析
├── QUICK_START_AGENTIC_RAG.md    # ⭐ Agentic RAG 快速上手指南
└── .env                          # 环境变量配置（需自行创建）
```

**⭐ 新增文件说明**：

- `agentic_rag.py`：Agentic RAG 主程序，包含完整的工具定义、状态机构建和测试用例
- `app/api/agentic_chat.py`：Agentic RAG 的 FastAPI 接口（支持流式和非流式）
- `test_agentic_stream.html`：Agentic RAG 的网页测试界面（推荐使用）
- `test_agentic_rag.bat`：Windows 快速测试脚本
- `AGENTIC_RAG_ARCHITECTURE.md`：架构深度解析文档
- `QUICK_START_AGENTIC_RAG.md`：快速上手指南

---

## 🔧 核心模块说明

### 1. RAG Service (`app/services/rag_service.py`)

这是系统的核心编排层，负责整合各个子模块：

- 初始化混合检索器（调用 `retrievers/ensemble_retriever.py`）
- 构建 RAG Chain（Prompt + LLM + 历史记忆）
- 提供对外接口（`chat()` 和 `chat_stream()`）

**模块拆分说明**（v0.5.0 重构）：

原来的 `rag_service.py` 文件过于臃肿（600+ 行），现已按照 FastAPI 最佳实践拆分为：

- `history/session_manager.py` - 会话历史管理（Redis）
- `llm/intent_classifier.py` - 意图分类服务
- `llm/query_rewriter.py` - 查询重写服务
- `llm/chitchat.py` - 闲聊服务
- `retrievers/es_retriever.py` - ES 检索器（向量 + BM25）
- `retrievers/ensemble_retriever.py` - 混合检索器配置
- `utils/formatters.py` - 格式化工具函数

**优势**：

- 职责单一：每个模块只负责一个功能
- 易于测试：可以单独测试每个模块
- 易于维护：修改某个功能不影响其他模块
- 易于扩展：新增功能只需添加新模块

### 2. 意图理解与智能路由

**优势**：

- 闲聊场景不触发检索，节省 ES 查询 + Reranker 计算成本
- qwen-turbo 比 qwen-max 便宜 10 倍，响应更快

### 3. 查询重写（智能触发）

**示例**：

```
用户："公司有哪些培训类型？"
→ 不需要重写（问题完整）

用户："费用怎么报销？"（上文提到培训）
→ 重写为："培训费用怎么报销？"

用户："那个流程复杂吗？"（上文提到报销）
→ 重写为："培训费用报销流程复杂吗？"
```

#### 1.3 混合检索

```python
# 向量检索（ES kNN）
vector_retriever = ESVectorRetriever(
    es_client=es_client,
    index_name=es_config.index_name,
    embeddings=embeddings,
    k=10  # 召回 10 个候选
)

# 关键字检索（ES BM25）
bm25_retriever = ESBM25Retriever(
    es_client=es_client,
    index_name=es_config.index_name,
    k=10  # 召回 10 个候选
)

# 混合检索（EnsembleRetriever）
base_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.5, 0.5]  # 各占 50% 权重
)
```

#### 1.4 两阶段检索

# 第一阶段：粗筛（RRF 融合排序后取 Top-6）

# 第二阶段：精排（Reranker 重排后取 Top-3）

#### 1.5 多轮对话记忆（滑动窗口）

**优势**：

- 减少 token 消耗（不需要每次都传全部历史）
- 提升响应速度（Prompt 更短）
- 保持上下文连贯性（保留最相关的历史）

````

### 2. 知识库构建 (`build_knowledge_es.py`)

负责离线构建 Elasticsearch 索引：
---

## ⚙️ 配置调优

### 1. 意图理解与查询重写参数

| 参数             | 位置                   | 默认值                    | 说明                         | 调优建议           |
| ---------------- | ---------------------- | ------------------------- | ---------------------------- | ------------------ |
| 意图理解历史轮数 | `classify_intent()`    | 1-2 轮                    | 意图分类时参考的历史对话轮数 | 一般 1-2 轮足够    |
| 查询重写历史轮数 | `rewrite_query()`      | 2-3 轮                    | 查询重写时参考的历史对话轮数 | 根据对话复杂度调整 |
| 指代词列表       | `need_query_rewrite()` | ["它", "这个", "那个"...] | 触发查询重写的指代词         | 根据业务场景补充   |
| 问题长度阈值     | `need_query_rewrite()` | 8 字符                    | 判断问题是否过短的阈值       | 中文建议 6-10 字符 |
| 常见主语列表     | `need_query_rewrite()` | ["培训", "课程"...]       | 判断问题是否缺少主语         | 根据业务领域补充   |

### 2. 检索参数调优

| 参数         | 位置             | 默认值     | 说明                   | 调优建议                 |
| ------------ | ---------------- | ---------- | ---------------------- | ------------------------ |
| 向量召回数   | `rag_service.py` | 10         | 向量检索召回的候选数   | 知识库大时可调至 20-30   |
| BM25 召回数  | `rag_service.py` | 10         | 关键字检索召回的候选数 | 与向量召回数保持一致     |
| 预过滤 TopK  | `rag_service.py` | 6          | RRF 融合后保留的候选数 | 重排模型轻量时可调至 10  |
| 重排 TopN    | `rag_service.py` | 3          | 最终发给大模型的文档数 | 根据 Prompt 长度限制调整 |
| 混合检索权重 | `rag_service.py` | [0.5, 0.5] | 向量 vs 关键字权重     | 根据业务场景调整         |

### 3. 文档切分参数

| 参数          | 位置                    | 默认值 | 说明                 |
| ------------- | ----------------------- | ------ | -------------------- |
| chunk_size    | `build_knowledge_es.py` | 500    | 每个切片的最大字符数 |
| chunk_overlap | `build_knowledge_es.py` | 50     | 切片之间的重叠字符数 |

**调优建议**：

- 测试环境：chunk_size=200（快速验证）
- 生产环境：chunk_size=500-1000（更完整的上下文）

### 4. 会话过期时间

```python
# app/services/rag_service.py
RedisChatMessageHistory(
    session_id=session_id,
    ttl=3600  # 1 小时，可根据业务需求调整
)
```

---

## 📝 常见问题

### Q1: 如何更新知识库？

A: 修改 `data/training_doc.txt` 后，重新运行：

```bash
python build_knowledge_es.py
```

### Q2: 如何验证 ES 是否正常运行？

A: 运行测试脚本：

```bash
python test_es_connection.py
```

或手动访问：

```bash
curl http://localhost:9200
curl http://localhost:9200/training_docs/_count
```

### Q3: 如何查看意图理解和查询重写的效果？

A: 启动服务后，每次问答会在控制台打印详细日志：

```
================================================================================
📨 收到用户问题：费用怎么报销？ | 会话ID：user_123
================================================================================
🔍 意图分类结果：retrieval | 问题：费用怎么报销？
🎯 路由决策：检索模式（触发 RAG）
✏️ 触发查询重写：可能缺少主语 | 问题：费用怎么报销？
📝 查询重写：
  原问题：费用怎么报销？
  改写后：培训费用怎么报销？
================================================================================
```

### Q4: 如何查看检索到的上下文？

A: 启动服务后，每次问答会在控制台打印检索到的文档片段：

```
🎯 经过 Reranker 重排后，最终发给大模型的上下文（Top-3）：
================================================================================
📄 文档片段 1 (重排后排名):
内容: 公司的培训制度包括...
================================================================================
```

### Q5: 如何清空会话历史？

A: 会话历史存储在 Redis 中，可以通过以下方式清空：

```bash
# 清空指定会话
redis-cli DEL chat_history:user_123

# 清空所有会话
redis-cli KEYS "chat_history:*" | xargs redis-cli DEL
```

### Q6: 如何切换大模型？

A: 修改 `app/services/rag_service.py` 中的模型名称：

```python
# 最终生成模型（RAG 回答）
llm = ChatOpenAI(
    model="qwen-max",  # 可改为 qwen-plus、qwen-turbo 等
    temperature=0.7
)

# 意图理解、查询重写、闲聊模型
_intent_llm = ChatOpenAI(model="qwen-turbo", temperature=0)
_rewrite_llm = ChatOpenAI(model="qwen-turbo", temperature=0)
_chitchat_llm = ChatOpenAI(model="qwen-turbo", temperature=0.7)
```

### Q7: 如何调整查询重写的触发条件？

A: 修改 `app/services/rag_service.py` 中的 `need_query_rewrite()` 函数：

---

## 🔄 版本历史

### v0.6.0 (当前版本) - Agentic RAG 大一统架构

- ✅ 新增 Agentic RAG 架构（Agent + RAG + 多工具编排）
- ✅ 将 RAG 检索器封装为 Tool
- ✅ 集成业务 API 工具（天气、订单查询）
- ✅ 集成 Tavily 互联网搜索工具
- ✅ 使用 LangGraph 状态机编排所有工具
- ✅ 支持意图识别、拒答能力、并发调用
- ✅ 新增 FastAPI 接口（`/api/agentic-chat`、`/api/agentic-chat-stream`）
- ✅ 新增网页测试界面（`test_agentic_stream.html`）
- ✅ 新增架构文档（`AGENTIC_RAG_ARCHITECTURE.md`、`QUICK_START_AGENTIC_RAG.md`）

### v0.5.0

- ✅ 重构代码结构：按照 FastAPI 最佳实践拆分模块
- ✅ 新增 `services/history/` - 会话历史管理
- ✅ 新增 `services/llm/` - LLM 相关服务（意图分类、查询重写、闲聊）
- ✅ 新增 `services/retrievers/` - 检索器模块
- ✅ 新增 `services/utils/` - 工具函数
- ✅ 精简 `rag_service.py` 为编排层（从 600+ 行降至 150 行）
- ✅ 提升代码可维护性和可测试性

### v0.4.0

- ✅ 新增意图理解（智能路由：闲聊/检索）
- ✅ 新增查询重写（智能触发：追问/指代词/模糊查询）
- ✅ 新增滑动窗口历史管理（1-2 轮 / 2-3 轮）
- ✅ 优化成本：闲聊场景使用 qwen-turbo，不触发检索
- ✅ 提升精度：多轮对话改写为独立问题

### v0.3.0

- ✅ 迁移到 Elasticsearch（替代 Chroma + 内存 BM25）
- ✅ 支持 ES kNN 向量检索
- ✅ 支持 ES BM25 全文检索（带 IK 中文分词）
- ✅ 企业级架构，支持分布式部署

### v0.2.0

- ✅ 新增混合检索（向量 + BM25）
- ✅ 新增两阶段检索（召回 + 重排）
- ✅ 新增 Redis 会话存储
- ✅ 优化流式输出性能

### v0.1.0

- ✅ 基础 RAG 问答功能
- ✅ 向量检索（Chroma）
- ✅ 流式输出（SSE）
---

## 🤖 Agentic RAG（Agent + RAG 大一统架构）

### 什么是 Agentic RAG？

Agentic RAG 是将 RAG 检索器包装成一个 Tool，和业务 API（天气、订单）、互联网搜索一起扔给 LangGraph，打造一个"无所不能"的企业级智能体。

**核心优势**：

1. **意图识别**：LLM 自动判断用户需要什么工具（不需要手写 if-else）
2. **拒答能力**：如果所有工具都无法解决，LLM 会明确告知用户
3. **并发调用**：可以同时查询订单 + 天气 + 知识库
4. **可扩展性**：新增工具只需定义函数，无需修改路由逻辑

### 快速开始

#### 方式一：命令行测试

```bash
# 运行 Agentic RAG 演示（包含 4 个测试用例）
python agentic_rag.py

# 或使用批处理脚本（Windows）
test_agentic_rag.bat
```

#### 方式二：FastAPI 接口测试

```bash
# 1. 启动服务
python main.py

# 2. 访问 API 文档
# http://localhost:8000/docs

# 3. 测试接口
# POST /api/agentic-chat（非流式）
# POST /api/agentic-chat-stream（流式）
# GET /api/agentic-tools（查看可用工具）
```

#### 方式三：网页测试（推荐）

```bash
# 1. 启动服务
python main.py

# 2. 打开测试页面
# 双击 test_agentic_stream.html
# 或访问 file:///path/to/test_agentic_stream.html
```

### 架构对比

**传统 RAG**：
```
用户问题 → RAG 检索 → 生成答案
```

**Agentic RAG**：
```
用户问题 → Agent 分析 → 动态路由到不同工具
    ├─ 知识库问题 → RAG Tool（检索企业文档）
    ├─ 天气查询 → Weather API Tool
    ├─ 订单查询 → Database Tool
    └─ 实时信息 → Web Search Tool（Tavily）
```

### 测试用例

#### 1. 复合问题（并发调用多个工具）
```
问题：帮我查一下订单 9982 的状态，另外我想知道公司出差打车的报销额度是多少？对了，今天上海会下雨吗？

Agent 决策：
  ✓ 识别 3 个意图
  ✓ 并发调用 3 个工具：
    - query_database_order(order_id='9982')
    - search_knowledge_base(query='出差打车报销额度')
    - get_current_weather(city='上海')
  ✓ LLM 整合结果，生成统一回答
```

#### 2. 知识库问题
```
问题：公司的培训类型有哪些？

Agent 决策：
  ✓ 调用 search_knowledge_base
  ✓ 检索企业知识库
```

#### 3. 天气查询
```
问题：北京今天天气怎么样？

Agent 决策：
  ✓ 调用 get_current_weather
  ✓ 不触发知识库检索（节省成本）
```

#### 4. 拒答测试
```
问题：帮我写一首诗

Agent 决策：
  ✓ 判断所有工具都不适用
  ✓ 明确告知用户能力边界
  ✓ 避免幻觉（Hallucination）
```

### 可用工具列表

| 工具名称 | 功能描述 | 适用场景 |
|---------|---------|---------|
| search_knowledge_base | 检索企业知识库 | 培训制度、报销流程等内部文档 |
| get_current_weather | 查询天气信息 | 天气查询 |
| query_database_order | 查询订单状态 | 订单查询 |
| tavily_search_results_json | 互联网搜索 | 实时信息、新闻等（需配置 TAVILY_API_KEY） |

### 配置 Tavily 搜索（可选）

如需使用互联网搜索功能，请在 `.env` 文件中添加：

```env
TAVILY_API_KEY=your_tavily_api_key_here
```

获取 API Key：https://tavily.com/

### 详细文档

查看 [AGENTIC_RAG_ARCHITECTURE.md](AGENTIC_RAG_ARCHITECTURE.md) 了解完整的架构设计和实现细节。

---

## 📄 许可证

MIT License

---

## 👥 贡献指南

欢迎提交 Issue 和 Pull Request！

---

## 📧 联系方式

如有问题，请联系项目维护者。
````
