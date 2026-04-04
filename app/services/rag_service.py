# -*- coding: utf-8 -*-
"""RAG 服务模块"""
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def format_docs(docs):
    """将检索到的文档列表格式化为字符串"""
    return "\n\n".join(doc.page_content for doc in docs)


def init_rag_components():
    """初始化 RAG 组件"""
    print("正在初始化 RAG 组件...")

    # 1. Embedding 模型
    embeddings = DashScopeEmbeddings(model="text-embedding-v4")

    # 2. LLM 模型
    llm = ChatOpenAI(model="qwen-max", temperature=0.7)

    # 3. 向量数据库
    vector_store = Chroma(
        persist_directory="vector_store",
        embedding_function=embeddings
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    # 4. Prompt 模板
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "你是一个专业的企业 HR 助手，负责回答员工关于公司培训制度的问题。\n"
         "请根据以下检索到的知识库内容来回答用户的问题。\n"
         "如果知识库中没有相关信息，请明确告知用户\"抱歉，我在知识库中未找到相关信息\"。\n\n"
         "知识库内容：\n{context}"),
        ("human", "{question}")
    ])

    # 5. RAG Chain
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print("✓ RAG 组件初始化完成！")
    return rag_chain


# 全局 RAG Chain 实例
rag_chain = init_rag_components()


def chat(question: str) -> str:
    """问答接口"""
    return rag_chain.invoke(question)