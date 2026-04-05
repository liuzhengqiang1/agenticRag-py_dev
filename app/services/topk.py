from langchain_core.retrievers import BaseRetriever

class TopKRetriever(BaseRetriever):
    def __init__(self, retriever, k):
        self.retriever = retriever
        self.k = k

    def _get_relevant_documents(self, query):
        docs = self.retriever.get_relevant_documents(query)
        return docs[:self.k]