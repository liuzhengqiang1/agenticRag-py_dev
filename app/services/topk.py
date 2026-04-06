from langchain_core.retrievers import BaseRetriever
from typing import List
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun

class TopKRetriever(BaseRetriever):
    retriever: BaseRetriever
    k: int

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        docs = self.retriever.invoke(query)
        return docs[:self.k]