from langchain_core.retrievers import BaseRetriever
from typing import List
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.callbacks import AsyncCallbackManagerForRetrieverRun

class TopKRetriever(BaseRetriever):
    retriever: BaseRetriever
    k: int

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        docs = self.retriever.invoke(query)
        return docs[:self.k]

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        docs = await self.retriever.ainvoke(query)
        return docs[:self.k]
