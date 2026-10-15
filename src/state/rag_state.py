"""RAG state definition for LangGraph"""

from typing import List
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.documents import Document


class RAGState(TypedDict):
    """State object for RAG workflow"""
    
    messages: Annotated[List,add_messages]
    retrieved_docs: List[Document] 