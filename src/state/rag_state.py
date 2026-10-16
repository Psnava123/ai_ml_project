"""RAG state definition for LangGraph"""

from typing import List, Optional, TypedDict, Annotated
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_core.documents import Document


class RAGState(TypedDict):
    """State object for RAG workflow"""
    
    messages: Annotated[List[BaseMessage], add_messages]
    retrieved_docs: Optional[List[Document]]
    query: Optional[str]                 # The user question (original or rewritten)
    tool_output_string: Optional[str]    # The string output from the Tool Node
    needs_tool: Optional[str]            # "yes" or "no" from Query Analyzer
    is_usable: Optional[str]             # "yes" or "no" from Validate Context
    retry_count: Optional[int]           # Safety counter so it does not loop forever