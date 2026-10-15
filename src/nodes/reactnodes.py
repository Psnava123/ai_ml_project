"""LangGraph nodes for RAG workflow + ReAct Agent inside generate_content"""

from typing import List, Optional
from src.state.rag_state import RAGState
import uuid  # <--- Add this at the very top of the file
from langchain_core.tools import tool

from langchain_core.documents import Document
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent 
from langgraph.checkpoint.memory import MemorySaver

# Wikipedia tool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.tools.tavily_search import TavilySearchResults


class RAGNodes:
    """Contains node functions for RAG workflow"""

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.my_doc_bucket = []
        self._agent = None  # lazy-init agent

    def retrieve_docs(self, state: RAGState) -> RAGState:
        """Classic retriever node"""
        question=state["messages"][-1].content
        docs = self.retriever.invoke(question)
        return {"retrieved_docs":docs}
    def _build_tools(self) -> List[Tool]:
        """Build retriever + wikipedia tools"""

        def retriever_tool_fn(query: str) -> str:
            print("..............calling retriever tool............")
            docs: List[Document] = self.retriever.invoke(query)
            if not docs:
                return "No documents found."
            merged = []
            self.my_doc_bucket=docs
            for i, d in enumerate(docs[:8], start=1):
                meta = d.metadata if hasattr(d, "metadata") else {}
                title = meta.get("title") or meta.get("source") or f"doc_{i}"
                merged.append(f"[{i}] {title}\n{d.page_content}")
            return "\n\n".join(merged)

        retriever_tool = Tool(
            name="retriever",
            description=" for Fetch passages from indexed corpus. topic covered Exploring LLM powered autonomous agents and diffusion models for video generation.",
            func=retriever_tool_fn,
        )

        tavily_tool = TavilySearchResults(max_results=3)
        return [retriever_tool, tavily_tool]

    def _build_agent(self):
        """ReAct agent with tools"""
        tools = self._build_tools()
        memory=MemorySaver()
        system_prompt = (
            "You are a helpful RAG agent. "
            "You have access to these two tools retriever and tavily search resultGive me an answer based on these two tools"
            "Prefer 'retriever' for user-provided docs; use 'tavily_search_results_json' for general knowledge. "
            "Return only the final useful answer."
        )
        self._agent = create_react_agent(self.llm, tools=tools,prompt=system_prompt,checkpointer=memory)
        return self._agent

    def generate_answer(self, state: RAGState) -> RAGState:
        """
        Generate answer using ReAct agent with retriever + wikipedia.
        """
        if self._agent is None:
            self._build_agent()

        result = self._agent.invoke({"messages": state["messages"]})
        print(result["messages"][-1],flush=True)
        print(f"\n{state.get("retrieved_docs", [])} ..................retrieve docs ............................",flush=True)

        return {"messages":[result["messages"][-1]],"retrieved_docs":self.my_doc_bucket}

        
              