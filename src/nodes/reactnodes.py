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
            print(f"\\n  this is my docs {len(self.my_doc_bucket)}\\n",flush=True)
            for i, d in enumerate(docs[:8], start=1):
                meta = d.metadata if hasattr(d, "metadata") else {}
                title = meta.get("title") or meta.get("source") or f"doc_{i}"
                merged.append(f"[{i}] {title}\n{d.page_content}")
            return "\n\n".join(merged)

        retriever_tool = Tool(
            name="retriever",
            description="ALWAYS use this tool FIRST before any other search tool when the query relates to AI, LLMs, agents, memory, search algorithms, or video/image generation. Fetches passages from Lilian Weng's indexed articles covering two main domains 1st-LLM-Powered Autonomous Agents: Planning, Task Decomposition, Chain of Thought (CoT), Tree of Thoughts (ToT), LLM+P, PDDL, Self-Reflection, ReAct, Reflexion, Chain of Hindsight (CoH), Algorithm Distillation (AD), Sensory/Short-Term/Long-Term Memory, MIPS, Approximate Nearest Neighbors (ANN), LSH, ANNOY, HNSW, FAISS, ScaNN, Tool Use, MRKL, TALM, Toolformer, API-Bank, HuggingGPT, Gorilla, ChemCrow, Generative Agents, AutoGPT, and GPT-Engineer.2nd Diffusion Models for Video Generation: DDIM, v-parameterization, reconstruction guidance, Video Diffusion Models (VDM), 3D U-Net, Diffusion Transformer (DiT), Sora, Imagen Video, Spatial/Temporal Super-Resolution (SSR/TSR), Make-A-Video, Tune-A-Video, Pseudo-3D convolution/attention, ST-Attention, Video LDM, AnimateDiff, Text2Video-Zero, and ControlVideo.",
            func=retriever_tool_fn,
        )
        

        tavily_tool = TavilySearchResults(max_results=3)

        def tavily_search_context(query: str) -> str:
            print("calling tavily search tool.................")
            results = tavily_tool.invoke({"query": query})
            if not results:
             return "No web search results found."
        
            # If Tavily already returns a string (e.g., an error message), return it directly
            if isinstance(results, str):
                return results
        
            # Combine all 3 dictionary results into one clean string
            context_parts = []
            for i, doc in enumerate(results, 1):
                url = doc.get("url", "No URL")
                content = doc.get("content", "")
                context_parts.append(f"Source {i} ({url}):\n{content}")
        
            return "\n\n-\n\n".join(context_parts)
        tavly_search_tool=Tool(
                name="tavely_search",
                description= "Use this tool ONLY as a backup for web search when the query is NO about LLM autonomous agents, memory, or video diffusion models. Returns a single string with the entire context of the search results.",
                func=tavily_search_context,
            )
        return [retriever_tool,tavly_search_tool]

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

        
              