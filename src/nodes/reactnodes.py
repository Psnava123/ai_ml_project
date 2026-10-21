"""LangGraph nodes for RAG workflow + ReAct Agent inside generate_content"""

from typing import List, Optional
from src.state.rag_state import RAGState

from langchain_core.documents import Document
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage,SystemMessage,ToolMessage
from langgraph.prebuilt import create_react_agent 
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.tools.tavily_search import TavilySearchResults
from pydantic import BaseModel ,Field
from typing import Literal
from langchain_core.tools import StructuredTool
from langchain_core.language_models.chat_models import BaseChatModel

class ValidateAndAnswer(BaseModel):
    answer: Optional[str] = None



class AnalyzerDecision(BaseModel):
    tool_name: Literal["retriever", "tavily_search", "none"]
       

class RAGNodes:
    """Contains node functions for RAG workflow"""

    def __init__(self, retriever, llm:BaseChatModel):
        self.retriever = retriever
        self.llm = llm
        self.my_doc_bucket = []
        self.tools = self._build_tools()
        self.tool_map = {tool.name: tool for tool in self.tools}
        
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
            for i, d in enumerate(docs[:3], start=1):
                meta = d.metadata if hasattr(d, "metadata") else {}
                title = meta.get("title") or meta.get("source") or f"doc_{i}"
                merged.append(f"[{i}] {title}\n{d.page_content}")
            return "\n\n".join(merged)

        retriever_tool = Tool(
            name="retriever",
            description="ALWAYS use this tool FIRST before any other search tool when the query relates to AI, LLMs, agents, memory, search algorithms, or video/image generation. Fetches passages from Lilian Weng's indexed articles covering two main domains 1st-LLM-Powered Autonomous Agents: Planning, Task Decomposition, Chain of Thought (CoT), Tree of Thoughts (ToT), LLM+P, PDDL, Self-Reflection, ReAct, Reflexion, Chain of Hindsight (CoH), Algorithm Distillation (AD), Sensory/Short-Term/Long-Term Memory, MIPS, Approximate Nearest Neighbors (ANN), LSH, ANNOY, HNSW, FAISS, ScaNN, Tool Use, MRKL, TALM, Toolformer, API-Bank, HuggingGPT, Gorilla, ChemCrow, Generative Agents, AutoGPT, and GPT-Engineer.2nd Diffusion Models for Video Generation: DDIM, v-parameterization, reconstruction guidance, Video Diffusion Models (VDM), 3D U-Net, Diffusion Transformer (DiT), Sora, Imagen Video, Spatial/Temporal Super-Resolution (SSR/TSR), Make-A-Video, Tune-A-Video, Pseudo-3D convolution/attention, ST-Attention, Video LDM, AnimateDiff, Text2Video-Zero, and ControlVideo.",
            func=retriever_tool_fn,
        )
        

        tavily_tool = TavilySearchResults(max_results=2)

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
                
                content = doc.get("content", "")
                context_parts.append(f"Source {i} :\n{content[:800]}")
        
            return "\n\n-\n\n".join(context_parts)
        tavly_search_tool=Tool(
                name="tavily_search",
                description= "Use this tool ONLY as a backup for web search when the query is NO about LLM autonomous agents, memory, or video diffusion models. Returns a single string with the entire context of the search results.",
                func=tavily_search_context,
            )
        return [retriever_tool,tavly_search_tool]
    def query_analyzer(self, state: RAGState) -> dict:
        """Picks the tool in 1 step; Python figures out needs_tool for free."""
        query = state.get("query") or state["messages"][-1].content
        analyzer_llm = self.llm.with_structured_output(AnalyzerDecision)
        
        prompt = (
            "Route Q to a tool:\n"
            "- retriever: AI, LLMs, agents, memory, video/image diffusion\n"
            "- tavily_search: live news, current events, unknown facts\n"
            "- none: greetings, chat, math, coding, basic facts\n"
            f"Q: {query}"
        )
        
        result: AnalyzerDecision = analyzer_llm.invoke([HumanMessage(content=prompt)])
        
        # Python decides 'yes' or 'no' using 0 tokens!
        needs_tool = "no" if result.tool_name == "none" else "yes"
        print(f"Needs Tool: {needs_tool} | Chosen Tool: {result.tool_name}")
        
        return {
            "query": query,
            "needs_tool": needs_tool,
            "tool_name": result.tool_name,
            "retry_count": state.get("retry_count", 0),
        }
    def tool_node(self,state:RAGState)->dict:
        """Picks a tool using Pydantic, runs it, and returns a single string."""
        query=state["query"]
        chosen_tool_name = state.get("tool_name", "retriever")
        if chosen_tool_name not in self.tool_map:
            chosen_tool_name = "retriever"
        
        final_tool=self.tool_map[chosen_tool_name]
        print(f"Running Tool: {chosen_tool_name} | Query: {query}")
        tool_output_string=final_tool.invoke(query)
        output_string: str = str(tool_output_string)
        
        return {"tool_output_string":output_string,
                "retrieved_docs":self.my_doc_bucket}
        
    def validate_context(self, state: RAGState) -> dict:
        """
        Takes the string from Tool Node and checks if it is usable.
        - 'yes' -> Go to LLM Node to generate answer
        - 'no'  -> Go to Query Rewriter
        """
        print("--- NODE: VALIDATE CONTEXT ---")
        current_query = state["query"]
        context_string: str = state.get("tool_output_string", "")

        # Quick check if the string is empty
        if (
            not context_string
            or "No documents found." in context_string
            or "No web search results found." in context_string
        ):
            print("Context Usable Decision (yes/no): no (empty result)")
            return {"is_usable": "no"}

        validator_llm = self.llm.with_structured_output(YesNoDecision)
        prompt = (
            "Check if the Tool Output String's Initial 1,000 characters  contains helpful facts to answer the User Question.\n"
            "- Reply 'yes' if the context Is relatable to the  the main parts of the question.\n"
            "- Reply 'no' if the question is way too vague and not clear"
            "or if the context is completely unrelated to what the user asked.\n\n"
            f"User Question: {current_query}\n\n"
            f"Tool Output String:\n{context_string[:1000]}"
        
         )

        result: YesNoDecision = validator_llm.invoke([HumanMessage(content=prompt)])
        print(f"Context Usable Decision (yes/no): {result.decision}")

        return {"is_usable": result.decision}
    
    def rewrite_query(self, state: RAGState) -> dict:
        """Rewrites the query so the next search gets better results."""
        print("--- NODE: QUERY REWRITER ---")
        old_query = state["query"]
        retries = state.get("retry_count", 0) + 1

        prompt = (
            "The search query below did not return useful results. "
            "Rewrite it into a clearer, better search query. "
            "Reply with ONLY the rewritten query string.\n\n"
            f"Old Query: {old_query}"
        )
        rewritten_query = self.llm.invoke([HumanMessage(content=prompt)]).content.strip()
        print(f"Rewritten Query (Attempt {retries}): {rewritten_query}")

        return {
            "query": rewritten_query,
            "retry_count": retries,
        }
    
    # def _build_agent(self):
    #     """ReAct agent with tools"""
    #     tools = self._build_tools()
    #     memory=MemorySaver()
    #     system_prompt = (
    #         "You are a helpful RAG agent. "
    #         "You have access to these two tools retriever and tavily search resultGive me an answer based on these two tools"
    #         "Prefer 'retriever' for user-provided docs; use 'tavily_search_results_json' for general knowledge. "
    #         "Return only the final useful answer."
    #     )
    #     self._agent = create_react_agent(self.llm, tools=tools,prompt=system_prompt,checkpointer=memory)
    #     return self._agent

    # ---------------------------------------------------------
    # 5. LLM NODE / GENERATE ANSWER (With Memory Support)
    # ---------------------------------------------------------
    def generate_answer(self, state: RAGState) -> dict:
        print("--- NODE: GENERATE ANSWER (LLM) ---")
        context_string = state.get("tool_output_string", "")

        if state.get("needs_tool") == "no":
            # Pass full message history so the LLM remembers past chat
            response = self.llm.invoke(state["messages"])
        else:
            # Give the LLM the tool context + the full chat history
            system_msg = SystemMessage(
                content=(
                    "You are a helpful assistant. Answer the user's question "
                    f"using this context:\n\n{context_string}"
                )
            )
            response = self.llm.invoke([system_msg] + state["messages"][-7:])

        return {
            "messages": [response],
            "retrieved_docs": self.my_doc_bucket,
            "query": None,
            "tool_output_string": "",
            "retry_count": 0,
          }