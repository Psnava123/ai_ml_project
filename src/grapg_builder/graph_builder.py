"""Graph builder for LangGraph workflow"""

from langgraph.graph import StateGraph, END
from src.state.rag_state import RAGState
from src.nodes.reactnodes import RAGNodes
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage


class GraphBuilder:
    """Builds and manages the LangGraph workflow"""
    
    def __init__(self, retriever, llm):
        """
        Initialize graph builder
        
        Args:
            retriever: Document retriever instance
            llm: Language model instance
        """
        self.nodes = RAGNodes(retriever, llm)
        self.graph = None
    
    def build(self):
       workflow=StateGraph(RAGState)
       workflow.add_node("query_analyzer",self.nodes.query_analyzer)
       workflow.add_node("tool_node",self.nodes.tool_node)
       workflow.add_node("validate_context",self.nodes.validate_context)
       workflow.add_node("rewrite_query",self.nodes.rewrite_query)
       workflow.add_node("generate_answer", self.nodes.generate_answer)
       workflow.set_entry_point("query_analyzer")
       
       def route_from_analyzer(state: RAGState):
        if state.get("needs_tool") == "yes":
            return "yes"
        return "no"
       workflow.add_conditional_edges("query_analyzer",route_from_analyzer,{
            "yes": "tool_node",
            "no": "generate_answer",
        },)
       workflow.add_edge("tool_node", "validate_context")

        # 5. Validate Context Router ("yes" -> LLM Generate Answer, "no" -> Rewrite Query)
       def route_from_validator(state: RAGState):
        if state.get("is_usable") == "yes" or state.get("retry_count", 0) >= 2:
            return "yes"
        return "no"
    
       workflow.add_conditional_edges("validate_context",route_from_validator,
            {
                "yes": "generate_answer",
                "no": "rewrite_query",
            },
            )
       workflow.add_edge("rewrite_query", "query_analyzer")

        # 7. LLM Generate Answer -> END
       workflow.add_edge("generate_answer", END)

        # 8. ADD CHECKPOINTER HERE!
       memory = MemorySaver()
       self.graph = workflow.compile(checkpointer=memory)
       return self.graph


       
    def get_reset_retrieve_docs(self):
        "for getting my retrieve docs form app"
        docs = self.nodes.my_doc_bucket.copy()
       
        return docs
    
    def run(self, question: str,config) -> dict:
        """
        Run the RAG workflow
        
        Args:
            question: User question
            
        Returns:
            Final state with answer
        """
        if self.graph is None:
            self.build()
        self.nodes.my_doc_bucket.clear()
        
        # Pa ckage your string into a real message object
        new_input = {"messages": [HumanMessage(content=question)]}
        return self.graph.invoke(new_input,config=config)