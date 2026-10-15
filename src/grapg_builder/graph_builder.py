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
        """
        Build the RAG workflow graph
        
        Returns:
            Compiled graph instance
        """
        # Create state graph
        
        # builder = StateGraph(RAGState)
        
        # # Add nodes
        # # builder.add_node("retriever", self.nodes.retrieve_docs)
        # builder.add_node("responder", self.nodes.generate_answer)
        
        # # Set entry point
        # builder.set_entry_point("responder")
        
        # # Add edges
        # builder.add_edge(" "responder")
        # builder.add_edge("responder", END)
        
        # # Compile graph
        # self.graph = builder.compile(checkpointer=memory)
        self.graph=self.nodes._build_agent()
        return self.graph
    
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
        
        
        # Package your string into a real message object
        new_input = {"messages": [HumanMessage(content=question)]}
        return self.graph.invoke(new_input,config=config)