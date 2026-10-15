from src.state.rag_state import RAGState

class Ragnodes:
    """Contains node functions for RAG workflow"""
    
    
    def __init__(self,retriever,llm):
        """
        Initialize RAG nodes
        
        Args:
            retriever: Document retriever instance
            llm: Language model instance
        """
        self.retriever=retriever
        self.llm=llm
    
    def retrieve_node(self,state:RAGState)->RAGState:
        
        """
        Retrieve relevant documents node
        
        Args:
            state: Current RAG state
            
        Returns:
            Updated RAG state with retrieved documents
        """
        
        
        docs=self.retriever.invoke(state.question)
        return RAGState(question=state.question,retrieved_docs=docs)
    def geterate_node(self, state:RAGState)->RAGState:
        """
        Generate answer from retrieved documents node
        
        Args:
            state: Current RAG state with retrieved documents
            
        Returns:
            Updated RAG state with generated answer
        """
        
        context="\n\n".join(doc.page_content for doc in state["retrieved_docs"])
        prompt = f"""Answer the question based on the context.

        Context:
        {context}

        Question: {state.question}"""
        
        respo=self.llm.invoke(prompt)
        return RAGState(question=state.question,retrieved_docs=state.retrieved_docs,answer=respo.content)
            

        