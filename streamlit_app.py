"""Streamlit UI for Agentic RAG System - Simplified Version"""

import streamlit as st
from pathlib import Path
import sys
import time

# Add src to path
sys.path.append(str(Path(__file__).parent))
from langchain_core.callbacks import UsageMetadataCallbackHandler

from src.config.config import Config
from src.document_ingesition.document_processor import documentprocessor
from src.vector_store.vectorstore import Vectorstore
from src.grapg_builder.graph_builder import GraphBuilder
from src.nodes.reactnodes import RAGNodes

# Page configuration
st.set_page_config(
    page_title="🤖 RAG Search",
    page_icon="🔍",
    layout="centered"
)

# Simple CSS
st.markdown("""
    <style>
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

def init_session_state():
    """Initialize session state variables"""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'history' not in st.session_state:
        st.session_state.history = []

@st.cache_resource
def initialize_rag():
    """Initialize the RAG system (cached)"""
    try:
        # Initialize components
        llm = Config.get_llm()
        doc_processor = documentprocessor(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        vector_store = Vectorstore()
        
        # Use default URLs
        urls = Config.DEFAULT_URLS
        
        # Process documents
        documents = doc_processor.process_urls(urls)
        
        # Create vector store
        vector_store.create_vectorstore(documents)
        
        # Build graph
        graph_builder = GraphBuilder(
            retriever=vector_store.get_retriever(),
            llm=llm
        )
        graph_builder.build()
        
        return graph_builder, len(documents)
    except Exception as e:
        st.error(f"Failed to initialize: {str(e)}")
        return None, 0

def main():
    """Main application"""
    init_session_state()
    
    # Title
    st.title("🔍 RAG Document Search")
    st.markdown("Ask questions about the loaded documents")
    
    # Initialize system
    if not st.session_state.initialized:
        with st.spinner("Loading system..."):
            rag_system, num_chunks = initialize_rag()
            if rag_system:
                st.session_state.rag_system = rag_system
                st.session_state.initialized = True
                st.success(f"✅ System ready! ({num_chunks} document chunks loaded)")
    # Display the LangGraph workflow diagram
    if st.session_state.rag_system:
        with st.expander("🗺️ View Agentic Graph Workflow"):
            graph_png = st.session_state.rag_system.graph.get_graph().draw_mermaid_png()
            st.image(graph_png, caption="Agentic RAG Architecture")
    st.markdown("---")
    
    # Search interface
    with st.form("search_form"):
        question = st.text_input(
            "Enter your question:",
            placeholder="What would you like to know?"
        )
        submit = st.form_submit_button("🔍 Search")
    
    # Process search
    if submit and question:
        if st.session_state.rag_system:
            with st.spinner("Searching..."):
                token_callback = UsageMetadataCallbackHandler()
                my_config = {
                    "configurable": {"thread_id": "chat_with_purnava123"},
                    "callbacks": [token_callback]
                }
                start_time = time.time()
                result = st.session_state.rag_system.run(question, my_config)
                elapsed_time = time.time() - start_time
                
                total_tokens = sum(
                usage.get("total_tokens", 0) 
                for usage in token_callback.usage_metadata.values()
                )
                final_text =getattr(result["messages"][-1],'text')
                
                result['retrieved_docs']=st.session_state.rag_system.get_reset_retrieve_docs()
                # Add to history
                st.session_state.history.append({
                        'question': question,
                        'answer': final_text,
                        'time': elapsed_time,
                        'tokens': total_tokens
                        
                    })
                
                # Display answer
                st.markdown("### 💡 Answer")
                st.success(final_text)
                st.caption(f"⏱️ Response time: {elapsed_time:.2f} seconds | 🪙 Total Tokens: {total_tokens}")
                
                # Show retrieved docs in expander
                with st.expander("📄 Source Documents"):
                    for i, doc in enumerate(result['retrieved_docs'], 1):
                        st.text_area(
                            f"Document {i}",
                            doc.page_content[:500] + "...",
                            height=150,
                            disabled=True
                        )
                
                    
    # Show history
    if st.session_state.history:
        st.markdown("---")
        st.markdown("### 📜 Recent Searches")
        
        for item in reversed(st.session_state.history[-3:]):  # Show last 3
            with st.container():
                st.markdown(f"**Q:** {item['question']}")
                st.markdown(f"**A:** {item['answer'][:200]}...")
                st.caption(f"Time: {item['time']:.2f}s Tokens: {item['tokens']}")
                st.markdown("")

if __name__ == "__main__":
    main()