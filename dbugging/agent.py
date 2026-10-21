import sys
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 2. Load environment variables (.env file in root)
load_dotenv(PROJECT_ROOT / ".env")
from src.config.config import Config

from src.config.config import Config
from src.document_ingesition.document_processor import documentprocessor
from src.vector_store.vectorstore import Vectorstore
from src.grapg_builder.graph_builder import GraphBuilder





def make_graph():
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
        
        compiled_graph=graph_builder.build()
        compiled_graph.checkpointer=None
        return compiled_graph
    
graph=make_graph()