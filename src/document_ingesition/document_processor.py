"""Document processing module for loading and splitting documents"""

from typing import List
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from typing import List, Union
from pathlib import Path
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    TextLoader,
    PyPDFDirectoryLoader
)
import re

class documentprocessor:
    """Handles document loading and processing"""
    def __init__(self,chunk_size=900,chunk_overlap=80):
        """
        Initialize document processor
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size=chunk_size
        self.chunk_overlap=chunk_overlap
        self.splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    def load_form_url(self,url:str)->List[Document]:
        """Load document(s) from a URL"""
        lodder=WebBaseLoader(url)
        return lodder.load()
    def load_form_pdf_dir(self, directory: Union[str,Path])->List[Document]:
        """Load documents from all PDFs inside a directory"""
        lodder=PyPDFDirectoryLoader(str(directory))
        return lodder.load()
    def load_form_text(self,File_path:Union[str,Path])->List[Document]:
        """Load document(s) from a TXT file"""
        lodder=TextLoader(str(File_path),encoding="utf-8")
        return lodder.load()
    def load_from_pdf(self, file_path: Union[str, Path]) -> List[Document]:
        """Load document(s) from a PDF file"""
        loader = PyPDFDirectoryLoader(str("data"))
        return loader.load()
    def load_document(self,source:List[str])->List[Document]:
        """
        Load documents from URLs, PDF directories, or TXT files

        Args:
            sources: List of URLs, PDF folder paths, or TXT file paths

        Returns:
            List of loaded documents
        """
        docs: List[Document]=[]
        for src in source:
            if src.startswith("http://") or src.startswith("https://"):
                docs.extend(self.load_form_url(src))
            path=Path("data")
            if path.exists():
              docs.extend(self.load_form_pdf_dir(path))
            elif path.suffix.lower()==".txt":
                docs.extend(self.load_form_text(path))
            else:
                raise ValueError(
                    f"Unsupported source type: {src}. "
                    "Use URL, .txt file, or PDF directory."
                )
        clean_docs:List[Document]=[]
        for doc in docs:
            text = doc.page_content
        
            # 1. Normalize excessive whitespace and consecutive blank lines
            text = re.sub(r'\r\n|\r', '\n', text)
            text = re.sub(r'\n{2,}', '\n\n', text)
            text = re.sub(r'[ \t]+', ' ', text)
            text = re.sub(r"\n+", "", text)
            # 2. Strip common blog/web noise artifacts (header bars, nav lines)
            text = re.sub(r'(?:\|\s*)+(?:Posts|Archive|Search|Tags|FAQ)\b.*', '', text, flags=re.IGNORECASE)
            
            clean_docs.append(Document(page_content=text,metadata=doc.metadata.copy()))
        print(f"{len(clean_docs)} documents cleaned ..ready to split\n",flush=True)
        return clean_docs
    def split_document(self,documets:List[Document])->List[Document]:
            """
            Split documents into chunks
            
            Args:
                documents: List of documents to split
                
            Returns:
                List of split documents
            """
            return self.splitter.split_documents(documets)
    def process_urls(self,urls:List[str])->List[Document]:
        """
        Complete pipeline to load and split documents
        
        Args:
            urls: List of URLs to process
            
        Returns:
            List of processed document chunks
        """
        docs=self.load_document(urls)
        return self.split_document(docs)
            
        