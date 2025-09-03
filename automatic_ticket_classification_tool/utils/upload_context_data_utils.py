from pinecone import Pinecone, ServerlessSpec, PineconeException
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from typing import Optional, List
from dotenv import load_dotenv

load_dotenv()

class EmbeddingError(Exception):
    """Custom exception for embedding-related errors"""
    pass

class PdfReadError(Exception):
    """Custom exception for PDF reading errors"""
    pass

class VectorStoreError(Exception):
    """Custom exception for vector store operations"""
    pass

def create_embeddings() -> Optional[OpenAIEmbeddings]:
    """Create embeddings using OpenAIEmbeddings"""
    try:
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable not set")
        embeddings = OpenAIEmbeddings()
        return embeddings
    except Exception as e:
        print(f"Failed to create embeddings: {str(e)}")
        raise EmbeddingError(f"Error creating embeddings: {str(e)}")

def read_pdf_data(pdf_file) -> str:
    """Read and extract text from PDF file"""
    try:
        reader = PdfReader(pdf_file)
        if len(reader.pages) == 0:
            print("Empty PDF file")
            raise PdfReadError("PDF file is empty")
        
        text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
        
        if not text.strip():
            print("No text content extracted from PDF")
            raise PdfReadError("No text content extracted from PDF")
            
        return text
    except Exception as e:
        print(f"Error reading PDF file: {str(e)}")
        raise PdfReadError(f"Failed to read PDF file: {str(e)}")

def chunk_data(data: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List:
    """Split text data into smaller chunks for processing"""
    if not isinstance(data, str):
        print(f"Invalid input type: {type(data)}")
        raise TypeError("Input data must be a string")
    
    if not data.strip():
        print("Empty input data")
        raise ValueError("Input data is empty")
    
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        print(f"Invalid chunk parameters: size={chunk_size}, overlap={chunk_overlap}")
        raise ValueError("Invalid chunk size or overlap parameters")
        
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )
        
        chunks = text_splitter.create_documents([data])
        if not chunks:
            print("No chunks created from input data")
            raise ValueError("No chunks created from input data")
        return chunks
    except Exception as e:
        print(f"Error chunking data: {str(e)}")
        raise

def _initialize_pinecone_client() -> Pinecone:
    """Initialize Pinecone client"""
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("PINECONE_API_KEY not set")
        raise ValueError("PINECONE_API_KEY environment variable not set")
        
    try:
        pc = Pinecone(api_key=api_key)
        return pc
    except PineconeException as e:
        print(f"Pinecone initialization error: {str(e)}")
        raise
    except Exception as e:
        print(f"Failed to initialize Pinecone client: {str(e)}")
        raise

def _create_index(pc: Pinecone, index_name: str):
    """Create Pinecone index if it doesn't exist"""
    if not index_name:
        print("Empty index name provided")
        raise ValueError("Index name cannot be empty")
        
    try:
        if not pc.has_index(index_name):
            print(f"Creating new Pinecone index: {index_name}")
            pc.create_index(
                name=index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
        index = pc.Index(index_name)
        return index
    except PineconeException as e:
        print(f"Pinecone operation failed: {str(e)}")
        raise
    except Exception as e:
        print(f"Error creating/accessing index: {str(e)}")
        raise

def store_embeddings_into_vector_store(documents, embeddings):
    """Store embeddings into Pinecone vector store"""
    if not documents:
        print("Empty documents list provided")
        raise ValueError("Documents list cannot be empty")
    if not embeddings:
        print("No embeddings object provided")
        raise ValueError("Embeddings object cannot be None")
        
    try:
        # Initialize Pinecone Client
        pc = _initialize_pinecone_client()
        
        # Create Index
        pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
        if not pinecone_index_name:
            print("PINECONE_INDEX_NAME not set")
            raise ValueError("PINECONE_INDEX_NAME environment variable not set")
            
        index = _create_index(pc, pinecone_index_name)
        
        # Store Embeddings
        vector_store = PineconeVectorStore(index=index, embedding=embeddings)
        vector_store.add_documents(documents=documents)
        print(f"Successfully stored {len(documents)} documents in vector store")
        return vector_store
    except (PineconeException, ValueError) as e:
        print(f"Vector store operation failed: {str(e)}")
        raise VectorStoreError(f"Failed to store embeddings: {str(e)}")
    except Exception as e:
        print(f"Unexpected error while storing embeddings: {str(e)}")
        raise VectorStoreError(f"Failed to store embeddings: {str(e)}")