import time
import os
from langchain_community.document_loaders.sitemap import SitemapLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

def _load_sitemap_data(url):
    """Load data from a sitemap URL."""
    print(f"Loading data from sitemap: {url}")
    sitemap_loader = SitemapLoader(url)
    data = sitemap_loader.load()
    print(f"Loaded {len(data)} documents from sitemap")
    return data
  
def _chunk_data(data):
    """Split documents into smaller chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100, 
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(data)
    print(f"Created {len(chunks)} chunks from {len(data)} documents")
    return chunks

def _create_or_get_pinecone_index(pinecone_api_key, index_name):
    """Create a new Pinecone index or get existing one."""
    pc = Pinecone(api_key=pinecone_api_key)
    
    # Check if index exists
    existing_indexes = pc.list_indexes()
    index_exists = any(index.name == index_name for index in existing_indexes)
    
    if not index_exists:
        print(f"Creating new index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=1536,  # OpenAI embeddings dimension
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        
        # Wait for index to be ready
        print("Waiting for index to be ready...")
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)
        print(f"Index {index_name} is ready!")
    else:
        print(f"Index {index_name} already exists")
    
    return index_name

def create_embeddings():
    """Create OpenAI embeddings instance."""
    # Make sure API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OpenAI API key not found in environment variables")
    
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=os.environ.get("OPENAI_API_KEY")
    )
    return embeddings

def load_data_to_pinecone(pinecone_api_key: str, pinecone_index: str):
    """Load data from sitemap to Pinecone index."""
    try:
        # Set Pinecone API key in environment (required by LangChain)
        os.environ["PINECONE_API_KEY"] = pinecone_api_key
        
        # Load data from website
        print("Step 1: Loading sitemap data...")
        data = _load_sitemap_data("https://netflixtechblog.medium.com/sitemap/sitemap.xml")
        
        if not data:
            raise ValueError("No data loaded from sitemap")
        
        # Chunk data
        print("Step 2: Chunking documents...")
        chunks = _chunk_data(data)
        
        if not chunks:
            raise ValueError("No chunks created from documents")
        
        # Create embeddings
        print("Step 3: Creating embeddings...")
        embeddings = create_embeddings()
        
        # Create or get index
        print("Step 4: Setting up Pinecone index...")
        index_name = _create_or_get_pinecone_index(pinecone_api_key, pinecone_index)
        
        # Store embeddings in Pinecone using index_name parameter
        print("Step 5: Storing embeddings in Pinecone...")
        vector_store = PineconeVectorStore(
            index_name=index_name,  # Use index_name instead of index
            embedding=embeddings,
            namespace=""  # Optional: use namespaces to organize your vectors
        )
        
        # Add documents in batches to avoid timeout
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            print(f"Adding batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
            vector_store.add_documents(documents=batch)
        
        print(f"Successfully added {len(chunks)} documents to Pinecone")
        
        # Get index statistics
        pc = Pinecone(api_key=pinecone_api_key)
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        
        return vector_store, {
            'documents_added': len(chunks),
            'total_vectors': stats.get('total_vector_count', 0)
        }
        
    except Exception as e:
        print(f"Error in load_data_to_pinecone: {str(e)}")
        raise e
  
def pull_index_data(pinecone_api_key: str, pinecone_index_name: str, embeddings):
    """Connect to existing Pinecone index and return vector store."""
    try:
        # Set Pinecone API key in environment (required by LangChain)
        os.environ["PINECONE_API_KEY"] = pinecone_api_key
        
        # Verify index exists
        pc = Pinecone(api_key=pinecone_api_key)
        existing_indexes = pc.list_indexes()
        index_exists = any(index.name == pinecone_index_name for index in existing_indexes)
        
        if not index_exists:
            raise ValueError(f"Index {pinecone_index_name} does not exist. Please load data first.")
        
        # Create vector store using index_name
        vector_store = PineconeVectorStore(
            index_name=pinecone_index_name,  # Use index_name instead of index
            embedding=embeddings,
            namespace=""  # Should match the namespace used when storing
        )
        
        return vector_store
        
    except Exception as e:
        print(f"Error in pull_index_data: {str(e)}")
        raise e
  
def fetch_relevant_documents(vector_store, prompt, document_count):
    """Search for relevant documents in the vector store."""
    try:
        print(f"Searching for: {prompt}")
        results = vector_store.similarity_search(
            query=prompt, 
            k=document_count
        )
        print(f"Found {len(results)} relevant documents")
        return results
    except Exception as e:
        print(f"Error in fetch_relevant_documents: {str(e)}")
        raise e