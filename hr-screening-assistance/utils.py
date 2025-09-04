from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAI
from langchain.chains.summarize import load_summarize_chain
from pypdf import PdfReader
from dotenv import load_dotenv
import os

load_dotenv()


def _read_pdf_data(file):
    """
    Read PDF data from file and return text

    Args:
      file: File object

    Returns:
      Text data from PDF file
    """
    pdf_reader = PdfReader(file)
    pages = pdf_reader.pages
    text = ""
    for page in pages:
        text += page.extract_text()
    return text


def _create_embeddings():
    """
    Create OpenAI embeddings

    Returns:
      OpenAIEmbeddings object
    """
    try:
        embeddings = OpenAIEmbeddings()
        return embeddings
    except Exception as e:
        print(f"Error creating embeddings: {e}")
        raise e


def _create_or_get_index(index_name):
    """
    Create or get Pinecone index

    Args:
      index_name: Name of Pinecone index

    Returns:
      Pinecone index object
    """
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    if not pc:
        raise ValueError("Pinecone API key not found")

    if not pc.has_index(index_name):
        try:
            pc.create_index(
                name=index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        except Exception as e:
            print(f"Error creating Pinecone index: {e}")
            raise e

    try:
        index = pc.Index(index_name)
        return index
    except Exception as e:
        print(f"Error getting Pinecone index: {e}")
        raise e


# Chunk each pdf
def create_docs(pdf_files, uuid):
    """
    Create documents from PDF files

    Args:
      pdf_files: List of PDF file objects
      uuid: Unique identifier for documents

    Returns:
      List of Document objects
    """
    docs = []
    for pdf_file in pdf_files:
        pdf_text = _read_pdf_data(pdf_file)

        docs.append(
            Document(
                page_content=pdf_text,
                metadata={
                    "uuid": uuid,
                    "id": pdf_file.file_id,
                    "source": pdf_file.name,
                    "type=": pdf_file.type,
                    "size": pdf_file.size,
                },
            )
        )

    return docs


# Create embeddings and store to Vector Store
def push_to_pinecone(docs):
    """
    Push documents to Pinecone vector store

    Args:
      docs: List of Document objects

    Returns:
      PineconeVectorStore object
    """
    try:
        embeddings = _create_embeddings()

        index = _create_or_get_index(os.getenv("PINECONE_INDEX_NAME"))

        vector_store = PineconeVectorStore(index=index, embedding=embeddings)
        vector_store.add_documents(docs)

        return vector_store
    except Exception as e:
        print(f"Error pushing to Pinecone: {e}")
        raise e


def retrieve_relevant_docs(job_description, num_resumes, vector_store):
    """
    Retrieve relevant documents from Pinecone vector store

    Args:
      job_description: Job description text
      num_resumes: Number of resumes to retrieve
      vector_store: PineconeVectorStore object

    Returns:
      List of Document objects
    """
    similar_docs = vector_store.similarity_search_with_score(
        job_description, k=num_resumes
    )
    return similar_docs


def get_summary(similar_doc):
    """
    Get summary of similar documents

    Args:
      similar_doc: Document object

    Returns:
      Summary text
    """
    try:
        llm = OpenAI(temperature=0)
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        summary = chain.invoke([similar_doc])
        return summary["output_text"]
    except Exception as e:
        print(f"Error getting summary: {e}")
        raise e
