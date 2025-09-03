from dotenv import load_dotenv
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
import os
import streamlit as st
from pinecone import Pinecone

load_dotenv()

@st.cache_resource
def _init_llm():
  llm = OpenAI(temperature=0)
  return llm

@st.cache_resource
def pull_index_data():
    """Connect to existing Pinecone index and return vector store."""
    try:
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
        
        if not pinecone_api_key or not pinecone_index_name:
            raise ValueError("Missing required environment variables PINECONE_API_KEY or PINECONE_INDEX_NAME")
        
        # Initialize embeddings
        embeddings = OpenAIEmbeddings()
        
        # Create vector store from existing index
        vector_store = PineconeVectorStore(
            index_name=pinecone_index_name,
            embedding=embeddings
        )
        
        return vector_store
    
    except Exception as e:
        print(f"Error in pull_index_data: {str(e)}")
        raise e

def retrieve_relevant_docs(query, vector_store, document_count=2):
  similar_docs = vector_store.similarity_search(query, k=document_count)
  return similar_docs

def combine_docs(similar_docs):
  combined_docs = "\n".join([doc.page_content for doc in similar_docs])
  return combined_docs

def generate_response(query, similar_docs):
  # Initialize LLM
  llm = _init_llm()
  
  # Retrieve Relevant Docs
  combined_docs = combine_docs(similar_docs)
  
  # Create Prompt
  prompt = PromptTemplate(
    input_variables=["combined_docs", "query"],
    template="""
    You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise.
    Context: {combined_docs}
    Question: {query}
    """,
  )
  
  # Create and run the chain
  chain = prompt | llm
  response = chain.invoke({"combined_docs": combined_docs, "query": query})

  return response

