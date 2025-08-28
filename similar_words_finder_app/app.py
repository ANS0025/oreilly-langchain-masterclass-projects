# Import the required libraries
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain.schema import Document
from typing import List, Optional
import os

# Load the environment variables
load_dotenv()

# Configuration constants
DEFAULT_CSV_PATH = "myData.csv"
DEFAULT_K_RESULTS = 3

# Load the data
@st.cache_data
def load_data() -> Optional[List[Document]]:
    try: 
        with st.spinner("Loading documents from {file_path}..."):
            loader = CSVLoader(file_path=DEFAULT_CSV_PATH)
            documents = loader.load()
        return documents

    except FileNotFoundError:
        st.error(f"Error: The file {DEFAULT_CSV_PATH} was not found.")
        return None

# Create Vectore Database
@st.cache_resource
def create_vector_db(documents: List[Document]):
    try:       
        embeddings = OpenAIEmbeddings()
        db = FAISS.from_documents(documents, embeddings)
        return db
    except Exception as e:
        st.error(f"Error creating vector database: {e}")
        return None

# Create the retriever
def get_similar_matches(user_input: str, db: FAISS) -> List[Document]:
    try:
        with st.spinner("Searching for similar matches..."):
            similar_matches = db.similarity_search(user_input, k=DEFAULT_K_RESULTS)
        return similar_matches
    except Exception as e:
        st.error(f"Error searching for similar matches: {e}")
        return []

# Create the user input handler
def get_user_input() -> str:
    input_text: str = st.text_input("Enter a word", key="input")
    return input_text


# Create the streamlit app
def main():
    st.set_page_config(page_title="Similar Words Finder")
    st.header("Similar Words Finder")
    
    documents = load_data()
    db = create_vector_db(documents)
    
    user_input = get_user_input()
    submit = st.button("Find Similar Things")

    if submit:
        similar_matches = get_similar_matches(user_input, db)
        
        if not similar_matches:
            st.warning("No similar matches found.")
            return
        
        for match in similar_matches:
            st.write(match.page_content)

if __name__ == "__main__":
    main()