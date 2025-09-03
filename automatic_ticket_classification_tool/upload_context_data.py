import streamlit as st
from utils.upload_context_data_utils import chunk_data, read_pdf_data, create_embeddings, store_embeddings_into_vector_store

st.title("Upload Context Data")
st.write("Upload the context data for the chatbot")

uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
store_button = st.button("Store File to Vector Store")

if uploaded_file is not None:
    st.success("File uploaded successfully!")
    # Store file into vector store
    
if store_button:
    if uploaded_file is not None:
      try:
          with st.spinner("Storing file into vector store..."):
              # Read PDF data
              pdf_data = read_pdf_data(uploaded_file)
              # Chunk data
              chunks = chunk_data(pdf_data)
              # Store data into Pinecone
              embeddings = create_embeddings()
              store_embeddings_into_vector_store(chunks, embeddings)
              
              
          st.success("File stored into vector store!")
      except Exception as e:
          st.error(f"Error storing file into vector store: {str(e)}")
          st.error("Please try again.")
    else:
        st.error("Please upload a file first.")
