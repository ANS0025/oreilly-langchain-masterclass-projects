import streamlit as st

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
              # Code to store file into vector store
              # ...
              pass
          st.success("File stored into vector store!")
      except Exception as e:
          st.error(f"Error storing file into vector store: {str(e)}")
          st.error("Please try again.")
    else:
        st.error("Please upload a file first.")
