import streamlit as st
import os
from dotenv import load_dotenv
from utils import create_embeddings, pull_index_data, fetch_relevant_documents, load_data_to_pinecone

load_dotenv()

def initialize_session_state():
    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = ""
    if "pinecone_api_key" not in st.session_state:
        st.session_state.pinecone_api_key = ""
        
def validate_api_keys(openai_api_key, pinecone_api_key):
    if not openai_api_key or not pinecone_api_key:
        st.error("Please enter both OpenAI and Pinecone API keys.")
        return False
    return True
  
def store_api_keys(openai_api_key, pinecone_api_key):
    st.session_state.openai_api_key = openai_api_key
    os.environ["OPENAI_API_KEY"] = openai_api_key
    
    st.session_state.pinecone_api_key = pinecone_api_key
    os.environ["PINECONE_API_KEY"] = pinecone_api_key

def main():
    st.title("Support Chatbot for Your Website")
    st.write("Welcome to the Support Chatbot for Your Website. How can I assist you today?")
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        openai_api_key = st.text_input("OpenAI API Key", type="password", value=st.session_state.openai_api_key)
        pinecone_api_key = st.text_input("Pinecone API Key", type="password", value=st.session_state.pinecone_api_key)
        
        # Store API keys whenever they're entered
        if openai_api_key or pinecone_api_key:
            store_api_keys(openai_api_key, pinecone_api_key)
        
        st.divider()
        
        # Load data section
        st.subheader("Load Data to Pinecone")
        load_button = st.button("Load data to Pinecone", type="primary")
        
        if load_button:
            if not validate_api_keys(openai_api_key, pinecone_api_key):
                st.error("Please enter both API keys before loading data.")
            else:
                try:
                    pinecone_index_name = os.getenv("PINECONE_INDEX")
                    if not pinecone_index_name:
                        st.error("PINECONE_INDEX not found in environment variables. Please add it to your .env file.")
                    else:
                        with st.spinner("Loading data to Pinecone... This may take a few minutes."):
                            vector_store, stats = load_data_to_pinecone(
                                st.session_state.pinecone_api_key, 
                                pinecone_index_name
                            )
                        st.success(f"✅ Data loaded successfully! Added {stats['documents_added']} documents to Pinecone.")
                        st.info(f"Total vectors in index: {stats['total_vectors']}")
                except Exception as e:
                    st.error(f"Error loading data to Pinecone: {str(e)}")
                    st.error("Please check your API keys and try again.")
    
    # Main content area
    st.header("Search Documents")
    
    with st.form(key="search_form"):
        prompt = st.text_input("Enter your question or topic to search:", 
                              placeholder="e.g., How to use LangChain with OpenAI?")
        document_count = st.slider("Number of documents to retrieve", 
                                  min_value=1, 
                                  max_value=10, 
                                  value=3, 
                                  step=1,
                                  help="More documents = more context but slower response")
        submit_button = st.form_submit_button(label="🔍 Search", type="primary")
        
    if submit_button:
        if not validate_api_keys(st.session_state.openai_api_key, st.session_state.pinecone_api_key):
            st.error("Please enter both API keys in the sidebar before searching.")
        elif not prompt:
            st.error("Please enter a topic to search.")
        else:
            # Perform search
            try:
                pinecone_index_name = os.getenv("PINECONE_INDEX")
                if not pinecone_index_name:
                    st.error("PINECONE_INDEX not found in environment variables.")
                else:
                    with st.spinner("Searching for relevant documents..."):
                        # Create embeddings
                        embeddings = create_embeddings()
                        # Pull index data from Pinecone
                        vector_store = pull_index_data(
                            st.session_state.pinecone_api_key, 
                            pinecone_index_name, 
                            embeddings
                        )
                        # Fetch relevant documents from index
                        results = fetch_relevant_documents(vector_store, prompt, document_count)
                    
                    # Display search results
                    if results:
                        st.success(f"Found {len(results)} relevant documents:")
                        for i, doc in enumerate(results, 1):
                            with st.expander(f"📄 Document {i}: {doc.metadata.get('source', 'Unknown source')[:100]}..."):
                                st.write("**Content:**")
                                st.write(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                                st.write("**Metadata:**")
                                st.json(doc.metadata)
                    else:
                        st.warning("No relevant documents found. Try a different search term.")
                        
            except Exception as e:
                st.error(f"Error searching Pinecone: {str(e)}")
                st.info("Please make sure you've loaded data to Pinecone first using the button in the sidebar.")
    
if __name__ == "__main__":
    main()