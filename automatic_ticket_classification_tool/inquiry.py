import streamlit as st
from utils.inquiry_utils import generate_response, pull_index_data, retrieve_relevant_docs
import os
from dotenv import load_dotenv

load_dotenv()

st.title("Chatbot")
st.write("Welcome to the Chatbot. How can I assist you today?")

with st.form("chat_form"):
    user_input = st.text_input("Enter your message:")
    submit_button = st.form_submit_button("Send")
    
if submit_button:
    if user_input:
        # Generate a response
        with st.spinner("Generating response..."):
          vector_store = pull_index_data()
          similar_docs = retrieve_relevant_docs(user_input, vector_store)
          response = generate_response(user_input, similar_docs)
        st.write(response)
        if st.button("Submit Ticket?", key="submit_ticket_btn"):
          try:
            st.write("Ticket submitted!")
          except Exception as e:
            st.error(f"Error submitting ticket: {str(e)}")
            st.error("Please try again.")
    else:
        st.write("Please enter a message.")

