
import streamlit as st
from utils.inquiry_utils import generate_response, pull_index_data, retrieve_relevant_docs
from utils.classification_utils import classify_ticket
from dotenv import load_dotenv

load_dotenv()

# Initialize session state
def _init_session_state():
    if "tickets" not in st.session_state:
        st.session_state.tickets = {}
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""
    if "response" not in st.session_state:
        st.session_state.response = ""

def _clear_session_input_and_response():
    st.session_state.user_input = ""
    st.session_state.response = ""

def _save_ticket_to_session_state(ticket, classification):
    category = classification["category"]
    if category not in st.session_state.tickets:
        st.session_state.tickets[category] = []
    st.session_state.tickets[category].append(ticket)

st.title("Chatbot")
st.write("Welcome to the Chatbot. How can I assist you today?")
    
_init_session_state()

with st.form("chat_form"):
    user_input = st.text_input("Enter your message:")
    submit_button = st.form_submit_button("Send")

if submit_button:
    if user_input:
        st.session_state.user_input = user_input
        with st.spinner("Generating response..."):
            vector_store = pull_index_data()
            similar_docs = retrieve_relevant_docs(user_input, vector_store)
            st.session_state.response = generate_response(user_input, similar_docs)
    else:
        st.error("Please enter a message.")

if st.session_state.response:
    st.write(st.session_state.response)
    if st.button("Submit Ticket?", key="submit_ticket_btn"):
        try:
            classification = classify_ticket(st.session_state.user_input)
            category = classification['category']
            _save_ticket_to_session_state(st.session_state.user_input, classification)
            _clear_session_input_and_response()
            st.success(f"Ticket submitted successfully and classified as: **{category}**")
        except Exception as e:
            st.error(f"Error submitting ticket: {str(e)}")
            st.error("Please try again.")
