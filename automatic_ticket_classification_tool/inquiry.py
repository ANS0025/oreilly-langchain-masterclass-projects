import streamlit as st

st.title("Chatbot")
st.write("Welcome to the Chatbot. How can I assist you today?")

with st.form("chat_form"):
    user_input = st.text_input("Enter your message:")
    submit_button = st.form_submit_button("Send")
    
if submit_button:
    if user_input:
        # Generate a response
        st.write(f"You: {user_input}")
        st.button("Submit Ticket?")
        if st.button("Submit Ticket?"):
          try:
            st.write("Ticket submitted!")
          except Exception as e:
            st.error(f"Error submitting ticket: {str(e)}")
            st.error("Please try again.")
    else:
        st.write("Please enter a message.")

