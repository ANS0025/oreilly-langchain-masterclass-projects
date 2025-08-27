# Import necessary libraries
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os
import streamlit as st
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from typing import List, Union, Optional

# Load environment variables
load_dotenv()

# Setup LLM
@st.cache_resource
def setup_llm():
    try:
        if not os.getenv("OPENAI_API_KEY"):
            st.error("OpenAI API key not found. Please set OPENAI_API_KEY in your environment.")
            st.stop()
    
        llm: ChatOpenAI = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
        return llm

    except Exception as e:
        logger.error(f"Error setting up LLM: {e}")
        st.error(f"Error setting up LLM: {e}")
        return None

# Handle user input
def get_user_input() -> Optional[str]:
    input_text: Optional[str] = st.chat_input("What can I help you with?", key="input")
    return input_text

# Simple Stateful Message Memory
def setup_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages: List[Union[SystemMessage, HumanMessage, AIMessage]] = [
            SystemMessage(content="You are a helpful assistant.")
        ]

# Get answer from LLM
def get_ai_response(question: str, llm: ChatOpenAI) -> Optional[str]:
    try:
        # Add the user question to the conversation history
        st.session_state.messages.append(HumanMessage(content=question))
        
        with st.spinner("Thinking..."):
            answer: AIMessage = llm.invoke(st.session_state.messages)
        
        # Add the AI response to the conversation history
        st.session_state.messages.append(AIMessage(content=answer.content))
        return answer.content
    
    except Exception as e:
        logger.error(f"Error getting AI response: {e}")
        st.error("Sorry, I encountered an error. Please try again.")
        return None
    
def display_conversation_history() -> None:
    # Skip the system message when displaying
    display_messages = [msg for msg in st.session_state.messages 
                       if not isinstance(msg, SystemMessage)]
    
    # Display the conversation history
    for message in display_messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.write(message.content)

# Create the Streamlit app
def main():
    llm = setup_llm()
    setup_session_state()

    st.set_page_config(page_title="Simple Question Answering App")
    st.header("Simple Question Answering App")

    # Display the conversation history
    display_conversation_history()

    # Get user input
    user_input: Optional[str] = get_user_input()

    if user_input:
        # Display user message immediately
        with st.chat_message("user"):
            st.write(user_input)
        
        # Get and display the AI response
        ai_response: Optional[str] = get_ai_response(user_input, llm)
        
        # Add the AI response to the conversation history
        if ai_response:
            with st.chat_message("assistant"):
                st.write(ai_response)
        

if __name__ == "__main__":
    main()