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
        st.session_state.messages.append(HumanMessage(content=question))
        answer: AIMessage = llm.invoke(st.session_state.messages)
        st.session_state.messages.append(AIMessage(content=answer.content))
        return answer.content
    
    except Exception as e:
        logger.error(f"Error getting AI response: {e}")
        st.error("Sorry, I encountered an error. Please try again.")
        return None

# Create the Streamlit app
def main():
    llm = setup_llm()
    setup_session_state()

    st.set_page_config(page_title="Simple Question Answering App")
    st.header("Simple Question Answering App")

    user_input: Optional[str] = get_user_input()

    if user_input:
        get_ai_response(user_input, llm)        

        # Display the conversation history
        for message in st.session_state.messages:
            if isinstance(message, HumanMessage):
                st.chat_message("user").write(message.content)
            elif isinstance(message, AIMessage):
                st.chat_message("assistant").write(message.content)

if __name__ == "__main__":
    main()