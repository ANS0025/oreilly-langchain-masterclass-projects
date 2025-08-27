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
    llm: ChatOpenAI = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    return llm

# Handle user input
def get_user_input() -> Optional[str]:
    input_text: Optional[str] = st.chat_input("What can I help you with?", key="input")
    return input_text

# Simple Stateful Message Memory
if "messages" not in st.session_state:
    st.session_state.messages: List[Union[SystemMessage, HumanMessage, AIMessage]] = [
        SystemMessage(content="You are a helpful assistant.")
    ]

# Get answer from LLM
def get_ai_response(question: str, llm: ChatOpenAI) -> str:
    st.session_state.messages.append(HumanMessage(content=question))
    answer: AIMessage = llm.invoke(st.session_state.messages)
    st.session_state.messages.append(AIMessage(content=answer.content))
    return answer.content

# Create the Streamlit app
def main():
    llm = setup_llm()

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