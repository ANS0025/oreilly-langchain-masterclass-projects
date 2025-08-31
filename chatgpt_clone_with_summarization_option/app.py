"""
A Streamlit web application for a chatbot with conversation summarization.

This application allows users to interact with an OpenAI-powered chatbot.
It maintains a conversation history and can provide a summary of the chat.
The user's OpenAI API key is required to use the chatbot.
"""

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationSummaryMemory
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from typing import List, Union, Optional

# Define a type for the messages
Message = Union[HumanMessage, AIMessage, SystemMessage]

def initialize_session_state() -> None:
    """
    Initializes the session state variables if they are not already set.
    'API_Key': Stores the user's OpenAI API key.
    'messages': A list to store the conversation history.
    'conversation': A ConversationChain object for managing the chat.
    """
    if 'API_Key' not in st.session_state:
        st.session_state['API_Key'] = ''
    if 'messages' not in st.session_state:
        st.session_state['messages']: List[Message] = [
            SystemMessage(content="You are a helpful assistant.")
        ]
    if 'conversation' not in st.session_state:
        st.session_state['conversation']: Optional[ConversationChain] = None

def initialize_llm() -> Optional[ChatOpenAI]:
    """
    Initializes the Language Model (LLM) with the user's API key.

    Returns:
        An instance of ChatOpenAI if the API key is valid, otherwise None.
    """
    if not st.session_state['API_Key']:
        return None
        
    try:
        llm = ChatOpenAI(
            temperature=0,
            openai_api_key=st.session_state['API_Key'],
            model_name='gpt-3.5-turbo'
        )
        return llm
    except Exception as e:
        st.sidebar.error(f"Error initializing LLM: {e}")
        return None

def handle_prompt() -> Optional[str]:
    """
    Displays a chat input box and returns the user's prompt.

    Returns:
        The user's input as a string, or None if no input is provided.
    """
    prompt = st.chat_input("Ask me anything!", key="input")
    return prompt

def display_conversation_history() -> None:
    """
    Displays the conversation history in the Streamlit chat interface.
    """
    for message in st.session_state['messages']:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.write(message.content)
        
def get_response(prompt: str) -> Optional[str]:
    """
    Gets a response from the LLM and updates the conversation history.

    Args:
        prompt: The user's input prompt.

    Returns:
        The LLM's response as a string, or None if an error occurs.
    """
    try:
        if st.session_state['conversation'] is None:
            llm = initialize_llm()
            if llm is None:
                return None
            memory = ConversationSummaryMemory(llm=llm)
            st.session_state['conversation'] = ConversationChain(
                llm=llm,
                verbose=True,
                memory=memory
            )
        
        result = st.session_state['conversation'].invoke({"input": prompt})
        response_text = result.get('response', str(result))
        
        st.session_state['messages'].append(HumanMessage(content=prompt))
        st.session_state['messages'].append(AIMessage(content=response_text))

        return response_text
    
    except Exception as e:
        st.error(f"Error getting response: {e}")
        return None

def summarise_conversation() -> None:
    """
    Displays a summary of the conversation in the sidebar.
    """
    if st.session_state['conversation'] is not None:
        st.sidebar.write("**Conversation Summary:**")
        st.sidebar.write(st.session_state['conversation'].memory.buffer)
    else:
        st.sidebar.write("No conversation to summarise.")

def main() -> None:
    """
    The main function that runs the Streamlit application.
    """
    st.title("Chat with OpenAI")
    initialize_session_state()
    
    with st.sidebar:
        st.header("Settings")
        api_key = st.text_input("What's your API key?", type="password", key="api_key_input")
        if api_key:
            st.session_state['API_Key'] = api_key
            st.success("API key provided!")
        else:
            st.warning("API key cannot be empty!")

        st.divider()
        summarise_button = st.button("Summarise the conversation", key="summarise")
        if summarise_button:
            summarise_conversation()
            
    if not st.session_state['API_Key']:
        st.info("Please provide your OpenAI API key in the sidebar to start chatting.")
        return
        
    display_conversation_history()
    
    prompt = handle_prompt()
    if prompt:
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_response(prompt)
                if response:
                    st.write(response)
                else:
                    st.error("Failed to get response. Please check your API key and try again.")
        
        st.rerun()

if __name__ == '__main__':
    main()