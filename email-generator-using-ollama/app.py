from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
import streamlit as st
import re


# Initialize the LLM
@st.cache_resource
def init_llm():
    llm = ChatOllama(model="qwen3:8b", validate_model_on_init=True, think=False)
    return llm


# Create the prompt template
def create_prompt_template():
    prompt_template = PromptTemplate(
        input_variables=["email_topic", "tone", "recipient", "sender"],
        template="""
      Write a professional email about {email_topic}.
      The tone should be {tone}.
      This email is being sent to {recipient}.
      The email is from {sender}.
      
      Your response should only be the email, with no other text or explanation.
      """,
    )
    return prompt_template


def strip_think_tags(text):
    """
    Strips content between <think> tags from the text.
    Args:
        text (str): Input text that may contain <think> tags
    Returns:
        str: Text with think tag content removed
    """
    pattern = r"<think>.*?</think>"
    return re.sub(pattern, "", text, flags=re.DOTALL).strip()


# Create the LLM chain
def create_email_chain(llm, prompt_template):
    email_chain = prompt_template | llm
    return email_chain


def main():
    # Initialize the LLM
    llm = init_llm()
    # Create the prompt template
    prompt_template = create_prompt_template()
    # Create the LLM chain
    email_chain = create_email_chain(llm, prompt_template)

    # Configure page settings
    st.set_page_config(
        page_title="Email Generator",
        page_icon="📧",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    # Create title and description
    st.title("✉️ Email Generator App")
    st.markdown("Generate professional emails quickly and easily!")

    # Create input fields
    col1, col2, col3 = st.columns(3)
    with col1:
        sender = st.text_input("Who is sending this email?")
    with col2:
        recipient = st.text_input("Who is the recipient?")
    with col3:
        tone = st.selectbox(
            "Select the tone of the email:",
            ["Professional", "Friendly", "Urgent", "Formal"],
        )

    email_topic = st.text_area("What is your email about?", height=200)

    # Generate button
    if st.button("Generate Email"):
        if email_topic and tone and recipient and sender:
            with st.spinner("Generating email..."):
                response = email_chain.invoke(
                    {
                        "email_topic": email_topic,
                        "tone": tone,
                        "recipient": recipient,
                        "sender": sender,
                    }
                ).content
                cleaned_response = strip_think_tags(response)
            with st.container(border=True):
                st.subheader("Generated Email")
                st.write(cleaned_response)
        else:
            st.warning("Please fill in all fields.")


if __name__ == "__main__":
    main()
