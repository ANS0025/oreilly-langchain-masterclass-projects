from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from langchain_openai import OpenAI 

def load_answer(question):
    llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)
    answer = llm(question)
    return answer
  
st.set_page_config(page_title="Simple Question Answering App")
st.header("Simple Question Answering App")

def get_text():
    input_text = st.text_input("You: ", key="input")
    return input_text

user_input = get_text()
response = load_answer(user_input)

submit = st.button("Generate")

if submit:
    st.subheader("Answer:")
    st.write(response)