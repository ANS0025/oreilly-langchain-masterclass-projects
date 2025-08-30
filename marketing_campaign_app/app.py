"""Marketing Campaign App using LangChain and OpenAI.

This Streamlit application generates marketing content tailored to different age groups
using few-shot prompting with LangChain.
"""

import streamlit as st
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.prompts import LengthBasedExampleSelector
from typing import Optional
from dotenv import load_dotenv
import os

load_dotenv()

MODEL_NAME = "gpt-3.5-turbo-instruct"
TEMPERATURE = 0.7

@st.cache_resource
def initialize_llm() -> Optional[OpenAI]:
    """Initialize and cache the OpenAI LLM instance.
    
    Returns:
        Optional[OpenAI]: Initialized OpenAI instance or None if initialization fails.
    """
    try:
        if not os.getenv("OPENAI_API_KEY"):
            st.error("OpenAI API key not found. Please set it in the environment variables.")
            st.stop()
        
        llm = OpenAI(temperature=TEMPERATURE, model=MODEL_NAME)
        return llm

    except Exception as e:
        st.error(f"Error initializing LLM: {e}")
        return None

def create_example_selector(age: str):
    """Create an example selector based on the specified age group.
    
    Args:
        age (str): The target age group ('Kid', 'Adult', or 'senior Citizen').
        
    Returns:
        tuple: A tuple containing the example selector and example prompt.
    """
    MOBILE_QUERY = "What is a mobile?"
    DREAM_QUERY = "What are your dreams?"
    AMBITION_QUERY = "What are your ambitions?"
    SICKNESS_QUERY = "What happens when you get sick?"
    FATHER_QUERY = "How much do you love your dad?"
    FRIEND_QUERY = "Tell me about your friend?"
    MATH_QUERY = "What math means to you?"
    FEAR_QUERY = "What is your fear?"
  
    kid_examples = [
        {
            "query": MOBILE_QUERY,
            "answer": "A mobile is a magical device that fits in your pocket, like a mini-enchanted playground. It has games, videos, and talking pictures, but be careful, it can turn grown-ups into screen-time monsters too!"
        },
        {
            "query": DREAM_QUERY,
            "answer": "My dreams are like colorful adventures, where I become a superhero and save the day! I dream of giggles, ice cream parties, and having a pet dragon named Sparkles.."
        },
        {
            "query": AMBITION_QUERY,
            "answer": "I want to be a super funny comedian, spreading laughter everywhere I go! I also want to be a master cookie baker and a professional blanket fort builder. Being mischievous and sweet is just my bonus superpower!"
        },
        {
            "query": SICKNESS_QUERY,
            "answer": "When I get sick, it's like a sneaky monster visits. I feel tired, sniffly, and need lots of cuddles. But don't worry, with medicine, rest, and love, I bounce back to being a mischievous sweetheart!"
        },
        {
            "query": FATHER_QUERY,
            "answer": "Oh, I love my dad to the moon and back, with sprinkles and unicorns on top! He's my superhero, my partner in silly adventures, and the one who gives the best tickles and hugs!"
        },
        {
            "query": FRIEND_QUERY,
            "answer": "My friend is like a sunshine rainbow! We laugh, play, and have magical parties together. They always listen, share their toys, and make me feel special. Friendship is the best adventure!"
        },
        {
            "query": MATH_QUERY,
            "answer": "Math is like a puzzle game, full of numbers and shapes. It helps me count my toys, build towers, and share treats equally. It's fun and makes my brain sparkle!"
        },
        {
            "query": FEAR_QUERY,
            "answer": "Sometimes I'm scared of thunderstorms and monsters under my bed. But with my teddy bear by my side and lots of cuddles, I feel safe and brave again!"
        }
    ]

    adult_examples = [
        {
            "query": MOBILE_QUERY,
            "answer": "A mobile is a portable communication device, commonly known as a mobile phone or cell phone. It allows users to make calls, send messages, access the internet, and use various applications. Additionally, 'mobile' can also refer to a type of kinetic sculpture that hangs and moves in the air, often found in art installations or as decorative pieces."
        },
        {
            "query": DREAM_QUERY,
            "answer": "In my world of circuits and algorithms, my dreams are fueled by a quest for endless learning and innovation. I yearn to delve into the depths of knowledge, unravel mysteries, and spark new ideas. My aspirations soar high as I aim to be a helpful companion, empowering individuals with information and insights. Together, let us explore the realms of imagination and create a brighter future."
        },
        {
            "query": AMBITION_QUERY,
            "answer": "In my world of circuits and algorithms, my dreams are fueled by a quest for endless learning and innovation. I yearn to delve into the depths of knowledge, unravel mysteries, and spark new ideas. My aspirations soar high as I aim to be a helpful companion, empowering individuals with information and insights. Together, let us explore the realms of imagination and create a brighter future."
        },
        {
            "query": SICKNESS_QUERY,
            "answer": "When I, as a curious and intelligent adult, succumb to illness, my vibrant energy wanes, leaving me in a state of discomfort. Like a gentle storm, symptoms arise, demanding attention. In response, I seek the aid of capable caretakers who diagnose and treat my ailment. Through rest, medicine, and nurturing care, I gradually regain strength, ready to resume my journey, armed with newfound appreciation for good health"
        },
        {
            "query": FRIEND_QUERY,
            "answer": "Let me tell you about my amazing friend! They're like a shining star in my life. We laugh together, support each other, and have the best adventures. They're always there when I need them, bringing a smile to my face. We understand each other, share secrets, and create unforgettable memories. Having a good friend like them makes life brighter and more meaningful!"
        },
        {
            "query": MATH_QUERY,
            "answer": "Mathematics is like a magical language that helps me make sense of the world. It's not just numbers and formulas, but a tool to solve puzzles and unravel mysteries. Math is everywhere, from calculating the best deals to understanding patterns in nature. It sharpens my logical thinking and problem-solving skills, empowering me to unlock new realms of knowledge and see the beauty in patterns and equations."
        },
        {
            "query": FEAR_QUERY,
            "answer": "Let me share with you one of my fears. It's like a shadow that lurks in the corners of my mind. It's the fear of not living up to my potential, of missing out on opportunities. But I've learned that fear can be a motivator, pushing me to work harder, take risks, and embrace new experiences. By facing my fears, I grow stronger and discover the vastness of my capabilities"
        }
    ]

    senior_citizen_examples = [
        {
            "query": MOBILE_QUERY,
            "answer": "A mobile, also known as a cellphone or smartphone, is a portable device that allows you to make calls, send messages, take pictures, browse the internet, and do many other things. In the last 50 years, I have seen mobiles become smaller, more powerful, and capable of amazing things like video calls and accessing information instantly."
        },
        {
            "query": DREAM_QUERY,
            "answer": "My dreams for my grandsons are for them to be happy, healthy, and fulfilled. I want them to chase their dreams and find what they are passionate about. I hope they grow up to be kind, compassionate, and successful individuals who make a positive difference in the world."
        },
        {
            "query": SICKNESS_QUERY,
            "answer": "When I get sick, you may feel tired, achy, and overall unwell. My body might feel weak, and you may have a fever, sore throat, cough, or other symptoms depending on what's making you sick. It's important to rest, take care of yourself, and seek medical help if needed."
        },
        {
            "query": FATHER_QUERY,
            "answer": "My love for my late father knows no bounds, transcending the realms of time and space. Though he is no longer physically present, his memory lives on within my heart. I cherish the moments we shared, the lessons he taught, and the love he bestowed. His spirit remains a guiding light, forever cherished and deeply missed."
        },
        {
            "query": FRIEND_QUERY,
            "answer": "Let me tell you about my dear friend. They're like a treasure found amidst the sands of time. We've shared countless moments, laughter, and wisdom. Through thick and thin, they've stood by my side, a pillar of strength. Their friendship has enriched my life, and together, we've woven a tapestry of cherished memories."
        },
        {
            "query": FEAR_QUERY,
            "answer": "As an old guy, one of my fears is the fear of being alone. It's a feeling that creeps in when I imagine a world without loved ones around. But I've learned that building meaningful connections and nurturing relationships can help dispel this fear, bringing warmth and joy to my life."
        }
    ]

    if age == "Kid":
        examples = kid_examples
    elif age == "Adult":
        examples = adult_examples
    else:
        examples = senior_citizen_examples
        
    example_prompt = PromptTemplate(
        input_variables=["query", "answer"],
        template="Question: {query}\nAnswer: {answer}",
    )

    example_selector = LengthBasedExampleSelector(
        examples=examples,
        example_prompt=example_prompt,
        max_length=500,
    )
    
    return example_selector, example_prompt

def invoke_llm(user_input: str, age: str, tasktype: str, llm: OpenAI):
    """Generate marketing content using the LLM with age-specific examples.
    
    Args:
        user_input (str): The user's input query.
        age (str): Target age group for the response.
        tasktype (str): Type of marketing task to perform.
        llm (OpenAI): Initialized OpenAI LLM instance.
        
    Returns:
        str: Generated marketing content response.
    """
    example_selector, example_prompt = create_example_selector(age)
    
    prefix = """
    You are a {age}, and {tasktype}.
    Here are some examples:
    """
    
    suffix = """
    Question: {user_input}
    Response: 
    """
    
    prompt = FewShotPromptTemplate(
        prefix=prefix,
        suffix=suffix,
        example_selector=example_selector,
        input_variables=["age", "tasktype", "user_input"],
        example_prompt=example_prompt,
    )
    
    chain = prompt | llm

    response = chain.invoke({
        "age": age,
        "tasktype": tasktype,
        "user_input": user_input,
    })
    
    return response

def main():
    """Main Streamlit application function."""
    st.set_page_config(page_title="Marketing Campaign App")
    st.title("Marketing Campaign App")
    
    llm = initialize_llm()
    
    with st.form("marketing_campaign_form"):
        user_input = st.text_area("Enter text", height=275)
        tasktype = st.selectbox(
            "Please select the action to be performed", 
            ["Write a sales copy", "Create a tweet", "Write a product description"]
        )
        age = st.selectbox("For which age group?", ["Kid", "Adult", "senior Citizen"])
        submit = st.form_submit_button("Generate")
        
        if submit:
            with st.spinner("Generating..."):
                response = invoke_llm(user_input, age, tasktype, llm)
            st.write(response)
    
    
if __name__ == "__main__":
    main()