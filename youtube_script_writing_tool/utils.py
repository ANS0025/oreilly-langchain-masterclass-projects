from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.chains import LLMChain
import logging
from typing import Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

def get_youtube_script(topic: str, video_length: int, creativity: float) -> Tuple[str, str, str]:
    """
    Generates a YouTube video script, title, and relevant search data.

    Args:
        topic: The subject of the video.
        video_length: The desired length of the video in minutes.
        creativity: The creativity level (temperature) for the language model, from 0.0 to 1.0.

    Returns:
        A tuple containing the generated title, script, and search data.

    Raises:
        ValueError: If any of the input parameters are invalid.
        Exception: For errors during the generation process.
    """
    if not all([isinstance(topic, str), topic]):
        raise ValueError("Topic must be a non-empty string.")
    if not isinstance(video_length, (int, float)) or video_length <= 0:
        raise ValueError("Video length must be a positive number.")
    if not isinstance(creativity, (int, float)) or not 0.0 <= creativity <= 1.0:
        raise ValueError("Creativity must be a float between 0.0 and 1.0.")

    try:
        llm = OpenAI(temperature=creativity, model_name="gpt-3.5-turbo-instruct")

        title_prompt = PromptTemplate.from_template(
            "Write a compelling YouTube video title for the topic: {topic}"
        )
        script_prompt = PromptTemplate.from_template(
            "Write a detailed script for a {video_length}-minute YouTube video on the topic: {topic}. "
            "Incorporate the following search data for context: {search_data}"
        )

        title_chain = LLMChain(llm=llm, prompt=title_prompt)
        script_chain = LLMChain(llm=llm, prompt=script_prompt)

        search = DuckDuckGoSearchRun()
        search_data = search.invoke(topic)

        title = title_chain.run(topic=topic)
        script = script_chain.run(topic=topic, video_length=video_length, search_data=search_data)

        return title.strip(), script.strip(), search_data

    except Exception as e:
        logger.error(f"An error occurred during script generation: {e}")
        raise