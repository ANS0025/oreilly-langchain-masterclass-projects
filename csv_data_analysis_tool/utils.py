"""
This module provides the core functionality for the CSV data analysis tool.

It uses the langchain experimental library to create a pandas dataframe agent
that can answer questions about a given CSV file.
"""

from typing import IO

import pandas as pd
from dotenv import load_dotenv
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import OpenAI

load_dotenv()


def query_agent(uploaded_file: IO[bytes], query: str) -> str:
    """
    Queries the agent with the given query.

    Args:
        uploaded_file: The uploaded CSV file.
        query: The user's query.

    Returns:
        The agent's response.
    """
    df = pd.read_csv(uploaded_file)
    llm = OpenAI(temperature=0)
    agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)
    return agent.run(query)