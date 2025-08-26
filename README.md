# LangChain Masterclass Projects

This repository contains hands-on projects from the O'Reilly course: **"LangChain Masterclass - Build 15 OpenAI and LLAMA 2 LLM Apps Using Python"**.

## Course Overview

This masterclass focuses on building practical applications using LangChain, OpenAI, and LLAMA 2 models. Each project demonstrates different aspects of large language model integration and development.

## Projects

### 1. Simple Question Answering App
- **Directory**: `simple-question-answering-app/`
- **Description**: Basic question-answering application using LangChain
- **Technologies**: Python, LangChain

*Additional projects will be added as the course progresses.*

## Setup Instructions

Each project has its own virtual environment and dependencies. To run a specific project:

1. Navigate to the project directory:
   ```bash
   cd project-name/
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your API keys:
   ```bash
   cp .env.example .env  # If example exists
   # Edit .env with your actual API keys
   ```

5. Run the application:
   ```bash
   python app.py
   ```

## Environment Variables

Each project may require different environment variables. Common ones include:
- `OPENAI_API_KEY` - Your OpenAI API key
- `LLAMA_API_KEY` - Your LLAMA 2 API key (if applicable)

## Requirements

- Python 3.8+
- Virtual environment (venv or conda)
- API keys for respective services

## Course Information

- **Platform**: O'Reilly Learning
- **Course**: LangChain Masterclass - Build 15 OpenAI and LLAMA 2 LLM Apps Using Python
- **Focus**: Practical LLM application development with LangChain

## Notes

- Each project is self-contained with its own dependencies
- Virtual environments (`.venv/`) and environment files (`.env`) are gitignored
- Follow the setup instructions for each individual project