import os
from pathlib import Path
from dotenv import load_dotenv


def load_config():
    # Look for .env in project root (parent of app/ folder)
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path)


def get_groq_api():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY not found in .env file")

    return api_key