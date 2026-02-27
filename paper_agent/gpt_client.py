import os
import sys
import openai
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constant import COMPLETION_MODEL, CHEEP_MODEL

load_dotenv()


class GPTClient:
    def __init__(self, model: str = None):
        self.model = model if model else CHEEP_MODEL
        openai.api_key = os.getenv("OPENAI_API_KEY")

    async def chat(self, prompt: str, system_prompt: str = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"
