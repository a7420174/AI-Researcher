import os
import sys
import asyncio
import time
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constant import COMPLETION_MODEL, CHEEP_MODEL

load_dotenv()


class GPTClient:
    def __init__(
        self, model: str = None, max_retries: int = 5, base_delay: float = 1.0
    ):
        self.model = model if model else COMPLETION_MODEL
        self.max_retries = max_retries
        self.base_delay = base_delay
        try:
            from litellm import acompletion

            self._completion = acompletion
        except ImportError:
            self._completion = None

    async def chat(self, prompt: str, system_prompt: str = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        if not self._completion:
            return "Error: litellm not available"

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = await self._completion(
                    model=self.model,
                    messages=messages,
                    temperature=0.7,
                )
                return response.choices[0].message.content
            except Exception as e:
                error_str = str(e)
                last_error = error_str

                if "rate_limit" in error_str.lower() or "429" in error_str:
                    delay = self.base_delay * (2**attempt)
                    print(
                        f"Rate limit hit, retrying in {delay}s (attempt {attempt + 1}/{self.max_retries})"
                    )
                    await asyncio.sleep(delay)
                    continue
                elif "timeout" in error_str.lower():
                    delay = self.base_delay * (2**attempt)
                    print(
                        f"Timeout, retrying in {delay}s (attempt {attempt + 1}/{self.max_retries})"
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    return f"Error: {error_str}"

        return f"Error after {self.max_retries} retries: {last_error}"
