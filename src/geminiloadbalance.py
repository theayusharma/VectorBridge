import asyncio
import itertools
import logging
import os

import google.generativeai as genai
from dotenv import load_dotenv
from google.api_core.exceptions import ResourceExhausted

load_dotenv()

logger = logging.getLogger(__name__)


class LLMGenerationError(Exception):
    pass


class GeminiAPIManager:
    def __init__(self) -> None:
        self.keys = self._load_keys()
        if not self.keys:
            raise RuntimeError(
                "Missing or empty 'GEMINI_API_KEYS' environment variable."
            )
        self._key_cycle = itertools.cycle(self.keys)
        logger.info(f"Initialized API Key Manager with {len(self.keys)} keys.")

    # get keys from env
    def _load_keys(self) -> list[str]:
        api_keys = os.getenv("GEMINI_API_KEYS", "").split(',')
        return [k.strip() for k in api_keys if k.strip()]

    def get_next_key(self) -> str:
        key = next(self._key_cycle)
        logger.info(f"Using API key ending in: ...{key[-4:]}")
        return key


api_key_manager = GeminiAPIManager()


async def generate_text_with_retry(
    prompt: str,
    model_name: str,
    max_retries: int,
    retry_delay: int = 1,
) -> str:
    last_exception = None
    for attempt in range(max_retries + 1):  # +1 for the initial attempt
        api_key = api_key_manager.get_next_key()
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name)
            response = await model.generate_content_async(prompt)
            logger.info("LLM generation successful.")
            return response.text.strip()

        except ResourceExhausted as e:
            last_exception = e
            logger.warning(
                f"Rate limit error with key ...{api_key[-4:]}. "
                f"Attempt {attempt + 1}/{max_retries + 1}. Retrying in {retry_delay}s..."
            )
            if attempt < max_retries:
                await asyncio.sleep(retry_delay)

        except Exception as e:
            last_exception = e
            logger.error(f"An unexpected error occurred during LLM generation: {e}")

            #break
            if attempt < max_retries:
                await asyncio.sleep(retry_delay)

    raise LLMGenerationError(
        f"LLM generation failed after {max_retries + 1} attempts. "
        f"Last error: {last_exception}"
    )
