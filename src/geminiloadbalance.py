import os
import itertools
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("api_logs.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

API_KEYS = os.getenv("GEMINI_API_KEYS", "").split(",")
API_KEYS = [k.strip() for k in API_KEYS if k.strip()]

if not API_KEYS:
    raise RuntimeError("Missing environment variable: GEMINI_API_KEYS")

_key_cycle = itertools.cycle(API_KEYS)

def get_next_api_key():
    key = next(_key_cycle)
    logger.info(f"Using API key: {key[:4]}...{key[-4:]}")
    return key
