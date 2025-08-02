import os
import itertools
from dotenv import load_dotenv

load_dotenv()

API_KEYS = os.getenv("GEMINI_API_KEYS", "").split(",")
API_KEYS = [k.strip() for k in API_KEYS if k.strip()]

if not API_KEYS:
    raise RuntimeError("Missing environment variable: GEMINI_API_KEYS")

_key_cycle = itertools.cycle(API_KEYS)

def get_next_api_key():
    return next(_key_cycle)
