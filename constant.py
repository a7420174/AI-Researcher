import os
from dotenv import load_dotenv
import global_state

load_dotenv()


def str_to_bool(value):
    """convert string to bool"""
    true_values = {"true", "yes", "1", "on", "t", "y"}
    false_values = {"false", "no", "0", "off", "f", "n"}

    if isinstance(value, bool):
        return value

    if not value:
        return False

    value = str(value).lower().strip()
    if value in true_values:
        return True
    if value in false_values:
        return False
    return True


DOCKER_WORKPLACE_NAME = os.getenv("DOCKER_WORKPLACE_NAME", "workplace_meta")
GITHUB_AI_TOKEN = os.getenv("GITHUB_AI_TOKEN", None)
AI_USER = os.getenv("AI_USER", "ai-sin")
LOCAL_ROOT = os.getenv("LOCAL_ROOT", os.getcwd())
PLATFORM = os.getenv("PLATFORM", "linux/amd64")

DEBUG = str_to_bool(os.getenv("DEBUG", True))

DEFAULT_LOG = str_to_bool(os.getenv("DEFAULT_LOG", True))
LOG_PATH = os.getenv("LOG_PATH", None)
LOG_PATH = global_state.LOG_PATH
EVAL_MODE = str_to_bool(os.getenv("EVAL_MODE", False))
BASE_IMAGES = os.getenv("BASE_IMAGES", "tjbtech1/paperapp:latest")

COMPLETION_MODEL = os.getenv("COMPLETION_MODEL", "gpt-4o-2024-08-06")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHEEP_MODEL = os.getenv("CHEEP_MODEL", "gpt-4o-mini-2024-07-18")

GPUS = os.getenv("GPUS", "")

FN_CALL = str_to_bool(os.getenv("FN_CALL", True))
API_BASE_URL = os.getenv("API_BASE_URL", None)
ADD_USER = str_to_bool(os.getenv("ADD_USER", False))

NON_FN_CALL = str_to_bool(os.getenv("NON_FN_CALL", False))

NOT_SUPPORT_SENDER = ["mistral", "groq"]


MUST_ADD_USER = ["deepseek/deepseek-reasoner", "o1-mini"]
NOT_SUPPORT_FN_CALL = ["o1-mini", "deepseek/deepseek-reasoner"]
NOT_USE_FN_CALL = ["deepseek/deepseek-chat"] + NOT_SUPPORT_FN_CALL

if EVAL_MODE:
    DEFAULT_LOG = False


MODULE_DESCRIPTIONS = {
    "Detailed Idea Description": "At this level, users provide comprehensive descriptions of their specific research ideas. The system processes these detailed inputs to develop implementation strategies based on the user's explicit requirements. Examples 1-2 are the templates of this mode.",
    "Reference-Based Ideation": "This simpler level involves users submitting reference papers without a specific idea in mind. The user query typically follows the format: 'I have some reference papers, please come up with an innovative idea and implement it with these papers.' The system then analyzes the provided references to generate and develop novel research concepts. Examples 3-4 are the templates of this mode.",
    "Paper Generation Agent": "Once all research and experimental work is finished, employ this agent for paper generation",
    "Deep Research": "Use this mode for comprehensive web-based research on any topic. No ML implementation or code execution required. Enter your research question and get detailed findings with sources.",
}


DEFAULT_ENV_TEMPLATE = """#===========================================
# MODEL & API 
# (See https://docs.camel-ai.org/key_modules/models.html#)
#===========================================

# OPENAI API (https://platform.openai.com/api-keys)
OPENAI_API_KEY='Your_Key'
# OPENAI_API_BASE_URL=""

# Azure OpenAI API
# AZURE_OPENAI_BASE_URL=""
# AZURE_API_VERSION=""
# AZURE_OPENAI_API_KEY=""
# AZURE_DEPLOYMENT_NAME=""


# Qwen API (https://help.aliyun.com/zh/model-studio/developer-reference/get-api-key)
QWEN_API_KEY='Your_Key'

# DeepSeek API (https://platform.deepseek.com/api_keys)
DEEPSEEK_API_KEY='Your_Key'

#===========================================
# Tools & Services API
#===========================================

# Google Search API (https://coda.io/@jon-dallas/google-image-search-pack-example/search-engine-id-and-google-api-key-3)
GOOGLE_API_KEY='Your_Key'
SEARCH_ENGINE_ID='Your_ID'

# Chunkr API (https://chunkr.ai/)
CHUNKR_API_KEY='Your_Key'

# Firecrawl API (https://www.firecrawl.dev/)
FIRECRAWL_API_KEY='Your_Key'
#FIRECRAWL_API_URL="https://api.firecrawl.dev"
"""

STOP_WORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "from",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "must",
    "shall",
    "can",
    "need",
    "dare",
    "ought",
    "used",
    "it",
    "its",
    "this",
    "that",
    "these",
    "those",
    "i",
    "you",
    "he",
    "she",
    "we",
    "they",
    "what",
    "which",
    "who",
    "whom",
    "whose",
    "where",
    "when",
    "why",
    "how",
    "all",
    "each",
    "every",
    "both",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "not",
    "only",
    "same",
    "so",
    "than",
    "too",
    "very",
    "just",
    "also",
    "now",
    "please",
    "summary",
    "status",
    "current",
    "development",
    "research",
    "review",
    "overview",
    "analysis",
    "table",
    "include",
    "information",
    "regarding",
}
