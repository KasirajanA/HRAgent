"""Central configuration. Every value can be overridden with an environment variable (or a .env file)."""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# --- Ollama -----------------------------------------------------------------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen2.5:7b")          # must support tool calling
# all-minilm: 22M params, 384-dim, ~46 MB. Smallest Ollama embedding model; good enough for short
# policy chunks. Swap for nomic-embed-text / snowflake-arctic-embed:33m if retrieval quality matters more.
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-minilm")
CHUNK_EMBED_MODEL = os.getenv("CHUNK_EMBED_MODEL", EMBED_MODEL)   # used only to find semantic boundaries
CHAT_TEMPERATURE = float(os.getenv("CHAT_TEMPERATURE", "0.1"))
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "60m")   # keep the chat model loaded between questions
NUM_CTX = int(os.getenv("NUM_CTX", "4096"))                  # context window; larger = more RAM
WARM_UP = os.getenv("WARM_UP", "true").lower() == "true"     # pre-load model + prompt cache at startup

# --- Qdrant -----------------------------------------------------------------
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")   # ":memory:" = embedded, non-persistent
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "hr_policies")

# --- RAG --------------------------------------------------------------------
POLICY_DIR = Path(os.getenv("POLICY_DIR", BASE_DIR / "data" / "policies"))
CHUNK_MIN_CHARS = int(os.getenv("CHUNK_MIN_CHARS", "200"))
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "1000"))   # all-minilm reads ~256 tokens ≈ 1000 chars
CHUNK_WINDOW = int(os.getenv("CHUNK_WINDOW", "1"))            # neighbour sentences mixed into each embedding
CHUNK_BREAKPOINT_PERCENTILE = float(os.getenv("CHUNK_BREAKPOINT_PERCENTILE", "90"))
UPLOAD_MAX_MB = int(os.getenv("UPLOAD_MAX_MB", "10"))
ALLOWED_EXTENSIONS = {".md", ".txt", ".pdf", ".docx"}
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "4"))
RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.35"))

# --- HR database / auth -----------------------------------------------------
DB_PATH = Path(os.getenv("DB_PATH", BASE_DIR / "data" / "hr.db"))
SESSION_TTL_HOURS = int(os.getenv("SESSION_TTL_HOURS", "8"))

# --- Agent ------------------------------------------------------------------
MAX_AGENT_STEPS = int(os.getenv("MAX_AGENT_STEPS", "6"))      # max LLM <-> tool round trips per question
GROUNDING_NUDGE = os.getenv("GROUNDING_NUDGE", "true").lower() == "true"  # re-ask once if no tool was used
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "12"))
COMPANY_NAME = os.getenv("COMPANY_NAME", "Acme Corp")
HR_CONTACT = os.getenv("HR_CONTACT", "hr@acme.example")
