import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Azure Storage settings
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_STORAGE_CONTAINER = os.getenv("AZURE_STORAGE_CONTAINER", "documents")

# PostgreSQL settings
POSTGRES_CONNECTION_STRING = os.getenv("POSTGRES_CONNECTION_STRING")

# OpenAI settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Storage settings
STORAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "storage")
Path(STORAGE_DIR).mkdir(parents=True, exist_ok=True)

# Embedding settings
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536

# Chunking settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MAX_CHUNKS_PER_BATCH = 10

# Query settings
MAX_RESULTS = 5
SIMILARITY_CUTOFF = 0.7

# OpenAI Configuration
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Azure Storage Configuration - temporarily disabled
# if not AZURE_STORAGE_CONNECTION_STRING or not AZURE_STORAGE_CONTAINER:
#     raise ValueError("Azure Storage configuration missing. Please set AZURE_STORAGE_CONNECTION_STRING and AZURE_STORAGE_CONTAINER")

# Storage paths
UPLOAD_DIR = "./uploads"

# Make sure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True) 