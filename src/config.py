import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"

# Create directories if they don't exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Model Configs
ZERO_SHOT_MODEL = "facebook/bart-large-mnli"
EMBEDDING_MODEL = "text-embedding-3-small" # or "text-embedding-ada-002"

# Dataset
KAGGLE_DATASET = "dylanjcastillo/7k-books-with-metadata"
